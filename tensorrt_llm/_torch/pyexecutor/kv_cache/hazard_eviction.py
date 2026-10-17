# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversation-aware KV eviction for KVCacheManagerV2 (opt-in: ``TLLM_KV_EVICT_HAZARD=<table.json>``).

With LRU the pages that leave GPU/host memory first are those of the conversation that has been idle the
longest. In agentic traffic that is often a conversation about to come back (a user after a long think time),
while the recently released pages of finished sub-agents stay resident. Here every request the KV cache manager
releases is classified into a *cell* -- sub-agent or main conversation x how long the conversation was idle before
this turn -- and the C++ eviction queues keep one LRU list per cell. They evict the list head with the lowest
hazard, the probability (learned offline from a trace replay) that a conversation of that cell, idle that long,
comes back within the table's horizon. Without the variable nothing here runs and eviction is plain LRU.

Every tensor-parallel rank has to make the same eviction choices, so no rank-local clock is used: rank 0 stamps a
request when it is enqueued (``py_kv_evict_time``, carried by the request broadcast), the eviction clock is the
latest stamp seen, and cells depend only on the request stream.

Table format (JSON)::

    {"format": "trtllm-kv-evict-hazard-v1", "step_s": 10.0, "horizon_s": 60.0,
     "features": ["sub", "gap"], "gap_edges_s": [5, 30, 120],
     "rows": {"*": [...], "sub=0|gap=first": [...], "sub=1|gap=<5": [...], ...}}

``rows[key][b]`` is the hazard after ``b * step_s`` seconds idle (the last entry covers everything beyond). ``"*"``
is the unclassified cell (requests without a conversation id, unknown keys). Keys join ``sub=<0|1>`` and
``gap=<bucket>`` for the listed features; gap buckets are ``first`` (no earlier turn seen) and ``<e0``, ``e0-e1``,
..., ``>eN`` from ``gap_edges_s``.
"""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

ENV_TABLE = "TLLM_KV_EVICT_HAZARD"
TABLE_FORMAT = "trtllm-kv-evict-hazard-v1"
UNCLASSIFIED = "*"
KNOWN_FEATURES = ("sub", "gap")


def hazard_eviction_enabled() -> bool:
    """Whether the opt-in is set (rank 0 then stamps requests for the eviction clock)."""
    return bool(os.environ.get(ENV_TABLE, "").strip())


def _fmt(edge: float) -> str:
    return f"{edge:g}"


def gap_bucket(prev_gap: Optional[float], edges: Sequence[float]) -> str:
    """Bucket label of the idle time before a turn (``first`` when there was no earlier turn)."""
    if prev_gap is None:
        return "first"
    for i, edge in enumerate(edges):
        if prev_gap < edge:
            return f"<{_fmt(edge)}" if i == 0 else f"{_fmt(edges[i - 1])}-{_fmt(edge)}"
    return f">{_fmt(edges[-1])}"


@dataclass(frozen=True)
class HazardTable:
    step_s: float
    features: Tuple[str, ...]
    gap_edges_s: Tuple[float, ...]
    # Row i is cell i; row 0 is the unclassified cell.
    rows: List[List[float]]
    cell_ids: Dict[str, int]

    @classmethod
    def from_dict(cls, data: dict) -> "HazardTable":
        if data.get("format") != TABLE_FORMAT:
            raise ValueError(f"KV eviction hazard table: format must be {TABLE_FORMAT!r}")
        step_s = float(data["step_s"])
        if not step_s > 0:
            raise ValueError("KV eviction hazard table: step_s must be positive")
        features = tuple(data.get("features", ()))
        unknown = [f for f in features if f not in KNOWN_FEATURES]
        if unknown:
            raise ValueError(f"KV eviction hazard table: unknown features {unknown}")
        edges = tuple(float(e) for e in data.get("gap_edges_s", ()))
        if "gap" in features and (not edges or list(edges) != sorted(set(edges))):
            raise ValueError(
                "KV eviction hazard table: gap_edges_s must be increasing and non-empty"
            )
        raw_rows = data["rows"]
        if UNCLASSIFIED not in raw_rows:
            raise ValueError(
                f"KV eviction hazard table: rows need the unclassified cell {UNCLASSIFIED!r}"
            )
        keys = [UNCLASSIFIED] + sorted(k for k in raw_rows if k != UNCLASSIFIED)
        rows = []
        for key in keys:
            row = [float(v) for v in raw_rows[key]]
            if not row:
                raise ValueError(f"KV eviction hazard table: row {key!r} is empty")
            rows.append(row)
        return cls(
            step_s=step_s,
            features=features,
            gap_edges_s=edges,
            rows=rows,
            cell_ids={key: i for i, key in enumerate(keys)},
        )

    @classmethod
    def load(cls, path: str) -> "HazardTable":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def cell_key(self, is_sub: bool, prev_gap: Optional[float]) -> str:
        parts = []
        if "sub" in self.features:
            parts.append(f"sub={int(is_sub)}")
        if "gap" in self.features:
            parts.append(f"gap={gap_bucket(prev_gap, self.gap_edges_s)}")
        return "|".join(parts) if parts else UNCLASSIFIED

    def cell(self, is_sub: bool, prev_gap: Optional[float]) -> int:
        return self.cell_ids.get(self.cell_key(is_sub, prev_gap), 0)


def _conversation_id(request) -> Optional[str]:
    params = getattr(request, "py_conversation_params", None)
    conversation_id = getattr(params, "conversation_id", None)
    if not conversation_id:
        return None
    conversation_id = str(conversation_id).strip()
    return conversation_id or None


class ConversationEvictionHints:
    """Eviction clock and per-conversation state behind the cells; one per KV cache manager.

    The state is a pure function of the request stream (conversation ids, sub-agent marks and rank-0 stamps), so
    every rank that sees the same requests produces the same hints.
    """

    def __init__(self, table: HazardTable, max_conversations: int = 1 << 16) -> None:
        self.table = table
        self.clock = 0.0
        self._max_conversations = max_conversations
        # conversation id -> eviction clock when its previous turn was released (oldest first)
        self._last_release: "OrderedDict[str, float]" = OrderedDict()
        # Releases per cell since the last summary() (diagnostics only).
        self._cell_counts = [0] * len(table.rows)

    def observe(self, request) -> bool:
        """Advance the clock to the request's rank-0 stamp. Returns whether the clock moved."""
        stamp = getattr(request, "py_kv_evict_time", None)
        if stamp is None or not stamp > self.clock:
            return False
        self.clock = float(stamp)
        return True

    def release_hint(self, request) -> Optional[Tuple[int, float]]:
        """(cell, release time) for the pages a finished request releases; None for dummy requests."""
        if getattr(request, "is_dummy_request", False) or getattr(request, "is_dummy", False):
            return None
        self.observe(request)
        conversation_id = _conversation_id(request)
        if conversation_id is None:
            self._cell_counts[0] += 1
            return 0, self.clock
        arrival = getattr(request, "py_kv_evict_time", None)
        last = self._last_release.pop(conversation_id, None)
        prev_gap = None if last is None or arrival is None else max(0.0, float(arrival) - last)
        is_sub = getattr(request, "py_subagent_affinity_id", None) is not None
        cell = self.table.cell(is_sub, prev_gap)
        self._cell_counts[cell] += 1
        self._last_release[conversation_id] = self.clock
        while len(self._last_release) > self._max_conversations:
            self._last_release.popitem(last=False)
        return cell, self.clock

    @property
    def num_releases(self) -> int:
        return sum(self._cell_counts)

    def summary(self) -> str:
        """Releases per cell since the previous summary, then reset the counters."""
        names = {i: key for key, i in self.table.cell_ids.items()}
        counts = ", ".join(f"{names[i]}: {n}" for i, n in enumerate(self._cell_counts) if n)
        self._cell_counts = [0] * len(self.table.rows)
        return f"clock {self.clock:.1f}, {len(self._last_release)} conversations, releases by cell {{{counts}}}"


def load_hazard_eviction_from_env() -> Optional[ConversationEvictionHints]:
    path = os.environ.get(ENV_TABLE, "").strip()
    if not path:
        return None
    return ConversationEvictionHints(HazardTable.load(path))
