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
"""Python side of the opt-in conversation-aware KV eviction (TLLM_KV_EVICT_HAZARD): table parsing, cells, the
rank-0 clock and its plumbing. The C++ ranking itself is covered by
tests/unittest/kv_cache_manager_v2_tests/test_eviction_hazard.py."""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import hazard_eviction as he
from tensorrt_llm._torch.pyexecutor.request_utils import attach_py_objects_to_requests

ROWS = {
    "*": [0.5, 0.4],
    "sub=0|gap=first": [0.9, 0.8],
    "sub=0|gap=<5": [0.95, 0.9],
    "sub=1|gap=first": [0.3, 0.2],
    "sub=1|gap=>120": [0.1, 0.05],
}


def _table(**overrides) -> dict:
    data = {
        "format": he.TABLE_FORMAT,
        "step_s": 10.0,
        "horizon_s": 60.0,
        "features": ["sub", "gap"],
        "gap_edges_s": [5, 30, 120],
        "rows": ROWS,
    }
    data.update(overrides)
    return data


def _request(conv=None, stamp=None, sub=None, dummy=False):
    params = None if conv is None else SimpleNamespace(conversation_id=conv)
    return SimpleNamespace(
        py_conversation_params=params,
        py_kv_evict_time=stamp,
        py_subagent_affinity_id=sub,
        is_dummy_request=dummy,
    )


@pytest.mark.parametrize(
    "gap,label",
    [
        (None, "first"),
        (0.0, "<5"),
        (4.99, "<5"),
        (5.0, "5-30"),
        (29.0, "5-30"),
        (30.0, "30-120"),
        (120.0, ">120"),
        (1e6, ">120"),
    ],
)
def test_gap_bucket_matches_the_simulator_labels(gap, label):
    assert he.gap_bucket(gap, (5.0, 30.0, 120.0)) == label


def test_table_cells():
    table = he.HazardTable.from_dict(_table())
    # Row 0 is the unclassified cell, the rest in sorted key order.
    assert table.rows[0] == ROWS["*"]
    assert sorted(table.cell_ids) == sorted(ROWS)
    assert table.cell_ids["*"] == 0
    first_main = table.cell(False, None)
    assert table.rows[first_main] == ROWS["sub=0|gap=first"]
    assert table.rows[table.cell(True, 500.0)] == ROWS["sub=1|gap=>120"]
    # A cell the training run never saw is unclassified.
    assert table.cell(False, 60.0) == 0


def test_table_without_sub_feature():
    rows = {"*": [0.5], "gap=first": [0.7], "gap=<5": [0.9]}
    table = he.HazardTable.from_dict(_table(features=["gap"], rows=rows))
    assert table.cell_key(True, 1.0) == "gap=<5"
    assert table.rows[table.cell(True, 1.0)] == [0.9]


@pytest.mark.parametrize(
    "overrides",
    [
        {"format": "other"},
        {"step_s": 0},
        {"features": ["sub", "osl"]},
        {"gap_edges_s": [30, 5]},
        {"rows": {"sub=0|gap=first": [0.5]}},
        {"rows": {"*": []}},
    ],
)
def test_table_rejects_bad_input(overrides):
    with pytest.raises(ValueError):
        he.HazardTable.from_dict(_table(**overrides))


def test_load_from_env(tmp_path, monkeypatch):
    path = tmp_path / "hazard.json"
    path.write_text(json.dumps(_table()))
    monkeypatch.delenv(he.ENV_TABLE, raising=False)
    assert not he.hazard_eviction_enabled()
    assert he.load_hazard_eviction_from_env() is None
    monkeypatch.setenv(he.ENV_TABLE, str(path))
    assert he.hazard_eviction_enabled()
    hints = he.load_hazard_eviction_from_env()
    assert hints is not None and len(hints.table.rows) == len(ROWS)


def test_clock_follows_rank0_stamps_and_never_goes_back():
    hints = he.ConversationEvictionHints(he.HazardTable.from_dict(_table()))
    assert hints.observe(_request("a", stamp=100.0))
    assert not hints.observe(_request("b", stamp=90.0))
    assert not hints.observe(_request("c", stamp=None))
    assert hints.clock == 100.0


def test_release_hints_follow_the_conversation():
    table = he.HazardTable.from_dict(_table())
    hints = he.ConversationEvictionHints(table)
    # First turn of a main conversation.
    assert hints.release_hint(_request("main", stamp=100.0)) == (table.cell(False, None), 100.0)
    # It comes back 3 s after that release: gap <5.
    assert hints.release_hint(_request("main", stamp=103.0)) == (table.cell(False, 3.0), 103.0)
    # A sub-agent (its parent session id is set) that returns after 200 s.
    assert hints.release_hint(_request("sub", stamp=110.0, sub="main"))[0] == table.cell(True, None)
    assert hints.release_hint(_request("sub", stamp=310.0, sub="main")) == (
        table.cell(True, 200.0),
        310.0,
    )
    # No conversation id: unclassified, stamped with the clock.
    assert hints.release_hint(_request(None, stamp=None)) == (0, 310.0)
    # Dummy requests release nothing worth classifying.
    assert hints.release_hint(_request("x", stamp=400.0, dummy=True)) is None
    # Diagnostics: releases per cell, reset by summary().
    assert hints.num_releases == 5
    text = hints.summary()
    assert "sub=1|gap=>120: 1" in text and "*: 1" in text and "2 conversations" in text
    assert hints.num_releases == 0


def test_conversation_state_is_bounded():
    hints = he.ConversationEvictionHints(he.HazardTable.from_dict(_table()), max_conversations=2)
    for i, conv in enumerate("abc"):
        hints.release_hint(_request(conv, stamp=float(i)))
    # "a" was dropped (oldest): its next turn counts as a first turn again.
    table = hints.table
    assert hints.release_hint(_request("a", stamp=3.0))[0] == table.cell(False, None)
    assert hints.release_hint(_request("c", stamp=3.5))[0] == table.cell(False, 1.5)


def test_identical_streams_give_identical_hints():
    table = he.HazardTable.from_dict(_table())
    stream = [
        ("a", 1.0, None),
        ("b", 2.0, "a"),
        ("a", 4.0, None),
        ("b", 300.0, "a"),
        (None, 301.0, None),
    ]
    out = []
    for _ in range(2):
        hints = he.ConversationEvictionHints(table)
        out.append([hints.release_hint(_request(c, stamp=t, sub=s)) for c, t, s in stream])
    assert out[0] == out[1]


def test_stamp_travels_with_the_request_broadcast():
    """Rank 0's stamp is one of the Python-only request objects that are broadcast and re-attached."""
    from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster

    # (Conversation params are collected for every request, so the attribute must exist.)
    request = SimpleNamespace(py_kv_evict_time=12.5, py_conversation_params=None)
    item = SimpleNamespace(id=7, request=request, is_normal_request=True)
    collected = RequestBroadcaster._collect_py_objects(None, [item])
    assert ("py_kv_evict_time", {7: 12.5}) in collected
    receiver = SimpleNamespace(id=7, request=SimpleNamespace(), is_normal_request=True)
    attach_py_objects_to_requests([receiver], collected)
    assert receiver.request.py_kv_evict_time == 12.5


def test_hint_only_subagent_affinity_routes_by_conversation(monkeypatch):
    from tensorrt_llm.serve.conversation_id import get_request_routing_id

    request = SimpleNamespace(
        conversation_params=SimpleNamespace(conversation_id="child", subagent_affinity_id="parent")
    )
    monkeypatch.delenv("TLLM_SUBAGENT_AFFINITY_HINT_ONLY", raising=False)
    assert get_request_routing_id(request) == "parent"
    monkeypatch.setenv("TLLM_SUBAGENT_AFFINITY_HINT_ONLY", "1")
    assert get_request_routing_id(request) == "child"
