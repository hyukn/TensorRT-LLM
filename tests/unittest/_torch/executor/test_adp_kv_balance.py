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
"""KVBalancedADPRouter: choose which waiting requests to admit so ADP ranks carry similar context tokens."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.request_utils import pop_adp_admission_window
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ADPRouter,
    DefaultADPRouter,
    KVBalancedADPRouter,
    RankState,
)

pytestmark = pytest.mark.cpu_only


def _dist(tp_size=4):
    dist = MagicMock()
    dist.tp_rank = 0
    dist.tp_size = tp_size
    dist.has_cp_helix = False
    return dist


def _item(req_id, num_tokens, target_dp_rank=None, children=None):
    params = SimpleNamespace(
        attention_dp_rank=target_dp_rank, attention_dp_relax=target_dp_rank is None
    )
    request = SimpleNamespace(num_input_tokens=num_tokens, py_scheduling_params=params)
    return SimpleNamespace(id=req_id, request=request, child_req_ids=children)


def _states(tokens, counts):
    return [
        RankState(rank=r, num_active_requests=c, num_active_tokens=t)
        for r, (t, c) in enumerate(zip(tokens, counts))
    ]


def _ids(assigned):
    return {rank: [item.id for item in items] for rank, items in assigned.items()}


def test_factory_selects_router_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_ADP_KV_BALANCE_WINDOW", raising=False)
    assert type(ADPRouter.create(_dist(), False)) is DefaultADPRouter
    monkeypatch.setenv("TRTLLM_ADP_KV_BALANCE_WINDOW", "6")
    monkeypatch.setenv("TRTLLM_ADP_KV_BALANCE_MAX_SKIPS", "3")
    router = ADPRouter.create(_dist(), False)
    assert isinstance(router, KVBalancedADPRouter)
    assert (router.admission_window, router.max_skips) == (6, 3)


def test_heavy_rank_gets_a_short_request_and_light_rank_a_long_one() -> None:
    router = KVBalancedADPRouter(_dist(), admission_window=4)
    # cap 3: rank 0 (heavy) and rank 3 (light) each have one free slot. Expected mean load after admitting two
    # requests: (2400 + 100 + 120) / 4 = 655, so rank 3 wants ~355 tokens and rank 0 as few as possible.
    states = _states([900, 600, 600, 300], [2, 3, 3, 2])
    fcfs = [_item(1, 100), _item(2, 120)]
    window = [_item(3, 600), _item(4, 50), _item(5, 400)]
    assigned, expected, leftover = router.route_requests_with_window(states, fcfs, window, 3)
    assert expected == 3
    assert _ids(assigned) == {0: [4], 1: [], 2: [], 3: [5]}
    assert [item.id for item in leftover] == [1, 2, 3]  # queue order kept
    assert router._skips == {1: 1, 2: 1}


def test_admits_as_many_as_the_fcfs_pop() -> None:
    router = KVBalancedADPRouter(_dist(), admission_window=8)
    states = _states([0, 0, 0, 0], [0, 0, 0, 0])
    fcfs = [_item(i, 100 * (i + 1)) for i in range(4)]
    window = [_item(10 + i, 50) for i in range(8)]
    assigned, _, leftover = router.route_requests_with_window(states, fcfs, window, 4)
    admitted = [item for items in assigned.values() for item in items]
    assert len(admitted) == 4 and len(leftover) == 8
    assert max(len(items) for items in assigned.values()) <= 4


def test_skipped_request_is_forced_after_max_skips() -> None:
    router = KVBalancedADPRouter(_dist(tp_size=2), admission_window=2, max_skips=2)
    states = _states([1000, 0], [1, 2])  # only rank 0 (heavy) has a free slot (cap 2)
    long_req = _item(7, 900)
    for round_idx in range(2):
        assigned, _, leftover = router.route_requests_with_window(
            states, [long_req], [_item(100 + round_idx, 10)], 2
        )
        assert _ids(assigned)[0] == [100 + round_idx]
        assert [item.id for item in leftover] == [7]
    assert router._skips == {7: 2}
    assigned, _, leftover = router.route_requests_with_window(
        states, [long_req], [_item(200, 10)], 2
    )
    assert _ids(assigned)[0] == [7]
    assert [item.id for item in leftover] == [200]
    assert router._skips == {}


def test_explicit_rank_requests_keep_their_rank() -> None:
    router = KVBalancedADPRouter(_dist(tp_size=2), admission_window=2)
    states = _states([0, 500], [0, 0])
    fcfs = [_item(1, 10, target_dp_rank=1), _item(2, 10)]
    assigned, _, leftover = router.route_requests_with_window(states, fcfs, [_item(3, 800)], 4)
    assert 1 in _ids(assigned)[1]
    assert sum(len(v) for v in assigned.values()) == 2
    assert len(leftover) == 1


def test_routing_is_deterministic() -> None:
    def run():
        router = KVBalancedADPRouter(_dist(), admission_window=6)
        states = _states([700, 100, 400, 400], [3, 2, 3, 2])
        fcfs = [_item(i, 37 * i % 500 + 10) for i in range(4)]
        window = [_item(10 + i, 91 * i % 700 + 5) for i in range(6)]
        assigned, expected, leftover = router.route_requests_with_window(states, fcfs, window, 4)
        return _ids(assigned), expected, [item.id for item in leftover], dict(router._skips)

    assert run() == run()


def test_pop_window_stops_at_children_or_explicit_rank() -> None:
    queue = FCFSWaitingQueue()
    for item in [_item(1, 10), _item(2, 10), _item(3, 10, target_dp_rank=0), _item(4, 10)]:
        queue.append(item)
    assert [item.id for item in pop_adp_admission_window(queue, 8)] == [1, 2]
    assert [item.id for item in queue] == [3, 4]
    queue = FCFSWaitingQueue()
    for item in [_item(1, 10), _item(2, 10, children=[9]), _item(3, 10)]:
        queue.append(item)
    assert [item.id for item in pop_adp_admission_window(queue, 1)] == [1]
    assert [item.id for item in pop_adp_admission_window(queue, 4)] == []
