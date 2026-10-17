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
"""MiniMax-M3 MoE: shared expert forked at a MoE scheduler stage.

``TRTLLM_M3_SHARED_EXPERT_FORK=pre_dispatch|pre_combine`` starts the shared expert on its stream
when the routed path reaches that stage of the external-comm MoE scheduler (the scheduler records
the fork event), instead of next to the router GEMM. Same kernels and inputs, so the output must be
bitwise the default placement's.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.moe.fused_moe import moe_scheduler
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import (
    MOE_STAGE_PRE_COMBINE,
    MOE_STAGE_PRE_DISPATCH,
    fire_stage_event,
)
from tensorrt_llm._torch.utils import EventType

HIDDEN = 256
NUM_EXPERTS = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _FakeEvent:
    def __init__(self) -> None:
        self.records = 0

    def record(self) -> None:
        self.records += 1


@pytest.mark.cpu_only
def test_fire_stage_event_records_only_the_registered_stage() -> None:
    ev = _FakeEvent()
    moe = SimpleNamespace(stage_events={MOE_STAGE_PRE_COMBINE: ev}, stage_events_fired=set())
    fire_stage_event(moe, MOE_STAGE_PRE_DISPATCH)
    assert ev.records == 0 and moe.stage_events_fired == set()
    fire_stage_event(moe, MOE_STAGE_PRE_COMBINE)
    assert ev.records == 1 and moe.stage_events_fired == {MOE_STAGE_PRE_COMBINE}
    # A MoE without registered events (every other model) is untouched.
    fire_stage_event(SimpleNamespace(), MOE_STAGE_PRE_COMBINE)
    fire_stage_event(SimpleNamespace(stage_events={}), MOE_STAGE_PRE_COMBINE)


@pytest.mark.cpu_only
def test_the_external_comm_scheduler_fires_both_stages_in_order() -> None:
    """The fork points sit right before the dispatch calls and before the combine."""
    src = inspect.getsource(moe_scheduler.ExternalCommMoEScheduler._forward_chunk_impl)
    lines = [ln.strip() for ln in src.splitlines()]
    fires = [
        i for i, ln in enumerate(lines) if ln == "fire_stage_event(moe, MOE_STAGE_PRE_DISPATCH)"
    ]
    dispatches = [i for i, ln in enumerate(lines) if "moe.comm.dispatch(" in ln]
    assert len(fires) == len(dispatches) == 2
    for fire, call in zip(fires, dispatches):
        assert call == fire + 1  # the event is recorded right before each dispatch is enqueued
    pre_combine = lines.index("fire_stage_event(moe, MOE_STAGE_PRE_COMBINE)")
    combine = next(i for i, ln in enumerate(lines) if "moe.comm.combine(" in ln)
    assert max(dispatches) < pre_combine < combine


class _Gate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((NUM_EXPERTS, HIDDEN), dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states.float(), self.weight)


class _Experts(nn.Module):
    """Stands in for the external-comm fused MoE: main-stream work around the two fork stages."""

    def __init__(self, fire: bool = True) -> None:
        super().__init__()
        self.comm = object()
        self.fire = fire
        self.w = torch.randn((HIDDEN, HIDDEN), dtype=torch.bfloat16, device="cuda") / 16

    def forward(self, x, router_logits, **kwargs) -> torch.Tensor:
        y = x @ self.w + router_logits.softmax(-1)[:, :1].to(x.dtype)  # routing + quantize
        if self.fire:
            fire_stage_event(self, MOE_STAGE_PRE_DISPATCH)
        for _ in range(4):  # dispatch + expert GEMMs
            y = torch.tanh(y @ self.w)
        if self.fire:
            fire_stage_event(self, MOE_STAGE_PRE_COMBINE)
        return y * 2 + 1  # combine


class _Shared(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.randn((HIDDEN, HIDDEN), dtype=torch.bfloat16, device="cuda") / 16

    def forward(self, hidden_states: torch.Tensor, lora_params=None) -> torch.Tensor:
        return torch.relu(hidden_states @ self.w) @ self.w


def _make_moe(fire: bool = True) -> MiniMaxM3MoE:
    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.gate = _Gate()
    moe.experts = _Experts(fire)
    moe.shared_experts = _Shared()
    moe.routed_output_is_global = False
    moe.allreduce = None
    moe.aux_stream = torch.cuda.Stream()
    moe.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}
    moe._prequant_event = torch.cuda.Event()
    return moe


@requires_cuda
@pytest.mark.parametrize("fire", [True, False], ids=["scheduler-fires", "never-fires"])
@pytest.mark.parametrize("stage", ["post_router", "pre_dispatch", "pre_combine"])
@pytest.mark.parametrize("num_tokens", [4, 104])
def test_late_fork_matches_default_placement_in_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, stage: str, fire: bool, num_tokens: int
) -> None:
    """Under a captured CUDA graph with two streams, the late fork is bitwise the default path."""
    torch.manual_seed(1)
    moe = _make_moe(fire)
    meta = SimpleNamespace(all_rank_num_tokens=[num_tokens])
    static_in = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")

    def run_graph(fork: str):
        monkeypatch.setattr(m3, "_SHARED_EXPERT_FORK", fork)
        with with_multi_stream(True):
            assert moe._use_late_shared_fork(None) == (fork != "")
            moe(static_in, meta)  # warm up outside capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = moe(static_in, meta)
        assert not getattr(moe.experts, "stage_events", {})  # cleared after the forward
        outs = []
        for _ in range(3):
            static_in.copy_(torch.randn_like(static_in))
            graph.replay()
            torch.cuda.synchronize()
            outs.append((static_in.clone(), static_out.clone()))
        return outs

    torch.manual_seed(2)
    ref = run_graph("")
    torch.manual_seed(2)
    got = run_graph(stage)
    for (ref_in, ref_out), (got_in, got_out) in zip(ref, got):
        assert torch.equal(ref_in, got_in)
        assert torch.equal(ref_out, got_out)
    if (
        fire and stage != "post_router"
    ):  # post_router is recorded by the MoE module, not the scheduler
        assert moe.experts.stage_events_fired == {stage}


@requires_cuda
def test_late_fork_eager_step_keeps_the_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside multi-stream graph capture the forward is unchanged (and gives the same result)."""
    torch.manual_seed(3)
    moe = _make_moe()
    meta = SimpleNamespace(all_rank_num_tokens=[8])
    x = torch.randn((8, HIDDEN), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(m3, "_SHARED_EXPERT_FORK", "")
    ref = moe(x, meta)
    monkeypatch.setattr(m3, "_SHARED_EXPERT_FORK", "pre_combine")
    assert not moe._use_late_shared_fork(None)
    assert torch.equal(moe(x, meta), ref)


@pytest.mark.cpu_only
def test_late_fork_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only multi-stream forwards with a MoE all-to-all, a shared expert and no LoRA fork late."""
    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.shared_experts = object()
    moe.aux_stream = object()
    moe.experts = SimpleNamespace(comm=object())

    monkeypatch.setattr(m3, "_SHARED_EXPERT_FORK", "")
    with with_multi_stream(True):
        assert not moe._use_late_shared_fork(None)
    monkeypatch.setattr(m3, "_SHARED_EXPERT_FORK", "after_router")  # not a scheduler stage
    with with_multi_stream(True):
        assert not moe._use_late_shared_fork(None)

    for stage in ("post_router", "pre_dispatch", "pre_combine"):
        monkeypatch.setattr(m3, "_SHARED_EXPERT_FORK", stage)
        assert not moe._use_late_shared_fork(None)  # eager step
        with with_multi_stream(True):
            assert moe._use_late_shared_fork(None)
            assert not moe._use_late_shared_fork({"lora": 1})
            moe.experts = SimpleNamespace(comm=None)  # TP: no all-to-all, default placement
            assert not moe._use_late_shared_fork(None)
            moe.experts = SimpleNamespace(comm=object())
            moe.shared_experts = None
            assert not moe._use_late_shared_fork(None)
            moe.shared_experts = object()
