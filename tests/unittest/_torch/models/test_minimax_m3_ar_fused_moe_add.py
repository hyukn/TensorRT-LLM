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
"""MiniMax-M3: the MoE's shared+routed add folded into the boundary AllReduce (input_add), plumbing and fallbacks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, AllReduceStrategy

pytestmark = pytest.mark.cpu_only


def _fake_allreduce_module(tp_size=2, mnnvl=None, symm=None):
    seen = {}

    def all_reduce_op(**kwargs):
        seen["input"] = kwargs["input"]
        seen["op"] = kwargs["op"]
        return [kwargs["input"] * 2]

    return (
        SimpleNamespace(
            mapping=SimpleNamespace(tp_size=tp_size, tp_rank=0, tp_group=[0, 1]),
            strategy=AllReduceStrategy.NCCL,
            symm_mem_allreduce=symm,
            mnnvl_allreduce=mnnvl,
            _disable_mpi=False,
            all_reduce_op=all_reduce_op,
            workspace=None,
        ),
        seen,
    )


def test_input_add_without_a_reduction_is_a_plain_add() -> None:
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    module, _ = _fake_allreduce_module(tp_size=1)
    out = AllReduce.forward(module, a, all_reduce_params=AllReduceParams(input_add=b))
    assert torch.equal(out, a + b)
    module, _ = _fake_allreduce_module()
    params = AllReduceParams(enable_allreduce=False, input_add=b)
    assert torch.equal(AllReduce.forward(module, a, all_reduce_params=params), a + b)


def test_input_add_is_folded_when_mnnvl_is_absent_or_declines() -> None:
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    params = AllReduceParams(input_add=b)
    module, seen = _fake_allreduce_module()
    out = AllReduce.forward(module, a, all_reduce_params=params)
    assert torch.equal(seen["input"], a + b) and torch.equal(out, (a + b) * 2)
    assert params.input_add is b  # the caller's params are not mutated

    calls = []

    def declining_mnnvl(x, all_reduce_params):
        calls.append(all_reduce_params.input_add)
        return None

    module, seen = _fake_allreduce_module(mnnvl=declining_mnnvl)
    AllReduce.forward(module, a, all_reduce_params=params)
    assert calls[0] is b  # MNNVL was offered the unfused pair first
    assert torch.equal(seen["input"], a + b)


def test_input_add_reaches_mnnvl_unfolded() -> None:
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    got = {}

    def mnnvl(x, all_reduce_params):
        got["x"], got["add"] = x, all_reduce_params.input_add
        return x + all_reduce_params.input_add

    # symm-mem only takes the plain op; a fused op must not be pre-added for it
    module, _ = _fake_allreduce_module(mnnvl=mnnvl, symm=lambda x: None)
    params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=torch.zeros(3, 8),
        norm_weight=torch.ones(8),
        input_add=b,
    )
    AllReduce.forward(module, a, all_reduce_params=params)
    assert torch.equal(got["x"], a) and got["add"] is b


class _FakeMoE:
    def __init__(self, shared, routed, shared_experts=True, routed_global=False):
        self.shared, self.routed = shared, routed
        self.shared_experts = object() if shared_experts else None
        self.routed_output_is_global = routed_global
        self.kwargs = None

    def __call__(self, hidden_states, attn_metadata, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("return_partials"):
            return self.shared.clone(), self.routed
        return self.shared.clone().add_(self.routed)


class _FakeAR:
    def __init__(self, mnnvl=True):
        self.mnnvl_allreduce = object() if mnnvl else None
        self.params = []

    def __call__(self, x, all_reduce_params):
        self.params.append(all_reduce_params)
        if all_reduce_params.input_add is not None:
            x = x + all_reduce_params.input_add
        return x * 2, x + all_reduce_params.residual


def _layer(moe, ar, *, post_fusion=True, wired=True) -> MiniMaxM3DecoderLayer:
    layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.block_sparse_moe = moe
    layer.allreduce = ar
    layer.pre_feed_forward_fusion = False
    layer.post_feed_forward_fusion = post_fusion
    layer.post_attention_layernorm = lambda h, r: (h, r)
    layer.next_layer_layernorm = (
        SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-5) if wired else None
    )
    return layer


def test_fused_moe_add_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    h = torch.zeros(4, 8)
    moe = _FakeMoE(h, h)
    monkeypatch.setattr(m3, "_AR_FUSED_MOE_ADD", False)
    assert not _layer(moe, _FakeAR())._fused_moe_add(h, None)
    monkeypatch.setattr(m3, "_AR_FUSED_MOE_ADD", True)
    assert _layer(moe, _FakeAR())._fused_moe_add(h, None)
    assert not _layer(moe, _FakeAR())._fused_moe_add(h, {"lora": 1})
    assert not _layer(moe, _FakeAR())._fused_moe_add(h.unsqueeze(0), None)
    assert not _layer(moe, _FakeAR(), post_fusion=False)._fused_moe_add(h, None)
    assert not _layer(moe, _FakeAR(), wired=False)._fused_moe_add(h, None)
    assert not _layer(moe, _FakeAR(mnnvl=False))._fused_moe_add(h, None)
    assert not _layer(moe, None)._fused_moe_add(h, None)
    assert not _layer(_FakeMoE(h, h, shared_experts=False), _FakeAR())._fused_moe_add(h, None)
    assert not _layer(_FakeMoE(h, h, routed_global=True), _FakeAR())._fused_moe_add(h, None)


@pytest.mark.parametrize("fused", [False, True])
def test_forward_moe_hands_the_routed_output_to_the_allreduce(
    monkeypatch: pytest.MonkeyPatch, fused: bool
) -> None:
    torch.manual_seed(0)
    hidden, residual = torch.randn(4, 8), torch.randn(4, 8)
    shared, routed = torch.randn(4, 8), torch.randn(4, 8)
    monkeypatch.setattr(m3, "_AR_FUSED_MOE_ADD", fused)
    moe, ar = _FakeMoE(shared, routed), _FakeAR()
    out, res = _layer(moe, ar).forward_MoE(hidden, SimpleNamespace(), residual)
    assert torch.equal(out, (shared + routed) * 2)
    assert torch.equal(res, shared + routed + residual)
    assert bool(moe.kwargs.get("return_partials")) is fused
    assert moe.kwargs["final_all_reduce_params"].enable_allreduce is False
    (params,) = ar.params
    assert params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    assert (params.input_add is routed) if fused else (params.input_add is None)


def test_next_layer_norm_without_post_fusion_adds_up_front() -> None:
    a, b, r = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    layer = _layer(_FakeMoE(a, b), _FakeAR(), post_fusion=False)
    layer.next_layer_layernorm = lambda h, res: (h, res)
    h, res = layer._apply_next_layer_layernorm(a, r, input_add=b)
    assert torch.equal(h, a + b) and res is r
