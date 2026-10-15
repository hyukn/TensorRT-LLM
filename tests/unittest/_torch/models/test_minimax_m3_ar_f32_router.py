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
"""MiniMax-M3: the post-attention AllReduce hands the MoE router an fp32 copy of the norm output (plumbing)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer, MiniMaxM3Gate

pytestmark = pytest.mark.cpu_only


def _gate() -> MiniMaxM3Gate:
    gate = MiniMaxM3Gate(hidden_size=16, num_experts=8, top_k=2, routed_scaling_factor=1.0)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(8, 16))
    return gate


def test_staged_fp32_input_gives_the_same_logits() -> None:
    torch.manual_seed(0)
    gate = _gate()
    hidden = torch.randn(4, 16).to(torch.bfloat16)
    ref = gate(hidden)
    gate.stage_fp32_input(hidden, hidden.float())
    assert torch.equal(gate(hidden), ref)
    # consumed, and never applied to another tensor
    gate.stage_fp32_input(hidden, torch.zeros(4, 16))
    other = hidden.clone()
    assert torch.equal(gate(other), ref)
    assert gate._staged_fp32_input is None


def test_staged_fp32_input_is_used() -> None:
    gate = _gate()
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    marker = torch.ones(2, 16)
    gate.stage_fp32_input(hidden, marker)
    assert torch.equal(gate(hidden), torch.nn.functional.linear(marker, gate.weight))


def _layer(moe, mnnvl=True) -> MiniMaxM3DecoderLayer:
    layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.block_sparse_moe = moe
    layer.allreduce = SimpleNamespace(mnnvl_allreduce=object() if mnnvl else None)
    return layer


def test_fp32_router_gate_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.zeros(4, 16)
    gate = _gate()
    moe = SimpleNamespace(gate=gate)
    monkeypatch.setattr(m3, "_AR_F32_ROUTER", False)
    assert _layer(moe)._fp32_router_gate(hidden) is None
    monkeypatch.setattr(m3, "_AR_F32_ROUTER", True)
    assert _layer(moe)._fp32_router_gate(hidden) is gate
    assert _layer(None)._fp32_router_gate(hidden) is None  # dense-MLP layer
    assert _layer(moe, mnnvl=False)._fp32_router_gate(hidden) is None
    assert _layer(SimpleNamespace(gate=object()))._fp32_router_gate(hidden) is None
    assert _layer(moe)._fp32_router_gate(hidden.unsqueeze(0)) is None
