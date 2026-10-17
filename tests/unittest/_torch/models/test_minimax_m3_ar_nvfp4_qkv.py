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
"""MiniMax-M3 TRTLLM_M3_AR_NVFP4_QKV: the layer-boundary AllReduce hands an NVFP4-repacked QKV GEMM its NVFP4 input
(plumbing; the kernels are covered by multi_gpu/test_m3_ar_nvfp4_qkv.py)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.distributed import AllReduceFusionOp
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3Attention,
    MiniMaxM3DecoderLayer,
    MiniMaxM3MsaSparseAttention,
)
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.functional import AllReduceStrategy

pytestmark = pytest.mark.cpu_only


def _nvfp4(swizzled: bool = True) -> Fp4QuantizedTensor:
    return Fp4QuantizedTensor(
        fp4_tensor=torch.zeros(4, 4, dtype=torch.uint8),
        scaling_factor=torch.zeros(512, dtype=torch.uint8),
        is_sf_swizzled=swizzled,
    )


def _attention(
    repacked: bool = True, fused: bool = True, sparse: bool = True
) -> MiniMaxM3Attention:
    attn = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
    nn.Module.__init__(attn)
    attn.enable_fused_qkv_index_projection = fused
    # Only the backend's type matters here (the MSA sparse backend is not an nn.Module).
    attn.attn = (
        MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention) if sparse else object()
    )
    attn.qkv_proj = SimpleNamespace(
        lora=None, _nvfp4_repacked=repacked, _nvfp4_input_scale=torch.tensor([0.5])
    )
    return attn


def test_staged_nvfp4_input_is_taken_only_for_the_same_tensor() -> None:
    attn = _attention()
    hidden = torch.zeros(4, 8)
    quantized = _nvfp4()
    attn.stage_nvfp4_qkv_input(hidden, quantized)
    assert attn._take_staged_mxfp8_qkv_input(hidden) is quantized
    assert attn._take_staged_mxfp8_qkv_input(hidden) is hidden  # consumed
    attn.stage_nvfp4_qkv_input(hidden, quantized)
    assert attn._take_staged_mxfp8_qkv_input(hidden.clone()) is not quantized
    assert attn._take_staged_mxfp8_qkv_input(hidden) is hidden  # the stale entry was dropped
    with pytest.raises(ValueError, match="swizzled"):
        attn.stage_nvfp4_qkv_input(hidden, _nvfp4(swizzled=False))


def test_accepts_nvfp4_qkv_input() -> None:
    assert _attention().accepts_nvfp4_qkv_input()
    assert torch.equal(_attention().nvfp4_qkv_input_scale, torch.tensor([0.5]))
    assert not _attention(repacked=False).accepts_nvfp4_qkv_input()  # MXFP8 projection
    assert not _attention(fused=False).accepts_nvfp4_qkv_input()
    assert not _attention(sparse=False).accepts_nvfp4_qkv_input()  # dense-attention layer
    attn = _attention()
    attn.qkv_proj.lora = object()
    assert not attn.accepts_nvfp4_qkv_input()


class _FakeAllReduce:
    """Records the fused-op parameters and returns outputs shaped like the MNNVL op's."""

    def __init__(self, mnnvl: bool = True) -> None:
        self.mnnvl_allreduce = object() if mnnvl else None
        self.strategy = AllReduceStrategy.AUTO
        self.calls = []

    def __call__(self, hidden_states, all_reduce_params):
        self.calls.append(all_reduce_params)
        norm = hidden_states + 1
        residual = all_reduce_params.residual + 2
        if all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
            quant = torch.zeros(
                hidden_states.shape[0], hidden_states.shape[1] // 2, dtype=torch.uint8
            )
            return norm, quant, torch.zeros(512, dtype=torch.uint8), residual
        assert all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
        return norm, residual


def _layer(next_attn, mnnvl: bool = True) -> MiniMaxM3DecoderLayer:
    layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.next_layer_attn_ref = [next_attn] if next_attn is not None else []
    layer.allreduce = _FakeAllReduce(mnnvl)
    layer.post_feed_forward_fusion = True
    layer.next_layer_layernorm = SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-5)
    return layer


def test_next_layer_nvfp4_attention_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.zeros(4, 8)
    accepting = _attention()
    monkeypatch.setattr(m3, "_AR_NVFP4_QKV", False)
    assert _layer(accepting)._next_layer_nvfp4_attention(hidden) is None
    monkeypatch.setattr(m3, "_AR_NVFP4_QKV", True)
    assert _layer(accepting)._next_layer_nvfp4_attention(hidden) is accepting
    assert _layer(None)._next_layer_nvfp4_attention(hidden) is None  # last layer: final norm
    assert _layer(accepting, mnnvl=False)._next_layer_nvfp4_attention(hidden) is None  # MNNVL only
    assert _layer(_attention(repacked=False))._next_layer_nvfp4_attention(hidden) is None
    assert _layer(accepting)._next_layer_nvfp4_attention(hidden.unsqueeze(0)) is None


@pytest.mark.parametrize("with_input_add", [False, True])
def test_boundary_allreduce_hands_over_the_nvfp4_activation(
    monkeypatch: pytest.MonkeyPatch, with_input_add: bool
) -> None:
    monkeypatch.setattr(m3, "_AR_NVFP4_QKV", True)
    monkeypatch.setattr(m3, "_AR_MXFP8_QKV", True)
    attn = _attention()
    layer = _layer(attn)
    hidden, residual = torch.zeros(4, 8), torch.zeros(4, 8)
    input_add = torch.ones(4, 8) if with_input_add else None
    norm, res = layer._apply_next_layer_layernorm(hidden, residual, input_add)
    (params,) = layer.allreduce.calls
    assert params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4
    assert params.scale is attn.qkv_proj._nvfp4_input_scale
    assert params.input_add is input_add
    assert params.norm_weight is layer.next_layer_layernorm.weight
    assert torch.equal(norm, hidden + 1) and torch.equal(res, residual + 2)
    staged = attn._take_staged_mxfp8_qkv_input(norm)
    assert isinstance(staged, Fp4QuantizedTensor) and staged.is_sf_swizzled
    assert staged.fp4_tensor.shape == (4, 4)


def test_boundary_allreduce_keeps_bf16_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m3, "_AR_NVFP4_QKV", False)
    monkeypatch.setattr(m3, "_AR_MXFP8_QKV", True)
    attn = _attention()  # repacked: the MXFP8 handoff does not apply either
    layer = _layer(attn)
    hidden, residual = torch.zeros(4, 8), torch.zeros(4, 8)
    norm, _ = layer._apply_next_layer_layernorm(hidden, residual)
    (params,) = layer.allreduce.calls
    assert params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    assert attn._take_staged_mxfp8_qkv_input(norm) is norm
