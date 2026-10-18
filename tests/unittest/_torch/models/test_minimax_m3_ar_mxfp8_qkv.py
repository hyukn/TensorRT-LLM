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
"""MiniMax-M3 MXFP8 activation handoffs: AllReduce -> QKV / shared expert, attention -> o_proj."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3Attention,
    MiniMaxM3DecoderLayer,
    MiniMaxM3MoE,
)
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor
from tensorrt_llm.functional import AllReduceStrategy

pytestmark = pytest.mark.cpu_only


def _attention() -> MiniMaxM3Attention:
    attn = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
    nn.Module.__init__(attn)
    return attn


def test_staged_input_is_taken_only_for_the_same_tensor() -> None:
    attn = _attention()
    hidden = torch.zeros(4, 8)
    quantized = MxFp8QuantizedTensor(fp8_tensor=torch.zeros(4, 8), scaling_factor=torch.zeros(4))
    attn.stage_mxfp8_qkv_input(hidden, quantized)
    assert attn._take_staged_mxfp8_qkv_input(hidden) is quantized
    # consumed: a second call (or another layer step) sees the bf16 tensor again
    assert attn._take_staged_mxfp8_qkv_input(hidden) is hidden
    attn.stage_mxfp8_qkv_input(hidden, quantized)
    other = hidden.clone()
    assert attn._take_staged_mxfp8_qkv_input(other) is other  # a different tensor never gets it
    assert attn._take_staged_mxfp8_qkv_input(hidden) is hidden  # and the stale entry was dropped


def _layer(next_attn, mnnvl=True, strategy=AllReduceStrategy.AUTO) -> MiniMaxM3DecoderLayer:
    layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.next_layer_attn_ref = [next_attn] if next_attn is not None else []
    layer.allreduce = SimpleNamespace(
        mnnvl_allreduce=object() if mnnvl else None, strategy=strategy
    )
    return layer


class _NextAttn:
    def __init__(self, accepts: bool) -> None:
        self.accepts = accepts

    def accepts_mxfp8_qkv_input(self) -> bool:
        return self.accepts


def test_next_layer_mxfp8_attention_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.zeros(4, 8)
    accepting = _NextAttn(True)
    monkeypatch.setattr(m3, "_AR_MXFP8_QKV", False)
    assert _layer(accepting)._next_layer_mxfp8_attention(hidden) is None
    monkeypatch.setattr(m3, "_AR_MXFP8_QKV", True)
    assert _layer(accepting)._next_layer_mxfp8_attention(hidden) is accepting
    assert _layer(None)._next_layer_mxfp8_attention(hidden) is None  # last layer: final norm
    assert _layer(accepting, mnnvl=False)._next_layer_mxfp8_attention(hidden) is None
    assert _layer(_NextAttn(False))._next_layer_mxfp8_attention(hidden) is None
    assert _layer(accepting)._next_layer_mxfp8_attention(hidden.unsqueeze(0)) is None
    # Without MNNVL: the non-fused NCCL paths emit the MXFP8 output too (context servers use NCCL); the
    # one-shot / two-shot kernels AUTO may pick do not.
    for strategy in (AllReduceStrategy.NCCL, AllReduceStrategy.NCCL_SYMMETRIC):
        layer = _layer(accepting, mnnvl=False, strategy=strategy)
        assert layer._next_layer_mxfp8_attention(hidden) is accepting
    for strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.UB, AllReduceStrategy.ONESHOT):
        assert (
            _layer(accepting, mnnvl=False, strategy=strategy)._next_layer_mxfp8_attention(hidden)
            is None
        )


def test_moe_staged_shared_input_is_taken_only_for_the_same_tensor() -> None:
    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    hidden = torch.zeros(4, 8)
    quantized = MxFp8QuantizedTensor(fp8_tensor=torch.zeros(4, 8), scaling_factor=torch.zeros(4))
    moe.stage_mxfp8_shared_input(hidden, quantized)
    assert moe._take_staged_mxfp8_shared_input(hidden) is quantized
    assert moe._take_staged_mxfp8_shared_input(hidden) is hidden  # consumed
    moe.stage_mxfp8_shared_input(hidden, quantized)
    assert moe._take_staged_mxfp8_shared_input(hidden.clone()) is not quantized
    assert moe._take_staged_mxfp8_shared_input(hidden) is hidden  # the stale entry was dropped


class _Moe:
    def __init__(self, accepts: bool) -> None:
        self.accepts = accepts

    def accepts_mxfp8_shared_input(self) -> bool:
        return self.accepts


def _moe_layer(moe, mnnvl=False, strategy=AllReduceStrategy.NCCL) -> MiniMaxM3DecoderLayer:
    layer = _layer(None, mnnvl=mnnvl, strategy=strategy)
    layer.block_sparse_moe = moe
    return layer


def test_shared_expert_mxfp8_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.zeros(4, 8)
    accepting = _Moe(True)
    monkeypatch.setattr(m3, "_AR_MXFP8_SHARED", False)
    assert _moe_layer(accepting)._mxfp8_shared_expert_moe(hidden) is None
    monkeypatch.setattr(m3, "_AR_MXFP8_SHARED", True)
    assert _moe_layer(accepting)._mxfp8_shared_expert_moe(hidden) is accepting
    assert (
        _moe_layer(accepting, strategy=AllReduceStrategy.NCCL_SYMMETRIC)._mxfp8_shared_expert_moe(
            hidden
        )
        is accepting
    )
    assert (
        _moe_layer(accepting, mnnvl=True, strategy=AllReduceStrategy.AUTO)._mxfp8_shared_expert_moe(
            hidden
        )
        is accepting
    )
    assert (
        _moe_layer(accepting, strategy=AllReduceStrategy.AUTO)._mxfp8_shared_expert_moe(hidden)
        is None
    )
    assert _moe_layer(None)._mxfp8_shared_expert_moe(hidden) is None  # dense layers (no MoE)
    assert _moe_layer(_Moe(False))._mxfp8_shared_expert_moe(hidden) is None
    assert _moe_layer(accepting)._mxfp8_shared_expert_moe(hidden.unsqueeze(0)) is None


class _DecodeGraphMxfp8(MXFP8LinearMethod):
    def __init__(self, on_graph: bool) -> None:
        self.on_graph = on_graph

    def on_decode_graph_path(self) -> bool:
        return self.on_graph


def _sparse_attention(o_proj_method, with_o_lora=True) -> MiniMaxM3Attention:
    attn = _attention()
    attn.is_sparse_attention_layer = True
    attn.attn = m3.MiniMaxM3MsaSparseAttention.__new__(m3.MiniMaxM3MsaSparseAttention)
    attn.register_to_config = False
    attn.attn_activation_dtype = torch.bfloat16
    attn.num_heads, attn.head_dim = 2, 64
    # The base Attention always builds o_proj with its o_lora LoraLayer (inactive without lora_params).
    attn.o_proj = SimpleNamespace(
        quant_method=o_proj_method,
        lora=LoraLayer([LoraModuleType.ATTENTION_DENSE], [128]) if with_o_lora else None,
    )
    return attn


def test_attn_out_mxfp8_request_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    q = torch.zeros(3, 2 * 64, dtype=torch.bfloat16)
    decode = SimpleNamespace(num_contexts=0)
    monkeypatch.setattr(m3, "_ATTN_OUT_MXFP8", False)
    assert _sparse_attention(_DecodeGraphMxfp8(True))._attn_out_mxfp8_request(q, decode) is None
    monkeypatch.setattr(m3, "_ATTN_OUT_MXFP8", True)
    for with_o_lora in (True, False):  # the inactive o_lora must not decline the request
        attn = _sparse_attention(_DecodeGraphMxfp8(True), with_o_lora=with_o_lora)
        request = attn._attn_out_mxfp8_request(q, decode)
        assert request is not None
        assert tuple(request.fp8.shape) == (3, 128) and request.fp8.dtype == torch.float8_e4m3fn
        assert not request.written and request.out_ptr is None
    attn = _sparse_attention(_DecodeGraphMxfp8(True))
    assert attn._attn_out_mxfp8_request(q, SimpleNamespace(num_contexts=1)) is None  # mixed step
    assert attn._attn_out_mxfp8_request(q.view(3, 2, 64), decode) is None
    assert _sparse_attention(_DecodeGraphMxfp8(False))._attn_out_mxfp8_request(q, decode) is None
    assert _sparse_attention(object())._attn_out_mxfp8_request(q, decode) is None  # not MXFP8
    attn = _sparse_attention(_DecodeGraphMxfp8(True))
    attn.attn_activation_dtype = torch.float16
    assert attn._attn_out_mxfp8_request(q, decode) is None
    attn = _sparse_attention(_DecodeGraphMxfp8(True))
    attn.is_sparse_attention_layer = False
    assert attn._attn_out_mxfp8_request(q, decode) is None


@pytest.mark.parametrize("enabled", [False, True])
def test_swiglu_mxfp8_flag_reaches_the_mlps(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig

    monkeypatch.setattr(m3, "_SWIGLU_MXFP8", enabled)
    pretrained = PretrainedConfig(hidden_size=64)
    pretrained.torch_dtype = torch.bfloat16
    config = ModelConfig(pretrained_config=pretrained, skip_create_weights_in_init=True)
    for shared in (True, False):
        mlp = m3._build_swiglu_oai_dense_mlp(
            model_config=config, intermediate_size=128, layer_idx=0, is_shared_expert=shared
        )
        assert mlp.swiglu_mxfp8_out is enabled
