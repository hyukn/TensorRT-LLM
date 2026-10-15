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
"""MiniMax-M3: the layer-boundary AllReduce hands the next layer's QKV GEMM its MXFP8 input (plumbing)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention, MiniMaxM3DecoderLayer
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor

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


def _layer(next_attn, mnnvl=True) -> MiniMaxM3DecoderLayer:
    layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.next_layer_attn_ref = [next_attn] if next_attn is not None else []
    layer.allreduce = SimpleNamespace(mnnvl_allreduce=object() if mnnvl else None)
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
