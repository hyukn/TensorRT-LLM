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
"""SwiGLU with the MXFP8 quantize of its output fused in (trtllm::silu_and_mul_mxfp8, GatedMLP.swiglu_mxfp8_out).

The fused kernel must give the bytes of mxfp8_quantize(silu_and_mul(x), True), and an MXFP8 GatedMLP that hands
them to down_proj must give the output of the unfused path, bit for bit.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.custom_ops.torch_custom_ops import IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import (
    MXFP8LinearMethod,
    _mxfp8_cutlass_op_available,
    flashinfer_mxfp8_autotune,
    flashinfer_mxfp8_decode_graph_capture,
)
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not _mxfp8_cutlass_op_available(), reason="needs the MXFP8 quantize / GEMM ops (SM100+)"
)

SWIGLU_OAI = dict(swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
PLAIN = dict(swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None)


def _scale_offsets(rows, cols, device):
    """Offsets of the valid (row, 32-column block) scales in the 128x4 swizzled layout."""
    nb = cols // 32
    r = torch.arange(rows, device=device)[:, None]
    b = torch.arange(nb, device=device)[None, :]
    return (
        (r // 128) * (((nb + 3) // 4) * 512)
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (b % 4)
    ).reshape(-1)


def _input(rows, d, kind, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * d, device="cuda", generator=gen)
    if kind == "wide":  # exponents spread over many binades, some past the limit
        x = x * torch.exp2(torch.randint(-20, 6, x.shape, device="cuda", generator=gen).float())
    elif kind == "sparse":  # all-zero blocks and isolated values
        x = x * (torch.rand(x.shape, device="cuda", generator=gen) < 0.05)
    elif kind == "ties":  # bf16 values on few levels: many equal amax candidates
        x = torch.round(x * 4) / 4
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("params", [SWIGLU_OAI, PLAIN], ids=["swiglu_oai", "plain"])
@pytest.mark.parametrize("kind", ["normal", "wide", "sparse", "ties"])
@pytest.mark.parametrize(("rows", "d"), [(1, 32), (9, 1536), (32, 1056), (130, 96), (300, 3072)])
def test_matches_silu_and_mul_then_mxfp8_quantize(rows, d, kind, params):
    x = _input(rows, d, kind, seed=rows + d)
    q, sf = torch.ops.trtllm.silu_and_mul_mxfp8(x, **params)
    h = torch.ops.trtllm.silu_and_mul(x, **params)
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(h, True)
    assert q.dtype == torch.float8_e4m3fn and tuple(q.shape) == (rows, d)
    assert torch.equal(q.view(torch.uint8), ref_q.view(rows, d).view(torch.uint8))
    off = _scale_offsets(rows, d, x.device)
    assert torch.equal(sf[off], ref_sf.view(torch.uint8).reshape(-1)[off])


def test_rejects_unaligned_width():
    with pytest.raises(AssertionError):
        torch.ops.trtllm.silu_and_mul_mxfp8(
            torch.zeros(2, 2 * 48, device="cuda", dtype=torch.bfloat16)
        )


HIDDEN, INTERMEDIATE = 512, 1024


def _mxfp8_mlp(**params) -> GatedMLP:
    torch.manual_seed(0)
    config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32))
    mlp = GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        bias=False,
        activation=F.silu,
        dtype=torch.bfloat16,
        config=config,
        **{k: v for k, v in params.items() if v is not None},
    ).cuda()

    def mx(out_f, in_f):
        w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.05
        q, s = quant_bf16_to_mxfp8(w, 32)
        return {"weight": q, "weight_scale_inv": s}

    mlp.gate_up_proj.load_weights([mx(INTERMEDIATE, HIDDEN), mx(INTERMEDIATE, HIDDEN)])
    mlp.down_proj.load_weights([mx(HIDDEN, INTERMEDIATE)])
    assert isinstance(mlp.down_proj.quant_method, MXFP8LinearMethod)
    return mlp


def _count_activation(monkeypatch, mlp):
    calls = []
    orig = mlp._apply_activation

    def counted(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(mlp, "_apply_activation", counted)
    return calls


@pytest.mark.parametrize("params", [SWIGLU_OAI, PLAIN], ids=["swiglu_oai", "plain"])
@pytest.mark.parametrize("num_tokens", [1, 9, 130])
def test_gated_mlp_eager_matches_unfused(monkeypatch, num_tokens, params):
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    mlp = _mxfp8_mlp(**params)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    calls = _count_activation(monkeypatch, mlp)
    ref = mlp(x)
    assert len(calls) == 1
    mlp.swiglu_mxfp8_out = True
    assert mlp.down_proj.quant_method.on_eager_quantize_path()
    out = mlp(x)
    assert len(calls) == 1  # the fused kernel ran instead of silu_and_mul + quantize
    assert torch.equal(out, ref)


@pytest.mark.skipif(
    not IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE, reason="needs FlashInfer CuTeDSL MXFP8"
)
@pytest.mark.parametrize("num_tokens", [4, 9, 32])
def test_gated_mlp_decode_graph_matches_unfused(monkeypatch, num_tokens):
    """The decode-graph quantize + GEMM path (the one M3 decode steps run) under CUDA graph capture."""
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    mlp = _mxfp8_mlp(**SWIGLU_OAI)
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        if not lin.quant_method.enable_flashinfer_auto():
            pytest.skip("FlashInfer MXFP8 auto backend unavailable")
        lin.quant_method.tune_decode_graph_backends = True
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
        mlp(x)  # tunes this bucket's tactics
    outs = []
    graphs = []
    for fused in (False, True):
        mlp.swiglu_mxfp8_out = fused
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            with flashinfer_mxfp8_decode_graph_capture():
                assert mlp.down_proj.quant_method.on_decode_graph_path()
                assert mlp._swiglu_mxfp8_out_ok(mlp.gate_up_proj(x)) == fused
                outs.append(mlp(x))
        graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


def test_gating(monkeypatch):
    mlp = _mxfp8_mlp(**SWIGLU_OAI)
    h1 = torch.zeros(4, 2 * INTERMEDIATE, dtype=torch.bfloat16, device="cuda")
    assert not mlp._swiglu_mxfp8_out_ok(h1)  # opt-in
    mlp.swiglu_mxfp8_out = True
    assert mlp._swiglu_mxfp8_out_ok(h1)
    assert not mlp._swiglu_mxfp8_out_ok(h1.view(2, 2, -1))
    assert not mlp._swiglu_mxfp8_out_ok(h1.float())
    assert not mlp._swiglu_mxfp8_out_ok(h1[:, : 2 * 48])
    monkeypatch.setattr(mlp.down_proj.quant_method, "accepts_prequantized_input", lambda m: False)
    assert not mlp._swiglu_mxfp8_out_ok(h1)
    bf16 = GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        bias=False,
        activation=F.silu,
        dtype=torch.bfloat16,
    ).cuda()
    bf16.swiglu_mxfp8_out = True
    assert not bf16._swiglu_mxfp8_out_ok(h1)  # down_proj is not an MXFP8 GEMM
