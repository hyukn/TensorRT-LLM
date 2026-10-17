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
"""MXFP8 Linear re-quantized to NVFP4 W4A4 with a static activation scale (MiniMax-M3 QKV,
TRTLLM_M3_QKV_NVFP4), its exact-M decode-graph tactics, and the activation amax calibration."""

import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4_GEMM_EXACT_M_OP
from tensorrt_llm._torch.models import minimax_m3_qkv_nvfp4 as qkv_nvfp4
from tensorrt_llm._torch.modules import linear as linear_mod
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    _mxfp8_cutlass_op_available,
)
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm._torch.modules.nvfp4_fake_quant import nvfp4_fake_quant
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, MxFp8QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="NVFP4 / MXFP8 GEMMs need SM100-family",
    ),
]

_ENVS = (
    qkv_nvfp4.ENABLE_ENV,
    qkv_nvfp4.SCALES_ENV,
    qkv_nvfp4.MARGIN_ENV,
    qkv_nvfp4.BACKENDS_ENV,
    qkv_nvfp4.KEEP_MXFP8_ENV,
    qkv_nvfp4.CALIBRATE_ENV,
    "TRTLLM_FAKEQUANT_DENSE_NVFP4",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    if not _mxfp8_cutlass_op_available():
        pytest.skip("needs the MXFP8 GEMM ops")
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


def _mxfp8_linear(out_f, in_f, seed=2):
    torch.manual_seed(seed)
    weight = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.05
    w_e4m3, w_scale = quant_bf16_to_mxfp8(weight, 32)
    lin = Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
    ).cuda()
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": w_scale}])
    assert isinstance(lin.quant_method, MXFP8LinearMethod)
    return lin


def _rel(a, b):
    return ((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm()).item()


def _nvfp4_roundtrip(t, global_scale):
    """fp4_quantize (swizzled scales) then the reference dequantizer."""
    fp4, sf = torch.ops.trtllm.fp4_quantize(t, global_scale, 16, False, True)
    return torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        fp4.cpu(), sf.view(torch.uint8).flatten().cpu(), (1.0 / global_scale).cpu(), 16, 1, True
    ).reshape(t.shape)


@pytest.mark.parametrize("out_f, in_f", [(2560, 6144), (384, 512)])
@pytest.mark.parametrize("backends", [("cublaslt", "cutedsl"), ("cutlass",)])
def test_repack_runs_the_nvfp4_rounded_operands(out_f, in_f, backends):
    lin = _mxfp8_linear(out_f, in_f)
    w_deq = lin.quant_method.dequantized_weight(lin)
    x = torch.randn(24, in_f, dtype=torch.bfloat16, device="cuda") * torch.logspace(
        -1, 1, in_f, device="cuda"
    ).to(torch.bfloat16)
    native = lin(x)
    amax = float(x.float().abs().max()) * 2.0
    lin.quant_method.repack_to_nvfp4(lin, amax, allowed_backends=backends)
    assert lin.weight.numel() == 0 and lin.weight_scale.numel() == 0  # MXFP8 weight freed
    assert not lin.quant_method.accepts_prequantized_input(lin)
    out = lin(x)
    assert out.shape == native.shape and out.dtype == torch.bfloat16
    # The operands are the production NVFP4 quantizations of the activation (static global scale
    # 448 * 6 / amax) and of the dequantized MXFP8 weight: only the GEMM's accumulation differs.
    gs_x = torch.tensor([448 * 6 / amax], device="cuda")
    gs_w = (448 * 6 / w_deq.abs().amax()).reshape(1).float()
    ref = F.linear(_nvfp4_roundtrip(x, gs_x), _nvfp4_roundtrip(w_deq.to(torch.bfloat16), gs_w))
    assert _rel(out, ref) < 5e-3
    # Same format as the fake-quant study; its rounding differs only at exact ties (x / s vs
    # x * (1 / s)), a few tenths of a percent of the elements.
    fake = F.linear(
        nvfp4_fake_quant(x, global_amax=torch.tensor(amax)),
        nvfp4_fake_quant(w_deq).to(torch.bfloat16),
    )
    assert _rel(out, fake) < 3e-2
    assert 0.02 < _rel(out, native) < 0.3  # a 4-bit GEMM, not today's MXFP8 one
    # 3-D input and pre-quantized MXFP8 input.
    torch.testing.assert_close(lin(x.view(2, 12, in_f)), out.view(2, 12, out_f), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="bf16 activation"):
        lin(MxFp8QuantizedTensor(*torch.ops.trtllm.mxfp8_quantize(x, True), is_sf_swizzled=True))


def test_repack_keeps_mxfp8_weight_on_request():
    lin = _mxfp8_linear(256, 512)
    shape = tuple(lin.weight.shape)
    lin.quant_method.repack_to_nvfp4(lin, 10.0, free_mxfp8=False)
    assert tuple(lin.weight.shape) == shape
    with pytest.raises(ValueError, match="positive"):
        _mxfp8_linear(256, 512).quant_method.repack_to_nvfp4(lin, 0.0)


def test_decode_graph_path_tunes_and_replays_the_exact_m_tactic():
    lin = _mxfp8_linear(2560, 6144, seed=3)
    # CUTLASS + cuBLASLt keep the profiling short (CuTe DSL tactics JIT-compile); same mechanism.
    lin.quant_method.repack_to_nvfp4(lin, 40.0, allowed_backends=("cublaslt", "cutlass"))
    AutoTuner.get().clear_cache()
    xs = {m: torch.randn(m, 6144, dtype=torch.bfloat16, device="cuda") for m in (8, 24, 104)}
    with linear_mod.flashinfer_mxfp8_decode_graph_capture():
        token = linear_mod._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.set(True)  # the warmup-only pass
        try:
            tuned = {m: lin(x) for m, x in xs.items()}
        finally:
            linear_mod._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.reset(token)
        entries = AutoTuner.get().profiling_cache.get_specific_custom_op(NVFP4_GEMM_EXACT_M_OP)
        tuned_ms = {key[3][0][0] for key in entries}
        assert tuned_ms == set(xs), tuned_ms  # one pick per exact M (not per power-of-two bucket)
        # Capture pass: cache lookups only, replay the picks.
        static_x = xs[104].clone()
        lin(static_x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = lin(static_x)
        graph.replay()
        torch.cuda.synchronize()
        eager = lin(xs[104])
    torch.testing.assert_close(static_out, eager, rtol=0, atol=0)
    assert _rel(eager, tuned[104]) < 1e-2
    # Outside the decode-graph context the unified (power-of-two bucket) op runs.
    assert _rel(lin(xs[24]), tuned[24]) < 1e-2


def test_calibration_records_the_running_input_amax_in_graphs():
    lin = _mxfp8_linear(256, 512)
    lin.quant_method.enable_input_amax_calibration(lin)
    assert not lin.quant_method.accepts_prequantized_input(lin)
    x = torch.randn(16, 512, dtype=torch.bfloat16, device="cuda")
    native = lin(x)
    lin(x * 0.5)
    assert lin._nvfp4_calib_amax.item() == pytest.approx(x.float().abs().max().item())
    static_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = lin(static_x)
    static_x.mul_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert lin._nvfp4_calib_amax.item() == pytest.approx(3.0 * x.float().abs().max().item())
    torch.testing.assert_close(static_out, lin(static_x), rtol=0, atol=0)  # the GEMM is unchanged
    torch.testing.assert_close(lin(x), native, rtol=0, atol=0)


class _Attn(nn.Module):
    def __init__(self, proj, sparse=False):
        super().__init__()
        self.qkv_proj = proj
        self.is_sparse_attention_layer = sparse
        self.enable_fused_qkv_index_projection = sparse


class _Layer(nn.Module):
    def __init__(self, proj, sparse=False, hidden=512):
        super().__init__()
        self.self_attn = _Attn(proj, sparse)
        self.input_layernorm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16).cuda()


def test_setup_calibrates_then_repacks_from_the_merged_file(monkeypatch, tmp_path):
    layers = nn.ModuleList([_Layer(_mxfp8_linear(256, 512, seed=s)) for s in (4, 5)])
    x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
    assert qkv_nvfp4.setup(layers, rank=0) == ("off", 0)

    prefix = str(tmp_path / "calib.json")
    monkeypatch.setenv(qkv_nvfp4.CALIBRATE_ENV, prefix)
    assert qkv_nvfp4.setup(layers, rank=3) == ("calibrate", 2)
    for i, layer in enumerate(layers):
        layer.self_attn.qkv_proj(x * (i + 1))
    layers[0].self_attn.qkv_proj._nvfp4_calib_flush(True)
    merged_path = str(tmp_path / "merged.json")
    merged = qkv_nvfp4.merge_calibration([f"{prefix}.rank3"], merged_path)
    amax = x.float().abs().max().item()
    assert merged["0"] == pytest.approx(amax) and merged["1"] == pytest.approx(2 * amax)
    assert qkv_nvfp4.load_scales(merged_path) == pytest.approx({0: amax, 1: 2 * amax})

    monkeypatch.setenv(qkv_nvfp4.ENABLE_ENV, "1")
    with pytest.raises(ValueError, match="exclusive"):
        qkv_nvfp4.setup(layers, rank=0)
    monkeypatch.delenv(qkv_nvfp4.CALIBRATE_ENV)
    with pytest.raises(ValueError, match=qkv_nvfp4.SCALES_ENV):
        qkv_nvfp4.setup(layers, rank=0)
    partial = str(tmp_path / "partial.json")
    with open(partial, "w") as f:
        json.dump({"amax": {"0": amax}}, f)
    monkeypatch.setenv(qkv_nvfp4.SCALES_ENV, partial)
    with pytest.raises(ValueError, match=r"\[1\]"):
        qkv_nvfp4.setup(layers, rank=0)

    monkeypatch.setenv(qkv_nvfp4.SCALES_ENV, merged_path)
    monkeypatch.setenv(qkv_nvfp4.MARGIN_ENV, "4")
    monkeypatch.setenv(qkv_nvfp4.BACKENDS_ENV, "cutlass")
    assert qkv_nvfp4.setup(layers, rank=0) == ("nvfp4", 2)
    proj = layers[1].self_attn.qkv_proj
    assert proj._nvfp4_repacked and proj._nvfp4_backends == ("cutlass",)
    assert proj._nvfp4_input_scale.item() == pytest.approx(448 * 6 / (2 * amax * 4), rel=1e-6)
    assert qkv_nvfp4.setup(layers, rank=0) == ("nvfp4", 0)  # already repacked: left alone


@pytest.mark.parametrize("repack", [False, True])
def test_linear_traces_under_torch_compile_fullgraph(monkeypatch, repack):
    """Piecewise-compiled CTX servers trace MXFP8LinearMethod.apply: the fake-quant mode / NVFP4 checks must not
    break the graph (Dynamo cannot trace ContextVar.get)."""
    torch._dynamo.reset()
    lin = _mxfp8_linear(256, 512)
    if repack:
        lin.quant_method.repack_to_nvfp4(lin, 20.0, allowed_backends=("cutlass",))
    x = torch.randn(16, 512, dtype=torch.bfloat16, device="cuda")
    eager = lin(x)
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "1")  # ignored while tracing
    compiled = torch.compile(lambda t: lin(t), backend="eager", fullgraph=True)
    assert _rel(compiled(x), eager) < 1e-2  # the compiled MXFP8 path may pick another GEMM backend


def test_fused_input_norm_feeds_the_repacked_projection(monkeypatch, tmp_path):
    """Sparse layers: the input layernorm emits NVFP4 with the projection's static scale (fused add + RMSNorm +
    quantize); the projection consumes it. Dense layers keep the bf16 norm."""
    torch.manual_seed(9)
    layers = nn.ModuleList(
        [
            _Layer(_mxfp8_linear(384, 512, seed=7), sparse=False),
            _Layer(_mxfp8_linear(384, 512, seed=8), sparse=True),
        ]
    )
    for layer in layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
    w_deq_sparse = layers[1].self_attn.qkv_proj.quant_method.dequantized_weight(
        layers[1].self_attn.qkv_proj
    )
    path = str(tmp_path / "amax.json")
    with open(path, "w") as f:
        json.dump({"amax": {"0": 8.0, "1": 8.0}}, f)
    monkeypatch.setenv(qkv_nvfp4.ENABLE_ENV, "1")
    monkeypatch.setenv(qkv_nvfp4.SCALES_ENV, path)
    assert qkv_nvfp4.setup(layers, rank=0) == ("nvfp4", 2)
    dense_norm, sparse_norm = layers[0].input_layernorm, layers[1].input_layernorm
    assert not dense_norm.is_nvfp4 and sparse_norm.is_nvfp4
    proj = layers[1].self_attn.qkv_proj
    h = torch.randn(24, 512, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(24, 512, dtype=torch.bfloat16, device="cuda")
    fp4, res_fused = sparse_norm(h, residual.clone())
    assert isinstance(fp4, Fp4QuantizedTensor)
    sparse_norm.is_nvfp4 = False  # the unfused reference: bf16 norm, then the projection quantizes
    normed, res_ref = sparse_norm(h, residual.clone())
    sparse_norm.is_nvfp4 = True
    torch.testing.assert_close(res_fused, res_ref, rtol=0, atol=0)
    # The fused kernel quantizes the fp32 normed value, the unfused path its bf16 rounding: some elements land on
    # the neighbouring E2M1 point. Same scales, same accuracy vs the bf16 GEMM of the normed activation.
    exact = F.linear(normed, w_deq_sparse.to(torch.bfloat16))
    fused_err, unfused_err = _rel(proj(fp4), exact), _rel(proj(normed), exact)
    assert fused_err < 0.25 and fused_err < 1.15 * unfused_err, (fused_err, unfused_err)
    assert _rel(proj(fp4), proj(normed)) < 5e-2

    monkeypatch.setenv(qkv_nvfp4.FUSE_NORM_ENV, "0")
    fresh = nn.ModuleList([_Layer(_mxfp8_linear(384, 512, seed=10), sparse=True)])
    with open(path, "w") as f:
        json.dump({"amax": {"0": 8.0}}, f)
    assert qkv_nvfp4.setup(fresh, rank=0) == ("nvfp4", 1)
    assert not fresh[0].input_layernorm.is_nvfp4
