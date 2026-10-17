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
"""NVFP4 fake quantization (accuracy study for the MXFP8 dense linears, TRTLLM_FAKEQUANT_DENSE_NVFP4)."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    _mxfp8_cutlass_op_available,
)
from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight, quant_bf16_to_mxfp8
from tensorrt_llm._torch.modules.nvfp4_fake_quant import (
    NVFP4_BLOCK,
    ROLES,
    _round_e2m1,
    activation_fake_quant,
    fake_quant_mode,
    fake_quant_rows,
    fake_quant_suspended,
    fp8_per_tensor_fake_quant,
    mx_fake_quant,
    nvfp4_fake_quant,
    tag_linear_roles,
    weight_fake_quant,
)
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@pytest.mark.cpu_only
def test_round_e2m1_is_nearest_with_ties_to_even_and_saturates():
    a = torch.tensor([0.0, 0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, 6.0, 7.0, 100.0])
    want = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0])
    assert torch.equal(_round_e2m1(a), want)


@pytest.mark.cpu_only
@pytest.mark.parametrize("shape", [(7, 64), (3, 5, 256)])
@pytest.mark.parametrize("block, target", [(NVFP4_BLOCK, 448.0 * 6.0), (32, 448.0)])
def test_blocks_are_e2m1_times_an_e4m3_scale(shape, block, target):
    torch.manual_seed(0)
    x = torch.randn(shape) * torch.logspace(-3, 2, shape[-1])
    y = nvfp4_fake_quant(x, block=block, global_target=target)
    if (block, target) == (NVFP4_BLOCK, 448.0 * 6.0):
        assert torch.equal(y, nvfp4_fake_quant(x))  # the NVFP4 defaults
    assert y.shape == x.shape and y.dtype == x.dtype
    gs = target / x.abs().amax()
    yb = y.reshape(-1, block).float()
    xb = x.reshape(-1, block).float()
    sf = (
        (xb.abs().amax(-1, keepdim=True) / 6.0 * gs)
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    scale = sf / gs
    ok = scale.squeeze(-1) > 0
    q = (yb[ok] / scale[ok]).abs()
    # Every element is a grid point times its block's scale (up to fp32 rounding of the product) ...
    on_grid = _GRID[torch.argmin((q.unsqueeze(-1) - _GRID).abs(), dim=-1)]
    torch.testing.assert_close(q, on_grid, rtol=1e-6, atol=1e-6)
    assert torch.all(yb[~ok] == 0)
    # ... namely the grid point nearest to |x| / scale, saturated at 6 (the E4M3 rounding of a block scale can
    # leave the block amax above 6 x scale); only exact ties may pick the other neighbour.
    a = (xb[ok] / scale[ok]).abs().clamp(max=6.0)
    nearest = _GRID[torch.argmin((a.unsqueeze(-1) - _GRID).abs(), dim=-1)]
    assert ((q - nearest).abs() > 1e-5).float().mean() < 1e-3
    # Error bound away from saturation: half a grid step (at most 1 at the top of the grid) times the block scale.
    unsat = (xb.abs() <= 6.0 * scale) & ok.unsqueeze(-1)
    assert torch.all((yb - xb).abs()[unsat] <= (scale.expand_as(xb) * (1.0 + 1e-5))[unsat])


@pytest.mark.cpu_only
def test_fp8_per_tensor_fake_quant_and_the_mode_operands():
    torch.manual_seed(3)
    xf = torch.randn(5, 96) * torch.logspace(-4, 1, 96)
    yf = fp8_per_tensor_fake_quant(xf)
    scale = xf.abs().amax() / 448.0
    q = yf / scale
    # On the E4M3 grid (up to the fp32 rounding of the product) and the amax maps to 448.
    torch.testing.assert_close(q.to(torch.float8_e4m3fn).float(), q, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(q.abs().max(), torch.tensor(448.0), rtol=1e-6, atol=0.0)
    x = xf.to(torch.bfloat16)
    y = fp8_per_tensor_fake_quant(x)
    assert y.dtype == x.dtype and y.shape == x.shape
    scale = (
        (x.float().abs().amax() / 448.0).to(torch.bfloat16).float()
    )  # the scale is kept in x's dtype
    q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    assert torch.equal(y, (q * scale).to(x.dtype))
    # Modes: 1 = W4A16, 2 = W4A4 (NVFP4 activations), 3 = W4A8 (block-32 weights, FP8 per-tensor activations).
    w = torch.randn(48, 96)
    assert torch.equal(weight_fake_quant(w, 1), nvfp4_fake_quant(w))
    assert torch.equal(weight_fake_quant(w, 2), nvfp4_fake_quant(w))
    assert torch.equal(weight_fake_quant(w, 3), nvfp4_fake_quant(w, block=32, global_target=448.0))
    assert torch.equal(activation_fake_quant(x, 1), x)
    assert torch.equal(activation_fake_quant(x, 2), nvfp4_fake_quant(x))
    assert torch.equal(activation_fake_quant(x, 3), y)
    # 4 = MXFP4 weights x MXFP8 activations.
    assert torch.equal(weight_fake_quant(w, 4), mx_fake_quant(w, "e2m1"))
    assert torch.equal(activation_fake_quant(x, 4), mx_fake_quant(x, "e4m3"))
    # 5 = the reference: weight and activation untouched (the MXFP8 weight, bf16 activations).
    assert torch.equal(weight_fake_quant(w, 5), w)
    assert torch.equal(activation_fake_quant(x, 5), x)


@pytest.mark.cpu_only
def test_mx_fake_quant_formats():
    torch.manual_seed(5)
    x = torch.randn(6, 128) * torch.logspace(-4, 2, 128)
    # MXFP8: identical to the reference MXFP8 quantize + dequantize (the checkpoint / activation format).
    assert torch.equal(
        mx_fake_quant(x, "e4m3"), dequant_mxfp8_weight(*quant_bf16_to_mxfp8(x, 32), 32)
    )
    # MXFP4: power-of-two block scales that just fit the block amax (amax / scale in (3, 6]), E2M1 elements.
    y = mx_fake_quant(x, "e2m1").reshape(-1, 32)
    xb = x.reshape(-1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(xb.abs().amax(-1, keepdim=True) / 6.0)))
    assert torch.equal(torch.log2(scale), torch.round(torch.log2(scale)))
    r = xb.abs().amax(-1) / scale.squeeze(-1)
    assert torch.all((r > 3.0) & (r <= 6.0))
    q = (y / scale).abs()
    torch.testing.assert_close(q, _round_e2m1((xb / scale).abs()), rtol=0, atol=0)
    assert torch.all((y - xb).abs() <= scale * 1.0 + 1e-12)


@requires_cuda
@pytest.mark.parametrize("rows, cols", [(64, 256), (256, 6144)])
def test_matches_the_nvfp4_quantize_dequantize_ops(rows, cols):
    """The production NVFP4 quantizer + dequantizer give (practically) the same values."""
    torch.manual_seed(1)
    x = (torch.randn(rows, cols, device="cuda") * torch.logspace(-2, 1, cols, device="cuda")).to(
        torch.bfloat16
    )
    gs = (448.0 * 6.0 / x.float().abs().amax()).reshape(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(x, gs, NVFP4_BLOCK, False, False)
    ref = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        fp4.cpu(), sf.view(torch.uint8).flatten().cpu(), (1.0 / gs).cpu(), NVFP4_BLOCK, 1, False
    ).reshape(rows, cols)
    got = nvfp4_fake_quant(x.float()).cpu()
    # Same grid point and block scale; the products differ by fp32 rounding (scale x q vs q x sf / gs), and only
    # exact ties may round to the other neighbour (the op multiplies by a reciprocal scale).
    mismatch = (~torch.isclose(got, ref, rtol=1e-5, atol=0.0)).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2e} of the elements differ"
    assert (got - ref).abs().max() <= (ref.abs().amax() / 6.0 * 2.0)


@requires_cuda
def test_w4a8_activation_matches_quantize_e4m3_per_tensor():
    """Mode 3's activation is the dynamic per-tensor FP8 of the W4A8 NVFP4 x FP8 path (quantize_e4m3_per_tensor).
    Its weight format (32-element E4M3 block scales, 448 / amax global scale) has no GPU quantizer in the tree
    (fp4_quantize takes E4M3 scales with 16-element blocks only; such checkpoints come from ModelOpt), so the block-32
    case of test_blocks_are_e2m1_times_an_e4m3_scale covers it."""
    torch.manual_seed(4)
    x = (torch.randn(24, 512, device="cuda") * torch.logspace(-3, 1, 512, device="cuda")).to(
        torch.bfloat16
    )
    x8, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    ref_x = (x8.view(torch.float8_e4m3fn).float() * x_scale.float().reshape(1)).to(torch.bfloat16)
    got_x = activation_fake_quant(x, 3)
    torch.testing.assert_close(got_x, ref_x, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("mode", [2, 3, 4])
def test_activation_fake_quant_is_cuda_graph_capturable(mode):
    """The per-call activation rounding runs inside the decode CUDA graphs (no host tensors / syncs)."""
    torch.manual_seed(7)
    x = (torch.randn(16, 256, device="cuda") * 3).to(torch.bfloat16)
    eager = activation_fake_quant(x, mode)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        activation_fake_quant(x, mode)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = activation_fake_quant(x, mode)
    x.mul_(2.0)  # replay reads the new values
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, activation_fake_quant(x, mode))
    assert not torch.equal(out, eager)


@requires_cuda
def test_mx_operands_match_the_mxfp8_and_mxfp4_quantizers():
    """Mode 4 rounds like the production quantizers: mxfp8_quantize for the activation (so mode 4's activation is
    today's) and fp4_quantize with UE8M0 32-element scales for the MXFP4 weight."""
    torch.manual_seed(6)
    x = (torch.randn(24, 512, device="cuda") * torch.logspace(-3, 1, 512, device="cuda")).to(
        torch.bfloat16
    )
    x8, x_sf = torch.ops.trtllm.mxfp8_quantize(x, False)
    ref_x = dequant_mxfp8_weight(
        x8.view(torch.float8_e4m3fn), x_sf.view(torch.uint8).reshape(24, 16), 32
    )
    torch.testing.assert_close(mx_fake_quant(x, "e4m3").float(), ref_x, rtol=0, atol=0)
    w = (torch.randn(64, 512, device="cuda") * torch.logspace(-2, 1, 512, device="cuda")).to(
        torch.bfloat16
    )
    fp4, sf = torch.ops.trtllm.fp4_quantize(w, None, 32, True, False)
    ref_w = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        fp4.cpu(), sf.view(torch.uint8).flatten().cpu(), None, 32, 0, False
    ).reshape(w.shape)
    got_w = mx_fake_quant(w, "e2m1").float().cpu()
    assert (~torch.isclose(got_w, ref_w, rtol=1e-5, atol=0.0)).float().mean() < 1e-3


def _mxfp8_linear(out_f, in_f):
    torch.manual_seed(2)
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
    w_deq = dequant_mxfp8_weight(w_e4m3.cuda(), w_scale.cuda(), 32)
    return lin, w_deq


@requires_cuda
@pytest.mark.skipif(not _mxfp8_cutlass_op_available(), reason="needs the MXFP8 GEMM ops")
@pytest.mark.parametrize("out_f, in_f", [(6144, 2048), (1536, 6144), (224, 96)])
def test_mxfp8_linear_runs_the_nvfp4_rounded_weight(monkeypatch, out_f, in_f):
    lin, w_deq = _mxfp8_linear(out_f, in_f)
    x = torch.randn(24, in_f, dtype=torch.bfloat16, device="cuda")
    monkeypatch.delenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", raising=False)
    monkeypatch.delenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", raising=False)
    native = lin(x)
    assert not hasattr(lin, "_fq_nvfp4_weight")

    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "1")
    w_fq = nvfp4_fake_quant(w_deq).to(torch.bfloat16)
    out = lin(x)
    # The weight was rebuilt from the swizzled MXFP8 scales: identical to rounding the reference dequant.
    assert torch.equal(lin._fq_nvfp4_weight, w_fq)
    torch.testing.assert_close(out, F.linear(x, w_fq), rtol=0, atol=0)
    rel = ((out.float() - native.float()).norm() / native.float().norm()).item()
    assert 0.0 < rel < 0.3
    assert not lin.quant_method.accepts_prequantized_input(lin)

    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "2")
    torch.testing.assert_close(lin(x), F.linear(nvfp4_fake_quant(x), w_fq), rtol=0, atol=0)
    assert lin._fq_nvfp4_weight is not None and torch.equal(
        lin._fq_nvfp4_weight, w_fq
    )  # shared with mode 1

    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "3")
    w_fq3 = weight_fake_quant(w_deq, 3).to(torch.bfloat16)
    torch.testing.assert_close(
        lin(x), F.linear(fp8_per_tensor_fake_quant(x), w_fq3), rtol=0, atol=0
    )
    assert torch.equal(lin._fq_nvfp4_weight, w_fq3)

    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "4")
    w_fq4 = mx_fake_quant(w_deq, "e2m1").to(torch.bfloat16)
    torch.testing.assert_close(lin(x), F.linear(mx_fake_quant(x, "e4m3"), w_fq4), rtol=0, atol=0)

    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "5")
    torch.testing.assert_close(lin(x), F.linear(x, w_deq.to(torch.bfloat16)), rtol=0, atol=0)
    assert torch.equal(lin._fq_nvfp4_weight.float(), w_deq)  # MXFP8 values are exact in bf16

    # Back to mode 1: the weight is rebuilt for the NVFP4 format.
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "1")
    torch.testing.assert_close(lin(x), F.linear(x, w_fq), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="bf16 activation"):
        lin(MxFp8QuantizedTensor(*torch.ops.trtllm.mxfp8_quantize(x, True), is_sf_swizzled=True))

    monkeypatch.delenv("TRTLLM_FAKEQUANT_DENSE_NVFP4")
    assert torch.equal(lin(x), native)  # default path untouched


@pytest.mark.cpu_only
def test_roles_tagging_and_row_selection(monkeypatch):
    class Block(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.proj = torch.nn.Linear(4, n, bias=False)

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(4, 14, bias=False)
            self.qkv_proj.local_output_sizes = (8, 2, 2, 1, 1)  # [Q | K | V | index-Q | index-K]
            self.o_proj = torch.nn.Linear(8, 4, bias=False)

    class MoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_experts = Block(6)
            self.gate = torch.nn.Linear(4, 3, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self, dense):
            super().__init__()
            self.self_attn = Attn()
            if dense:
                self.mlp = Block(6)
            else:
                self.block_sparse_moe = MoE()

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Layer(True), Layer(False)])
    assert tag_linear_roles(model) == 6
    l0, l1 = model.layers
    assert (l0.self_attn.qkv_proj._fq_role, l0.self_attn.o_proj._fq_role, l0.mlp.proj._fq_role) == (
        "qkv",
        "o",
        "dense",
    )
    assert l1.block_sparse_moe.shared_experts.proj._fq_role == "shared"
    assert not hasattr(l1.block_sparse_moe.gate, "_fq_role")

    qkv, o, shared = (
        l1.self_attn.qkv_proj,
        l1.self_attn.o_proj,
        l1.block_sparse_moe.shared_experts.proj,
    )
    monkeypatch.delenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", raising=False)
    assert fake_quant_rows(qkv) is None and fake_quant_rows(o) is None  # no filter: everything
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", ",".join(ROLES))
    assert fake_quant_rows(qkv) is None
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "index")
    assert (
        fake_quant_rows(qkv) == [(12, 14)]
        and fake_quant_rows(o) == []
        and fake_quant_rows(shared) == []
    )
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "qkv, shared")
    assert (
        fake_quant_rows(qkv) == [(0, 12)]
        and fake_quant_rows(shared) is None
        and fake_quant_rows(o) == []
    )
    assert fake_quant_rows(l0.mlp.proj) == []
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "qkv,index")
    assert fake_quant_rows(qkv) is None
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "qkv+index+o")  # '+' separates too
    assert (
        fake_quant_rows(qkv) is None
        and fake_quant_rows(o) is None
        and fake_quant_rows(shared) == []
    )
    del qkv.local_output_sizes  # a dense-attention layer's plain fused QKV: no index rows
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "index")
    assert fake_quant_rows(qkv) == []
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "qkv")
    assert fake_quant_rows(qkv) is None
    with pytest.raises(RuntimeError, match="tag_linear_roles"):
        fake_quant_rows(l1.block_sparse_moe.gate)
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "qkv,router")
    with pytest.raises(ValueError, match="router"):
        fake_quant_rows(qkv)

    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "2")
    assert fake_quant_mode() == 2
    with fake_quant_suspended():
        assert fake_quant_mode() == 0
    assert fake_quant_mode() == 2


@requires_cuda
@pytest.mark.skipif(not _mxfp8_cutlass_op_available(), reason="needs the MXFP8 GEMM ops")
def test_mxfp8_linear_role_filter_recomputes_only_the_selected_rows(monkeypatch):
    lin, w_deq = _mxfp8_linear(6144, 2048)
    lin._fq_role = "qkv"
    lin.local_output_sizes = (4096, 512, 512, 512, 512)
    x = torch.randn(24, 2048, dtype=torch.bfloat16, device="cuda")
    monkeypatch.delenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", raising=False)
    monkeypatch.delenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", raising=False)
    native = lin(x)
    w_fq = nvfp4_fake_quant(w_deq).to(torch.bfloat16)
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4", "2")
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "index")
    out = lin(x)
    assert torch.equal(out[:, :5120], native[:, :5120])  # Q / K / V: today's MXFP8 GEMM
    torch.testing.assert_close(
        out[:, 5120:], F.linear(nvfp4_fake_quant(x), w_fq[5120:]), rtol=0, atol=0
    )
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "o,shared")
    assert torch.equal(lin(x), native)  # role not selected: today's GEMM
    monkeypatch.setenv("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "qkv")
    out = lin(x)
    torch.testing.assert_close(
        out[:, :5120], F.linear(nvfp4_fake_quant(x), w_fq[:5120]), rtol=0, atol=0
    )
    assert torch.equal(out[:, 5120:], native[:, 5120:])
