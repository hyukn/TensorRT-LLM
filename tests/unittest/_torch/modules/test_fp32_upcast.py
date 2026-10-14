# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vectorized upcast must produce exactly what tensor.to(torch.float32) produces."""

import pytest
import torch

from tensorrt_llm._torch.modules.fp32_upcast import MIN_NUMEL, upcast_to_fp32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(16384, 6144), (1537, 6144), (MIN_NUMEL + 3,), (7, 5)])
def test_bitwise_equal_to_torch_cast(dtype, shape):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(shape, generator=g, device="cuda") * 30).to(dtype)
    flat = x.view(-1)
    specials = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1e-40, -1e-40], device="cuda"
    ).to(dtype)
    flat[: specials.numel()] = specials
    flat[-1] = torch.finfo(dtype).max
    out = upcast_to_fp32(x)
    ref = x.to(torch.float32)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def test_falls_back_for_unsupported_inputs():
    x = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16).t()  # non-contiguous
    assert torch.equal(upcast_to_fp32(x), x.to(torch.float32))
    y = torch.randn(2048, 1024, device="cuda", dtype=torch.float32)
    assert torch.equal(upcast_to_fp32(y), y)


def test_minimax_m3_gate_unchanged(monkeypatch):
    from tensorrt_llm._torch.models import modeling_minimaxm3 as m

    gate = m.MiniMaxM3Gate(hidden_size=6144, num_experts=128, top_k=4, routed_scaling_factor=2.0)
    gate = gate.cuda()
    with torch.no_grad():
        gate.weight.copy_(torch.randn(128, 6144, device="cuda") * 0.02)
    x = torch.randn(4096, 6144, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(m, "_ROUTER_FAST_UPCAST", False)
    ref = gate(x)
    monkeypatch.setattr(m, "_ROUTER_FAST_UPCAST", True)
    out = gate(x)
    assert torch.equal(out, ref)
