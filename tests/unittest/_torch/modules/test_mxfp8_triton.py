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
"""The Triton MXFP8 quantize building blocks match torch.ops.trtllm.mxfp8_quantize bit for bit."""

import pytest
import torch

from tensorrt_llm._torch.modules.mxfp8_triton import mxfp8_quantize_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 (UE8M0 / E4M3 cvt) needs SM100+",
)


def swizzled_valid_offsets(m: int, k: int, device) -> torch.Tensor:
    """Offsets of the (row < m) scales in the 128x4 swizzled layout, row-major over (row, block)."""
    num_blocks = k // 32
    num_k_tiles = (num_blocks + 3) // 4
    rows = torch.arange(m, device=device)[:, None]
    blks = torch.arange(num_blocks, device=device)[None, :]
    off = (
        (rows // 128) * (num_k_tiles * 512)
        + (blks // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (blks % 4)
    )
    return off.reshape(-1)


def _inputs(kind: str, m: int, k: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(m * 131 + k)
    x = torch.randn((m, k), generator=g, device="cuda", dtype=torch.float32)
    if kind == "normal":
        pass
    elif kind == "wide":
        # Block magnitudes from 2^-30 to 2^30: every scale exponent, saturation, subnormal E4M3.
        e = torch.randint(-30, 31, (m, k // 32, 1), generator=g, device="cuda").float()
        x = (x.view(m, k // 32, 32) * torch.exp2(e)).view(m, k)
    elif kind == "sparse":
        x = x * (torch.rand((m, k), generator=g, device="cuda") < 0.05)
        x[:, :32] = 0.0  # an all-zero block
        x[:, 32:64] = -0.0
    elif kind == "ties":
        # Values on E4M3 rounding ties and at the 448 boundary after scaling.
        base = torch.tensor([448.0, 464.0, 480.0, 1.0625, 1.1875, -2.125, 0.0, 17.0], device="cuda")
        x = base.repeat(m, k // 8) * torch.exp2(
            torch.randint(-4, 5, (m, 1), generator=g, device="cuda").float()
        )
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("kind", ["normal", "wide", "sparse", "ties"])
@pytest.mark.parametrize("m, k", [(1, 2048), (4, 2048), (24, 768), (37, 6144), (130, 2048)])
def test_matches_trtllm_mxfp8_quantize(kind: str, m: int, k: int) -> None:
    x = _inputs(kind, m, k)
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    got_q, got_sf = mxfp8_quantize_triton(x)
    assert got_q.shape == ref_q.reshape(m, k).shape
    assert got_sf.numel() == ref_sf.numel()
    assert torch.equal(got_q.view(torch.uint8), ref_q.reshape(m, k).view(torch.uint8))
    off = swizzled_valid_offsets(m, k, x.device)
    assert torch.equal(got_sf[off], ref_sf.view(torch.uint8).reshape(-1)[off])


@pytest.mark.parametrize("kind", ["normal", "wide", "sparse"])
@pytest.mark.parametrize("m, k", [(4, 2048), (24, 768), (33, 6144)])
def test_matches_flashinfer_cute_dsl_mxfp8_quantize(kind: str, m: int, k: int) -> None:
    """The other tactic of MXFP8QuantizeRunner: the autotuner may pick either."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner

    x = _inputs(kind, m, k)
    try:
        ref_q, ref_sf = MXFP8QuantizeRunner(x.dtype)([x], tactic=MXFP8QuantizeRunner.CUTE_DSL)
    except Exception as e:  # pragma: no cover - FlashInfer build without the CuTeDSL quantizer
        pytest.skip(f"FlashInfer CuTeDSL MXFP8 quantize unavailable: {e}")
    got_q, got_sf = mxfp8_quantize_triton(x)
    assert torch.equal(got_q.view(torch.uint8), ref_q.reshape(m, k).view(torch.uint8))
    off = swizzled_valid_offsets(m, k, x.device)
    assert torch.equal(got_sf[off], ref_sf.view(torch.uint8).reshape(-1)[off])
