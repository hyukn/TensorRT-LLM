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
"""Triton building blocks that reproduce ``torch.ops.trtllm.mxfp8_quantize`` bit for bit.

``mxfp8_quantize_rows`` turns blocks of 32 bf16 values (already rounded to bf16,
then widened to fp32) into E4M3 values and one UE8M0 scale per block with the same
instruction sequence as ``cvt_warp_fp16_to_mxfp8`` (cpp/tensorrt_llm/kernels/
quantization.cuh, built with --use_fast_math):

    sf     = cvt.rp.satfinite.ue8m0(amax *ftz rcp.approx.ftz(448))
    scale  = amax != 0 ? rcp.approx.ftz(float(sf)) : 0
    e4m3   = cvt.rn.satfinite.e4m3(x *ftz scale)

so a producer kernel can emit the quantized activation of an MXFP8 GEMM directly.
``sf_offset_128x4`` is ``get_sf_out_offset_128x4`` (the swizzled scale layout the
MXFP8 GEMMs read).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul_ftz(a, b):
    return tl.inline_asm_elementwise(
        "mul.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _f32_to_ue8m0_rp(x):
    """``__nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundPosInf)`` as a uint8 code."""
    code = tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rp.satfinite.ue8m0x2.f32 h, 0f00000000, $1; cvt.u32.u16 $0, h; }",
        "=r,f",
        [x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return (code & 0xFF).to(tl.uint8)


@triton.jit
def _ue8m0_to_f32(code):
    """``static_cast<float>(__nv_fp8_e8m0)``: 2^(code - 127); code 0 is the subnormal 2^-127."""
    c = code.to(tl.uint32)
    bits = tl.where(c == 0, 0x00400000, c << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_e4m3_rn_sat(x):
    """``__nv_fp8x2_e4m3(float2)``: cvt.rn.satfinite.e4m3x2.f32, one value at a time."""
    code = tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, $1; cvt.u32.u16 $0, h; }",
        "=r,f",
        [x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return (code & 0xFF).to(tl.uint8)


@triton.jit
def mxfp8_quantize_rows(x):
    """Quantize ``x`` [R, 32] (fp32 holding bf16 values) block-wise.

    Returns (e4m3 bytes [R, 32] as uint8, UE8M0 scale codes [R] as uint8).
    """
    amax = tl.max(tl.abs(x), axis=1)
    sf_code = _f32_to_ue8m0_rp(
        _mul_ftz(amax, _rcp_approx_ftz(tl.full(amax.shape, 448.0, tl.float32)))
    )
    scale = tl.where(amax != 0.0, _rcp_approx_ftz(_ue8m0_to_f32(sf_code)), 0.0)
    q = _f32_to_e4m3_rn_sat(_mul_ftz(x, scale[:, None]))
    return q, sf_code


@triton.jit
def sf_offset_128x4(m_idx, k_idx, num_k_tiles):
    """Byte offset of scale (row ``m_idx``, 32-column block ``k_idx``) in the 128x4 swizzled layout."""
    inner_k = k_idx % 4
    inner_m = (m_idx % 128) // 32
    outer_m = m_idx % 32
    k_tile = k_idx // 4
    m_tile = m_idx // 128
    return m_tile * (num_k_tiles * 512) + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k


@triton.jit
def _mxfp8_quantize_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    num_cols,
    stride_x,
    stride_q,
    num_k_tiles,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0)
    blk0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    offs_b = blk0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    offs_e = tl.arange(0, 32)
    cols = offs_b[:, None] * 32 + offs_e[None, :]
    mask = cols < num_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    q, sf = mxfp8_quantize_rows(x)
    tl.store(q_ptr + row * stride_q + cols, q, mask=mask)
    tl.store(sf_ptr + sf_offset_128x4(row, offs_b, num_k_tiles), sf, mask=offs_b * 32 < num_cols)


def swizzled_sf_numel(num_rows: int, num_cols: int) -> int:
    """``computeSwizzledLayoutSFSize(num_rows, num_cols / 32)``."""
    return ((num_rows + 127) // 128 * 128) * ((num_cols // 32 + 3) // 4 * 4)


def mxfp8_quantize_triton(x):
    """Reference-shaped wrapper: (e4m3 [M, K], swizzled UE8M0 [numel]) like mxfp8_quantize(x, True).

    Scale rows past ``M`` (the 128-row padding) are left as allocated; the MXFP8
    GEMMs never combine them with a valid row.
    """
    assert x.dim() == 2 and x.shape[1] % 32 == 0 and x.dtype == torch.bfloat16
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((swizzled_sf_numel(m, k),), dtype=torch.uint8, device=x.device)
    bpp = 4
    grid = (m, triton.cdiv(k // 32, bpp))
    _mxfp8_quantize_kernel[grid](
        x,
        q.view(torch.uint8),
        sf,
        k,
        x.stride(0),
        q.stride(0),
        (k // 32 + 3) // 4,
        BLOCKS_PER_PROGRAM=bpp,
    )
    return q, sf


@dataclass
class Mxfp8OutputRequest:
    """Buffers a producer kernel may fill with the MXFP8 form of its bf16 output.

    A consumer that would quantize the producer's output for an MXFP8 GEMM opens a
    request (``mxfp8_output_request``) around the producer. A producer that can fuse
    the quantization writes ``fp8`` / ``sf`` (128x4 swizzled scales) in addition to
    its bf16 output and sets ``written``; otherwise the consumer quantizes as before.
    ``out_ptr`` names the bf16 output buffer the request belongs to, so only the
    producer writing exactly that buffer (all rows) fills it.
    """

    fp8: torch.Tensor
    sf: torch.Tensor
    out_ptr: Optional[int] = None
    written: bool = False

    @classmethod
    def allocate(cls, num_rows: int, num_cols: int, device) -> "Mxfp8OutputRequest":
        return cls(
            fp8=torch.empty((num_rows, num_cols), dtype=torch.float8_e4m3fn, device=device),
            sf=torch.empty(
                (swizzled_sf_numel(num_rows, num_cols),), dtype=torch.uint8, device=device
            ),
        )

    def matches(self, output: torch.Tensor) -> bool:
        """Whether ``output`` (any view of [rows, cols]) is the whole buffer this request is for."""
        rows, cols = self.fp8.shape
        return (
            not self.written
            and self.out_ptr is not None
            and output.data_ptr() == self.out_ptr
            and output.is_contiguous()
            and output.numel() == rows * cols
            and output.shape[0] == rows
        )


_MXFP8_OUTPUT_REQUEST: ContextVar[Optional[Mxfp8OutputRequest]] = ContextVar(
    "trtllm_mxfp8_output_request", default=None
)


@contextmanager
def mxfp8_output_request(
    request: Optional[Mxfp8OutputRequest],
) -> Iterator[Optional[Mxfp8OutputRequest]]:
    token = _MXFP8_OUTPUT_REQUEST.set(request)
    try:
        yield request
    finally:
        _MXFP8_OUTPUT_REQUEST.reset(token)


def current_mxfp8_output_request() -> Optional[Mxfp8OutputRequest]:
    return _MXFP8_OUTPUT_REQUEST.get()
