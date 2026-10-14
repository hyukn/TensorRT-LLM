# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vectorized bf16/fp16 -> fp32 upcast.

`tensor.to(torch.float32)` on CUDA runs PyTorch's dynamic-casting elementwise path
(unrolled_elementwise_kernel), which is not vectorized: on a [16384, 6144] bf16 activation it
moves ~600 MB at ~2.4 TB/s (~250 us on GB300). This Triton kernel does the same exact
conversion (every bf16/fp16 value is exactly representable in fp32) with 16-byte accesses at
close to HBM bandwidth. Used for the fp32 MoE router input of MiniMax-M3 context steps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 8192
# Below this the launch overhead dominates and torch's cast is just as fast.
MIN_NUMEL = 1 << 20


@triton.jit
def _upcast_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, eviction_policy="evict_first")
    tl.store(dst + offs, x.to(tl.float32), mask=mask)


def upcast_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """Return x.to(torch.float32) (bitwise the same values) for a contiguous CUDA bf16/fp16 x."""
    if (
        not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or x.numel() < MIN_NUMEL
    ):
        return x.to(torch.float32)
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    _upcast_kernel[(triton.cdiv(n, _BLOCK),)](x, out, n, BLOCK=_BLOCK, num_warps=8)
    return out


__all__ = ["MIN_NUMEL", "upcast_to_fp32"]
