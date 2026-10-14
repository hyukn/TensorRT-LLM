# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer attention-op block offsets for KVCacheManagerV2, written straight into the batch table.

KVCacheManagerV2 converts its per-pool page indices into per-layer attention-op block offsets
(``_copy_swa_block_offsets_with_scratch_compiled``) when the layers of a pool differ in page
stride (e.g. MiniMax-M3's dense / sparse / draft layers) or SWA scratch reuse is on. That compiled
path computes every one of the ``capacity`` index slots (2 x max_num_sequences + reserved) at full
``max_blocks_per_seq`` width into a staging tensor and copies the batch's rows out afterwards:
with 61 layers, capacity 61 and a 1M-token max_seq_len that is ~244 MB of int32 per step
(~0.9 ms of GPU time in front of every decode step).

``copy_per_layer_block_offsets`` computes the same values for the batch's ``num_seqs`` rows only,
at full width (no assumption about which block columns a kernel reads), in one Triton kernel that
writes into the destination table. Bitwise equal to the compiled path on every row it writes
(tests/unittest/_torch/executor/test_kv_cache_v2_per_layer_offsets.py).

Opt-in with TRTLLM_KV_PER_LAYER_OFFSETS_TRITON=1.

With TRTLLM_KV_PER_LAYER_OFFSETS_LIVE_COLUMNS=1 as well, the caller bounds the columns to the ones
the batch's sequences own (their largest block count, and any context scratch range): at a 1M-token
max_seq_len and ~150K-token contexts that is ~1/7 of the full width (TP4 decode B48: 92 us -> ~15
us per step). Columns past the bound keep whatever an earlier step wrote instead of being rewritten
with the BAD_PAGE_INDEX padding, which is equivalent for any consumer that, like every attention
kernel, walks a sequence's pages only up to its own length.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

ENV = "TRTLLM_KV_PER_LAYER_OFFSETS_TRITON"
LIVE_COLUMNS_ENV = "TRTLLM_KV_PER_LAYER_OFFSETS_LIVE_COLUMNS"


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def live_columns_enabled() -> bool:
    return os.environ.get(LIVE_COLUMNS_ENV, "0") == "1"


@triton.jit
def _per_layer_offsets_kernel(
    block_offsets,  # [num_pools, capacity, 2, M] int32 (only slot 0 of dim 2 is read)
    copy_idx,  # [>= num_seqs] int64: index slot of batch row s
    pool_ids,  # [L, 2] int64
    scales,  # [L, 2] int32
    layer_offsets,  # [L, 2] int32
    scratch_pages,  # [L, 2] int32
    scratch_begs,  # [num_pools, ctx_cap] int32
    scratch_ends,  # [num_pools, ctx_cap] int32
    scratch_slots,  # [num_pools, ctx_cap, S] int32
    num_contexts,  # [] int32 (device)
    dst,  # [>= L, >= num_seqs, 2, >= M] int32
    M,
    src_s0,
    src_s1,
    beg_s0,
    slot_s0,
    slot_s1,
    S,
    dst_s0,
    dst_s1,
    dst_s2,
    bad,
    BLOCK: tl.constexpr,
):
    lr = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    m = b < M
    pool = tl.load(pool_ids + lr)
    sc = tl.load(scales + lr)
    lo = tl.load(layer_offsets + lr)
    idx = tl.load(copy_idx + s)
    base = tl.load(block_offsets + pool * src_s0 + idx * src_s1 + b, mask=m, other=0)
    conv = tl.where(base == bad, bad, base * sc + lo)
    n_ctx = tl.load(num_contexts)
    if s < n_ctx:
        sp = tl.load(scratch_pages + lr)
        beg = tl.load(scratch_begs + pool * beg_s0 + s)
        end = tl.load(scratch_ends + pool * beg_s0 + s)
        in_range = (b >= beg) & (b < end)
        total = tl.where(in_range, b - beg, 0) * sp
        slot_idx = tl.minimum(total // sc, S - 1)
        slot_id = tl.load(scratch_slots + pool * slot_s0 + s * slot_s1 + slot_idx, mask=m, other=0)
        scratch_index = slot_id * sc + ((total % sc) + lo) % sc
        conv = tl.where(in_range, scratch_index, conv)
    layer = lr // 2
    role = lr % 2
    tl.store(dst + layer * dst_s0 + s * dst_s1 + role * dst_s2 + b, conv, mask=m)


def copy_per_layer_block_offsets(
    dst: torch.Tensor,
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
    scratch_pages: torch.Tensor,
    scratch_begs: torch.Tensor,
    scratch_ends: torch.Tensor,
    scratch_slots: torch.Tensor,
    num_contexts: torch.Tensor,
    num_seqs: int,
    bad_page_index: int,
    num_cols: Optional[int] = None,
) -> None:
    """dst[:L, :num_seqs] = the compiled path's staging[:L, :num_seqs] (see module docstring).

    With num_cols, only block columns [0, num_cols) are written.
    """
    num_layers = int(pool_ids.shape[0])
    if num_seqs <= 0 or num_layers == 0:
        return
    m = int(block_offsets.shape[-1])
    if num_cols is not None:
        m = min(m, int(num_cols))
        if m <= 0:
            return
    assert block_offsets.is_contiguous() and block_offsets.shape[2] == 2
    assert dst.stride(-1) == 1 and dst.shape[2] == 2 and dst.shape[-1] >= m
    assert scratch_begs.is_contiguous() and scratch_ends.is_contiguous()
    assert scratch_slots.is_contiguous()
    for t in (pool_ids, scales, layer_offsets, scratch_pages):
        assert t.is_contiguous() and tuple(t.shape) == (num_layers, 2)
    block = 1024
    grid = (num_layers * 2, num_seqs, triton.cdiv(m, block))
    _per_layer_offsets_kernel[grid](
        block_offsets,
        copy_idx,
        pool_ids,
        scales,
        layer_offsets,
        scratch_pages,
        scratch_begs,
        scratch_ends,
        scratch_slots,
        num_contexts,
        dst,
        m,
        block_offsets.stride(0),
        block_offsets.stride(1),
        scratch_begs.stride(0),
        scratch_slots.stride(0),
        scratch_slots.stride(1),
        int(scratch_slots.shape[-1]),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        int(bad_page_index),
        BLOCK=block,
        num_warps=4,
    )


__all__ = [
    "ENV",
    "LIVE_COLUMNS_ENV",
    "copy_per_layer_block_offsets",
    "enabled",
    "live_columns_enabled",
]
