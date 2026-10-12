# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Triton per-layer block-offset conversion must reproduce KVCacheManagerV2's compiled path.

For every (layer, role) the batch rows' block offsets -- BAD page passthrough, page-index scale and
layer offset, and SWA scratch slots for context rows -- must be bitwise equal to what
``_copy_swa_block_offsets_with_scratch_compiled`` + the staging copy write into the destination
table, and rows past num_seqs must stay untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _copy_swa_block_offsets_with_scratch_compiled,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.per_layer_block_offsets import (
    copy_per_layer_block_offsets,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

SENTINEL = -7


def _case(num_pools, capacity, m, layers, num_seqs, num_contexts, slots, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def ri(lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=g, device="cuda", dtype=torch.int32)

    offsets = ri(0, 50_000, (num_pools, capacity, 2, m))
    # Each slot's valid prefix ends somewhere; the rest is BAD (as in the host table).
    lens = ri(0, m + 1, (num_pools, capacity, 1, 1))
    cols = torch.arange(m, device="cuda", dtype=torch.int32)
    offsets = torch.where(cols < lens, offsets, BAD_PAGE_INDEX).contiguous()
    copy_idx = torch.zeros(capacity, dtype=torch.long, device="cuda")
    copy_idx[:num_seqs] = torch.randperm(capacity, generator=g, device="cuda")[:num_seqs]
    pool_ids = torch.randint(0, num_pools, (layers, 2), generator=g, device="cuda")
    scales = torch.tensor([1, 2, 4, 8], device="cuda", dtype=torch.int32)[
        torch.randint(0, 4, (layers, 2), generator=g, device="cuda")
    ]
    layer_offsets = (ri(0, 1 << 20, (layers, 2)) % scales).to(torch.int32)
    scratch_pages = ri(1, 4, (layers, 2))
    begs = torch.zeros(num_pools, capacity, dtype=torch.int32, device="cuda")
    ends = torch.zeros_like(begs)
    a = ri(0, m, (num_pools, num_contexts))
    b = ri(0, m, (num_pools, num_contexts))
    begs[:, :num_contexts] = torch.minimum(a, b)
    ends[:, :num_contexts] = torch.maximum(a, b) + 1
    scratch_slots = ri(0, 100_000, (num_pools, capacity, slots))
    n_ctx = torch.tensor(num_contexts, dtype=torch.int32, device="cuda")
    return (
        offsets,
        copy_idx,
        pool_ids,
        scales,
        layer_offsets,
        scratch_pages,
        begs,
        ends,
        scratch_slots,
        n_ctx,
    )


@pytest.mark.parametrize(
    "num_pools,capacity,m,layers,num_seqs,num_contexts,slots",
    [
        (3, 13, 300, 7, 5, 0, 4),  # pure decode
        (3, 13, 300, 7, 5, 2, 4),  # mixed: two context rows with scratch ranges
        (2, 9, 1024, 5, 9, 9, 16),  # all slots, all context, width = one Triton block
        (4, 65, 2048, 11, 1, 0, 8),  # single decode row
        (1, 6, 77, 3, 6, 3, 1),  # single scratch slot (slot index clamped)
    ],
)
def test_per_layer_offsets_match_compiled_path(
    num_pools, capacity, m, layers, num_seqs, num_contexts, slots
):
    args = _case(num_pools, capacity, m, layers, num_seqs, num_contexts, slots, seed=capacity + m)
    (offsets, copy_idx, pool_ids, scales, layer_offsets, scratch_pages, begs, ends, sl, n_ctx) = (
        args
    )
    max_seqs = capacity + 3
    staging = torch.empty(layers, capacity, 2, m, dtype=torch.int32, device="cuda")
    _copy_swa_block_offsets_with_scratch_compiled(
        offsets,
        copy_idx,
        pool_ids,
        scales,
        layer_offsets,
        scratch_pages,
        torch.arange(m, dtype=torch.int32, device="cuda"),
        begs,
        ends,
        sl,
        n_ctx,
        staging,
    )
    ref = torch.full((layers + 2, max_seqs, 2, m + 5), SENTINEL, dtype=torch.int32, device="cuda")
    ref[:layers, :num_seqs, :, :m].copy_(staging[:layers, :num_seqs])
    got = torch.full_like(ref, SENTINEL)
    copy_per_layer_block_offsets(
        got,
        offsets,
        copy_idx,
        pool_ids,
        scales,
        layer_offsets,
        scratch_pages,
        begs,
        ends,
        sl,
        n_ctx,
        num_seqs,
        BAD_PAGE_INDEX,
    )
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


def test_per_layer_offsets_empty_batch_is_noop():
    args = _case(2, 5, 64, 3, 0, 0, 2, seed=1)
    dst = torch.full((3, 5, 2, 64), SENTINEL, dtype=torch.int32, device="cuda")
    copy_per_layer_block_offsets(dst, *args, 0, BAD_PAGE_INDEX)
    assert bool((dst == SENTINEL).all())
