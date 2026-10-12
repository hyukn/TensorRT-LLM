# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tcgen05 context-row index scorer must reproduce the fmha_sm100 proxy pass.

Both write raw Q.K per-128-block maxima into the [heads, k_tiles, tokens] buffer the block
selector reads; on every entry the selector reads the two must be bitwise equal, and the selected
top-k blocks identical, for varlen context batches over a paged FP8 index-K pool.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    build_kv_page_indices,
    per_token_valid_blocks,
    require_msa_module,
    select_blocks_from_maxscore,
)

PAGE = 128
D = 128


def _case(qo_lens, kv_lens, heads, seed, page_pad=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(qo_lens)
    nblk = [math.ceil(k / PAGE) for k in kv_lens]
    max_blocks = max(nblk)
    num_pages = sum(nblk) + 3
    perm = torch.randperm(num_pages, device="cuda", generator=g).to(torch.int32)
    bt = torch.zeros(batch, max_blocks + 2, dtype=torch.int32, device="cuda")
    o = 0
    for b, n in enumerate(nblk):
        bt[b, :n] = perm[o : o + n]
        o += n
    pool_full = torch.zeros(
        num_pages, 1 + page_pad, PAGE, D, device="cuda", dtype=torch.float8_e4m3fn
    )
    pool_full[:, 0] = (torch.randn(num_pages, PAGE, D, device="cuda", generator=g) * 1.5).to(
        torch.float8_e4m3fn
    )
    pool = pool_full[:, :1]
    total_q = sum(qo_lens)
    q = (torch.randn(total_q, heads, D, device="cuda", generator=g) * 1.5).to(torch.float8_e4m3fn)
    cu = torch.zeros(batch + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(qo_lens, dtype=torch.int32), 0)
    qo_cpu = torch.tensor(qo_lens, dtype=torch.int32)
    kv_cpu = torch.tensor(kv_lens, dtype=torch.int32)
    return q, pool, bt, cu.cuda(), qo_cpu, kv_cpu


@pytest.fixture
def ctx_scorer_enabled(monkeypatch):
    monkeypatch.setenv(msa_indexer.CTX_SCORER_ENV, "1")
    msa_indexer.ctx_scorer_module.cache_clear()
    mod = msa_indexer.ctx_scorer_module()
    if mod is None:
        pytest.skip("tcgen05 context scorer unavailable")
    yield mod
    msa_indexer.ctx_scorer_module.cache_clear()


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize(
    "qo_lens,kv_lens",
    [
        ([140], [150_000]),
        ([16, 700, 1], [300, 20_000, 129]),
        ([2000, 127, 128, 129], [60_000, 127, 5_000, 40_000]),
        ([4096], [4096]),
    ],
)
def test_ctx_scorer_matches_fmha_proxy(ctx_scorer_enabled, heads, qo_lens, kv_lens):
    mod = ctx_scorer_enabled
    q, pool, bt, cu, qo_cpu, kv_cpu = _case(qo_lens, kv_lens, heads, seed=sum(qo_lens) + heads)
    off = (kv_cpu - qo_cpu).to(torch.int32)
    # Today's path.
    fmha = require_msa_module()
    kv_indices = build_kv_page_indices(bt.cpu(), kv_cpu, PAGE).cuda()
    proxy = msa_indexer._proxy_max_score(
        q,
        pool,
        qo_lens_cpu=qo_cpu,
        kv_lens_cpu=kv_cpu,
        qo_offset_cpu=off,
        kv_indices=kv_indices,
        sm_scale=D**-0.5,
        causal=True,
    )
    del fmha
    # New path.
    max_k_tiles = max(math.ceil(k / PAGE) for k in kv_lens)
    args = msa_indexer.CtxScoreArgs(
        cu_seqlens_q=cu,
        seq_lens=kv_cpu.cuda(),
        block_table=bt,
        plan=mod.ctx_plan_tc(qo_cpu, kv_cpu, heads),
        max_k_tiles=max_k_tiles,
    )
    got = msa_indexer._ctx_scores(q, pool, args)
    torch.cuda.synchronize()
    nvb = (
        per_token_valid_blocks(qo_cpu, kv_cpu, off, causal=True, block_size=PAGE)
        .cuda()
        .to(torch.int32)
    )
    blk = torch.arange(max_k_tiles, device="cuda")[:, None]
    valid = (blk < nvb[None, :])[None].expand(heads, -1, -1)
    a = got[:, :max_k_tiles][valid]
    b = proxy[:, :max_k_tiles][valid]
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    sel_new = select_blocks_from_maxscore(
        got, topk=16, n_valid_blocks=nvb, init_blocks=0, local_blocks=1
    )
    sel_old = select_blocks_from_maxscore(
        proxy[:, :max_k_tiles].contiguous(),
        topk=16,
        n_valid_blocks=nvb,
        init_blocks=0,
        local_blocks=1,
    )
    assert torch.equal(sel_new, sel_old)
