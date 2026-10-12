# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""_sparse_atten_nvfp4_kv_into must equal MSA's sparse_atten_nvfp4_kv_func bitwise.

Builds a paged NVFP4 cache with random packed E2M1 bytes and finite E4M3 block scales, random
causal top-16 block tables, then runs both paths on the same CSR schedule.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import _sparse_atten_nvfp4_kv_into
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    require_msa_module,
)
from tensorrt_llm._utils import is_sm_100f

PAGE, D, TOPK = 128, 128, 16


def make(qo_lens, kv_lens, hq, hkv, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(qo_lens)
    max_blocks = max(math.ceil(k / PAGE) for k in kv_lens)
    pages = batch * max_blocks
    table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).reshape(batch, max_blocks)
    )
    kv = torch.randint(
        0, 256, (pages, 2, hkv, PAGE, D // 2), device="cuda", dtype=torch.uint8, generator=g
    )
    # finite E4M3 scales in [0.25, 2): exponent bits 0b0101..0b1000 -> bytes 0x28..0x47
    sc = torch.randint(
        0x28, 0x48, (pages, 2, hkv, PAGE, D // 16), device="cuda", dtype=torch.uint8, generator=g
    )
    total_q = sum(qo_lens)
    q = (torch.randn(total_q, hq, D, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    # Causal top-k per query token: its own block plus random earlier blocks,
    # ascending, -1 padded.
    q2k = torch.full((hkv, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    t = 0
    for qo, kvl in zip(qo_lens, kv_lens):
        off = kvl - qo
        for i in range(qo):
            nblk = (off + i) // PAGE + 1
            for h in range(hkv):
                n = min(TOPK, nblk)
                cand = (
                    torch.randperm(nblk - 1, device="cuda", generator=g)[: n - 1]
                    if nblk > 1
                    else torch.empty(0, device="cuda", dtype=torch.long)
                )
                sel = torch.cat([cand, torch.tensor([nblk - 1], device="cuda")]).sort().values
                q2k[h, t, :n] = sel.to(torch.int32)
            t += 1
    cu_q = torch.tensor(
        [0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device="cuda"
    )
    cu_kv = torch.tensor(
        [0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device="cuda"
    )
    return dict(
        q=q,
        k=kv[:, 0],
        v=kv[:, 1],
        ks=sc[:, 0],
        vs=sc[:, 1],
        table=table,
        q2k=q2k,
        cu_q=cu_q,
        cu_kv=cu_kv,
        seq=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        max_q=max(qo_lens),
        max_kv=max(kv_lens),
        total_k=sum(kv_lens),
        total_rows=sum(math.ceil(k / PAGE) for k in kv_lens),
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(),
    reason="MiniMax-M3 NVFP4 MSA requires SM100/SM103.",
)


@pytest.mark.parametrize(
    "qo,kvl,hq,hkv",
    [([700], [20000], 32, 2), ([300, 900], [5000, 40000], 16, 1), ([129], [129], 16, 1)],
)
def test_nvfp4_sparse_into_matches_packaged(qo, kvl, hq, hkv):
    require_msa_module()
    from fmha_sm100 import sparse

    gs_k = torch.full((1,), 0.02, device="cuda")
    gs_v = torch.full((1,), 0.03, device="cuda")
    c = make(qo, kvl, hq, hkv)
    row_ptr, q_idx, sched = sparse.build_k2q_csr(
        c["q2k"],
        c["cu_q"],
        c["cu_kv"],
        PAGE,
        total_k=c["total_k"],
        max_seqlen_k=c["max_kv"],
        max_seqlen_q=c["max_q"],
        total_rows=c["total_rows"],
        qhead_per_kv=hq // hkv,
        return_schedule=True,
    )
    args = (c["q"], c["k"], c["v"], c["ks"], c["vs"], gs_k, gs_v, row_ptr, q_idx, TOPK)
    kw = dict(
        cu_seqlens_q=c["cu_q"],
        cu_seqlens_k=c["cu_kv"],
        max_seqlen_q=c["max_q"],
        max_seqlen_k=c["max_kv"],
        blk_kv=PAGE,
        causal=True,
        softmax_scale=D**-0.5,
        page_table=c["table"],
        seqused_k=c["seq"],
        schedule=sched,
    )
    ref = sparse.sparse_atten_nvfp4_kv_func(
        *args, partial_dtype=torch.bfloat16, return_softmax_lse=False, **kw
    )
    out = torch.empty_like(ref)
    assert _sparse_atten_nvfp4_kv_into(sparse.sparse_atten_nvfp4_kv_func, out, *args, **kw)
    assert not torch.isnan(ref).any()
    assert torch.equal(out, ref)
