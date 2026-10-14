# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-gen context attention for MiniMax-M3 dense layers over M3's flat sub-page pool.

Not bitwise equal to fmha_sm100's dense plan (another kernel); checked against an fp32
reference of the same FP8 inputs and against fmha_sm100 at the same tolerance.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM100-family GPU required",
)

PAGE = 128
D = 128


class _SubpagePool:
    """Stand-in for MiniMaxM3KVCacheManagerV2.get_kv_subpage_pool: slot s packs `scale`
    sub-pages; this layer's K is sub-page s * scale, its V the next one."""

    def __init__(self, k_pages, v_pages, scale=3):
        n, hkv = int(k_pages.shape[0]), int(k_pages.shape[1])
        flat = torch.zeros(n * scale, hkv, PAGE, D, dtype=k_pages.dtype, device=k_pages.device)
        flat[0::scale] = k_pages
        flat[1::scale] = v_pages
        self.flat = flat[: (n - 1) * scale + 2]
        self.scale = scale

    def get_kv_subpage_pool(self, layer_idx, kv_layout="HND"):
        return self.flat, self.scale


def _case(qo_lens, cached, hq=32, hkv=2, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv_lens = [q + c for q, c in zip(qo_lens, cached)]
    pages = [(k + PAGE - 1) // PAGE for k in kv_lens]
    npool = sum(pages) + 5
    k = (torch.randn(npool, hkv, PAGE, D, generator=g, device="cuda") * 1.5).to(torch.float8_e4m3fn)
    v = torch.randn(npool, hkv, PAGE, D, generator=g, device="cuda").to(torch.float8_e4m3fn)
    perm = torch.randperm(npool, generator=g, device="cuda").to(torch.int32)
    bt = torch.zeros(len(qo_lens), max(pages), dtype=torch.int32, device="cuda")
    o = 0
    for b, p in enumerate(pages):
        bt[b, :p] = perm[o : o + p]
        o += p
    q = (torch.randn(sum(qo_lens), hq, D, generator=g, device="cuda") * 1.5).to(torch.float8_e4m3fn)
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        q=q,
        k=k,
        v=v,
        bt=bt,
        qo=qo_lens,
        kv=kv_lens,
        cached=cached,
        hq=hq,
        hkv=hkv,
        seq=torch.tensor(kv_lens, **i32),
        cuq=torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), **i32),
        cukv=torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)), **i32),
    )


def _reference_rows(c, rows):
    out = []
    grp = c["hq"] // c["hkv"]
    starts = [0] + list(torch.tensor(c["qo"]).cumsum(0).tolist())
    for r in rows:
        b = max(i for i in range(len(c["qo"])) if starts[i] <= r)
        pos = c["cached"][b] + (r - starts[b])
        ids = c["bt"][b, : (c["kv"][b] + PAGE - 1) // PAGE].long()
        kk = c["k"][ids].float().permute(1, 0, 2, 3).reshape(c["hkv"], -1, D)[:, : pos + 1]
        vv = c["v"][ids].float().permute(1, 0, 2, 3).reshape(c["hkv"], -1, D)[:, : pos + 1]
        kk, vv = kk.repeat_interleave(grp, 0), vv.repeat_interleave(grp, 0)
        s = torch.einsum("hd,hkd->hk", c["q"][r].float(), kk) * D**-0.5
        out.append(torch.einsum("hk,hkd->hd", torch.softmax(s, -1), vv))
    return torch.stack(out)


def _run_trtllm_gen(c):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
        minimax_m3_trtllm_gen_dense_prefill,
    )

    out = torch.empty(c["q"].shape, dtype=torch.bfloat16, device="cuda")
    minimax_m3_trtllm_gen_dense_prefill(
        c["q"],
        _SubpagePool(c["k"], c["v"]),
        0,
        block_table=c["bt"],
        seq_lens=c["seq"],
        cu_seqlens_q=c["cuq"],
        cu_seqlens_kv=c["cukv"],
        max_q_len=max(c["qo"]),
        max_kv_len=max(c["kv"]),
        sm_scale=D**-0.5,
        output=out,
    )
    return out


def _run_fmha_sm100(c):
    from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import run_msa_sparse_gqa
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        build_kv_page_indices,
    )

    qo = torch.tensor(c["qo"], dtype=torch.int32)
    kv = torch.tensor(c["kv"], dtype=torch.int32)
    out = torch.empty(c["q"].shape, dtype=torch.bfloat16, device="cuda")
    run_msa_sparse_gqa(
        c["q"],
        c["k"],
        c["v"],
        None,
        kv_indices=build_kv_page_indices(c["bt"].cpu(), kv, PAGE).cuda(),
        sm_scale=D**-0.5,
        qo_lens_cpu=qo,
        kv_lens_cpu=kv,
        qo_offset_cpu=kv - qo,
        causal=True,
        out=out,
        use_fp8=True,
    )
    return out


@pytest.mark.parametrize(
    "qo_lens,cached",
    [
        ([2048], [0]),
        ([1000], [149503]),
        ([4096], [40960 + 17]),
        ([300, 2048, 77], [5000, 70000, 129]),
    ],
)
@pytest.mark.parametrize("heads", [(32, 2), (16, 1)], ids=["tp2", "tp4"])
def test_trtllm_gen_dense_prefill_matches_reference(qo_lens, cached, heads):
    c = _case(qo_lens, cached, hq=heads[0], hkv=heads[1])
    got = _run_trtllm_gen(c)
    ref_sm100 = _run_fmha_sm100(c)
    assert torch.isfinite(got).all()
    total = sum(qo_lens)
    g = torch.Generator().manual_seed(1)
    rows = sorted(
        set(torch.randint(0, total, (48,), generator=g).tolist())
        | {0, 1, total - 1, qo_lens[0] - 1, min(total - 1, qo_lens[0])}
    )
    ref = _reference_rows(c, rows)
    err_gen = (got[rows].float() - ref).abs()
    err_sm100 = (ref_sm100[rows].float() - ref).abs()
    # Both kernels quantize P to FP8 for the second GEMM and round the output to bf16, so neither
    # is exact; the trtllm-gen path must be at least as close to the fp32 reference.
    assert err_gen.mean().item() <= 1.25 * err_sm100.mean().item() + 1e-5, (
        err_gen.mean().item(),
        err_sm100.mean().item(),
    )
    assert err_gen.max().item() <= max(2 * err_sm100.max().item(), 2e-2)
    # Gross-error guard (a wrong page, mask or row offset is O(100 %) of the output magnitude).
    d = (got.float() - ref_sm100.float()).abs().mean().item()
    scale = ref_sm100.float().abs().mean().item()
    assert d < 0.1 * scale, (d, scale)
