# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the v2 MiniMax-M3 NVFP4 sparse decode kernel (kernels/sparse_decode_v2).

Reuses the NVFP4 input builder and fp32 oracle of test_minimax_m3_sparse_attn_decode.py and the
same tolerances (3e-2 against the oracle, 1e-2 for split-K invariance), and additionally checks v2
against the Triton kernel and the dispatch through minimax_m3_sparse_attn_decode.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_minimax_m3_sparse_attn_decode import (  # noqa: E402
    HEAD_DIM,
    PAGE_SIZE,
    _make_nvfp4_inputs,
    _reference_sparse_decode,
    _run_nvfp4,
)

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.sparse_decode_v2 import (  # noqa: E402
    minimax_m3_sparse_attn_decode_v2,
    select_v2_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="NVFP4 sparse decode needs SM100/SM103",
)

SM_SCALE = HEAD_DIM**-0.5


def _run_v2(case, *, force="v2", **kwargs):
    out = torch.empty(case.q.shape, device=case.q.device, dtype=torch.bfloat16)
    minimax_m3_sparse_attn_decode_v2(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=SM_SCALE,
        output=out,
        decode_query_len=case.decode_query_len,
        _force=force,
        **case.scales,
        **kwargs,
    )
    return out


def _reference(case):
    return _reference_sparse_decode(
        case.q.to(torch.bfloat16) if case.q.dtype == torch.float8_e4m3fn else case.q,
        case.k_ref,
        case.v_ref,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=SM_SCALE,
        decode_query_len=case.decode_query_len,
    )


def _eagle_overlap_topk(case, *, shared: int, seed: int):
    """Rewrite the top-k table so the queries of a request share `shared` pages.

    Query 0 keeps its random sorted selection; each later query swaps
    16 - shared of them for pages query 0 did not pick (when it has 16 valid
    entries and enough pages to choose from), re-sorted ascending with the
    -1 tail untouched, like the selector emits.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk = case.topk_idx.cpu()
    dql = case.decode_query_len
    num_kv_heads, total_q, k = topk.shape
    for r, seq_len in enumerate(case.seq_lens.tolist()):
        for h in range(num_kv_heads):
            base = topk[h, r * dql]
            nvalid0 = int((base >= 0).sum())
            for j in range(1, dql):
                row = topk[h, r * dql + j]
                kv_len = max(seq_len - dql + j + 1, 0)
                nb = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
                real = min(k, nb)
                if real < k or nvalid0 < k or nb <= k:
                    continue
                keep = base[:k].clone()
                pool = torch.tensor([b for b in range(nb) if b not in set(keep.tolist())])
                n_swap = min(k - shared, len(pool))
                if n_swap > 0:
                    drop = torch.randperm(k, generator=gen)[:n_swap]
                    add = pool[torch.randperm(len(pool), generator=gen)[:n_swap]]
                    keep[drop] = add.to(keep.dtype)
                row[:k] = keep.sort().values
    case.topk_idx = topk.to(case.topk_idx.device)
    return case


SHAPES = [
    # (num_kv_heads, group, decode_query_len)
    (1, 16, 1),
    (1, 16, 2),
    (1, 16, 4),
    (1, 8, 1),
    (1, 8, 2),
    (1, 8, 4),
    (2, 16, 4),
    (4, 16, 4),
    (2, 8, 2),
    (4, 8, 1),
    (2, 16, 3),
]
SEQS = {
    "one-block": [128],
    "short-mixed": [1, 128, 129],
    "two-req": [1025, 4097],
    "batch5": [300, 1500, 2049, 4096, 5000],
}


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(("num_kv_heads", "group", "dql"), SHAPES)
@pytest.mark.parametrize("seqs", list(SEQS.values()), ids=list(SEQS.keys()))
def test_v2_matches_reference_and_old(num_kv_heads, group, dql, seqs, q_dtype):
    seq_lens = [max(s, dql) for s in seqs]
    case = _make_nvfp4_inputs(
        seq_lens, num_kv_heads=num_kv_heads, group=group, decode_query_len=dql
    )
    case.q = case.q.to(q_dtype)
    out = _run_v2(case)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), _reference(case), rtol=3e-2, atol=3e-2)
    old = _run_nvfp4(case)
    torch.testing.assert_close(out.float(), old.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shared", [16, 14, 13, 8, 0])
@pytest.mark.parametrize(
    ("num_kv_heads", "group", "dql"), [(1, 16, 4), (4, 16, 4), (1, 8, 4), (1, 16, 2)]
)
def test_v2_eagle_overlapping_selection(num_kv_heads, group, dql, shared):
    seq_lens = [5000, 3000, 2600, 9000, 129 * 128 + 7]
    case = _make_nvfp4_inputs(
        seq_lens, num_kv_heads=num_kv_heads, group=group, decode_query_len=dql, seed=5
    )
    _eagle_overlap_topk(case, shared=shared, seed=shared)
    case.q = case.q.to(torch.float8_e4m3fn)
    out = _run_v2(case)
    torch.testing.assert_close(out.float(), _reference(case), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out.float(), _run_nvfp4(case).float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("nsplit", [1, 2, 3, 4, 5, 6, 8, 12, 16])
@pytest.mark.parametrize("dql", [1, 4])
def test_v2_split_invariant(nsplit, dql):
    case = _make_nvfp4_inputs(
        [4097, 300, 8192], num_kv_heads=2, group=16, decode_query_len=dql, seed=83
    )
    reference = _run_v2(case, num_topk_chunks=1)
    out = _run_v2(case, num_topk_chunks=nsplit)
    torch.testing.assert_close(out.float(), reference.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out.float(), _reference(case), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    ("wpt", "stages", "nsplit"),
    [(4, 1, 1), (4, 2, 3), (2, 3, 2), (2, 1, 1), (4, 6, 4), (2, 8, 1)],
)
def test_v2_ring_and_warp_configs(wpt, stages, nsplit):
    """Ring reuse (stages < pages per CTA) and both warp splits stay exact."""
    case = _make_nvfp4_inputs(
        [6000, 2049, 4500], num_kv_heads=1, group=16, decode_query_len=4, seed=9
    )
    _eagle_overlap_topk(case, shared=13, seed=1)
    cfg = dict(ntiles=4, wpt=wpt, stages=stages, nsplit=nsplit)
    out = _run_v2(case, _config=cfg)
    torch.testing.assert_close(out.float(), _reference(case), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    ("wpt", "nsplit", "group"),
    [(4, 1, 16), (2, 1, 16), (4, 3, 16), (2, 16, 16), (4, 5, 8), (2, 2, 8)],
)
def test_v2_single_query_configs(wpt, nsplit, group):
    case = _make_nvfp4_inputs(
        [6000, 2049, 129, 4500], num_kv_heads=2, group=group, decode_query_len=1, seed=19
    )
    cfg = dict(ntiles=1, wpt=wpt, stages=min(6, (16 + nsplit - 1) // nsplit), nsplit=nsplit)
    out = _run_v2(case, _config=cfg)
    torch.testing.assert_close(out.float(), _reference(case), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dql", [1, 4])
@pytest.mark.parametrize("nsplit", [1, 4])
def test_v2_zero_length_rows_are_zero_not_nan(dql, nsplit):
    seq_lens = [1024, 0, 512, 0]
    case = _make_nvfp4_inputs(seq_lens, num_kv_heads=1, group=16, decode_query_len=dql, seed=21)
    out = _run_v2(case, num_topk_chunks=nsplit)
    assert torch.isfinite(out).all()
    for r in (1, 3):
        assert torch.equal(
            out[r * dql : (r + 1) * dql], torch.zeros_like(out[r * dql : (r + 1) * dql])
        )
    assert out[0].abs().sum() > 0
    expected = _reference(case)
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


def test_v2_garbage_past_sequence_end_is_ignored():
    """Slots past seq_len (never written) may hold NaN-encoding E4M3 scales."""
    seq_lens = [1000, 2049, 129, 300]
    dql = 4
    case = _make_nvfp4_inputs(seq_lens, num_kv_heads=1, group=16, decode_query_len=dql, seed=29)
    clean = _run_v2(case)
    for r, s in enumerate(seq_lens):
        last_block = (s - 1) // PAGE_SIZE
        page = int(case.block_table[r, last_block])
        off = s % PAGE_SIZE
        if off == 0:
            continue
        case.k_paged[page, :, off:] = 0xFF
        case.v_paged[page, :, off:] = 0xFF
        for role in (0, 1):
            flat = case.scale_pool[page, role].reshape(-1, PAGE_SIZE * (HEAD_DIM // 16))
            if role == 0:
                flat[:, off * (HEAD_DIM // 16) :] = 0xFF
            else:
                tok = torch.arange(PAGE_SIZE, device="cuda")
                col = torch.arange(HEAD_DIM // 16, device="cuda")
                idx = (
                    (tok[:, None] // 4) * (4 * (HEAD_DIM // 16))
                    + 4 * col[None, :]
                    + tok[:, None] % 4
                )
                bad = idx[off:].reshape(-1)
                flat[:, bad] = 0xFF
    dirty = _run_v2(case)
    assert torch.isfinite(dirty).all()
    torch.testing.assert_close(dirty, clean, rtol=0, atol=0)


def test_v2_ignores_padded_topk_entries():
    seq_lens = [1, 64, 128, 129, 200]
    case = _make_nvfp4_inputs(seq_lens, num_kv_heads=1, group=16, decode_query_len=1, seed=13)
    assert (case.topk_idx == -1).any()
    out = _run_v2(case)
    torch.testing.assert_close(out.float(), _reference(case), rtol=3e-2, atol=3e-2)


def test_v2_accepts_token_major_topk_table():
    case = _make_nvfp4_inputs([1025, 4097], num_kv_heads=2, group=16, decode_query_len=2, seed=31)
    out_hm = _run_v2(case)
    case.topk_idx = case.topk_idx.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    assert not case.topk_idx.is_contiguous()
    out_tm = _run_v2(case)
    assert torch.equal(out_hm, out_tm)


def test_v2_fp8_q_matches_prewidened_bf16_q():
    """FP8 q is widened exactly; bf16 q of the same values gives the same output."""
    case = _make_nvfp4_inputs(
        [300, 1500, 4097], num_kv_heads=1, group=16, decode_query_len=4, seed=61
    )
    case.q = case.q.to(torch.float8_e4m3fn)
    out_fp8 = _run_v2(case)
    case.q = case.q.to(torch.bfloat16)
    out_bf16 = _run_v2(case)
    assert torch.equal(out_fp8, out_bf16)


@pytest.mark.parametrize("nsplit", [1, 4])
def test_v2_cuda_graph_replay_tracks_inputs(nsplit):
    seq_lens = [2048, 4097, 900, 3000]
    dql = 4
    case = _make_nvfp4_inputs(seq_lens, num_kv_heads=1, group=16, decode_query_len=dql, seed=41)
    out = torch.empty(case.q.shape, device="cuda", dtype=torch.bfloat16)

    def run():
        minimax_m3_sparse_attn_decode_v2(
            case.q,
            case.k_paged,
            case.v_paged,
            case.topk_idx,
            case.block_table,
            case.seq_lens,
            sm_scale=SM_SCALE,
            output=out,
            decode_query_len=dql,
            num_topk_chunks=nsplit,
            _force="v2",
            **case.scales,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    gen = torch.Generator(device="cuda").manual_seed(99)
    for it in range(3):
        case.q.copy_(torch.randn(case.q.shape, device="cuda", generator=gen, dtype=torch.float32))
        # Change the selection and the lengths too: replay must read them live.
        if it == 1:
            nkv, tq, k = case.topk_idx.shape
            flipped = case.topk_idx.view(nkv, tq // dql, dql, k).flip(2).reshape(nkv, tq, k)
            case.topk_idx.copy_(flipped.clone())
        if it == 2:
            case.seq_lens.copy_(case.seq_lens - 5)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(case)
        torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


def test_select_v2_config_is_shape_only():
    common = dict(
        num_kv_heads=1, group=16, decode_query_len=4, topk=16, sm_count=152, capability=(10, 3)
    )
    a = select_v2_config(num_reqs=24, **common)
    assert a == select_v2_config(num_reqs=24, **common) and a is not None
    assert a["nsplit"] == 6 and a["ntiles"] == 4
    assert select_v2_config(num_reqs=64, **common)["nsplit"] == 2
    # Attention DP (4 KV heads per rank): one wave of groups, three splits between half and
    # two thirds of a wave, then one split.
    kv4 = common | {"num_kv_heads": 4}
    assert [select_v2_config(num_reqs=b, **kv4)["nsplit"] for b in (8, 18, 20, 24, 25, 28, 32)] == [
        4,
        2,
        3,
        3,
        3,
        1,
        1,
    ]
    # Measured slower than the old kernel -> dispatch keeps the old kernel.
    assert select_v2_config(num_reqs=8, **common) is None
    assert select_v2_config(num_reqs=16, **(common | {"decode_query_len": 1})) is None
    assert select_v2_config(num_reqs=24, **(common | {"capability": (10, 0)})) is None
    assert select_v2_config(num_reqs=24, **(common | {"group": 4})) is None


def test_dispatch_falls_back_to_old_kernel_bitwise():
    """Shapes the dispatch leaves to the old kernel produce exactly its output."""
    case = _make_nvfp4_inputs([300, 1500], num_kv_heads=1, group=16, decode_query_len=4, seed=3)
    out = _run_v2(case, force=None)  # B=2: below the v2 threshold
    assert torch.equal(out, _run_nvfp4(case))
