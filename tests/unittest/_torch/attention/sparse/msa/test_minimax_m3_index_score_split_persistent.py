# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent-grid MiniMax-M3 split index scorer: bitwise equal to the unsplit scorer.

The split op (one rank's share of the KV blocks, written into per-head score tensors) is driven for
every rank of a simulated tensor-parallel group on one GPU, the way ``tp_split_score`` drives it,
and the union of the ranks' writes must equal the unsplit op word for word -- computed scores,
fully masked blocks and untouched entries alike -- eagerly and under CUDA graph replay. The
persistent kernel balances blocks through a device work counter that it resets itself, so the
counter must read zero after every launch.
"""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs an SM100-family GPU"
)

SENTINEL = 0x7FBADBAD
DQL = 4


def _ops():
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import index_tp_split
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops_mod

    if not getattr(ops_mod, "IS_CUTLASS_DSL_AVAILABLE", False):
        pytest.skip("CuTe DSL not available")
    return (
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score_split,
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score,
        ops_mod.CuteDSLMiniMaxM3IndexDecodeScoreSplitRunner,
        index_tp_split,
    )


def _case(batch, seqs, heads, seed=0, pages=6000):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seqs = torch.as_tensor(seqs, dtype=torch.int32)
    nbmax = int(((seqs.long() + 127) // 128).max())
    pool = torch.randint(0, 256, (pages, 128, 128), generator=g, device="cuda", dtype=torch.uint8)
    q = torch.randint(
        0, 256, (batch * DQL, heads, 128), generator=g, device="cuda", dtype=torch.uint8
    )
    # Garbage past each request's last block on purpose: the kernels must not read it.
    bt = torch.randint(0, pages, (batch, nbmax), generator=g, device="cuda", dtype=torch.int32)
    return q.view(torch.float8_e4m3fn), pool.view(torch.float8_e4m3fn), bt, seqs.cuda(), nbmax


def _split_all_ranks(split, index_tp_split, q, pool, bt, sl, buf, world):
    heads = buf.shape[0]
    outs = [buf[h].t() for h in range(heads)]
    outs = outs + [outs[0]] * (4 - len(outs))
    for r in range(world):
        split(
            q,
            pool,
            bt,
            sl,
            outs[0],
            outs[1],
            outs[2],
            outs[3],
            heads,
            DQL,
            r,
            world,
            index_tp_split.split_k_for_batch(int(sl.shape[0])),
        )


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "batch,seqs,heads",
    [
        (3, [5, 129, 70000], 4),
        (16, [65536] * 16, 4),
        (24, list(range(1000, 240001, 10000))[:24], 2),
        (48, None, 4),
    ],
    ids=["tiny-ragged", "b16-fixed", "b24-ramp-2heads", "b48-random"],
)
def test_split_scorer_matches_unsplit(world, batch, seqs, heads):
    split, unsplit, runner, index_tp_split = _ops()
    if seqs is None:
        g = torch.Generator().manual_seed(7)
        seqs = torch.randint(20000, 200001, (batch,), generator=g)
    q, pool, bt, sl, nbmax = _case(batch, seqs, heads)
    tokens = q.shape[0]

    ref = torch.full((heads, nbmax, tokens), SENTINEL, dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    unsplit(q, pool, bt, sl, ref.transpose(1, 2), DQL)

    got = torch.full_like(ref.view(torch.int32), SENTINEL).view(torch.float32)
    _split_all_ranks(split, index_tp_split, q, pool, bt, sl, got, world)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(2):
                _split_all_ranks(split, index_tp_split, q, pool, bt, sl, got, world)
    for _ in range(3):
        got.view(torch.int32).fill_(SENTINEL)
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    assert all(int(c.item()) == 0 for c in runner._work_counters.values())
