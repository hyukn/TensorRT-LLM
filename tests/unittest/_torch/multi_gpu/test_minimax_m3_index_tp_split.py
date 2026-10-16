# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 index decode scoring split across TP ranks must equal the unsplit scorer bitwise.

Every rank scores all index heads on blocks rank::tp and writes each head into its owner's peer
buffer; after the barrier a rank's buffer must hold exactly what the production scorer writes for
that rank's heads over all blocks. Covers consecutive layers reusing the buffers and CUDA graph
replay, and TP wider than the index heads (replicated head groups: scores land in each group's
first rank and the replicas read them in place).

Local index-Q (TRTLLM_M3_INDEX_TP_SPLIT_LOCAL_Q): every rank passes all index heads and the split
skips the index-Q gather and its barrier. In the model the layer's tensor-parallel AllReduce
keeps the next layer's remote score stores out of a buffer a peer still reads; here a host
barrier (eager) or an extra flag barrier (CUDA graph) stands in for it.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

TOTAL_HEADS = 4
DQL = 4


def _inputs(batch, ctx, seed, total_heads=TOTAL_HEADS):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = torch.randint(
        max(1, ctx // 2), ctx + 1, (batch,), generator=g, device="cuda", dtype=torch.int32
    )
    seq[0] = ctx
    if batch > 2:
        seq[1] = DQL
    nb = (ctx + 127) // 128
    pages = batch * nb + 8
    pool = (torch.randn(pages, 1, 128, 128, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
    q = (torch.randn(batch * DQL, total_heads, 128, generator=g, device="cuda") * 2).to(
        torch.float8_e4m3fn
    )
    bt = (
        torch.randperm(pages - 8, generator=g, device="cuda")
        .to(torch.int32)[: batch * nb]
        .view(batch, nb)
        .contiguous()
    )
    return q, pool, bt, seq, nb


def _valid_mask(seq, nb, tq):
    nblk = (seq + 127) // 128
    req = torch.arange(tq, device="cuda") // DQL
    return torch.arange(nb, device="cuda")[:, None] < nblk[req][None, :]


def _run_rank(tp, total_heads=TOTAL_HEADS, local_q=False):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.index_tp_split import (
        IndexTpSplitWorkspace,
        tp_split_score,
    )
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops  # noqa: F401

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=tp, rank=rank, gpus_per_node=tp, tp_size=tp)
    # TP wider than the heads replicates head groups over consecutive ranks (index_head_range).
    replicas = max(1, tp // total_heads)
    groups = tp // replicas
    local = total_heads // groups
    group = rank // replicas
    heads = slice(group * local, (group + 1) * local)
    max_batch, max_ctx = 24, 200000
    k_tiles = (max_ctx + 127) // 128
    ws = IndexTpSplitWorkspace.get(
        mapping,
        local_heads=local,
        k_tiles=k_tiles,
        q_row_bytes=0 if local_q else local * 128,
        max_tokens=max_batch * DQL,
        replicas=replicas,
    )

    def q_arg(q_full):
        """What the split gets: this rank's heads, or with local index-Q every head (as a
        strided view, which the scorer must take as is)."""
        if not local_q:
            return q_full[:, heads].contiguous()
        backing = torch.empty(
            q_full.shape[0], total_heads + 1, 128, dtype=q_full.dtype, device=q_full.device
        )
        backing[:, :total_heads] = q_full
        return backing[:, :total_heads]

    def split(q_in):
        return tp_split_score(
            ws,
            q_in,
            pool,
            block_table=bt,
            seq_lens_cuda=seq,
            decode_query_len=DQL,
            local_heads=local,
            k_tiles=k_tiles,
            q_all_heads=local_q,
        )

    op = torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score
    try:
        cases = [
            (1, 300, 0),
            (3, 5000, 1),
            (8, 98304, 2),
            (24, 98304, 3),
            (5, 200000, 4),
            (24, 131072, 5),
        ]
        for batch, ctx, seed in cases:
            q, pool, bt, seq, nb = _inputs(batch, ctx, seed, total_heads)
            tq = batch * DQL
            ref = torch.empty(total_heads, nb, tq, device="cuda")
            op(q, pool.squeeze(1), bt, seq, ref.transpose(1, 2), DQL)
            valid = _valid_mask(seq, nb, tq)
            # Two consecutive "layers" with different Q reuse the same buffers.
            for layer in range(2):
                q_l = q if layer == 0 else (q.float() * -1).to(torch.float8_e4m3fn)
                ref_l = ref
                if layer == 1:
                    ref_l = torch.empty_like(ref)
                    op(q_l, pool.squeeze(1), bt, seq, ref_l.transpose(1, 2), DQL)
                got = split(q_arg(q_l)).clone()
                torch.cuda.synchronize()
                if local_q:
                    # No barrier A: keep the next layer's stores out until every rank has
                    # copied this layer's scores (the model's AllReduce does this).
                    tensorrt_llm.mpi_barrier()
                for lh in range(local):
                    a = got[lh, :nb][valid].view(torch.int32)
                    b = ref_l[group * local + lh][valid].view(torch.int32)
                    assert torch.equal(a, b), (
                        f"rank {rank} B={batch} ctx={ctx} layer={layer} head={lh}"
                    )

        # CUDA graph: three layers per replay, inputs refreshed in place between replays.
        batch, ctx = 16, 65536
        q, pool, bt, seq, nb = _inputs(batch, ctx, 7, total_heads)
        tq = batch * DQL
        q_static = q_arg(q)
        outs = []

        def body():
            outs.clear()
            for _ in range(3):
                outs.append(split(q_static))
                if local_q:
                    ws.barrier()  # stands in for the layer's AllReduce
            return outs[-1]

        body()
        torch.cuda.synchronize()
        tensorrt_llm.mpi_barrier()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                body()
        torch.cuda.synchronize()
        valid = _valid_mask(seq, nb, tq)
        for rep in range(3):
            gen = torch.Generator(device="cuda").manual_seed(100 + rep)
            q_new = (torch.randn(tq, total_heads, 128, device="cuda", generator=gen) * 2).to(
                torch.float8_e4m3fn
            )
            q_static.copy_(q_new if local_q else q_new[:, heads])
            g.replay()
            torch.cuda.synchronize()
            ref = torch.empty(total_heads, nb, tq, device="cuda")
            op(q_new, pool.squeeze(1), bt, seq, ref.transpose(1, 2), DQL)
            got = ws.local_scores(local, k_tiles, tq)
            for lh in range(local):
                a = got[lh, :nb][valid].view(torch.int32)
                b = ref[group * local + lh][valid].view(torch.int32)
                assert torch.equal(a, b), f"rank {rank} graph replay {rep} head {lh}"
            # The next replay's first stores must not land before every rank has compared.
            tensorrt_llm.mpi_barrier()
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_index_tp_split_matches_unsplit(mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(tp,)] * tp))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_index_tp_split_replicated_heads(mpi_pool_executor):
    # TP = 2 x heads: every head group spans two ranks (as TP8 with 4 index heads).
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(tp, tp // 2)] * tp))
    for r in results:
        assert r is True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
@pytest.mark.parametrize("replicated_heads", [False, True])
def test_index_tp_split_local_q(mpi_pool_executor, replicated_heads):
    # Local index-Q: every rank holds all index heads; no gather, one barrier per call.
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    heads = tp // 2 if replicated_heads else TOTAL_HEADS
    results = mpi_pool_executor.map(_run_rank, *zip(*[(tp, heads, True)] * tp))
    for r in results:
        assert r is True


if __name__ == "__main__":
    # Direct launch, one process per rank under an MPI launcher (srun --mpi=pmix / mpirun),
    # where spawning an MPI pool is not available.
    n = int(sys.argv[1])
    assert _run_rank(n) is True
    assert _run_rank(n, n // 2) is True  # replicated head groups
    assert _run_rank(n, local_q=True) is True  # local index-Q
    assert _run_rank(n, n // 2, local_q=True) is True  # local index-Q, replicated head groups
    print(f"rank {tensorrt_llm.mpi_rank()} OK", flush=True)
