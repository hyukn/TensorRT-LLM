# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eagle3 draft LM head vocab-sharded across attention-DP ranks must reproduce the local argmax.

Each rank owns a different number of draft rows (padded to max_num_requests), the group
all-gathers them, projects onto its vocabulary shard and the production greedy sampler
(greedy_sample_draft_with_tp_gather) combines the shard maxima. Every rank's tokens must equal
argmax(full-vocab logits of its own rows) -- except on exact bf16 ties, where either token is a
valid greedy pick. Covers several row counts (incl. an empty rank), an LM_HEAD_TP_SIZE override
(two groups of two), the fallback when the vocabulary does not split, and CUDA graph replay.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
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

HIDDEN = 1024
VOCAB = 32768


def _sampler(mapping):
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    class _Sampler:
        _get_local_max_and_combined = SpecWorkerBase._get_local_max_and_combined
        _get_draft_tokens_from_gathered = SpecWorkerBase._get_draft_tokens_from_gathered
        greedy = SpecWorkerBase.greedy_sample_draft_with_tp_gather

    s = _Sampler()
    s.mapping = mapping
    return s


def _all_rows(world, counts, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(counts[r], HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        for r in range(world)
    ]


def _check(tokens, rows, weight):
    """tokens == argmax of this rank's full-vocab logits, up to exact-value ties."""
    ref_logits = F.linear(rows, weight).float()
    ref = torch.argmax(ref_logits, dim=-1).to(torch.int32)
    assert tokens.shape == ref.shape, (tokens.shape, ref.shape)
    bad = tokens != ref
    if bool(bad.any()):
        rows_idx = bad.nonzero().flatten()
        got_val = ref_logits[rows_idx, tokens[rows_idx].long()]
        ref_val = ref_logits[rows_idx, ref[rows_idx].long()]
        # Different GEMM shapes may round a near-tie differently; anything else is a bug.
        ulp = ref_val.abs() * 2.0**-7
        assert bool((ref_val - got_val <= 2 * ulp).all()), (rows_idx, got_val, ref_val)
        assert int(bad.sum()) <= max(1, rows.shape[0] // 8)


def _run_rank(world):
    from tensorrt_llm._torch.speculative import eagle3_lm_head_tp

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(
            world_size=world,
            rank=rank,
            gpus_per_node=world,
            tp_size=world,
            enable_attention_dp=True,
        )
        os.environ.pop(eagle3_lm_head_tp.ENV, None)
        assert not eagle3_lm_head_tp.enabled(mapping)
        os.environ[eagle3_lm_head_tp.ENV] = "1"
        assert eagle3_lm_head_tp.enabled(mapping)
        assert not eagle3_lm_head_tp.enabled(
            Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        )

        g = torch.Generator(device="cuda").manual_seed(7)
        weight = (torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
        sampler = _sampler(mapping)

        # (max_rows, per-rank row counts, LM_HEAD_TP_SIZE override)
        cases = [
            (18, [18 - r for r in range(world)], None),
            (18, [1] + [0] * (world - 1), None),
            (8, [8] * world, None),
            (24, [r * 5 % 24 for r in range(world)], "2" if world % 2 == 0 else None),
        ]
        for ci, (max_rows, counts, tp_override) in enumerate(cases):
            if tp_override is None:
                os.environ.pop("LM_HEAD_TP_SIZE", None)
            else:
                os.environ["LM_HEAD_TP_SIZE"] = tp_override
            rows = _all_rows(world, counts, seed=100 + ci)[rank]
            out = eagle3_lm_head_tp.sharded_draft_logits(rows, weight, mapping, max_rows)
            assert out is not None
            logits, lm_map = out
            tp = lm_map.tp_size
            assert tp == (int(tp_override) if tp_override else min(world, 256 // max_rows))
            assert logits.shape == (tp * max_rows, VOCAB // tp) and logits.dtype == torch.float32
            tokens = sampler.greedy(logits, None, lm_map)[: counts[rank]]
            _check(tokens, rows, weight)
        os.environ.pop("LM_HEAD_TP_SIZE", None)

        # A vocabulary that does not split evenly falls back (same decision on every rank).
        odd = weight[: VOCAB - 1]
        assert eagle3_lm_head_tp.sharded_draft_logits(rows[:1], odd, mapping, 18) is None

        # CUDA graph capture + replay with new rows.
        max_rows = 16
        static_rows = torch.zeros(max_rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        n_live = max_rows - rank
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm up (compiles the sampler's combine kernel)
                lg, lm_map = eagle3_lm_head_tp.sharded_draft_logits(
                    static_rows[:n_live], weight, mapping, max_rows
                )
                sampler.greedy(lg, None, lm_map)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lg, lm_map = eagle3_lm_head_tp.sharded_draft_logits(
                static_rows[:n_live], weight, mapping, max_rows
            )
            static_tokens = sampler.greedy(lg, None, lm_map)
        for it in range(3):
            rows = _all_rows(world, [max_rows] * world, seed=500 + it)[rank]
            static_rows.copy_(rows)
            graph.replay()
            torch.cuda.synchronize()
            _check(static_tokens[:n_live].clone(), rows[:n_live], weight)
        os.environ.pop(eagle3_lm_head_tp.ENV, None)
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_eagle3_draft_lm_head_tp_matches_local_argmax(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world,)] * world))
    for r in results:
        assert r is True


if __name__ == "__main__":
    # Direct launch, one process per rank under an MPI launcher (srun --mpi=pmix / mpirun),
    # where spawning an MPI pool is not available.
    assert _run_rank(int(sys.argv[1])) is True
    print(f"rank {tensorrt_llm.mpi_rank()} OK", flush=True)
