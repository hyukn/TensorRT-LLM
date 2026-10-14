# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy verify/draft tokens from vocab shards under plain TP (TRTLLM_SPEC_TP_VOCAB_SHARD_GREEDY).

The Eagle3 one-model verify head and draft head are vocab-parallel. With the opt-in each TP rank
projects onto its own vocab shard only (``LMHead.forward_vocab_shard``) and the tokens come from
the shards' (index, max) pairs (``SpecWorkerBase.greedy_sample_draft_with_tp_gather`` with a
plain-TP mapping) instead of argmax over all-gathered [rows, vocab] logits. These tests check,
on one GPU with the all-gathers emulated, that the shard projection is bitwise the gathered
projection's columns and that the shard argmax picks exactly the full-vocab argmax, ties included
(lowest vocab id wins in both).
"""

import pytest
import torch

import tensorrt_llm._torch.distributed as dist_pkg
import tensorrt_llm._torch.distributed.ops as dist_ops
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import (
    TP_VOCAB_SHARD_GREEDY_ENV_VAR,
    tp_vocab_shard_greedy_enabled,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _worker(mapping):
    worker = Eagle3OneModelWorker.__new__(Eagle3OneModelWorker)
    torch.nn.Module.__init__(worker)
    worker.mapping = mapping
    worker.guided_decoder = None
    worker.is_mtp_eagle = False
    return worker


def _tp_mapping(tp, rank, adp=False):
    return Mapping(world_size=tp, tp_size=tp, rank=rank, gpus_per_node=8, enable_attention_dp=adp)


def _logits_with_ties(rows, vocab, tp, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, vocab, device="cuda", generator=g).to(dtype)
    shard = vocab // tp
    top = logits.float().max().item() + 4.0
    for r in range(rows):
        kind = r % 5
        if kind == 1:  # tie across two shards: the lower id must win
            a, b = 3 + r, (tp - 1) * shard + 7
            logits[r, a] = top
            logits[r, b] = top
        elif kind == 2:  # tie inside one shard
            a = shard + 11
            logits[r, a] = top
            logits[r, a + 5] = top
        elif kind == 3:  # max at a shard boundary
            logits[r, shard] = top
            logits[r, shard - 1] = top
        elif kind == 4:  # tie in every shard at the same local offset
            for s in range(tp):
                logits[r, s * shard + 2] = top
    return logits


def _emulated_gather(worker_by_rank, shards):
    """allgather(combined, mapping, dim=-1) of every rank's packed (index, max)."""
    packed = [
        worker_by_rank[r]._get_local_max_and_combined(shards[r], worker_by_rank[r].mapping)
        for r in range(len(shards))
    ]
    return torch.cat(packed, dim=-1)


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 48, 192, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_argmax_is_full_argmax(tp, rows, dtype, monkeypatch):
    vocab = 200064
    logits = _logits_with_ties(rows, vocab, tp, dtype, seed=rows * 31 + tp)
    expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
    shard = vocab // tp
    shards = [logits[:, r * shard : (r + 1) * shard].contiguous() for r in range(tp)]
    workers = [_worker(_tp_mapping(tp, r)) for r in range(tp)]
    gathered = _emulated_gather(workers, shards)

    calls = []

    def fake_allgather(t, mapping, dim=-1, sizes=None):
        calls.append((mapping.tp_size, dim))
        return gathered

    monkeypatch.setattr(dist_ops, "allgather", fake_allgather)
    for r in range(tp):
        tokens = workers[r].greedy_sample_draft_with_tp_gather(shards[r], None, workers[r].mapping)
        assert tokens.dtype == torch.int32
        assert torch.equal(tokens, expected), f"rank {r}"
    assert calls == [(tp, -1)] * tp


def test_adp_mapping_keeps_row_segments(monkeypatch):
    """An LM-head-TP-in-ADP mapping still selects this rank's row segment."""
    tp, rows_per_rank, vocab = 4, 3, 4096
    logits = _logits_with_ties(tp * rows_per_rank, vocab, tp, torch.bfloat16, seed=5)
    expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
    shard = vocab // tp
    shards = [logits[:, r * shard : (r + 1) * shard].contiguous() for r in range(tp)]
    workers = [_worker(_tp_mapping(tp, r, adp=True)) for r in range(tp)]
    gathered = _emulated_gather(workers, shards)
    monkeypatch.setattr(dist_ops, "allgather", lambda t, mapping, dim=-1, sizes=None: gathered)
    for r in range(tp):
        tokens = workers[r].greedy_sample_draft_with_tp_gather(shards[r], None, workers[r].mapping)
        assert torch.equal(tokens, expected[r * rows_per_rank : (r + 1) * rows_per_rank])


def _lm_heads(vocab, hidden, tp, adp=False):
    heads = []
    g = torch.Generator(device="cuda").manual_seed(11)
    full = (torch.randn(vocab, hidden, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    for r in range(tp):
        # reduce_output only matters for ROW mode; off, this single-process test needs no
        # all-reduce workspace for its TP mappings.
        head = LMHead(
            vocab,
            hidden,
            dtype=torch.bfloat16,
            mapping=_tp_mapping(tp, r, adp),
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
        ).cuda()
        if head.tp_mode == TensorParallelMode.COLUMN:
            rows = head.out_features
            w = torch.zeros(rows, hidden, dtype=torch.bfloat16, device="cuda")
            part = full[r * rows : min((r + 1) * rows, vocab)]
            w[: part.shape[0]] = part
        else:
            w = full
        with torch.no_grad():
            head.weight.copy_(w)
        heads.append(head)
    return heads


@pytest.mark.parametrize("rows", [4, 192])
def test_forward_vocab_shard_is_gathered_columns(rows, monkeypatch):
    tp, vocab, hidden = 4, 200064, 1024
    heads = _lm_heads(vocab, hidden, tp)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    local = [h.forward_vocab_shard(x) for h in heads]
    monkeypatch.setattr(
        dist_pkg, "allgather", lambda t, mapping, dim=-1, sizes=None: torch.cat(local, dim=-1)
    )
    for r, head in enumerate(heads):
        assert head.supports_vocab_shard_projection()
        full = head(x)
        assert full.shape == (rows, vocab)
        shard = vocab // tp
        assert torch.equal(full[:, r * shard : (r + 1) * shard], head.forward_vocab_shard(x))


def test_vocab_shard_projection_gates():
    # Padded vocab (not divisible by TP): the gathered output drops padding columns.
    assert not _lm_heads(1001, 256, 4)[0].supports_vocab_shard_projection()
    # Attention DP replicates the head.
    assert not _lm_heads(1024, 256, 4, adp=True)[0].supports_vocab_shard_projection()
    # A single rank has nothing to gather.
    assert not _lm_heads(1024, 256, 1)[0].supports_vocab_shard_projection()
    assert _lm_heads(1024, 256, 2)[0].supports_vocab_shard_projection()


def test_env_gate(monkeypatch):
    monkeypatch.delenv(TP_VOCAB_SHARD_GREEDY_ENV_VAR, raising=False)
    assert not tp_vocab_shard_greedy_enabled()
    draft = torch.nn.Module()
    draft.lm_head = _lm_heads(1024, 256, 4)[0]
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    draft.logits_processor = LogitsProcessor()
    worker = _worker(_tp_mapping(4, 0))
    assert not worker._draft_from_tp_vocab_shards(draft)
    monkeypatch.setenv(TP_VOCAB_SHARD_GREEDY_ENV_VAR, "1")
    assert tp_vocab_shard_greedy_enabled()
    assert worker._draft_from_tp_vocab_shards(draft)
    worker.guided_decoder = object()
    assert not worker._draft_from_tp_vocab_shards(draft)
