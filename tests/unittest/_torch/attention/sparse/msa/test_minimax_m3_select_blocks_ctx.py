# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-step block selector must be bitwise what the packaged selector returns."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_utils
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import (
    select_blocks_ctx as sbc,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM100-family GPU required",
)

TOPK = 16
PAGE = 128


def _packaged(scores, nvb, init_blocks, local_blocks, head_major):
    return torch.ops.trtllm.minimax_m3_select_blocks(
        scores, nvb, TOPK, init_blocks, local_blocks, head_major
    )


def _causal_n_valid(total_q, cached, qo_lens=None):
    """Per-token valid-block counts of a context batch (as per_token_valid_blocks makes them)."""
    qo_lens = qo_lens or [total_q]
    out = []
    for qo, c in zip(qo_lens, cached):
        pos = torch.arange(qo, dtype=torch.int64) + c
        out.append((pos + 1 + PAGE - 1) // PAGE)
    return torch.cat(out).to(torch.int32).cuda()


def _scores(kind, heads, blocks, tokens, gen):
    if kind == "randn":
        s = torch.randn(heads, blocks, tokens, generator=gen, device="cuda")
    elif kind == "recency":
        # Later blocks score higher on average (the order the streaming pass sees worst).
        trend = torch.linspace(-3, 3, blocks, device="cuda").view(1, -1, 1)
        s = torch.randn(heads, blocks, tokens, generator=gen, device="cuda") + trend
    elif kind == "ties":
        # Few distinct values: many exact key ties, overflowing the candidate buffer.
        s = torch.randint(0, 3, (heads, blocks, tokens), generator=gen, device="cuda").float()
    elif kind == "all_equal":
        s = torch.zeros(heads, blocks, tokens, device="cuda")
    elif kind == "specials":
        s = torch.randn(heads, blocks, tokens, generator=gen, device="cuda")
        r = torch.rand(heads, blocks, tokens, generator=gen, device="cuda")
        s = torch.where(r < 0.05, torch.full_like(s, float("-inf")), s)
        s = torch.where((r >= 0.05) & (r < 0.06), torch.full_like(s, float("inf")), s)
        s = torch.where((r >= 0.06) & (r < 0.07), torch.zeros_like(s), s)
        s = torch.where((r >= 0.07) & (r < 0.08), -torch.zeros_like(s), s)
        s = torch.where((r >= 0.08) & (r < 0.085), torch.full_like(s, float("nan")), s)
    elif kind == "neg_inf":
        s = torch.full((heads, blocks, tokens), float("-inf"), device="cuda")
    else:
        raise ValueError(kind)
    return s.contiguous()


@pytest.mark.parametrize("head_major", [True, False])
@pytest.mark.parametrize("init_local", [(0, 1), (1, 2), (0, 0), (3, 0)])
@pytest.mark.parametrize("kind", ["randn", "recency", "ties", "all_equal", "specials", "neg_inf"])
@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_matches_packaged_selector(kind, init_local, head_major, variant):
    heads, tokens = 2, 1000
    cached = 150 * PAGE + 37
    blocks = (cached + tokens + PAGE - 1) // PAGE  # 1181
    gen = torch.Generator(device="cuda").manual_seed(hash((kind, init_local)) % 2**31)
    scores = _scores(kind, heads, blocks, tokens, gen)
    nvb = _causal_n_valid(tokens, [cached])
    init_blocks, local_blocks = init_local
    assert sbc.supports(scores, nvb, TOPK)
    ref = _packaged(scores, nvb, init_blocks, local_blocks, head_major)
    got = sbc.select_blocks_ctx(
        scores,
        nvb,
        topk=TOPK,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        head_major_output=head_major,
        variant=variant,
    )
    assert got.shape == ref.shape and got.stride() == ref.stride()
    assert torch.equal(got, ref)


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("blocks", [129, 300, 2800, 4100])
def test_varlen_batches_and_ragged_n_valid(heads, blocks):
    """Varlen context rows, n_valid past num_blocks, zero and tiny rows, a non-multiple-of-32 tail."""
    gen = torch.Generator(device="cuda").manual_seed(blocks * 7 + heads)
    tokens = 777
    scores = _scores("randn", heads, blocks, tokens, gen)
    nvb = torch.randint(0, blocks + 40, (tokens,), generator=gen, device="cuda", dtype=torch.int32)
    nvb[:5] = torch.tensor([0, 1, 15, 16, 17], dtype=torch.int32)
    for head_major in (True, False):
        ref = _packaged(scores, nvb, 0, 1, head_major)
        got = sbc.select_blocks_ctx(
            scores, nvb, topk=TOPK, init_blocks=0, local_blocks=1, head_major_output=head_major
        )
        assert torch.equal(got, ref)


def test_scorer_layout_ignores_unwritten_blocks():
    """The scorers' [H, max_k_tiles, total_q] buffer is only written below each token's
    n_valid; the rest is garbage (NaN here) and must never be read."""
    heads, tokens = 2, 2048
    qo_lens = [1500, 548]
    cached = [200 * PAGE, 90 * PAGE + 5]
    nvb = _causal_n_valid(tokens, cached, qo_lens)
    blocks = int(nvb.max()) + 64
    gen = torch.Generator(device="cuda").manual_seed(3)
    clean = _scores("randn", heads, blocks, tokens, gen)
    mask = torch.arange(blocks, device="cuda").view(1, -1, 1) < nvb.view(1, 1, -1)
    dirty = torch.where(mask, clean, torch.full_like(clean, float("nan")))
    ref = _packaged(clean, nvb, 0, 1, True)
    got = sbc.select_blocks_ctx(
        dirty, nvb, topk=TOPK, init_blocks=0, local_blocks=1, head_major_output=True
    )
    assert torch.equal(got, ref)


def test_large_context_chunk_shape():
    """A 16K-token chunk over ~350K cached tokens on one TP2 rank (2 KV heads)."""
    heads, tokens = 2, 16384
    cached = 2704 * PAGE
    nvb = _causal_n_valid(tokens, [cached])
    blocks = int(nvb.max())
    gen = torch.Generator(device="cuda").manual_seed(11)
    scores = _scores("recency", heads, blocks, tokens, gen)
    ref = _packaged(scores, nvb, 0, 1, True)
    got = sbc.select_blocks_ctx(
        scores, nvb, topk=TOPK, init_blocks=0, local_blocks=1, head_major_output=True
    )
    assert torch.equal(got, ref)


def test_supports_gates():
    s = torch.randn(2, 300, 512, device="cuda")
    nvb = torch.full((512,), 300, dtype=torch.int32, device="cuda")
    assert sbc.supports(s, nvb, 16)
    assert not sbc.supports(s, nvb, 8)  # topk
    assert not sbc.supports(s.transpose(1, 2).contiguous().transpose(1, 2), nvb, 16)  # stride
    assert not sbc.supports(torch.randn(2, 128, 512, device="cuda"), nvb, 16)  # bitonic range
    assert not sbc.supports(torch.randn(2, 300, 64, device="cuda"), nvb[:64], 16)  # decode-sized


def test_dispatch_through_select_blocks_from_maxscore(monkeypatch):
    """With the env gate on, the production entry point returns the packaged result."""
    monkeypatch.setattr(msa_utils, "ctx_select_enabled", lambda: True)
    monkeypatch.setattr(msa_utils, "ctx_select_verify", lambda: True)
    heads, tokens = 2, 4096
    cached = 600 * PAGE
    nvb = _causal_n_valid(tokens, [cached])
    blocks = int(nvb.max())
    scores = _scores("randn", heads, blocks, tokens, torch.Generator(device="cuda").manual_seed(5))
    got = msa_utils.select_blocks_from_maxscore(
        scores,
        topk=TOPK,
        n_valid_blocks=nvb,
        init_blocks=0,
        local_blocks=1,
        head_major_output=True,
    )
    assert torch.equal(got, _packaged(scores, nvb, 0, 1, True))
