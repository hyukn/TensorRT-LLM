# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 horizontal producers with local index-Q (every index head on every TP rank).

With TRTLLM_M3_INDEX_TP_SPLIT_LOCAL_Q a rank's packed row is [Q | K | V | all index-Q | index-K]
and the producer runs with num_heads_index a multiple of num_heads_kv. Every output must be
bitwise what the KV-group-sharded producers give: the rank's Q and caches as in its own sharded
call, and index-Q head h as in the sharded call of the rank that owns h.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOTAL_Q_HEADS, TOTAL_KV_HEADS, TOTAL_INDEX_HEADS, DIM = 64, 4, 4, 128


def _rope_cache(max_positions, rotary_dim=64, base=5_000_000.0):
    positions = torch.arange(max_positions, dtype=torch.float32, device="cuda")
    inverse_frequency = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    frequency = torch.outer(positions, inverse_frequency)
    return torch.stack((frequency.cos(), frequency.sin()), dim=1).contiguous()


def _caches(kind, num_pages, num_kv_heads):
    index = torch.zeros(num_pages, 1, 128, 128, dtype=torch.float8_e4m3fn, device="cuda")
    if kind == "fp8":
        main = torch.zeros(
            num_pages, 2, num_kv_heads, 128, 128, dtype=torch.float8_e4m3fn, device="cuda"
        )
        return (main,), index
    data = torch.zeros(num_pages, 2, num_kv_heads, 128, 64, dtype=torch.uint8, device="cuda")
    scale = torch.zeros(num_pages, 2, num_kv_heads, 128, 8, dtype=torch.uint8, device="cuda")
    return (data, scale), index


def _run(kind, packed, main, index, slots, heads_q, heads_kv, heads_index, weights, rope, pos):
    common = (heads_q, heads_kv, heads_index, 128, 64, 1e-5, *weights, rope, pos)
    if kind == "fp8":
        return torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
            packed, main[0], index, slots, *common
        )
    inv_scales = torch.tensor([1.0, 1.75, 0.625], dtype=torch.float32, device="cuda")
    return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
        packed, main[0], main[1], index, slots, inv_scales, *common
    )


def _requires(kind):
    from tensorrt_llm._utils import get_sm_version

    if kind == "nvfp4" and get_sm_version() not in (100, 103):
        pytest.skip("NVFP4 quantization requires Blackwell")


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 32, 129])
def test_local_index_q_producer_matches_sharded(kind, tp_size, num_tokens):
    _requires(kind)
    gen = torch.Generator(device="cuda").manual_seed(1000 * tp_size + num_tokens)
    kv_groups = min(tp_size, TOTAL_KV_HEADS)
    heads_q = TOTAL_Q_HEADS // tp_size
    heads_kv = TOTAL_KV_HEADS // kv_groups
    heads_index = TOTAL_INDEX_HEADS // kv_groups
    num_pages = (num_tokens + 127) // 128 + 4

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16)

    # The projection outputs of one token batch: every rank sees the same index-Q / index-K
    # columns (identical projection input), its own Q / K / V.
    index_q_all = randn(num_tokens, TOTAL_INDEX_HEADS * DIM)
    index_k = randn(num_tokens, DIM)
    weights = [randn(DIM) for _ in range(4)]
    rope = _rope_cache(4096)
    pos = torch.randint(0, 4096, (num_tokens,), generator=gen, device="cuda", dtype=torch.int32)
    slots = torch.randperm((num_pages - 1) * 128, generator=gen, device="cuda")[:num_tokens]
    slots = slots.to(torch.int32)
    if num_tokens > 2:
        slots[-1] = -1

    def run_rank(rank, local_q):
        group = rank // max(tp_size // TOTAL_KV_HEADS, 1)
        g = torch.Generator(device="cuda").manual_seed(rank)
        qkv = torch.randn(
            num_tokens, (heads_q + 2 * heads_kv) * DIM, generator=g, device="cuda"
        ).to(torch.bfloat16)
        own = index_q_all[:, group * heads_index * DIM : (group + 1) * heads_index * DIM]
        packed = torch.cat([qkv, index_q_all if local_q else own, index_k], dim=1).contiguous()
        main, index = _caches(kind, num_pages, heads_kv)
        q, index_q = _run(
            kind,
            packed,
            main,
            index,
            slots,
            heads_q,
            heads_kv,
            TOTAL_INDEX_HEADS if local_q else heads_index,
            weights,
            rope,
            pos,
        )
        return q, index_q, main, index

    sharded = [run_rank(r, False) for r in range(tp_size)]
    for rank in range(tp_size):
        q, index_q, main, index = run_rank(rank, True)
        ref_q, _, ref_main, ref_index = sharded[rank]
        assert index_q.shape == (num_tokens, TOTAL_INDEX_HEADS, DIM)
        assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
        for cache, ref in zip(main, ref_main):
            assert torch.equal(cache.view(torch.uint8), ref.view(torch.uint8))
        assert torch.equal(index.view(torch.uint8), ref_index.view(torch.uint8))
        # Head h equals what its owner's sharded producer emits.
        for owner in range(0, tp_size, max(tp_size // TOTAL_KV_HEADS, 1)):
            group = owner // max(tp_size // TOTAL_KV_HEADS, 1)
            got = index_q[:, group * heads_index : (group + 1) * heads_index]
            assert torch.equal(got.view(torch.uint8), sharded[owner][1].view(torch.uint8))


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_local_index_q_producer_rejects_non_multiple(kind):
    _requires(kind)
    num_tokens, heads_q, heads_kv, heads_index = 2, 32, 2, 3
    packed = torch.randn(
        num_tokens, (heads_q + 2 * heads_kv + heads_index + 1) * DIM, device="cuda"
    ).to(torch.bfloat16)
    main, index = _caches(kind, 4, heads_kv)
    weights = [torch.ones(DIM, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    with pytest.raises(RuntimeError, match="multiple of KV heads"):
        _run(
            kind,
            packed,
            main,
            index,
            torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
            heads_q,
            heads_kv,
            heads_index,
            weights,
            _rope_cache(8),
            torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
        )
