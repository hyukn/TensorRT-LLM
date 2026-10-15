# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prepare() builds only the fmha_sm100 plans some layer reads.

With NVFP4 sparse KV the sparse layers run the k2q CSR kernel (no GQA plan), the tcgen05 context
scorer replaces the proxy plan, and trtllm-gen dense prefill replaces the dense plan. Skipping
them saves host time per context step, and the sparse-GQA plan's short-row path asserts
total_q * heads <= 65536, which an attention-DP rank (all heads) can exceed in warmup.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import (
    trtllm_gen_dense_decode as tgd,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttention,
)
from tensorrt_llm.bindings import DataType


def _metadata(*, num_contexts, qo_lens, kv_lens, nvfp4, staged_cu):
    cls = MiniMaxM3MsaSparseAttention.Metadata
    m = cls.__new__(cls)
    m.mapping = None
    m._seq_lens = torch.tensor(qo_lens, dtype=torch.int32)
    m.num_contexts = num_contexts
    m.kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    m.is_cuda_graph = False
    m.kv_cache_manager = SimpleNamespace(
        dtype=DataType.NVFP4 if nvfp4 else DataType.BF16,
        is_nvfp4_layer=lambda i: nvfp4,
        tokens_per_block=128,
        max_total_draft_tokens=3,
    )
    m._stage_host_lengths()
    m._set_decode_span()
    m.msa_cu_q_lens = torch.zeros(len(qo_lens) + 1, dtype=torch.int32) if staged_cu else None
    m._msa_fields_ready = True
    m._msa_params = SimpleNamespace(
        sharded_index_head_count=lambda mapping: 4,
        sharded_head_counts=lambda mapping: (64, 4),
        topk=16,
    )
    m._msa_prefill_n_valid_buf = None
    m.max_num_tokens = 32768
    return m


@pytest.fixture
def stubs(monkeypatch):
    calls = []

    class FakeFmha:
        @staticmethod
        def fmha_sm100_plan(qo, kv, num_qo_heads, **kw):
            calls.append(("plan", num_qo_heads, kw.get("kv_block_num"), kw.get("output_maxscore")))
            return ("plan",)

    state = SimpleNamespace(ctx=True, dense=True, calls=calls)

    class FakeCtx:
        @staticmethod
        def ctx_plan_tc(qo, kv, heads):
            return (1, 1)

    monkeypatch.setattr(msa_backend, "require_msa_module", lambda: FakeFmha)
    monkeypatch.setattr(msa_backend, "ctx_scorer_module", lambda: FakeCtx if state.ctx else None)
    monkeypatch.setattr(tgd, "dense_prefill_trtllm_gen_enabled", lambda: state.dense)
    monkeypatch.setattr(msa_backend, "_cache_device", lambda meta: torch.device("cpu"))
    return state


def _kinds(calls):
    out = set()
    for _, _, kv_block_num, maxscore in calls:
        out.add("proxy" if maxscore else ("gqa" if kv_block_num else "dense"))
    return out


def test_nvfp4_ctx_scorer_trtllm_gen_dense_builds_no_fmha_plan(stubs):
    m = _metadata(
        num_contexts=2, qo_lens=(3000, 900), kv_lens=(150000, 9000), nvfp4=True, staged_cu=True
    )
    assert tuple(m._msa_fmha_plan_consumers()) == (False, False, False)
    m._build_step_plans()
    assert stubs.calls == []
    assert m.msa_prefill_proxy_plan is None and m.msa_prefill_gqa_plan is None
    assert m.msa_prefill_dense_plan is None
    # The tcgen05 scorer plan and the valid-block counts are still staged.
    assert m._msa_ctx_score_plan is not None
    assert m.msa_prefill_n_valid_blocks is not None and m.msa_prefill_n_valid_blocks.numel() == 3900


def test_each_feature_off_brings_its_plan_back(stubs):
    stubs.ctx = False
    m = _metadata(num_contexts=1, qo_lens=(512,), kv_lens=(8192,), nvfp4=True, staged_cu=True)
    m._build_step_plans()
    assert _kinds(stubs.calls) == {"proxy"}

    stubs.calls.clear()
    stubs.ctx, stubs.dense = True, False
    m._build_step_plans()
    assert _kinds(stubs.calls) == {"dense"}

    stubs.calls.clear()
    stubs.dense = True
    m = _metadata(num_contexts=1, qo_lens=(512,), kv_lens=(8192,), nvfp4=False, staged_cu=True)
    m._build_step_plans()
    assert _kinds(stubs.calls) == {"gqa"}

    stubs.calls.clear()
    # trtllm-gen dense prefill without the staging it reads keeps the dense plan.
    m = _metadata(num_contexts=1, qo_lens=(512,), kv_lens=(8192,), nvfp4=True, staged_cu=False)
    m._build_step_plans()
    assert _kinds(stubs.calls) == {"dense"}


def test_mixed_step_and_attention_dp_head_count(stubs):
    """A mixed step (context prefix + decode rows) with all 64 heads on the rank, as attention DP
    runs it: no plan, so no 65536-row assertion; the context scorer still takes the prefix."""
    m = _metadata(
        num_contexts=1,
        qo_lens=(1500, 4, 4, 4),
        kv_lens=(120000, 5000, 6000, 7000),
        nvfp4=True,
        staged_cu=True,
    )
    assert m.msa_decode_span is not None and m.msa_decode_span.row_first == 1
    m._build_step_plans()
    assert stubs.calls == []
    assert m._msa_ctx_score_plan[0] == (0, 1)
