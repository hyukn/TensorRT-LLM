# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniMax-M3 MSA fmha_sm100 kernel prewarm.

The variant enumeration is checked against the variant fmha_sm100's own
planner and dispatch select for every context row length, and the concurrent
builds against both fmha_sm100's real variant manager and a stand-in for
fmha_sm100.jit shared by several processes. The JIT build log is checked to
reach the TensorRT-LLM logger.
"""

import fcntl
import logging
import math
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_utils
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_HEAD_DIM,
    MSA_REQUIRED_TOPK,
    fmha_context_variants,
    msa_package_available,
    prewarm_fmha_variants,
    require_msa_module,
)
from tensorrt_llm._utils import get_sm_version

PAGE_SIZE = 128
# MiniMax-M3 per rank under attention DP: every head stays on each rank.
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
NUM_INDEX_HEADS = 4
# Every row length selects the variant of some length up to this one.
MAX_DISTINCT_QO_LEN = 129

skip_without_msa = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_sm_version() not in (100, 103)
    or not msa_package_available(),
    reason="fmha_sm100 (MSA) requires SM100/SM103 and the packaged kernels.",
)


def _context_variants(fmha_sm100, max_qo_len):
    """The variants of a MiniMax-M3 context step with an FP8 cache."""
    return fmha_context_variants(
        fmha_sm100,
        max_qo_len=max_qo_len,
        page_size=PAGE_SIZE,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_index_heads=NUM_INDEX_HEADS,
        attn_dtype=torch.float8_e4m3fn,
        index_dtype=torch.bfloat16,
    )


class _DispatchRecorded(Exception):
    """Raised by the recording stand-in for get_fmha_variant."""


def _dispatched_variant(fmha_sm100, monkeypatch, *, qo_len, heads, kv_heads, dtype, site):
    """Return the variant key fmha_sm100 selects for one row of qo_len tokens.

    Runs the real planner and the real dispatch up to get_fmha_variant, which
    records the key and stops before any kernel is built or launched.
    """
    recorded = []

    def record(*key):
        recorded.append(key)
        raise _DispatchRecorded

    monkeypatch.setattr(fmha_sm100.api, "get_fmha_variant", record)
    device = torch.device("cuda")
    kv_len = qo_len + 2 * PAGE_SIZE
    qo_lens = torch.tensor([qo_len], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    num_pages = math.ceil(kv_len / PAGE_SIZE)
    q = torch.zeros(qo_len, heads, MSA_REQUIRED_HEAD_DIM, device=device).to(dtype)
    kv = torch.zeros(num_pages, kv_heads, PAGE_SIZE, MSA_REQUIRED_HEAD_DIM, device=device).to(dtype)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    plan_kwargs = {}
    run_kwargs = {"output_maxscore": False}
    kv_block_indexes = None
    if site == "proxy":
        plan_kwargs["output_maxscore"] = True
        run_kwargs = {"output_maxscore": True, "output_o": False}
    elif site == "sparse":
        plan_kwargs["kv_block_num"] = MSA_REQUIRED_TOPK
        kv_block_indexes = torch.zeros(
            qo_len, kv_heads, MSA_REQUIRED_TOPK, dtype=torch.int32, device=device
        )
    plan = fmha_sm100.fmha_sm100_plan(
        qo_lens,
        kv_lens,
        heads,
        num_kv_heads=kv_heads,
        qo_offset=kv_lens - qo_lens,
        page_size=PAGE_SIZE,
        num_kv_splits=1,
        causal=True,
        **plan_kwargs,
    )
    with pytest.raises(_DispatchRecorded):
        fmha_sm100.fmha_sm100(
            q, kv, kv, plan, kv_indices=kv_indices, kv_block_indexes=kv_block_indexes, **run_kwargs
        )
    return recorded[0]


@skip_without_msa
def test_context_variants_match_fmha_sm100_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fmha_sm100 = require_msa_module()
    sparse_qo_len_limit = fmha_sm100.api._prefill_qlen_threshold(True)
    dispatched = set()
    for qo_len in range(1, MAX_DISTINCT_QO_LEN + 1):
        dispatched.add(
            _dispatched_variant(
                fmha_sm100,
                monkeypatch,
                qo_len=qo_len,
                heads=NUM_INDEX_HEADS,
                kv_heads=1,
                dtype=torch.bfloat16,
                site="proxy",
            )
        )
        dispatched.add(
            _dispatched_variant(
                fmha_sm100,
                monkeypatch,
                qo_len=qo_len,
                heads=NUM_Q_HEADS,
                kv_heads=NUM_KV_HEADS,
                dtype=torch.float8_e4m3fn,
                site="dense",
            )
        )
        # Longer sparse rows run the CuTe DSL prefill, which is not a variant.
        if qo_len <= sparse_qo_len_limit:
            dispatched.add(
                _dispatched_variant(
                    fmha_sm100,
                    monkeypatch,
                    qo_len=qo_len,
                    heads=NUM_Q_HEADS,
                    kv_heads=NUM_KV_HEADS,
                    dtype=torch.float8_e4m3fn,
                    site="sparse",
                )
            )
        enumerated = _context_variants(fmha_sm100, qo_len)
        assert set(enumerated) == dispatched, f"row lengths up to {qo_len}"
    # A longer row selects one of the variants already listed.
    for qo_len in (MAX_DISTINCT_QO_LEN + 1, 8704):
        for heads, kv_heads, dtype, site in (
            (NUM_INDEX_HEADS, 1, torch.bfloat16, "proxy"),
            (NUM_Q_HEADS, NUM_KV_HEADS, torch.float8_e4m3fn, "dense"),
        ):
            assert (
                _dispatched_variant(
                    fmha_sm100,
                    monkeypatch,
                    qo_len=qo_len,
                    heads=heads,
                    kv_heads=kv_heads,
                    dtype=dtype,
                    site=site,
                )
                in enumerated
            )


@skip_without_msa
def test_prewarm_builds_every_variant_with_the_real_variant_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent builds through fmha_sm100's own variant manager all succeed.

    The manager loads the variant source templates lazily and without a lock,
    so builds fanned out over cold templates race on them. Only nvcc is
    stubbed out; every round starts from a fresh manager and an empty cache.
    """
    fmha_sm100 = require_msa_module()
    jit = fmha_sm100.jit
    variants = _context_variants(fmha_sm100, 8704)

    def write_ninja(self, cache_dir, variant_name, inst_cu, run_cu):
        pass

    def run_ninja(self, cache_dir):
        (cache_dir / f"{cache_dir.name}.so").write_bytes(b"")

    loaded = []
    monkeypatch.setattr(jit.FMHAVariantManager, "_write_ninja", write_ninja)
    monkeypatch.setattr(jit.FMHAVariantManager, "_run_ninja", run_ninja)
    monkeypatch.setattr(jit, "get_fmha_variant", lambda *key: loaded.append(key))
    for round_idx in range(50):
        monkeypatch.setattr(jit, "CACHE_BASE", tmp_path / str(round_idx))
        monkeypatch.setattr(jit, "_variant_manager", jit.FMHAVariantManager())
        loaded.clear()
        prewarm_fmha_variants(fmha_sm100, variants)
        assert sorted(loaded) == variants, f"round {round_idx}"


def _acquire_file_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


@pytest.mark.cpu_only
def test_prewarm_builds_each_variant_once_across_processes(tmp_path: Path) -> None:
    """Concurrent prewarms sharing a cache build each variant once; a failed
    build is reported without stopping the others or the caller."""
    builds, loads = [], []
    record_lock = threading.Lock()

    def compile_only(name, params):
        if name == "broken":
            raise RuntimeError("nvcc failed")
        so_path = tmp_path / name / f"{name}.so"
        if so_path.exists():
            return
        # Widen the window in which an unlocked second build would start.
        time.sleep(0.05)
        so_path.write_bytes(b"")
        with record_lock:
            builds.append(name)

    def get_fmha_variant(*key):
        with record_lock:
            loads.append(key[0])

    jit = SimpleNamespace(
        CACHE_BASE=tmp_path,
        _variant_key_from_runtime=lambda *key: (key[0], {}),
        _acquire_file_lock=_acquire_file_lock,
        _release_file_lock=_release_file_lock,
        _variant_manager=SimpleNamespace(_load_templates=lambda: None, _compile_only=compile_only),
        get_fmha_variant=get_fmha_variant,
    )
    fmha_sm100 = SimpleNamespace(jit=jit)
    variants = [("a",), ("b",), ("broken",), ("c",)]
    # Each thread stands in for one rank of a node sharing the kernel cache.
    ranks = [
        threading.Thread(target=prewarm_fmha_variants, args=(fmha_sm100, variants))
        for _ in range(4)
    ]
    for rank in ranks:
        rank.start()
    for rank in ranks:
        rank.join()
    assert sorted(builds) == ["a", "b", "c"]
    assert sorted(loads) == sorted(["a", "b", "c"] * len(ranks))


@pytest.mark.cpu_only
def test_fmha_jit_log_reaches_trtllm_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """fmha_sm100's per-variant build message is re-emitted by the TensorRT-LLM
    logger, once, and a warning keeps its severity."""
    emitted = []
    monkeypatch.setattr(msa_utils.logger, "info", lambda msg: emitted.append(("info", msg)))
    monkeypatch.setattr(msa_utils.logger, "warning", lambda msg: emitted.append(("warning", msg)))
    msa_utils._forward_fmha_jit_log()
    msa_utils._forward_fmha_jit_log()
    jit_logger = logging.getLogger("fmha_sm100.jit")
    jit_logger.info("JIT compiling FMHA variant: %s", "1_0_1_3_1_0_2")
    jit_logger.warning("cache miss")
    assert emitted == [
        ("info", "fmha_sm100.jit: JIT compiling FMHA variant: 1_0_1_3_1_0_2"),
        ("warning", "fmha_sm100.jit: cache miss"),
    ]
