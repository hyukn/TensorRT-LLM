# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMaxM3KVCacheManagerV2.get_block_ids_per_seq fast path must match the list-based path."""

import array
import time

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX


class _FakeKVCache:
    def __init__(self, indices, kind):
        self.num_blocks = len(indices)
        if kind == "array_i":
            self._seq = array.array("i", indices)
        elif kind == "array_q":
            self._seq = array.array("q", indices)
        else:
            self._seq = memoryview(array.array("i", indices))

    def get_base_page_indices(self, pool_id):
        assert pool_id == 0
        return self._seq


def _manager(kv_cache_map, cls=MiniMaxM3KVCacheManagerV2):
    mgr = object.__new__(cls)
    mgr.kv_cache_map = kv_cache_map
    return mgr


def _reference(mgr, request_ids):
    """The pre-fast-path implementation: Python lists, then a padded int32 tensor."""
    rows_src = MiniMaxM3KVCacheManagerV2._get_batch_cache_indices_by_pool_id(
        mgr, request_ids, pool_id=0
    )
    max_blocks = max((len(r) for r in rows_src), default=0)
    out = torch.zeros((len(rows_src), max_blocks), dtype=torch.int32)
    rows = out.numpy()
    for row, ids in zip(rows, rows_src):
        row[: len(ids)] = ids
    rows[rows == BAD_PAGE_INDEX] = 0
    return out


@pytest.mark.parametrize("kind", ["array_i", "array_q", "memoryview"])
def test_fast_path_matches_list_path(kind):
    torch.manual_seed(0)
    lengths = [0, 1, 127, 750, 1024, 5]
    kv_cache_map = {}
    for rid, n in enumerate(lengths):
        ids = torch.randint(0, 1 << 20, (n,)).tolist()
        if n > 3:
            ids[-2] = BAD_PAGE_INDEX
        kv_cache_map[100 + rid] = _FakeKVCache(ids, kind)
    mgr = _manager(kv_cache_map)
    assert mgr._block_ids_fast_path_ok()
    request_ids = [100 + i for i in (3, 0, 4, 1, 2, 5)]
    got = mgr.get_block_ids_per_seq(request_ids)
    want = _reference(mgr, request_ids)
    assert got.dtype == torch.int32
    assert torch.equal(got, want)
    assert torch.equal(mgr.get_block_ids_per_seq([]), _reference(mgr, []))


def test_override_falls_back_to_generic_path():
    class _Override(MiniMaxM3KVCacheManagerV2):
        def get_batch_cache_indices(self, request_ids, *args, **kwargs):
            return [[7, 8, 9] for _ in request_ids]

    mgr = _manager({1: _FakeKVCache([1, 2], "array_i")}, cls=_Override)
    assert not mgr._block_ids_fast_path_ok()
    assert mgr.get_block_ids_per_seq([1]).tolist() == [[7, 8, 9]]


def test_fast_path_is_faster_at_agentx_scale():
    kv_cache_map = {
        rid: _FakeKVCache(list(range(rid * 1000, rid * 1000 + 750)), "array_i") for rid in range(24)
    }
    mgr = _manager(kv_cache_map)
    ids = list(range(24))

    def best(fn, n=20):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    fast = best(lambda: mgr.get_block_ids_per_seq(ids))
    slow = best(lambda: _reference(mgr, ids))
    print(f"24 x 750 pages: list path {slow * 1e6:.0f} us, fast path {fast * 1e6:.0f} us")
    assert fast < slow
