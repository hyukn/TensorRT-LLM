# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMaxM3KVCacheManagerV2 caches its per-layer pool views and rebuilds them when the pool moves."""

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor


class _FakeImpl:
    def __init__(self, base, upper):
        self.base, self.upper = base, upper

    def get_mem_pool_base_address(self, layer_offset, role):
        return self.base + 4096 * layer_offset

    def get_page_index_upper_bound(self, layer_offset, role):
        return self.upper


def _manager(impl):
    mgr = object.__new__(MiniMaxM3KVCacheManagerV2)
    mgr.impl = impl
    mgr.layer_offsets = {3: 0, 4: 1}
    return mgr


def _geometry(mgr, calls):
    def fn(layer_idx, kv_layout):
        calls.append((layer_idx, kv_layout))
        off = mgr.layer_offsets[layer_idx]
        num_slots = mgr.impl.upper // 4
        return mgr.impl.base + 4096 * off, torch.uint8, num_slots, 4, [2, 128, 8]

    return fn


def test_pool_view_cached_and_rebuilt_on_move():
    pool = torch.zeros(1 << 22, dtype=torch.uint8, device="cuda")
    impl = _FakeImpl(pool.data_ptr(), 64)
    mgr = _manager(impl)
    calls = []
    geo = _geometry(mgr, calls)
    v1 = mgr._cached_pool_view("kv", 3, "HND", None, geo)
    v2 = mgr._cached_pool_view("kv", 3, "HND", None, geo)
    assert v1 is v2 and len(calls) == 1
    # Same view as building it directly.
    addr, dt, n, sc, shape = geo(3, "HND")
    ref = convert_to_torch_tensor(TensorWrapper(addr, dt, [n, sc, *shape]))[:, :2]
    assert v1.data_ptr() == ref.data_ptr() and v1.shape == ref.shape and v1.stride() == ref.stride()
    # Another layer / layout / kind is a separate entry.
    mgr._cached_pool_view("kv", 4, "HND", None, geo)
    mgr._cached_pool_view("kv", 3, "NHD", None, geo)
    mgr._cached_pool_view("kv_scale", 3, "HND", None, geo)
    n_calls = len(calls)
    # Pool re-laid out (rebalance): base address or page count changes -> rebuilt.
    impl.upper = 128
    v3 = mgr._cached_pool_view("kv", 3, "HND", None, geo)
    assert v3 is not v1 and v3.shape[0] == 128 // 4 and len(calls) == n_calls + 1
    impl.base += 1 << 20
    v4 = mgr._cached_pool_view("kv", 3, "HND", None, geo)
    assert v4.data_ptr() == v3.data_ptr() + (1 << 20) and len(calls) == n_calls + 2
    assert mgr._cached_pool_view("kv", 3, "HND", None, geo) is v4


def test_pool_view_built_in_inference_mode_is_mutable_outside():
    pool = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    mgr = _manager(_FakeImpl(pool.data_ptr(), 16))
    geo = _geometry(mgr, [])
    with torch.inference_mode():
        v = mgr._cached_pool_view("kv", 3, "HND", None, geo)
        v.fill_(1)
    assert not v.is_inference()
    v.zero_()  # the KV cache NaN check does this outside inference mode
    with torch.inference_mode():
        assert mgr._cached_pool_view("kv", 3, "HND", None, geo) is v
        v.add_(1)
