# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversation-aware eviction ranking (KVCacheManager.set_eviction_hazard / _KVCache.set_eviction_hint).

Two released conversations A (older) and B share a GPU-only pool; a third sequence then needs part of their
pages. Plain LRU evicts A. With a hazard model, the conversation whose cell has the lower hazard at its idle
time is evicted, whichever was released first.
"""

import gc
import itertools
import os
import sys
import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import CudaStream, KVCacheManager, TokenId, TokenIdExt
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        CudaStream,
        KVCacheManager,
        TokenId,
        TokenIdExt,
    )

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from cuda_test_utils import (  # noqa: E402
        TemporaryCudaStream,
        init_cuda_once,
        temporary_sys_path,
    )
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

with temporary_sys_path(_TEST_DIR):
    from test_kv_cache_manager_v2 import create_config

TPB = 8
HIGH, LOW = 0.9, 0.1


class TestEvictionHazard(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        self._tok = itertools.count(1)
        gc.collect()
        gc.disable()
        self.manager = self._new_manager()

    def tearDown(self) -> None:
        gc.enable()
        self.manager.shutdown()

    @staticmethod
    def _new_manager() -> KVCacheManager:
        return KVCacheManager(create_config(TPB, 4 << 20, 0, 0, 1, None, 0, 8192))

    def _tokens(self, n: int) -> list[TokenIdExt]:
        return [TokenId(next(self._tok)) for _ in range(n)]

    @staticmethod
    def _capacity_blocks(manager: KVCacheManager, stream: CudaStream) -> int:
        kv = manager.create_kv_cache(None, [])
        kv.resume(stream)
        n = 0
        while n < 100000 and kv.resize((n + 1) * TPB):
            n += 1
        kv.close()
        return n

    @staticmethod
    def _release(manager, prompt, stream, hint=None) -> None:
        """Prefill + commit a conversation turn, then release it (optionally with an eviction hint)."""
        kv = manager.create_kv_cache(None, prompt)
        kv.resume(stream)
        assert kv.resize(len(prompt))
        n = kv.num_committed_tokens
        if n < len(prompt):
            kv.commit(prompt[n:])
        kv.stop_committing()
        if hint is not None:
            kv.set_eviction_hint(*hint)
        kv.close()

    def _scenario(self, manager, hint_a=None, hint_b=None, clock=None):
        """Release A then B, then allocate a new sequence that needs part of their pages.

        Returns (reuse of A before, after, reuse of B before, after).
        """
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            capacity = self._capacity_blocks(manager, stream)
            k = capacity // 4
            self.assertGreaterEqual(k, 4, "pool too small for the scenario")
            prompt_a, prompt_b = self._tokens(k * TPB), self._tokens(k * TPB)
            self._release(manager, prompt_a, stream, hint_a)
            self._release(manager, prompt_b, stream, hint_b)
            if clock is not None:
                manager.set_eviction_clock(clock)
            before = manager.probe_reuse(None, prompt_a), manager.probe_reuse(None, prompt_b)
            # Free pages: capacity - 2k. Needing k / 2 more evicts k / 2 pages of one conversation.
            kv = manager.create_kv_cache(None, [])
            kv.resume(stream)
            self.assertTrue(kv.resize((capacity - 2 * k + k // 2) * TPB))
            after = manager.probe_reuse(None, prompt_a), manager.probe_reuse(None, prompt_b)
            kv.close()
        s.take_finish_event().synchronize()
        manager.clear_reusable_blocks()
        self.assertGreater(before[0], 0)
        self.assertGreater(before[1], 0)
        return before[0], after[0], before[1], after[1]

    def _install(self, manager, rows, step=10.0):
        manager.set_eviction_hazard(step, rows)
        self.assertEqual(manager.num_eviction_cells, len(rows))

    def test_lru_without_model(self) -> None:
        self.assertIsNone(self.manager.eviction_clock)
        self.assertEqual(self.manager.num_eviction_cells, 0)
        a0, a1, b0, b1 = self._scenario(self.manager)
        self.assertLess(a1, a0, "LRU evicts the conversation released first")
        self.assertEqual(b1, b0)

    def test_low_hazard_cell_evicted_first(self) -> None:
        # Cell 1 (A) likely returns, cell 2 (B) does not: B goes although A was released earlier.
        self._install(self.manager, [[0.5], [HIGH], [LOW]])
        a0, a1, b0, b1 = self._scenario(
            self.manager, hint_a=(1, 10.0), hint_b=(2, 20.0), clock=30.0
        )
        self.assertEqual(a1, a0)
        self.assertLess(b1, b0)

    def test_equal_cells_fall_back_to_lru(self) -> None:
        self._install(self.manager, [[0.5], [HIGH]])
        a0, a1, b0, b1 = self._scenario(
            self.manager, hint_a=(1, 10.0), hint_b=(1, 20.0), clock=30.0
        )
        self.assertLess(a1, a0)
        self.assertEqual(b1, b0)

    def test_hazard_depends_on_idle_time(self) -> None:
        # Cell 1 is likely to return within 20 s of its release, unlikely afterwards; cell 2 is in between.
        rows = [[0.5], [HIGH, HIGH, LOW], [0.5, 0.5, 0.5]]
        self._install(self.manager, rows)
        # A idle 5 s (hazard 0.9) vs B idle 5 s (0.5): B goes.
        a0, a1, b0, b1 = self._scenario(
            self.manager, hint_a=(1, 100.0), hint_b=(2, 100.0), clock=105.0
        )
        self.assertEqual(a1, a0)
        self.assertLess(b1, b0)
        # A idle 40 s (0.1) vs B idle 5 s (0.5): A goes.
        a0, a1, b0, b1 = self._scenario(
            self.manager, hint_a=(1, 200.0), hint_b=(2, 235.0), clock=240.0
        )
        self.assertLess(a1, a0)
        self.assertEqual(b1, b0)

    def test_rows_are_made_monotone(self) -> None:
        # Cell 1's raw row rises after 10 s; the model keeps its running minimum (0.1), so A still goes.
        self._install(self.manager, [[0.5], [LOW, HIGH], [0.5]])
        a0, a1, b0, b1 = self._scenario(self.manager, hint_a=(1, 0.0), hint_b=(2, 25.0), clock=30.0)
        self.assertLess(a1, a0)
        self.assertEqual(b1, b0)

    def test_unhinted_pages_use_the_unclassified_cell(self) -> None:
        # A is released without a hint: it is stamped with the clock on release and ranked by row 0.
        self._install(self.manager, [[HIGH], [LOW]])
        self.manager.set_eviction_clock(10.0)
        a0, a1, b0, b1 = self._scenario(self.manager, hint_a=None, hint_b=(1, 20.0), clock=30.0)
        self.assertEqual(a1, a0)
        self.assertLess(b1, b0)

    def test_identical_streams_make_identical_choices(self) -> None:
        # What tensor-parallel ranks rely on: the same request stream and clock give the same evictions.
        rows = [[0.5], [HIGH, LOW], [0.4, 0.4]]
        other = self._new_manager()
        try:
            results = []
            for manager in (self.manager, other):
                self._install(manager, rows)
                self._tok = itertools.count(1)
                results.append(
                    self._scenario(manager, hint_a=(1, 0.0), hint_b=(2, 12.0), clock=15.0)
                )
            self.assertEqual(results[0], results[1])
        finally:
            other.shutdown()

    def test_api_contract(self) -> None:
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv = self.manager.create_kv_cache(None, [])
            kv.resume(stream)
            with self.assertRaises(Exception):
                kv.set_eviction_hint(0, 1.0)  # no model installed
            self._install(self.manager, [[0.5], [0.4]])
            with self.assertRaises(IndexError):
                kv.set_eviction_hint(2, 1.0)
            with self.assertRaises(ValueError):
                kv.set_eviction_hint(1, -1.0)
            kv.set_eviction_hint(1, 1.0)
            kv.close()
        s.take_finish_event().synchronize()
        for step, rows in ((0.0, [[0.5]]), (10.0, [[]]), (10.0, [[float("nan")]])):
            with self.assertRaises(ValueError):
                self.manager.set_eviction_hazard(step, rows)
        # The clock never goes back, and survives replacing the model.
        self.manager.set_eviction_clock(50.0)
        self.manager.set_eviction_clock(40.0)
        self.assertEqual(self.manager.eviction_clock, 50.0)
        self._install(self.manager, [[0.5]])
        self.assertEqual(self.manager.eviction_clock, 50.0)
        # An empty table removes the model.
        self.manager.set_eviction_hazard(10.0, [])
        self.assertIsNone(self.manager.eviction_clock)
        self.assertEqual(self.manager.num_eviction_cells, 0)


if __name__ == "__main__":
    unittest.main()
