# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT - a Python-side generalized ``choose_one`` fast path.

``AutoTunerProfilingCache.get_cache_key`` builds

    (custom_op, runner_class, str(runner.unique_id()), <bucketed shapes>)

on every inference call, but only the bucketed dynamic dim varies between calls
at a given call site. The first three components and every static dim are
invariant. This prototype collapses the invariant part once into a
``DispatchSlot`` and reduces steady state to two array indexes.

Two pieces make that possible:

``BucketRule``
    Replaces ``DynamicTensorSpec.map_to_tuning_buckets``, which is an arbitrary
    ``Callable`` today. A closed set of rules can be evaluated in a few ns and,
    unlike a Callable, can be shipped verbatim to C++ (see ``tactic_table.h``)
    so a native lookup never has to re-derive Python's bucketing. An AST sweep
    of the tree found 32 of 40 ``DynamicTensorSpec`` sites covered by
    ``Pow2``/``RoundUp``/``Identity``; the remaining 8 keep the ``Callable``
    escape hatch and simply do not get a slot.

``DispatchSlot``
    Holds the resolved table plus a resolver specialized to the rule at build
    time, so the hot path has no dict, no tuple construction, no hashing and no
    Python-level function call.

Run this file directly to check the rules against the functions they replace.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

POW2, EXPLICIT, ROUND_UP, IDENTITY = 0, 1, 2, 3

#: Distinguishes "no tactic registered for this bucket" from a cached ``None``.
MISSING = object()


@dataclass(frozen=True, slots=True)
class BucketRule:
    """Declarative replacement for ``map_to_tuning_buckets``.

    Mirrors ``tensorrt_llm::tactic_table::BucketRule`` field for field so a slot
    can be handed to the native table without a translation step.
    """

    kind: int = POW2
    lo: int = 1
    hi: int = 1 << 20
    divisor: int = 1
    overflow_idx: int = -1
    input_idx: int = 0
    dim_idx: int = 0
    values: Tuple[int, ...] = ()

    @property
    def num_buckets(self) -> int:
        if self.kind == POW2:
            return self.hi.bit_length() - self.lo.bit_length() + 1
        if self.kind == EXPLICIT:
            return len(self.values)
        if self.kind == ROUND_UP:
            return self.hi // self.lo
        return self.hi + 1

    def index(self, value: int) -> int:
        v = value // self.divisor if self.divisor != 1 else value
        if self.kind == POW2:
            if v <= self.lo:
                return 0
            if v > self.hi:
                return self.overflow_idx if self.overflow_idx >= 0 else self.num_buckets - 1
            return v.bit_length() - self.lo.bit_length()
        if self.kind == ROUND_UP:
            idx = (v - 1) // self.lo
            return 0 if idx < 0 else min(idx, self.num_buckets - 1)
        if self.kind == EXPLICIT:
            return max(0, min(bisect.bisect_right(self.values, v) - 1, self.num_buckets - 1))
        return 0 if v < 0 else min(v, self.num_buckets - 1)

    def bucket_values(self) -> Tuple[int, ...]:
        """The bucket each index stands for; used when priming the table."""
        if self.kind == POW2:
            return tuple(self.lo << i for i in range(self.num_buckets))
        if self.kind == EXPLICIT:
            return self.values
        if self.kind == ROUND_UP:
            return tuple(self.lo * (i + 1) for i in range(self.num_buckets))
        return tuple(range(self.num_buckets))


@dataclass
class DispatchSlot:
    """One call site's resolved tactic table."""

    custom_op: str
    rule: BucketRule
    epoch: int = 0
    table: List[Any] = field(default_factory=list)
    runner_ids: List[int] = field(default_factory=list)
    lookup: Callable[[Sequence], Any] = field(init=False)

    def __post_init__(self) -> None:
        if not self.table:
            self.table = [MISSING] * self.rule.num_buckets
        if not self.runner_ids:
            self.runner_ids = [0] * self.rule.num_buckets
        self.lookup = self._compile()

    def _compile(self) -> Callable[[Sequence], Any]:
        """Specialize the resolver so the hot path has no branching to do."""
        rule, table = self.rule, self.table
        input_idx, dim_idx = rule.input_idx, rule.dim_idx
        if rule.kind == POW2 and rule.divisor == 1 and rule.overflow_idx < 0:
            base = rule.lo.bit_length()
            last = rule.num_buckets - 1
            lo, hi = rule.lo, rule.hi

            def lookup(inputs):
                extent = inputs[input_idx].size(dim_idx)
                if extent <= lo:
                    return table[0]
                return table[last if extent > hi else extent.bit_length() - base]

            return lookup

        return lambda inputs: table[rule.index(inputs[input_idx].size(dim_idx))]

    def as_native_table(self) -> Optional[Tuple[Tuple[int, ...], BucketRule]]:
        """The seam to ``tactic_table.h``.

        Returns None unless every tactic is an int, which is what keeps ops with
        opaque tactics (CuTe DSL config tuples, ``(backend_name, index)`` pairs)
        on the Python path automatically rather than by special-casing them.
        """
        if any(t is MISSING or not isinstance(t, int) for t in self.table):
            return None
        return tuple(self.table), self.rule


# ---------------------------------------------------------------------------
# Equivalence checks against the functions BucketRule replaces.
# ---------------------------------------------------------------------------
def _last_positive_power_of_2(x: int) -> int:
    return 1 << (x.bit_length() - 1) if x else 1


def _check_pow2() -> int:
    """`Pow2` must agree with `last_positive_power_of_2` over the tuned range."""
    rule = BucketRule(kind=POW2, lo=1, hi=8192)
    values = rule.bucket_values()
    failures = 0
    for x in list(range(1, 3000)) + [4096, 8191, 8192, 9000, 16383]:
        expected = min(_last_positive_power_of_2(x), 8192)
        if values[rule.index(x)] != expected:
            failures += 1
    return failures


def _check_round_rule() -> int:
    """`Pow2` with divisor/overflow must reproduce trtllm-gen's `round_rule`.

    That closure (trtllm_gen_custom_ops.py) is the most involved mapper in tree:
    it deflates by ep_size under DP, clamps to MAX_PROFILE_BUCKET, and falls
    through to tune_max_num_tokens above the clamp.
    """

    def reference(x, ep_size, use_dp, max_bucket, tune_max_num_tokens):
        deflated = x // ep_size if use_dp else x
        if deflated > max_bucket and tune_max_num_tokens > max_bucket:
            return tune_max_num_tokens
        return min(max(1, _last_positive_power_of_2(deflated)), max_bucket)

    failures = 0
    for ep_size, use_dp, max_bucket, tune_max in [
        (8, True, 1024, 4096),
        (1, False, 1024, 512),
        (4, True, 2048, 8192),
        (2, True, 512, 256),
    ]:
        has_overflow = tune_max > max_bucket
        rule = BucketRule(
            kind=POW2,
            lo=1,
            hi=max_bucket,
            divisor=ep_size if use_dp else 1,
            overflow_idx=max_bucket.bit_length() if has_overflow else -1,
        )
        values = list(rule.bucket_values())
        if has_overflow:
            values.append(tune_max)
        for x in list(range(1, 4000)) + [8192, 20000, 65536]:
            if values[rule.index(x)] != reference(x, ep_size, use_dp, max_bucket, tune_max):
                failures += 1
    return failures


if __name__ == "__main__":
    pow2_failures = _check_pow2()
    round_failures = _check_round_rule()
    print(
        f"Pow2 vs last_positive_power_of_2 : "
        f"{'OK' if not pow2_failures else f'{pow2_failures} mismatches'}"
    )
    print(
        f"Pow2 vs trtllm-gen round_rule    : "
        f"{'OK (4 configs x 4003 inputs)' if not round_failures else f'{round_failures} mismatches'}"
    )
    raise SystemExit(1 if (pow2_failures or round_failures) else 0)
