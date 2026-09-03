# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT - full host cost of one tunable-op call, and what each option removes.

Reproduces the ``trtllm::tunable_allreduce`` call chain and times five variants
end to end, so the two candidate optimizations can be compared on the same
footing rather than argued about separately:

  L1  dispatcher + @torch.library.custom_op wrapper
  L2  Python body constructs the runner
  L3  AutoTuner.choose_one                  <- what PR #16902 bypasses
  L4  runner.forward
  L5  torch.ops.trtllm.allreduce (native)

Variants:

  V0  current main: custom_op + full choose_one
  V1  fast_custom_op + full choose_one          (the wrapper layer only)
  V2  custom_op + DispatchSlot                  (the lookup layer only)
  V3  both
  V4  fully native, lookup inside C++           (the PR #16902 shape)

The reference point is a kernel, not a percentage: a TRT-LLM kernel call runs
roughly 20-30us, so ``--kernel-us`` normalizes everything against that.

Usage::

    python tests/microbenchmarks/autotuner_native_dispatch/bench_host_breakdown.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.utils.cpp_extension as cpp_extension
from dispatch_slot import POW2, BucketRule, DispatchSlot
from torch.library import Library, infer_schema, register_fake

_HERE = Path(__file__).resolve().parent
HIDDEN = 8192
GROUP = list(range(8))


# ---------------------------------------------------------------------------
# A faithful copy of the AutoTuner structures on the inference hot path.
# ---------------------------------------------------------------------------
def _last_positive_power_of_2(x: int) -> int:
    return 1 << (x.bit_length() - 1) if x else 1


@dataclass(frozen=True, slots=True, unsafe_hash=True)
class DynSpec:
    input_idx: int
    dim_idx: int
    gen_tuning_buckets: Tuple
    map_to_tuning_buckets: Callable


@dataclass(frozen=True, slots=True, unsafe_hash=True)
class ConSpec:
    input_idx: int
    dim_idx: int


@dataclass(frozen=True, slots=True)
class TuningConfig:
    dynamic_tensor_specs: Tuple = ()
    constraint_specs: Tuple = ()
    tune_max_num_tokens: Optional[int] = None


@lru_cache(maxsize=None)  # main already memoizes this; keep it for fidelity
def _find_nearest_profile(shapes, dyn, con, tune_max=None, apply_map=True):
    profile = [list(shape) for shape in shapes]
    for spec in dyn:
        if apply_map:
            profile[spec.input_idx][spec.dim_idx] = spec.map_to_tuning_buckets(
                profile[spec.input_idx][spec.dim_idx]
            )
        if tune_max is not None:
            profile[spec.input_idx][spec.dim_idx] = min(
                profile[spec.input_idx][spec.dim_idx], tune_max
            )
    for spec in con:
        if profile[spec.input_idx] == [0]:
            continue
        profile[spec.input_idx][spec.dim_idx] = -1
    return tuple(tuple(shape) for shape in profile)


class AllReduceRunner:
    def __init__(self, tp_size, group, dtype, op, eps, trigger, window, handle, ext_op):
        self.tp_size, self.group, self.input_dtype = tp_size, group, dtype
        self.op, self.eps, self.trigger, self.window = op, eps, trigger, window
        self.handle, self.ext_op = handle, ext_op

    def unique_id(self):
        return (self.tp_size, tuple(self.group), self.input_dtype, self.op, self.window)

    def forward(self, inputs, tactic):
        inp, residual, norm_weight, scale, bias, workspace = inputs
        return self.ext_op(
            inp,
            residual,
            norm_weight,
            scale,
            bias,
            workspace,
            self.group,
            self.handle,
            self.op,
            self.eps,
            self.trigger,
        )


CUSTOM_OP = "trtllm::tunable_allreduce::allreduce"
TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynSpec(0, 0, tuple(1 << i for i in range(14)), _last_positive_power_of_2),
    ),
    constraint_specs=(ConSpec(1, 0),),
)
PROFILING_CACHE: dict = {}


def _get_input_sizes(inputs):
    return [t.size() if isinstance(t, torch.Tensor) else torch.Size((0,)) for t in inputs]


def _get_cache_key(runner, shapes):
    return (
        CUSTOM_OP,
        runner.__class__.__name__,
        str(runner.unique_id()),
        _find_nearest_profile(
            shapes,
            TUNING_CONFIG.dynamic_tensor_specs,
            TUNING_CONFIG.constraint_specs,
            TUNING_CONFIG.tune_max_num_tokens,
            True,
        ),
    )


def _choose_one(runners, inputs):
    shapes = tuple(_get_input_sizes(inputs))
    for idx, runner in enumerate(runners):
        key = _get_cache_key(runner, shapes)
        if key in PROFILING_CACHE:
            _, tactic, _ = PROFILING_CACHE[key]
            return runners[idx], tactic
    return runners[0], -1


def _bench(fn: Callable, iters: int) -> float:
    for _ in range(max(1000, iters // 10)):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1e6


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-us", type=float, default=25.0)
    parser.add_argument("--iters", type=int, default=40000)
    parser.add_argument("--tokens", type=int, default=96)
    args = parser.parse_args(argv)

    ext = cpp_extension.load(
        name="tactic_table_bench_ext",
        sources=[str(_HERE / "bench_ext.cpp")],
        extra_include_paths=[str(_HERE)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    num_buckets = 14
    handle = ext.register_pow2_table(1, 8192, 1, num_buckets - 1, [7] * num_buckets, 1, [-1])
    native_op = torch.ops.tactic_table_bench.fused_native_op

    runner = AllReduceRunner(8, GROUP, torch.bfloat16, 4, 1e-5, False, False, handle, native_op)

    def body_with_choose_one(
        inp, residual, norm_weight, scale, bias, workspace, group, strategy, op, eps, trigger
    ):
        local = AllReduceRunner(
            len(group), group, inp.dtype, op, eps, trigger, False, handle, native_op
        )
        _, tactic = _choose_one([local], [inp, residual, norm_weight, scale, bias, workspace])
        return local.forward([inp, residual, norm_weight, scale, bias, workspace], tactic)

    slot = DispatchSlot(CUSTOM_OP, BucketRule(kind=POW2, lo=1, hi=8192))
    for i in range(num_buckets):
        slot.table[i] = 7
    slot.lookup = slot._compile()

    def body_with_slot(
        inp, residual, norm_weight, scale, bias, workspace, group, strategy, op, eps, trigger
    ):
        slot.lookup([inp])
        return native_op(
            inp, residual, norm_weight, scale, bias, workspace, group, handle, op, eps, trigger
        )

    def register(ns: str, body: Callable, fast: bool):
        def impl(
            input: torch.Tensor,
            residual: Optional[torch.Tensor],
            norm_weight: Optional[torch.Tensor],
            scale: Optional[torch.Tensor],
            bias: Optional[torch.Tensor],
            workspace: Optional[torch.Tensor],
            group: List[int],
            strategy: int,
            op: int,
            eps: float,
            trigger_completion_at_end: bool,
        ) -> List[torch.Tensor]:
            return body(
                input,
                residual,
                norm_weight,
                scale,
                bias,
                workspace,
                group,
                strategy,
                op,
                eps,
                trigger_completion_at_end,
            )

        if fast:
            lib = Library(ns, "FRAGMENT")
            lib.define(infer_schema(impl, op_name="ar", mutates_args=()))
            lib.impl("ar", impl, "CompositeExplicitAutograd")
            register_fake(f"{ns}::ar", lambda i, *rest: [i.new_empty(0)])
            return lib
        torch.library.custom_op(f"{ns}::ar", mutates_args=())(impl)
        register_fake(f"{ns}::ar", lambda i, *rest: [i.new_empty(0)])
        return None

    keep = [
        register("v0", body_with_choose_one, fast=False),
        register("v1", body_with_choose_one, fast=True),
        register("v2", body_with_slot, fast=False),
        register("v3", body_with_slot, fast=True),
    ]
    assert keep is not None

    x = torch.empty(args.tokens, HIDDEN, dtype=torch.bfloat16)
    residual = torch.empty(args.tokens, HIDDEN, dtype=torch.bfloat16)
    norm_weight = torch.empty(HIDDEN, dtype=torch.bfloat16)
    workspace = torch.empty(16, dtype=torch.int64)
    inputs = [x, residual, norm_weight, None, None, workspace]
    PROFILING_CACHE[_get_cache_key(runner, tuple(_get_input_sizes(inputs)))] = (0, 7, 1.0)

    call = (x, residual, norm_weight, None, None, workspace, GROUP, 0, 4, 1e-5, False)
    results = {
        "V0 current (custom_op + choose_one)": _bench(lambda: torch.ops.v0.ar(*call), args.iters),
        "V1 fast_custom_op + choose_one": _bench(lambda: torch.ops.v1.ar(*call), args.iters),
        "V2 custom_op + DispatchSlot": _bench(lambda: torch.ops.v2.ar(*call), args.iters),
        "V3 fast_custom_op + DispatchSlot": _bench(lambda: torch.ops.v3.ar(*call), args.iters),
        "V4 fully native (#16902 shape)": _bench(
            lambda: native_op(
                x, residual, norm_weight, None, None, workspace, GROUP, handle, 4, 1e-5, False
            ),
            args.iters,
        ),
    }

    baseline = results["V0 current (custom_op + choose_one)"]
    print(f"reference kernel = {args.kernel_us:.0f}us, {args.tokens} tokens, {args.iters} iters\n")
    print(f"{'variant':44s} {'us':>8} {'% kernel':>9} {'saved':>8}")
    print("-" * 73)
    for name, value in results.items():
        print(
            f"{name:44s} {value:8.2f} {value / args.kernel_us * 100:8.1f}% {baseline - value:7.2f}"
        )

    print("\nsub-costs inside choose_one")
    shapes = tuple(_get_input_sizes(inputs))
    for name, fn in [
        ("choose_one total", lambda: _choose_one([runner], inputs)),
        ("  _get_input_sizes", lambda: tuple(_get_input_sizes(inputs))),
        ("  get_cache_key", lambda: _get_cache_key(runner, shapes)),
        ("    str(unique_id())", lambda: str(runner.unique_id())),
        ("DispatchSlot lookup", lambda: slot.lookup(inputs)),
    ]:
        print(f"  {name:42s} {_bench(fn, args.iters):8.2f}")


if __name__ == "__main__":
    main()
