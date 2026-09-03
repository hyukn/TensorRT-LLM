# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT - what a native tactic lookup costs, and whether it can be reached cheaply.

Answers two questions the design hinges on:

1. How much does the dense-handle table in ``tactic_table.h`` beat the
   ``std::map``-with-rebuilt-composite-key shape that PR #16902 uses? Both are
   timed inside C++ so no boundary cost is mixed in.

2. Can a Python-resident op call into that lookup and come out ahead? Three
   exposure forms are compared against simply doing the lookup in Python. This
   is the question that decides whether a native tactic table can be "applied to
   every op" or only rides along with ops that are already lowered to C++.

Usage::

    python tests/microbenchmarks/autotuner_native_dispatch/bench_native_dispatch.py
"""

import argparse
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.utils.cpp_extension as cpp_extension

_HERE = Path(__file__).resolve().parent


def _load_extension():
    return cpp_extension.load(
        name="tactic_table_bench_ext",
        sources=[str(_HERE / "bench_ext.cpp")],
        extra_include_paths=[str(_HERE)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _bench(fn: Callable, iters: int) -> float:
    """Microseconds per call."""
    for _ in range(max(1000, iters // 10)):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1e6


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument(
        "--tokens",
        type=int,
        default=96,
        help="Dynamic extent to look up (a non-power-of-2 by default).",
    )
    args = parser.parse_args(argv)

    ext = _load_extension()

    lo, hi = 1, 8192
    num_buckets = hi.bit_length() - lo.bit_length() + 1  # 14, matching allreduce
    handle = ext.register_pow2_table(lo, hi, 1, num_buckets - 1, [7] * num_buckets, 1, [-1])
    ext.setup_legacy([1 << i for i in range(num_buckets)])

    hidden = 8192
    x = torch.empty(args.tokens, hidden, dtype=torch.bfloat16)
    residual = torch.empty(args.tokens, hidden, dtype=torch.bfloat16)
    norm_weight = torch.empty(hidden, dtype=torch.bfloat16)
    workspace = torch.empty(16, dtype=torch.int64)
    group = list(range(8))

    print(f"{num_buckets} buckets, dynamic extent = {args.tokens}\n")

    print("=== lookup cost inside C++ (no boundary crossing) ===")
    dense = ext.bench_lookup_in_cpp(handle, args.tokens, 5_000_000) * 1000
    legacy = ext.bench_legacy_map_in_cpp(args.tokens, 500_000) * 1000
    print(f"  {'dense-handle table (this draft)':52s} {dense:8.1f} ns")
    print(f"  {'std::map + rebuilt key x2 (#16902 shape)':52s} {legacy:8.1f} ns")
    print(f"  {'speedup':52s} {legacy / dense:8.1f}x")

    print("\n=== reaching that lookup from Python ===")
    via_dispatcher = _bench(
        lambda: torch.ops.tactic_table_bench.lookup_op(handle, x, 0), args.iters
    )
    via_pybind = _bench(lambda: ext.lookup_raw(handle, x.size(0)), args.iters)
    table = [7] * num_buckets
    inputs = [x]
    in_python = _bench(lambda: table[inputs[0].size(0).bit_length() - 1], args.iters)
    size_only = _bench(lambda: x.size(0), args.iters)
    print(f"  {'torch.ops (full dispatcher)':52s} {via_dispatcher:8.3f} us")
    print(f"  {'raw pybind11 (+ Python-side size(), graph break)':52s} {via_pybind:8.3f} us")
    print(f"  {'same lookup done in Python':52s} {in_python:8.3f} us")
    print(f"  {'  (floor) x.size(0) alone':52s} {size_only:8.3f} us")

    print("\n=== lookup inside a native op (the shape that actually wins) ===")
    native = _bench(
        lambda: torch.ops.tactic_table_bench.fused_native_op(
            x, residual, norm_weight, None, None, workspace, group, handle, 4, 1e-5, False
        ),
        args.iters // 4,
    )
    print(f"  {'11-arg native op, lookup internal':52s} {native:8.3f} us")

    print(
        "\nTakeaway: the table is ~free once execution is already in C++, but "
        "every way of reaching it from Python costs more than just doing the "
        "lookup in Python. A native tactic table rides along with lowering an "
        "op to C++; it is not independently adoptable."
    )


if __name__ == "__main__":
    main()
