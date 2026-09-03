# AutoTuner native dispatch — draft and measurements

**Status: draft. Nothing here is wired into the build or used by any op.** It exists
to preserve a design and the measurements behind it, so the question does not have
to be re-litigated from scratch.

The question: **can PR #16902's approach — bypassing the Python AutoTuner lookup by
resolving the tactic in C++ — be generalized into something every tunable op adopts?**

Short answer: a generalized C++ table is worth building *and is ~80-100x faster than
the hand-rolled one already in tree*, but it cannot be adopted independently. It only pays
off for ops already lowered to C++, and the cost it removes is not where the host time
actually goes.

## Contents

| File | What it is |
|---|---|
| `tactic_table.h` | The generalized C++ tactic table. Header-only, no dependencies beyond the STL. |
| `bench_ext.cpp` | pybind/TORCH_LIBRARY harness: exposes the lookup three ways, plus a replica of #16902's `std::map` shape for a direct comparison. |
| `dispatch_slot.py` | Python-side generalized `choose_one` fast path: declarative `BucketRule` + `DispatchSlot`. Run directly to check the rules against the functions they replace. |
| `bench_native_dispatch.py` | Lookup cost in C++, and the cost of every way of reaching it from Python. |
| `bench_host_breakdown.py` | Full host cost of one tunable-op call across five variants. |

All measurements below: AMD EPYC 7313P, torch 2.10.0a0, CPU tensors (the cost being
measured is host-side), `-O3`. Ranges are the spread across repeat runs; absolute
values are pessimistic on this CPU, but the ratios are what the conclusions rest on.

## Finding 1 — the generalized table is ~80-100x faster than the in-tree one

```
dense-handle table (this draft)             5.0 - 6.2 ns
std::map + rebuilt key x2 (#16902 shape)        ~495 ns
                                             80x - 100x
```

(The dense figure moves run to run because it is only a few cache accesses; the
ratio is stable at roughly two orders of magnitude.)

`allreduceOp.cpp` rebuilds a composite key (group vector, fusion op, dtype, six input
shapes) on **every call** and does two red-black-tree lookups with lexicographic
comparison. This draft makes the handle a dense index into a `std::deque`, so the
lookup is a bounds-free index plus a `__builtin_clzll`.

So generalizing here does not trade performance for maintainability — it wins both.
The catch is what that 490ns is a fraction of (Finding 3).

## Finding 2 — a C++ lookup cannot be reached cheaply from Python

```
torch.ops (full dispatcher)                     1.445 us
raw pybind11 (+ Python-side size())             0.459 us   <- also a Dynamo graph break
same lookup done in Python                      0.272 us
  (floor) x.size(0) alone                       0.229 us
```

**Every way of calling into C++ costs more than doing the lookup in Python.** The
`x.size(0)` needed to pass the extent already costs 0.229us, which is 84% of the
whole Python lookup.

The consequence: a native tactic table is not an independently adoptable optimization.
It is free only when the caller is *already* in C++ — i.e. it rides along with lowering
an op, it does not motivate lowering one. PR #16902 wins because it made
`autotuned_allreduce` a native op, not because the lookup moved.

## Finding 3 — the lookup is not where the host time is

One `trtllm::tunable_allreduce` call, against a 25us reference kernel:

| Variant | us | % of kernel | saved |
|---|---|---|---|
| V0 current (`custom_op` + `choose_one`) | 33.4 - 36.0 | 134-144% | — |
| V1 `fast_custom_op` + `choose_one` | 16.4 - 16.8 | ~66% | 16.6 - 19.6 |
| V2 `custom_op` + `DispatchSlot` | 27.0 - 27.2 | ~108% | 6.3 - 9.0 |
| V3 both | 10.7 | ~43% | 22.7 - 25.3 |
| V4 fully native, lookup in C++ (#16902) | 3.8 | ~15% | 29.6 - 32.2 |

Inside `choose_one`:

```
choose_one total                                3.67 us
  _get_input_sizes                              1.41 us
  get_cache_key                                 1.60 us
    str(unique_id())                            0.99 us
DispatchSlot lookup (replacement)               0.32 us
```

**Host cost (33-36us) exceeds the kernel it launches (25us).** But the `@custom_op`
wrapper is ~27us of that and `choose_one` is only ~3.7us — the layer #16902 targets is
**~10% of host time**, while the decorator layer is ~77%.

This is why the plan that came out of this investigation put `fast_custom_op` adoption
first (landed as a separate change) and dropped the native table: V1 alone captures
most of what is available, for a one-line change per op.

## Finding 4 — the floor for any op that stays in Python

After V3 (both Python-side optimizations), 10.7us remains: roughly 7us for the
dispatcher marshaling arguments and entering Python, plus ~3.8us for the inner native
op. That only disappears if the function body itself becomes C++.

For the 28 CuTe DSL custom ops that is structurally impossible — their bodies are
`cute.compile(...)` JIT callables living in Python, with no C++ entry point to dispatch
to. **~11us/call, or ~45% of a kernel, is the hard floor for those ops.**

## What `BucketRule` buys, and why it is the load-bearing piece

`DynamicTensorSpec.map_to_tuning_buckets` is an arbitrary `Callable` today. That single
fact causes both problems the in-tree native caches have: C++ cannot evaluate it, so
each implementation hardcodes its own copy of the bucketing and of
`_find_nearest_profile`'s semantics — and when those drift from Python, the lookup
silently misses and falls back to a *different* tactic with no diagnostic.

Making the rule declarative (a closed set of kinds plus integer parameters) means
Python can ship it to C++ as data, and there is nothing left to drift. An AST sweep of
the tree found **32 of 40** `DynamicTensorSpec` sites covered by `Pow2`/`RoundUp`/
`Identity`; the other 8 keep the `Callable` escape hatch and simply never get a handle.

`dispatch_slot.py` checks the two non-trivial rules against the functions they replace:

```
Pow2 vs last_positive_power_of_2 : OK
Pow2 vs trtllm-gen round_rule    : OK (4 configs x 4003 inputs)
```

`round_rule` is the hardest case in tree — it deflates by `ep_size` under DP, clamps to
`MAX_PROFILE_BUCKET`, and falls through to `tune_max_num_tokens` above the clamp.

## Before any of this could ship

- **Not integrated.** Nothing calls `tactic_table.h`; no Python registration path wires
  `DispatchSlot.as_native_table()` to `registerTable`.
- **`lookup()` (multi-dim) is untested.** Only `lookup1d` is exercised; 33 of 33 literal
  `TuningConfig`s in tree have one dynamic dim, so the general path has no caller yet.
- **No warn-once on a miss.** The table returns its registered fallback rather than
  another bucket's tactic, which is the important half, but a silent fallback still
  deserves a one-time diagnostic.
- **Uses pybind + `torch/extension.h`.** In-tree this would be nanobind and the CMake
  build.
- Registration is assumed to happen at warmup and lookup at serve time. A `std::deque`
  is used so a concurrent registration cannot invalidate a reference a reader holds,
  but that ordering is a convention, not something enforced.

## Reproducing

```bash
cd tests/microbenchmarks/autotuner_native_dispatch
python3 dispatch_slot.py            # BucketRule equivalence checks
python3 bench_native_dispatch.py    # Findings 1 and 2
python3 bench_host_breakdown.py     # Findings 3 and 4
```

The C++ extension is built on demand via `torch.utils.cpp_extension.load`; no GPU and
no CMake build are required.
