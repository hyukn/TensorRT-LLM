/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DRAFT - benchmark harness for tactic_table.h. Built on demand by the bench
 * scripts via torch.utils.cpp_extension.load; not part of the CMake build.
 *
 * Exposes the lookup three ways so the cost of each Python->C++ boundary can be
 * compared against doing the same lookup in Python, and provides a replica of
 * the PR #16902 std::map shape to compare the two C++ designs directly.
 */
#include "tactic_table.h"

#include <torch/extension.h>

#include <chrono>
#include <map>
#include <tuple>

namespace tt = tensorrt_llm::tactic_table;

namespace
{

// ---------------------------------------------------------------------------
// Registration helper: flatten a rule into ints so Python ships plain data.
// ---------------------------------------------------------------------------
int64_t registerPow2Table(int64_t lo, int64_t hi, int64_t divisor, int64_t overflowIdx, std::vector<int64_t> tactics,
    int64_t tacticsPerEntry, std::vector<int64_t> fallback)
{
    tt::BucketRule rule;
    rule.kind = tt::BucketKind::Pow2;
    rule.lo = lo;
    rule.hi = hi;
    rule.divisor = divisor;
    rule.overflowIdx = static_cast<int32_t>(overflowIdx);
    return tt::registerTable(
        {rule}, {1}, std::move(tactics), static_cast<int32_t>(tacticsPerEntry), std::move(fallback), "bench::pow2");
}

// ---------------------------------------------------------------------------
// (A) exposed as a torch op -> full PyTorch dispatcher
// ---------------------------------------------------------------------------
int64_t lookupOp(int64_t handle, at::Tensor const& x, int64_t dim)
{
    return tt::lookup1d(handle, x.size(dim))[0];
}

// ---------------------------------------------------------------------------
// (B) raw pybind11 -> no dispatcher, but opaque to Dynamo (graph break)
// ---------------------------------------------------------------------------
int64_t lookupRaw(int64_t handle, int64_t extent)
{
    return tt::lookup1d(handle, extent)[0];
}

// ---------------------------------------------------------------------------
// (C) a native op doing the lookup internally -- the #16902 shape. 11 args to
//     match autotuned_allreduce so the marshaling cost is representative.
// ---------------------------------------------------------------------------
std::vector<at::Tensor> fusedNativeOp(at::Tensor const& input, std::optional<at::Tensor> const& residual,
    std::optional<at::Tensor> const& normWeight, std::optional<at::Tensor> const& scale,
    std::optional<at::Tensor> const& bias, std::optional<at::Tensor> const& workspace,
    std::vector<int64_t> const& group, int64_t handle, int64_t op, double eps, bool triggerCompletionAtEnd)
{
    int64_t const tactic = tt::lookup1d(handle, input.size(0))[0];
    // The real op would branch on `tactic` here; keep it trivial so only host
    // cost is measured.
    (void) tactic;
    (void) residual;
    (void) normWeight;
    (void) scale;
    (void) bias;
    (void) workspace;
    (void) group;
    (void) op;
    (void) eps;
    (void) triggerCompletionAtEnd;
    return {input.new_empty({0})};
}

// ---------------------------------------------------------------------------
// (D) in-C++ timing loops: the lookup cost with no boundary crossing at all
// ---------------------------------------------------------------------------
double benchLookupInCpp(int64_t handle, int64_t extent, int64_t iters)
{
    auto const start = std::chrono::steady_clock::now();
    int64_t sink = 0;
    for (int64_t i = 0; i < iters; ++i)
    {
        sink += tt::lookup1d(handle, extent + (i & 1))[0];
    }
    auto const stop = std::chrono::steady_clock::now();
    if (sink == INT64_MIN)
    {
        throw std::runtime_error("unreachable; keeps the loop alive");
    }
    return std::chrono::duration<double, std::micro>(stop - start).count() / iters;
}

// ---------------------------------------------------------------------------
// (E) replica of the PR #16902 / autotuned_allreduce shape, for comparison:
//     a std::map keyed on a composite rebuilt on every call, looked up twice
//     (NCCL-window and non-window).
// ---------------------------------------------------------------------------
struct LegacyKey
{
    std::vector<int64_t> group;
    int64_t fusionOp;
    bool window;
    int32_t dtype;
    std::vector<std::vector<int64_t>> shapes;

    bool operator<(LegacyKey const& other) const
    {
        return std::tie(group, fusionOp, window, dtype, shapes)
            < std::tie(other.group, other.fusionOp, other.window, other.dtype, other.shapes);
    }
};

std::map<LegacyKey, std::vector<int64_t>>& legacyCache()
{
    static std::map<LegacyKey, std::vector<int64_t>> instance;
    return instance;
}

std::vector<int64_t>& legacyBuckets()
{
    static std::vector<int64_t> instance;
    return instance;
}

LegacyKey makeLegacyKey(bool window)
{
    LegacyKey key;
    key.group = {0, 1, 2, 3, 4, 5, 6, 7};
    key.fusionOp = 4;
    key.window = window;
    key.dtype = 15;
    key.shapes = {{8192}, {-1, 8192}, {8192}, {0}, {0}, {16}};
    return key;
}

void setupLegacy(std::vector<int64_t> buckets)
{
    legacyBuckets() = buckets;
    legacyCache()[makeLegacyKey(false)] = std::vector<int64_t>(buckets.size(), 7);
    legacyCache()[makeLegacyKey(true)] = std::vector<int64_t>(buckets.size(), 7);
    // Give the tree the depth a real model's cache would have.
    for (int i = 0; i < 600; ++i)
    {
        LegacyKey other = makeLegacyKey(true);
        other.fusionOp = 100 + i;
        legacyCache()[other] = std::vector<int64_t>(buckets.size(), 1);
    }
}

double benchLegacyMapInCpp(int64_t extent, int64_t iters)
{
    auto const start = std::chrono::steady_clock::now();
    int64_t sink = 0;
    for (int64_t i = 0; i < iters; ++i)
    {
        // Rebuild the key every call, exactly as allreduceOp.cpp does.
        LegacyKey key = makeLegacyKey(false);
        auto const nonWindow = legacyCache().find(key);
        int64_t const idx = (std::upper_bound(legacyBuckets().begin(), legacyBuckets().end(), extent + (i & 1))
                                - legacyBuckets().begin())
            - 1;
        if (nonWindow != legacyCache().end())
        {
            sink += nonWindow->second[idx];
        }
        key.window = true;
        auto const windowed = legacyCache().find(key);
        if (windowed != legacyCache().end())
        {
            sink += windowed->second[idx];
        }
    }
    auto const stop = std::chrono::steady_clock::now();
    if (sink == INT64_MIN)
    {
        throw std::runtime_error("unreachable; keeps the loop alive");
    }
    return std::chrono::duration<double, std::micro>(stop - start).count() / iters;
}

} // namespace

TORCH_LIBRARY(tactic_table_bench, m)
{
    m.def("lookup_op(int handle, Tensor x, int dim) -> int");
    m.def(
        "fused_native_op(Tensor input, Tensor? residual, Tensor? norm_weight, Tensor? scale, "
        "Tensor? bias, Tensor? workspace, int[] group, int handle, int op, float eps, "
        "bool trigger_completion_at_end) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tactic_table_bench, CompositeExplicitAutograd, m)
{
    m.impl("lookup_op", &lookupOp);
    m.impl("fused_native_op", &fusedNativeOp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("register_pow2_table", &registerPow2Table);
    m.def("lookup_raw", &lookupRaw);
    m.def("bench_lookup_in_cpp", &benchLookupInCpp);
    m.def("bench_legacy_map_in_cpp", &benchLegacyMapInCpp);
    m.def("setup_legacy", &setupLegacy);
    m.attr("MISSING_TACTIC") = py::int_(tt::kMissingTactic);
}
