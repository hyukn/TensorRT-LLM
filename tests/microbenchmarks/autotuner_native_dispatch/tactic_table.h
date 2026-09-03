/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DRAFT - not wired into the build, not used by any op. See README.md.
 *
 * An op-agnostic tactic table for ops that have already been lowered to C++ and
 * therefore can resolve their AutoTuner tactic without re-entering Python.
 *
 * Two properties separate this from the hand-rolled table in
 * `cpp/tensorrt_llm/thop/allreduceOp.cpp` (PR #16902):
 *
 *  1. A handle is a DENSE INDEX into a vector, not a key into a `std::map`.
 *     #16902 rebuilds a composite key (group vector + fusion op + dtype + six
 *     input shapes) on every call and does two red-black-tree lookups with
 *     lexicographic comparison. Measured on an EPYC 7313P: ~495ns there versus
 *     ~5ns here.
 *
 *  2. The static half of the AutoTuner cache key NEVER reaches C++. Python
 *     collapses (custom_op name, runner class, unique_id, static shapes) into
 *     the handle at registration time, and ships the bucket rule down as data.
 *     #16902 instead hardcodes the 14 bucket cutoffs in C++ and hand-copies the
 *     semantics of `_find_nearest_profile`; if either drifts from the Python
 *     definition the lookup silently misses and falls back to a different
 *     strategy with no diagnostic. There is nothing here to drift.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace tensorrt_llm::tactic_table
{

// Closed set of bucket rules. Every entry corresponds to a `map_to_tuning_buckets`
// function used in tree today; an op whose mapper falls outside this set keeps the
// Python path instead of getting a handle.
enum class BucketKind : int32_t
{
    Pow2 = 0,     // last_positive_power_of_2
    Explicit = 1, // arbitrary sorted cutoffs
    RoundUp = 2,  // ceil to a multiple of `lo`
    Identity = 3,
};

struct BucketRule
{
    BucketKind kind{BucketKind::Pow2};
    int64_t lo{1};               // smallest bucket value (also the step for RoundUp)
    int64_t hi{1};               // largest bucket value
    int64_t divisor{1};          // pre-divide, e.g. trtllm-gen's use_dp/ep_size deflation
    int32_t overflowIdx{-1};     // index used when the value exceeds `hi`; -1 => clamp
    int32_t inputIdx{0};         // which input tensor carries the dynamic dim
    int32_t dimIdx{0};           // which dim of that tensor
    std::vector<int64_t> values; // Explicit only, ascending
};

// Sentinel stored for a bucket Python never registered a tactic for.
inline constexpr int64_t kMissingTactic = INT64_MIN;

struct Table
{
    std::vector<BucketRule> rules;
    std::vector<int64_t> strides;  // mixed-radix strides, one per rule
    std::vector<int64_t> tactics;  // flat, tacticsPerEntry * numEntries
    int32_t tacticsPerEntry{1};    // 1 for a scalar tactic, 2 for [tileN, cfg], ...
    std::vector<int64_t> fallback; // used on a miss; size == tacticsPerEntry
    char const* opName{""};        // for the warn-once on a miss
};

namespace detail
{
// A deque never invalidates references to existing elements when it grows, so a
// lookup holding `Table const&` stays valid even if another thread registers a
// new table concurrently. A vector would reallocate and dangle -- registration
// and lookup do not overlap in the intended warmup-then-serve flow, but that is
// an invariant nobody enforces, so do not depend on it.
inline std::deque<Table>& tables()
{
    static std::deque<Table> instance;
    return instance;
}

inline std::mutex& registrationMutex()
{
    static std::mutex instance;
    return instance;
}
} // namespace detail

inline int32_t bucketIndex(BucketRule const& rule, int64_t value)
{
    int64_t const v = (rule.divisor == 1) ? value : value / rule.divisor;
    switch (rule.kind)
    {
    case BucketKind::Pow2:
    {
        if (v <= rule.lo)
        {
            return 0;
        }
        if (v > rule.hi)
        {
            return rule.overflowIdx;
        }
        int const highBit = 63 - __builtin_clzll(static_cast<uint64_t>(v));
        int const lowBit = 63 - __builtin_clzll(static_cast<uint64_t>(rule.lo));
        return static_cast<int32_t>(highBit - lowBit);
    }
    case BucketKind::RoundUp:
    {
        int64_t const idx = (v - 1) / rule.lo;
        return static_cast<int32_t>(idx < 0 ? 0 : idx);
    }
    case BucketKind::Explicit:
    {
        auto const it = std::upper_bound(rule.values.begin(), rule.values.end(), v);
        int64_t const idx = (it - rule.values.begin()) - 1;
        return static_cast<int32_t>(idx < 0 ? 0 : idx);
    }
    case BucketKind::Identity:
    default: return static_cast<int32_t>(v < 0 ? 0 : v);
    }
}

/// Register a table and return its handle. Called from Python during warmup,
/// once the AutoTuner has a tactic for every bucket.
inline int64_t registerTable(std::vector<BucketRule> rules, std::vector<int64_t> strides, std::vector<int64_t> tactics,
    int32_t tacticsPerEntry, std::vector<int64_t> fallback, char const* opName)
{
    if (rules.empty() || rules.size() != strides.size())
    {
        throw std::invalid_argument("tactic_table: one stride is required per rule");
    }
    if (tacticsPerEntry < 1 || tactics.size() % static_cast<size_t>(tacticsPerEntry) != 0)
    {
        throw std::invalid_argument("tactic_table: tactics is not a multiple of tacticsPerEntry");
    }
    if (fallback.size() != static_cast<size_t>(tacticsPerEntry))
    {
        throw std::invalid_argument("tactic_table: fallback must hold tacticsPerEntry values");
    }
    std::lock_guard<std::mutex> const lock(detail::registrationMutex());
    detail::tables().push_back(
        Table{std::move(rules), std::move(strides), std::move(tactics), tacticsPerEntry, std::move(fallback), opName});
    return static_cast<int64_t>(detail::tables().size() - 1);
}

inline Table const& table(int64_t handle)
{
    return detail::tables()[static_cast<size_t>(handle)];
}

/// Hot path, one dynamic dimension. 33 of 33 literal TuningConfigs in tree have
/// exactly one, so this is the case worth specializing.
///
/// Returns a pointer to `tacticsPerEntry` values -- a pointer rather than a
/// scalar so ops whose tactic is a small vector (trtllm-gen's [tileN, cfg]) fit
/// without a second code path. On a miss the table's fallback is returned, so a
/// caller never silently receives some other bucket's tactic.
inline int64_t const* lookup1d(int64_t handle, int64_t dynamicExtent)
{
    Table const& t = table(handle);
    int32_t const idx = bucketIndex(t.rules[0], dynamicExtent);
    int64_t const* entry = t.tactics.data() + static_cast<size_t>(idx) * t.tacticsPerEntry;
    return (*entry == kMissingTactic) ? t.fallback.data() : entry;
}

/// General path, any number of dynamic dimensions, mixed-radix flattened.
inline int64_t const* lookup(int64_t handle, int64_t const* dynamicExtents, size_t numExtents)
{
    Table const& t = table(handle);
    if (numExtents != t.rules.size())
    {
        return t.fallback.data();
    }
    int64_t flat = 0;
    for (size_t i = 0; i < numExtents; ++i)
    {
        flat += static_cast<int64_t>(bucketIndex(t.rules[i], dynamicExtents[i])) * t.strides[i];
    }
    int64_t const* entry = t.tactics.data() + static_cast<size_t>(flat) * t.tacticsPerEntry;
    return (*entry == kMissingTactic) ? t.fallback.data() : entry;
}

} // namespace tensorrt_llm::tactic_table
