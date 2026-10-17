/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/evictionController.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/page.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// LRUEvictionPolicy
// ---------------------------------------------------------------------------

NodeRef LRUEvictionPolicy::push(SharedPtr<Page> page, bool evictFirst)
{
    TLLM_CHECK_DEBUG_WITH_INFO(!page->nodeRef.has_value(), "page must not already be scheduled for eviction");
    if (evictFirst)
    {
        mQueue.push_front(std::move(page));
        return mQueue.begin();
    }
    else
    {
        mQueue.push_back(std::move(page));
        auto it = mQueue.end();
        --it;
        return it;
    }
}

SharedPtr<Page> LRUEvictionPolicy::pop()
{
    TLLM_CHECK_DEBUG(!mQueue.empty());
    auto page = mQueue.front();
    mQueue.pop_front();
    return page;
}

SharedPtr<Page> LRUEvictionPolicy::remove(NodeRef node)
{
    auto page = *node;
    TLLM_CHECK_DEBUG_WITH_INFO(
        page->nodeRef.has_value() && page->nodeRef.value() == node, "node's page must reference this node");
    mQueue.erase(node);
    return page;
}

// ---------------------------------------------------------------------------
// HazardEvictionModel
// ---------------------------------------------------------------------------

HazardEvictionModel::HazardEvictionModel(double stepSec, std::vector<std::vector<float>> table)
    : mStepSec(stepSec)
    , mTable(std::move(table))
{
    if (!(mStepSec > 0.0) || !std::isfinite(mStepSec))
    {
        throw std::invalid_argument("HazardEvictionModel: stepSec must be positive and finite");
    }
    if (mTable.empty())
    {
        throw std::invalid_argument("HazardEvictionModel: the table needs at least the unclassified cell (row 0)");
    }
    for (auto& row : mTable)
    {
        if (row.empty())
        {
            throw std::invalid_argument("HazardEvictionModel: every cell needs at least one idle bucket");
        }
        // Running minimum: the list-head comparison in PrioritizedEvictionPolicy is exact only for rows that do
        // not increase with idle time.
        float runMin = std::numeric_limits<float>::infinity();
        for (auto& v : row)
        {
            if (!std::isfinite(v))
            {
                throw std::invalid_argument("HazardEvictionModel: table values must be finite");
            }
            runMin = std::min(runMin, v);
            v = runMin;
        }
    }
}

float HazardEvictionModel::hazard(EvictionCell cell, double idleSec) const
{
    auto const& row = mTable.at(static_cast<size_t>(cell));
    double const buckets = std::max(idleSec, 0.0) / mStepSec;
    size_t const last = row.size() - 1;
    size_t const b = buckets >= static_cast<double>(last) ? last : static_cast<size_t>(buckets);
    return row[b];
}

void HazardEvictionModel::advanceClock(double now) noexcept
{
    if (now > mNow)
    {
        mNow = now;
    }
}

// ---------------------------------------------------------------------------
// PrioritizedEvictionPolicy
// ---------------------------------------------------------------------------

PrioritizedEvictionPolicy::Key PrioritizedEvictionPolicy::keyOf(Page const& page) noexcept
{
    return {page.priority, page.evictionCell};
}

NodeRef PrioritizedEvictionPolicy::push(SharedPtr<Page> page, bool evictFirst)
{
    if (mModel != nullptr && page->evictionStamp < 0.0)
    {
        // Released without a hint (no conversation, or before the model existed): it ranks as released now.
        page->evictionStamp = mModel->now();
    }
    LRUEvictionPolicy& policy = mPolicies[keyOf(*page)]; // std::map default-constructs if not present
    return policy.push(std::move(page), evictFirst);
}

std::map<PrioritizedEvictionPolicy::Key, LRUEvictionPolicy>::iterator PrioritizedEvictionPolicy::selectVictimQueue()
{
    // Lowest priority key evicted first (std::map iterates in ascending key order)
    auto best = mPolicies.begin();
    if (mModel == nullptr)
    {
        return best;
    }
    // Among the cells of the lowest priority, the head with the lowest hazard. Within a cell the head is the
    // least recently released page, whose hazard is the cell's lowest (rows are non-increasing in idle time).
    Priority const lowest = best->first.first;
    double const now = mModel->now();
    auto headOf = [](auto const& it) -> Page const& { return **it->second.cbegin(); };
    float bestHazard = mModel->hazard(best->first.second, now - headOf(best).evictionStamp);
    double bestStamp = headOf(best).evictionStamp;
    for (auto it = std::next(best); it != mPolicies.end() && it->first.first == lowest; ++it)
    {
        double const stamp = headOf(it).evictionStamp;
        float const h = mModel->hazard(it->first.second, now - stamp);
        if (h < bestHazard || (h == bestHazard && stamp < bestStamp))
        {
            best = it;
            bestHazard = h;
            bestStamp = stamp;
        }
    }
    return best;
}

SharedPtr<Page> PrioritizedEvictionPolicy::pop()
{
    TLLM_CHECK_DEBUG(!mPolicies.empty());
    auto it = selectVictimQueue();
    auto page = it->second.pop();
    if (it->second.empty())
    {
        mPolicies.erase(it);
    }
    return page;
}

SharedPtr<Page> PrioritizedEvictionPolicy::remove(NodeRef node)
{
    auto page = *node;
    auto it = mPolicies.find(keyOf(*page));
    TLLM_CHECK_DEBUG(it != mPolicies.end());
    it->second.remove(node);
    if (it->second.empty())
    {
        mPolicies.erase(it);
    }
    return page;
}

SlotCount PrioritizedEvictionPolicy::size() const noexcept
{
    SlotCount total = 0;
    for (auto const& [p, policy] : mPolicies)
    {
        total += slotCountValueFromSize(policy.size());
    }
    return total;
}

// allPages() removed — use begin()/end() iterator instead.

// ---------------------------------------------------------------------------
// PerLevelEvictionController
// ---------------------------------------------------------------------------

PerLevelEvictionController::PerLevelEvictionController(
    TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cacheLevel)
    : mCacheLevel(cacheLevel)
    , mLifeCycleGrouping(lifeCycleGrouping)
{
    // Compute number of pool groups = max(grouping) + 1
    PoolGroupIndex numPoolGroups{0};
    for (auto g : mLifeCycleGrouping)
    {
        if (g + 1 > numPoolGroups)
            numPoolGroups = g + 1;
    }
    // Pool group indices must be contiguous 0..N-1.
    TLLM_CHECK_DEBUG(numPoolGroups
        == PoolGroupIndex{
            static_cast<int>(std::set<PoolGroupIndex>(mLifeCycleGrouping.begin(), mLifeCycleGrouping.end()).size())});
    mPolicies.resize(numPoolGroups);
}

PerLevelEvictionController::~PerLevelEvictionController()
{
    KVCM2_CHECK_FATAL_DEBUG_WITH_INFO(
        std::all_of(mPolicies.begin(), mPolicies.end(), [](auto const& p) { return p.empty(); }),
        "Eviction controller is not empty on destruction");
}

PrioritizedEvictionPolicy& PerLevelEvictionController::getPolicy(LifeCycleId lcId)
{
    PoolGroupIndex pgIdx = mLifeCycleGrouping.at(lcId);
    return mPolicies.at(pgIdx);
}

void PerLevelEvictionController::scheduleForEviction(Page& page, bool evictFirst)
{
    TLLM_CHECK_DEBUG(page.nodeRef == std::nullopt);
    TLLM_CHECK_DEBUG(page.cacheLevel == mCacheLevel);
    auto sharedPage = page.sharedFromThis();
    NodeRef ref = getPolicy(page.lifeCycle).push(sharedPage, evictFirst);
    page.nodeRef = ref;
    TLLM_CHECK_DEBUG_WITH_INFO(*ref == sharedPage, "stored iterator must dereference to this page");
}

TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> PerLevelEvictionController::evict(
    TypedVec<PoolGroupIndex, SlotCount> const& minNumPages)
{
    TLLM_CHECK_DEBUG(minNumPages.size() == numPoolGroups());

    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> ret(numPoolGroups());

    try
    {
        for (PoolGroupIndex pgIdx{0}; pgIdx < minNumPages.size(); ++pgIdx)
        {
            auto& policy = mPolicies.at(pgIdx);
            SlotCount const count = minNumPages.at(pgIdx);
            if (count < 0)
            {
                throw LogicError("PerLevelEvictionController::evict: page count must be non-negative");
            }
            SlotCount const available = policy.size() + slotCountValueFromSize(ret[pgIdx].size());
            if (available < count)
            {
                throw OutOfPagesError("Not enough pages to evict in group " + std::to_string(pgIdx.value()));
            }
            while (slotCountValueFromSize(ret[pgIdx].size()) < count)
            {
                auto page = policy.pop();
                page->nodeRef = std::nullopt;
                ret[pgIdx].push_back(page);
                // @TODO: evict dependencies (like Python _evict_dependencies)
            }
        }
    }
    catch (...)
    {
        // Re-queue evicted pages in reverse order (push to front so they are evicted first next time)
        for (PoolGroupIndex pgIdx = ret.size(); pgIdx > PoolGroupIndex{0};)
        {
            --pgIdx;
            auto& group = ret[pgIdx];
            while (!group.empty())
            {
                auto page = std::move(group.back());
                group.pop_back();
                scheduleForEviction(*page, /*evictFirst=*/true);
            }
        }
        throw;
    }

    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(ret.begin(), ret.end(),
                                   [this](auto const& group) {
                                       return std::all_of(group.begin(), group.end(),
                                           [this](auto const& p) { return p->cacheLevel == mCacheLevel; });
                                   }),
        "Corrupted eviction controller");

    return ret;
}

void PerLevelEvictionController::remove(NodeRef node)
{
    auto page = *node;
    TLLM_CHECK_DEBUG_WITH_INFO(
        page->nodeRef.has_value() && page->nodeRef.value() == node, "page's nodeRef must match the node being removed");
    getPolicy(page->lifeCycle).remove(node);
    page->nodeRef = std::nullopt;
}

SlotCount PerLevelEvictionController::numEvictablePages(PoolGroupIndex pgIdx) const
{
    return mPolicies.at(pgIdx).size();
}

void PerLevelEvictionController::setHazardModel(HazardEvictionModel const* model) noexcept
{
    for (auto& policy : mPolicies)
    {
        policy.setHazardModel(model);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
