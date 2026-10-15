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

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace tc = tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename Func>
bool loopedReadWrite(Func&& func, ssize_t size) noexcept
{
    ssize_t count = 0;
    while (count < size)
    {
        ssize_t bytes = func(count);
        if (bytes <= 0)
        {
            if (errno == EINTR)
            {
                continue; // Retry on interrupt
            }
            TLLM_LOG_ERROR("Disk read/write failed: %s\n", strerror(errno));
            return false;
        }
        count += bytes;
    }
    assert(count == size);
    return true;
}

bool writeAll(int fd, ssize_t pos, void const* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pwrite(fd, static_cast<std::byte const*>(data) + finished, size - finished, pos + finished); },
        size);
}

bool readAll(int fd, ssize_t pos, void* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pread(fd, static_cast<std::byte*>(data) + finished, size - finished, pos + finished); },
        size);
}

// ---------------------------------------------------------------------------
// Disk / host copies
//
// These copies are stream-ordered host work: a transfer may start only once everything enqueued
// before it on `stream` has finished, and work enqueued after it must not start before it is done.
// They used to run their pread/pwrite/memcpy loops inside a CUDA host function. The driver runs all
// host functions of a process on a single callback thread, so one multi-second transfer serialized
// every disk transfer of every stream behind it and stalled CUDA API calls made meanwhile.
//
// Now the host function only hands the job to DiskIoPool and returns. The pool splits the job's
// tasks over its worker threads, and the worker that finishes the last chunk publishes the job's
// completion in pinned host memory. The stream waits for that value with cuStreamWaitValue64, which
// keeps the ordering above without occupying the callback thread for the duration of the I/O.
//
// TRTLLM_KV_DISK_IO_THREADS sets the number of worker threads (default kDefaultDiskIoThreads);
// 0 restores the previous behaviour, which is also the fallback when the device lacks 64-bit stream
// memory operations or the pinned completion flags cannot be allocated.
// ---------------------------------------------------------------------------

namespace
{

constexpr size_t kDefaultDiskIoThreads = 8;
constexpr size_t kMaxDiskIoThreads = 64;
// Completion flags; a flag is busy from enqueue until its job completes.
constexpr size_t kNumFlagSlots = 4096;
// One flag per cache line, so that workers publishing different jobs do not share a line.
constexpr size_t kFlagStride = 64;
// Each job is cut into at most this many chunks per worker, for load balance across workers.
constexpr size_t kChunksPerThread = 4;

constexpr size_t divUp(size_t a, size_t b) noexcept
{
    return (a + b - 1) / b;
}

bool runTasks(
    std::vector<Task<DiskAddress, DiskAddress>> const& tasks, size_t begin, size_t end, ssize_t numBytes) noexcept
{
    std::unique_ptr<std::byte[]> buffer(new (std::nothrow) std::byte[numBytes]);
    if (!buffer)
    {
        TLLM_LOG_ERROR("Disk to disk copy: cannot allocate a %zd-byte bounce buffer\n", numBytes);
        return false;
    }
    bool success = true;
    for (size_t i = begin; i < end; ++i)
    {
        auto const& t = tasks[i];
        success = success && readAll(t.src.fd, t.src.pos, buffer.get(), numBytes);
        success = success && writeAll(t.dst.fd, t.dst.pos, buffer.get(), numBytes);
    }
    return success;
}

bool runTasks(
    std::vector<Task<MemAddress, DiskAddress>> const& tasks, size_t begin, size_t end, ssize_t numBytes) noexcept
{
    bool success = true;
    for (size_t i = begin; i < end; ++i)
    {
        auto const& t = tasks[i];
        success = success && readAll(t.src.fd, t.src.pos, reinterpret_cast<void*>(t.dst), numBytes);
    }
    return success;
}

bool runTasks(
    std::vector<Task<DiskAddress, MemAddress>> const& tasks, size_t begin, size_t end, ssize_t numBytes) noexcept
{
    bool success = true;
    for (size_t i = begin; i < end; ++i)
    {
        auto const& t = tasks[i];
        success = success && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), numBytes);
    }
    return success;
}

bool runTasks(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, size_t begin, size_t end, ssize_t numBytes) noexcept
{
    for (size_t i = begin; i < end; ++i)
    {
        auto const& t = tasks[i];
        memcpy(reinterpret_cast<void*>(t.dst), reinterpret_cast<void const*>(t.src), numBytes);
    }
    return true;
}

class DiskIoPool;

//! One disk / host copy. Its tasks can run in any order and on any thread.
class DiskIoJob
{
public:
    explicit DiskIoJob(char const* name) noexcept
        : mName(name)
    {
    }

    virtual ~DiskIoJob() = default;

    DiskIoJob(DiskIoJob const&) = delete;
    DiskIoJob& operator=(DiskIoJob const&) = delete;

    [[nodiscard]] virtual size_t numTasks() const noexcept = 0;

    //! Runs tasks [begin, end). Returns false if an I/O call failed.
    virtual bool run(size_t begin, size_t end) noexcept = 0;

    [[nodiscard]] char const* name() const noexcept
    {
        return mName;
    }

private:
    friend class DiskIoPool;

    char const* mName;
    // Set by DiskIoPool.
    DiskIoPool* mPool{nullptr};
    size_t mFlagSlot{0};
    uint64_t mFlagValue{0};
    std::atomic<size_t> mRemainingChunks{0};
    std::atomic<bool> mFailed{false};
};

template <typename DstAddr, typename SrcAddr>
class CopyJob final : public DiskIoJob
{
public:
    CopyJob(char const* name, std::vector<Task<DstAddr, SrcAddr>> tasks, ssize_t numBytes) noexcept
        : DiskIoJob(name)
        , mTasks(std::move(tasks))
        , mNumBytes(numBytes)
    {
    }

    [[nodiscard]] size_t numTasks() const noexcept override
    {
        return mTasks.size();
    }

    bool run(size_t begin, size_t end) noexcept override
    {
        return runTasks(mTasks, begin, end, mNumBytes);
    }

private:
    std::vector<Task<DstAddr, SrcAddr>> mTasks;
    ssize_t mNumBytes;
};

//! Previous behaviour: the whole job runs inside the host function, on the driver's callback thread.
CUDA_CB void hostFnRunJob(void* userData) noexcept
{
    auto const job = std::unique_ptr<DiskIoJob>(static_cast<DiskIoJob*>(userData));
    if (!job->run(0, job->numTasks()))
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] %s failed.\n", job->name());
    }
}

size_t diskIoThreadsFromEnv() noexcept
{
    char const* env = std::getenv("TRTLLM_KV_DISK_IO_THREADS");
    if (env == nullptr || *env == '\0')
    {
        return kDefaultDiskIoThreads;
    }
    char* end = nullptr;
    long const value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0)
    {
        TLLM_LOG_WARNING("Ignoring TRTLLM_KV_DISK_IO_THREADS=%s; using %zu", env, kDefaultDiskIoThreads);
        return kDefaultDiskIoThreads;
    }
    return std::min(static_cast<size_t>(value), kMaxDiskIoThreads);
}

bool deviceSupportsWaitValue64() noexcept
{
    constexpr int kMaxDevices = 64;
    // 0: unknown, 1: supported, 2: unsupported.
    static std::array<std::atomic<int8_t>, kMaxDevices> cache{};
    CUdevice device = 0;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS || device < 0 || device >= kMaxDevices)
    {
        return false;
    }
    int8_t state = cache[device].load(std::memory_order_relaxed);
    if (state == 0)
    {
        int supported = 0;
        bool const ok = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device)
            == CUDA_SUCCESS;
        state = ok && supported != 0 ? 1 : 2;
        cache[device].store(state, std::memory_order_relaxed);
        if (state == 2)
        {
            TLLM_LOG_WARNING(
                "KV cache disk I/O: device %d lacks 64-bit stream memory operations; disk transfers run inside "
                "CUDA host functions",
                static_cast<int>(device));
        }
    }
    return state == 1;
}

//! Process-wide worker pool for disk / host copies, with completion signalled to streams through
//! pinned flags. Never destroyed: host functions and workers may still reference it at exit.
class DiskIoPool
{
public:
    //! The pool, or nullptr when asynchronous disk I/O is disabled or unavailable. Call with a CUDA
    //! context current.
    static DiskIoPool* instance() noexcept
    {
        static DiskIoPool* const pool = create();
        return pool;
    }

    //! Enqueues the job on stream: a host function that hands it to the workers, then a wait for its
    //! completion flag. On success the pool owns the job. Returns nullopt, leaving the job with the
    //! caller, when every completion flag is in use.
    std::optional<CUresult> enqueue(std::unique_ptr<DiskIoJob>& job, CUstream stream) noexcept
    {
        size_t slot = 0;
        uint64_t value = 0;
        {
            std::lock_guard<std::mutex> lock(mSlotMutex);
            if (mFreeSlots.empty())
            {
                return std::nullopt;
            }
            slot = mFreeSlots.back();
            mFreeSlots.pop_back();
            // A flag only grows, and a slot is reused only after its previous job published, so a
            // stream still waiting for an earlier value of this flag is satisfied by every later one.
            value = ++mSlotValues[slot];
        }
        job->mPool = this;
        job->mFlagSlot = slot;
        job->mFlagValue = value;
        CUresult const launched = cuLaunchHostFunc(stream, &DiskIoPool::hostFnSubmit, job.get());
        if (launched != CUDA_SUCCESS)
        {
            releaseSlot(slot);
            return launched;
        }
        // The pool owns the job from here; the worker finishing its last chunk deletes it.
        job.release();
        return cuStreamWaitValue64(stream, mFlagsDevice + slot * kFlagStride, value, CU_STREAM_WAIT_VALUE_GEQ);
    }

private:
    struct Chunk
    {
        DiskIoJob* job;
        size_t begin;
        size_t end;
    };

    DiskIoPool(std::byte* flagsHost, CUdeviceptr flagsDevice)
        : mFlagsHost(flagsHost)
        , mFlagsDevice(flagsDevice)
        , mSlotValues(kNumFlagSlots, 0)
    {
        mFreeSlots.reserve(kNumFlagSlots);
        for (size_t slot = kNumFlagSlots; slot > 0; --slot)
        {
            mFreeSlots.push_back(slot - 1);
        }
    }

    //! Starts up to numThreads workers and returns how many started.
    size_t startWorkers(size_t numThreads) noexcept
    {
        try
        {
            mThreads.reserve(numThreads);
            for (size_t i = 0; i < numThreads; ++i)
            {
                mThreads.emplace_back([this] { workerLoop(); });
                pthread_setname_np(mThreads.back().native_handle(), "kvcm-disk-io");
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING(
                "KV cache disk I/O: started %zu of %zu worker threads (%s)", mThreads.size(), numThreads, e.what());
        }
        mNumThreads = mThreads.size();
        return mNumThreads;
    }

    static DiskIoPool* create() noexcept
    {
        size_t const numThreads = diskIoThreadsFromEnv();
        if (numThreads == 0)
        {
            return nullptr;
        }
        static_assert(std::atomic<uint64_t>::is_always_lock_free);
        static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t));
        void* host = nullptr;
        CUdeviceptr device = 0;
        if (cuMemHostAlloc(&host, kNumFlagSlots * kFlagStride, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)
            != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "KV cache disk I/O: cannot allocate completion flags; disk transfers run inside CUDA "
                "host functions");
            return nullptr;
        }
        if (cuMemHostGetDevicePointer(&device, host, 0) != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "KV cache disk I/O: completion flags are not device-mapped; disk transfers run inside "
                "CUDA host functions");
            cuMemFreeHost(host);
            return nullptr;
        }
        auto* flags = static_cast<std::byte*>(host);
        for (size_t slot = 0; slot < kNumFlagSlots; ++slot)
        {
            new (flags + slot * kFlagStride) std::atomic<uint64_t>(0);
        }
        DiskIoPool* pool = nullptr;
        try
        {
            pool = new DiskIoPool(flags, device);
        }
        catch (std::bad_alloc const&)
        {
        }
        if (pool == nullptr || pool->startWorkers(numThreads) == 0)
        {
            // With no worker running, nothing references the pool or the flags.
            TLLM_LOG_WARNING(
                "KV cache disk I/O: cannot start worker threads; disk transfers run inside CUDA host "
                "functions");
            delete pool;
            cuMemFreeHost(host);
            return nullptr;
        }
        TLLM_LOG_INFO(
            "KV cache disk I/O: %zu worker threads, completion via stream memory operations", pool->mNumThreads);
        return pool;
    }

    std::atomic<uint64_t>& flag(size_t slot) const noexcept
    {
        return *std::launder(reinterpret_cast<std::atomic<uint64_t>*>(mFlagsHost + slot * kFlagStride));
    }

    void releaseSlot(size_t slot) noexcept
    {
        std::lock_guard<std::mutex> lock(mSlotMutex);
        mFreeSlots.push_back(slot);
    }

    //! Runs on the driver's callback thread once the stream reaches the job: queue it and return.
    static CUDA_CB void hostFnSubmit(void* userData) noexcept
    {
        auto* job = static_cast<DiskIoJob*>(userData);
        job->mPool->submit(job);
    }

    void submit(DiskIoJob* job) noexcept
    {
        // enqueueCopy never submits an empty job.
        size_t const numTasks = job->numTasks();
        size_t const chunkSize = divUp(numTasks, std::min(numTasks, mNumThreads * kChunksPerThread));
        size_t const numChunks = divUp(numTasks, chunkSize);
        job->mRemainingChunks.store(numChunks, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            for (size_t begin = 0; begin < numTasks; begin += chunkSize)
            {
                mQueue.push_back(Chunk{job, begin, std::min(numTasks, begin + chunkSize)});
            }
        }
        if (numChunks == 1)
        {
            mQueueCv.notify_one();
        }
        else
        {
            mQueueCv.notify_all();
        }
    }

    void workerLoop() noexcept
    {
        while (true)
        {
            Chunk chunk{};
            {
                std::unique_lock<std::mutex> lock(mQueueMutex);
                mQueueCv.wait(lock, [this] { return !mQueue.empty(); });
                chunk = mQueue.front();
                mQueue.pop_front();
            }
            DiskIoJob* job = chunk.job;
            if (!job->run(chunk.begin, chunk.end))
            {
                job->mFailed.store(true, std::memory_order_relaxed);
            }
            if (job->mRemainingChunks.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                complete(job);
            }
        }
    }

    void complete(DiskIoJob* job) noexcept
    {
        if (job->mFailed.load(std::memory_order_relaxed))
        {
            TLLM_LOG_ERROR("[kvCacheManagerV2Utils] %s failed.\n", job->name());
        }
        size_t const slot = job->mFlagSlot;
        // Release: the job's data (from every worker, via the acq_rel countdown) is visible before
        // the stream can observe the flag.
        flag(slot).store(job->mFlagValue, std::memory_order_release);
        delete job;
        releaseSlot(slot);
    }

    std::byte* const mFlagsHost;
    CUdeviceptr const mFlagsDevice;
    size_t mNumThreads{0};

    std::mutex mSlotMutex;
    std::vector<uint64_t> mSlotValues; // last value handed out per flag
    std::vector<size_t> mFreeSlots;

    std::mutex mQueueMutex;
    std::condition_variable mQueueCv;
    std::deque<Chunk> mQueue;

    std::vector<std::thread> mThreads;
};

template <typename DstAddr, typename SrcAddr>
CUresult enqueueCopy(
    char const* name, std::vector<Task<DstAddr, SrcAddr>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    if (tasks.empty())
    {
        return CUDA_SUCCESS;
    }
    std::unique_ptr<DiskIoJob> job;
    try
    {
        job = std::make_unique<CopyJob<DstAddr, SrcAddr>>(name, std::move(tasks), numBytes);
    }
    catch (std::bad_alloc const&)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (auto* pool = deviceSupportsWaitValue64() ? DiskIoPool::instance() : nullptr; pool != nullptr)
    {
        if (auto const result = pool->enqueue(job, stream); result.has_value())
        {
            return *result;
        }
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed))
        {
            TLLM_LOG_WARNING(
                "KV cache disk I/O: %zu transfers in flight; running further ones inside CUDA host "
                "functions until some complete",
                kNumFlagSlots);
        }
    }
    CUresult const launched = cuLaunchHostFunc(stream, hostFnRunJob, job.get());
    if (launched == CUDA_SUCCESS)
    {
        job.release();
    }
    return launched;
}

} // namespace

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return enqueueCopy("disk to disk copy", std::move(tasks), numBytes, stream);
}

CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return enqueueCopy("disk to host copy", std::move(tasks), numBytes, stream);
}

CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return enqueueCopy("host to disk copy", std::move(tasks), numBytes, stream);
}

CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return enqueueCopy("host to host copy", std::move(tasks), numBytes, stream);
}

SizeType32 IndexMapper::addNewSequence(LlmRequest::RequestIdType requestId)
{
    TLLM_CHECK(indexMap_.find(requestId) == indexMap_.end());
    auto iter = freeIndices_.begin();
    TLLM_CHECK_WITH_INFO(iter != freeIndices_.end(), "No free index found");
    auto index = *iter;
    freeIndices_.erase(iter);
    indexMap_[requestId] = index;
    return index;
}

SizeType32 IndexMapper::getIndex(LlmRequest::RequestIdType requestId)
{
    auto iter = indexMap_.find(requestId);
    TLLM_CHECK_WITH_INFO(iter != indexMap_.end(), "Request ID not found in IndexMapper");
    return iter->second;
}

void IndexMapper::removeSequence(LlmRequest::RequestIdType requestId)
{
    auto iter = indexMap_.find(requestId);
    TLLM_CHECK(iter != indexMap_.end());
    auto index = iter->second;
    freeIndices_.insert(index);
    indexMap_.erase(iter);
}

at::Tensor IndexMapper::getCopyIndex(std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numContext,
    SizeType32 beamWidth, bool replicateBeamZero)
{
    TLLM_CHECK_WITH_INFO(beamWidth <= maxCopyBeamWidth_, "Requested beam width %d exceeds the copy-index capacity %d",
        beamWidth, maxCopyBeamWidth_);
    TLLM_CHECK_WITH_INFO(replicateBeamZero || beamWidth <= maxBeamWidth_,
        "Requested beam width %d exceeds the source page-table width %d", beamWidth, maxBeamWidth_);
    int numSeqs = numContext + beamWidth * (requestIds.size() - numContext);
    SizeType32 batchSize = static_cast<SizeType32>(requestIds.size());
    SizeType32 idx = 0;
    for (SizeType32 i = 0; i < batchSize; i++)
    {
        if (i < numContext)
        {
            copyIndex_[idx++] = this->getIndex(requestIds[i]) * maxBeamWidth_;
        }
        else
        {
            for (SizeType32 j = 0; j < beamWidth; j++)
            {
                auto const sourceBeam = replicateBeamZero ? 0 : j;
                copyIndex_[idx++] = this->getIndex(requestIds[i]) * maxBeamWidth_ + sourceBeam;
            }
        }
    }

    TLLM_CHECK_WITH_INFO(idx == numSeqs, "Index mapper failed to generate copy index");

    return copyIndex_.slice(0, 0, numSeqs);
}

void IndexMapper::gatherKBlockOffsets(at::Tensor const& source, at::Tensor destination,
    std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numBlocks)
{
    std::vector<int64_t> sourceRowsByRequest;
    sourceRowsByRequest.reserve(requestIds.size());
    for (auto const requestId : requestIds)
    {
        sourceRowsByRequest.push_back(static_cast<int64_t>(getIndex(requestId)) * maxBeamWidth_);
    }

    auto const* sourceData = source.data_ptr<int32_t>();
    auto* destinationData = destination.data_ptr<int32_t>();
    auto const sourceRows = source.size(1);
    auto const sourcePlanes = source.size(2);
    auto const sourceBlocks = source.size(3);
    auto const destinationRows = destination.size(1);
    auto const destinationPlanes = destination.size(2);
    auto const destinationBlocks = destination.size(3);
    auto const copyBytes = static_cast<size_t>(numBlocks) * sizeof(int32_t);

    for (int64_t pool = 0; pool < source.size(0); ++pool)
    {
        for (size_t destinationRow = 0; destinationRow < sourceRowsByRequest.size(); ++destinationRow)
        {
            auto const sourceRow = sourceRowsByRequest[destinationRow];
            auto const sourceOffset = ((pool * sourceRows + sourceRow) * sourcePlanes) * sourceBlocks;
            auto const destinationOffset
                = ((pool * destinationRows + static_cast<int64_t>(destinationRow)) * destinationPlanes)
                * destinationBlocks;
            std::memcpy(destinationData + destinationOffset, sourceData + sourceOffset, copyBytes);
        }
    }
}

IndexMapper::IndexMapper(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxCopyBeamWidth)
    : maxBeamWidth_(maxBeamWidth)
    , maxCopyBeamWidth_(maxCopyBeamWidth == 0 ? maxBeamWidth : maxCopyBeamWidth)
{
    TLLM_CHECK_WITH_INFO(maxCopyBeamWidth_ >= maxBeamWidth_,
        "Copy-index beam width %d must be at least the source page-table beam width %d", maxCopyBeamWidth_,
        maxBeamWidth_);
    indexMap_.reserve(maxBatchSize);
    for (SizeType32 i = 0; i < maxBatchSize; i++)
    {
        freeIndices_.insert(i);
    }
    // Allocate copyIndex_ memory as pinned (page-locked) host memory
    copyIndex_ = at::empty(
        {maxBatchSize * maxCopyBeamWidth_}, at::TensorOptions().dtype(at::ScalarType::Int).pinned_memory(true));
}

IndexMapper::~IndexMapper()
{
    indexMap_.clear();
    freeIndices_.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
