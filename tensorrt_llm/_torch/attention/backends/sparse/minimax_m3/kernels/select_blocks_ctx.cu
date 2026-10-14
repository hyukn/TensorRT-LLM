/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MiniMax-M3 top-k block selection for wide, query-contiguous score tensors (context steps).
 *
 * Contract: identical to torch.ops.trtllm.minimax_m3_select_blocks (minimaxM3SelectBlocks.cu) on
 * every input it accepts -- scores [H, num_blocks, total_q] fp32 with stride(2) == 1 (the layout the
 * MSA index scorers write), n_valid_blocks [total_q] int32, topk 16. Per (query, kv head) the result
 * is the 16 best blocks of [0, clamp(n_valid, 0, num_blocks)) under the total order
 * (orderedScoreKey(forced score) ascending, block ascending), forced scores per the init / local
 * rules, blocks whose forced score is -inf dropped, ascending block ids, -1 padded. That is exactly
 * the order the histogram kernel resolves (its fp16 / radix bins only pre-partition by the same key
 * and its boundary sort breaks ties by smaller block), so the outputs are bitwise identical.
 *
 * Why another kernel: the histogram kernel gives each (query, head) row one CTA and walks the row
 * along the block axis, whose stride is total_q (16K on a 16K-token chunk). Every 4-byte score
 * read is then its own 32-byte sector, read at least twice. Here a CTA owns 32 consecutive queries
 * of one head, and warp w reads block rows w, w + SLICES, ... for all 32 queries at once, i.e. one
 * fully used 128-byte line per load:
 *   pass 1  per thread, the minimum key of each of GROUPS interleaved groups of its blocks;
 *           T[q] = 16th smallest of the query's SLICES * GROUPS group minima. Those minima are
 *           distinct real elements, so at least 16 elements have key <= T[q]: every top-16
 *           element has key <= T[q].
 *   pass 2  collect the (key, block) pairs with key <= T[q] into a shared buffer (CAP per query).
 *           If a query overflows, T[q] is lowered to the 16th best of its first CAP candidates
 *           (again 16 real elements) and the pass repeats for the overflowing queries only; after
 *           kMaxRounds a single thread per query falls back to an exact streaming insertion.
 *   select  exact rank of every candidate by (key, block); ranks < 16 are the result.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>

namespace
{

constexpr int kTopK = 16;
constexpr int kWarp = 32;
constexpr int kCap = 128; // candidate slots per query
constexpr int kMaxRounds = 3;
constexpr uint32_t kEmptyKey = 0xFFFFFFFFu;
constexpr uint64_t kEmptyComp = ~0ull;

__device__ __forceinline__ uint32_t orderedScoreKey(float x)
{
    uint32_t const bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

__device__ __forceinline__ uint32_t forcedKey(
    float raw, int32_t block, int32_t initBlocks, int64_t localStart, uint32_t initKey, uint32_t localKey)
{
    // Local forcing wins where the two ranges overlap (matches applyForcing).
    if (block >= localStart)
    {
        return localKey;
    }
    if (block < initBlocks)
    {
        return initKey;
    }
    return orderedScoreKey(raw);
}

__device__ __forceinline__ uint64_t makeComp(uint32_t key, int32_t block)
{
    return (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(block);
}

template <int SLICES, int GROUPS, int UNROLL, bool HeadMajorOutput>
__global__ void __launch_bounds__(SLICES* kWarp) selectBlocksCtxKernel(float const* __restrict__ scores,
    int64_t headStride, int64_t blockStride, int32_t const* __restrict__ nValidBlocks, int32_t* __restrict__ output,
    int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries, int32_t initBlocks, int32_t localBlocks)
{
    constexpr int kMinima = SLICES * GROUPS;
    static_assert(kMinima >= kTopK, "need at least topk group minima");
    static_assert(UNROLL % GROUPS == 0, "an unrolled batch must cover whole group cycles");

    __shared__ uint32_t sMin[kWarp][kMinima + 1];
    __shared__ uint64_t sBuf[kWarp][kCap + 1]; // +1: 2-way instead of 32-way bank conflicts
    __shared__ uint32_t sThresh[kWarp];
    __shared__ int32_t sCount[kWarp];
    __shared__ int32_t sDone[kWarp];
    __shared__ int32_t sSel[kWarp][kTopK];

    int const lane = threadIdx.x % kWarp;
    int const slice = threadIdx.x / kWarp;
    int const head = blockIdx.y;
    int64_t const query = static_cast<int64_t>(blockIdx.x) * kWarp + lane;
    bool const live = query < totalQueries;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int32_t const rawValid = live ? nValidBlocks[query] : 0;
    int32_t const valid = max(0, min(rawValid, numBlocks));
    int64_t const localStart = max(static_cast<int64_t>(rawValid) - static_cast<int64_t>(localBlocks), int64_t{0});
    uint32_t const initKey = orderedScoreKey(1.0e30F);
    uint32_t const localKey = orderedScoreKey(1.0e29F);
    float const* __restrict__ base = scores + static_cast<int64_t>(head) * headStride + (live ? query : 0);

    // CTA-wide row extent (all warps hold the same 32 queries).
    int32_t warpMax = valid;
#pragma unroll
    for (int off = 16; off > 0; off /= 2)
    {
        warpMax = max(warpMax, __shfl_xor_sync(0xffffffffu, warpMax, off));
    }
    int32_t const rowsPerSlice = (warpMax + SLICES - 1) / SLICES;

    // ---- pass 1: group minima -----------------------------------------------------------
    uint32_t gmin[GROUPS];
#pragma unroll
    for (int g = 0; g < GROUPS; ++g)
    {
        gmin[g] = kEmptyKey;
    }
    {
        int32_t j = 0;
        for (; j + UNROLL <= rowsPerSlice; j += UNROLL)
        {
            float v[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                int32_t const b = slice + (j + u) * SLICES;
                v[u] = b < valid ? base[static_cast<int64_t>(b) * blockStride] : 0.F;
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                int32_t const b = slice + (j + u) * SLICES;
                if (b < valid)
                {
                    gmin[u % GROUPS]
                        = min(gmin[u % GROUPS], forcedKey(v[u], b, initBlocks, localStart, initKey, localKey));
                }
            }
        }
        for (; j < rowsPerSlice; ++j)
        {
            int32_t const b = slice + j * SLICES;
            if (b < valid)
            {
                float const v = base[static_cast<int64_t>(b) * blockStride];
                gmin[j % GROUPS] = min(gmin[j % GROUPS], forcedKey(v, b, initBlocks, localStart, initKey, localKey));
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GROUPS; ++g)
    {
        sMin[lane][slice * GROUPS + g] = gmin[g];
    }
    if (slice == 0)
    {
        sCount[lane] = 0;
        sDone[lane] = live ? 0 : 1;
    }
    __syncthreads();

    // T[q] = 16th smallest group minimum: the element whose (#less) <= 15 < (#less-or-equal).
    for (int i = slice; i < kMinima; i += SLICES)
    {
        uint32_t const x = sMin[lane][i];
        int less = 0;
        int lessEq = 0;
#pragma unroll 8
        for (int k = 0; k < kMinima; ++k)
        {
            uint32_t const y = sMin[lane][k];
            less += y < x;
            lessEq += y <= x;
        }
        if (less <= kTopK - 1 && lessEq > kTopK - 1)
        {
            sThresh[lane] = x; // every writer writes the same value
        }
    }
    __syncthreads();

    // ---- pass 2 (+ refinement rounds): collect candidates with key <= T -----------------------
    bool fallback = false;
    for (int round = 0;; ++round)
    {
        uint32_t const thresh = sThresh[lane];
        bool const active = !sDone[lane];
        if (active)
        {
            int32_t j = 0;
            for (; j + UNROLL <= rowsPerSlice; j += UNROLL)
            {
                float v[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                {
                    int32_t const b = slice + (j + u) * SLICES;
                    v[u] = b < valid ? base[static_cast<int64_t>(b) * blockStride] : 0.F;
                }
#pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                {
                    int32_t const b = slice + (j + u) * SLICES;
                    if (b < valid)
                    {
                        uint32_t const key = forcedKey(v[u], b, initBlocks, localStart, initKey, localKey);
                        if (key <= thresh)
                        {
                            int const slot = atomicAdd(&sCount[lane], 1);
                            if (slot < kCap)
                            {
                                sBuf[lane][slot] = makeComp(key, b);
                            }
                        }
                    }
                }
            }
            for (; j < rowsPerSlice; ++j)
            {
                int32_t const b = slice + j * SLICES;
                if (b < valid)
                {
                    uint32_t const key = forcedKey(
                        base[static_cast<int64_t>(b) * blockStride], b, initBlocks, localStart, initKey, localKey);
                    if (key <= thresh)
                    {
                        int const slot = atomicAdd(&sCount[lane], 1);
                        if (slot < kCap)
                        {
                            sBuf[lane][slot] = makeComp(key, b);
                        }
                    }
                }
            }
        }
        __syncthreads();
        bool const overflow = active && sCount[lane] > kCap;
        if (!__syncthreads_or(overflow))
        {
            break;
        }
        if (round + 1 >= kMaxRounds)
        {
            fallback = overflow;
            break;
        }
        // Lower T for overflowing queries: 16th best (by comp) of the CAP buffered candidates.
        if (overflow)
        {
            for (int i = slice; i < kCap; i += SLICES)
            {
                uint64_t const x = sBuf[lane][i];
                int rank = 0;
                for (int k = 0; k < kCap; ++k)
                {
                    rank += sBuf[lane][k] < x;
                }
                if (rank == kTopK - 1)
                {
                    sThresh[lane] = static_cast<uint32_t>(x >> 32);
                }
            }
        }
        __syncthreads();
        if (slice == 0 && overflow)
        {
            sCount[lane] = 0;
        }
        if (slice == 0 && !overflow)
        {
            sDone[lane] = 1;
        }
        __syncthreads();
    }

    // ---- exact selection among the candidates ---------------------------------------------------
    if (slice == 0)
    {
#pragma unroll
        for (int r = 0; r < kTopK; ++r)
        {
            sSel[lane][r] = -1;
        }
    }
    __syncthreads();
    int const count = min(sCount[lane], kCap);
    if (!fallback && live)
    {
        for (int i = slice; i < count; i += SLICES)
        {
            uint64_t const x = sBuf[lane][i];
            int rank = 0;
            for (int k = 0; k < count; ++k)
            {
                rank += sBuf[lane][k] < x;
            }
            if (rank < kTopK)
            {
                sSel[lane][rank] = static_cast<int32_t>(static_cast<uint32_t>(x));
                if (static_cast<uint32_t>(x >> 32) == orderedScoreKey(-INFINITY))
                {
                    sSel[lane][rank] = -1; // forced score -inf: dropped, as the reference does
                }
            }
        }
    }
    __syncthreads();

    if (slice != 0 || !live)
    {
        return;
    }
    int32_t sel[kTopK];
    if (fallback)
    {
        // Pathological ties (more than CAP candidates share the 16th key): exact streaming insertion.
        uint64_t best[kTopK];
#pragma unroll
        for (int r = 0; r < kTopK; ++r)
        {
            best[r] = kEmptyComp;
        }
        for (int32_t b = 0; b < valid; ++b)
        {
            uint64_t const c = makeComp(
                forcedKey(base[static_cast<int64_t>(b) * blockStride], b, initBlocks, localStart, initKey, localKey),
                b);
            if (c < best[kTopK - 1])
            {
                uint64_t carry = c;
#pragma unroll
                for (int r = 0; r < kTopK; ++r)
                {
                    uint64_t const cur = best[r];
                    bool const take = carry < cur;
                    best[r] = take ? carry : cur;
                    carry = take ? cur : carry;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kTopK; ++r)
        {
            uint32_t const key = static_cast<uint32_t>(best[r] >> 32);
            sel[r] = (best[r] == kEmptyComp || key == orderedScoreKey(-INFINITY))
                ? -1
                : static_cast<int32_t>(static_cast<uint32_t>(best[r]));
        }
    }
    else
    {
#pragma unroll
        for (int r = 0; r < kTopK; ++r)
        {
            sel[r] = sSel[lane][r];
        }
    }
    // Ascending block ids, -1 padding last: sort with -1 mapped above every id.
    uint32_t k[kTopK];
#pragma unroll
    for (int r = 0; r < kTopK; ++r)
    {
        k[r] = sel[r] < 0 ? 0xFFFFFFFFu : static_cast<uint32_t>(sel[r]);
    }
#pragma unroll
    for (int size = 2; size <= kTopK; size *= 2)
    {
#pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2)
        {
#pragma unroll
            for (int i = 0; i < kTopK; ++i)
            {
                int const p = i ^ stride;
                if (p > i)
                {
                    bool const asc = (i & size) == 0;
                    uint32_t const a = k[i];
                    uint32_t const c = k[p];
                    bool const swap = asc ? (a > c) : (a < c);
                    k[i] = swap ? c : a;
                    k[p] = swap ? a : c;
                }
            }
        }
    }
    int64_t const row = HeadMajorOutput ? static_cast<int64_t>(head) * totalQueries + query : query * numKvHeads + head;
    int4* dst = reinterpret_cast<int4*>(output + row * kTopK);
#pragma unroll
    for (int v = 0; v < kTopK / 4; ++v)
    {
        int4 o;
        o.x = k[4 * v + 0] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(k[4 * v + 0]);
        o.y = k[4 * v + 1] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(k[4 * v + 1]);
        o.z = k[4 * v + 2] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(k[4 * v + 2]);
        o.w = k[4 * v + 3] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(k[4 * v + 3]);
        dst[v] = o;
    }
}

template <int SLICES, int GROUPS, int UNROLL>
void launch(torch::Tensor const& scores, torch::Tensor const& nValid, torch::Tensor& out, int64_t initBlocks,
    int64_t localBlocks, bool headMajor, cudaStream_t stream)
{
    int32_t const numKvHeads = static_cast<int32_t>(scores.size(0));
    int32_t const numBlocks = static_cast<int32_t>(scores.size(1));
    int32_t const totalQueries = static_cast<int32_t>(scores.size(2));
    dim3 const grid((totalQueries + kWarp - 1) / kWarp, numKvHeads);
    dim3 const block(SLICES * kWarp);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    char const* pdl = std::getenv("TRTLLM_ENABLE_PDL");
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = (pdl != nullptr && pdl[0] == '1') ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    auto kernel = headMajor ? selectBlocksCtxKernel<SLICES, GROUPS, UNROLL, true>
                            : selectBlocksCtxKernel<SLICES, GROUPS, UNROLL, false>;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, scores.data_ptr<float>(), scores.stride(0), scores.stride(1),
        nValid.data_ptr<int32_t>(), out.data_ptr<int32_t>(), numKvHeads, numBlocks, totalQueries,
        static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks)));
}

} // namespace

// Same arguments and result as torch.ops.trtllm.minimax_m3_select_blocks; requires
// scores.stride(2) == 1. `variant` picks (slices, groups, unroll); 0 is the tuned default.
torch::Tensor selectBlocksCtx(torch::Tensor const& scores, torch::Tensor const& nValidBlocks, int64_t topK,
    int64_t initBlocks, int64_t localBlocks, bool headMajorOutput, int64_t variant)
{
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3);
    TORCH_CHECK(scores.stride(2) == 1, "select_blocks_ctx needs query-contiguous scores");
    TORCH_CHECK(scores.size(1) <= 65535, "select_blocks_ctx supports at most 65535 blocks");
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.dim() == 1
        && nValidBlocks.size(0) == scores.size(2) && nValidBlocks.is_contiguous());
    TORCH_CHECK(topK == kTopK, "select_blocks_ctx supports topk=16");
    TORCH_CHECK(initBlocks >= 0 && localBlocks >= 0);
    c10::cuda::CUDAGuard const guard{scores.device()};
    auto const opts = scores.options().dtype(torch::kInt32);
    auto out = headMajorOutput ? torch::empty({scores.size(0), scores.size(2), kTopK}, opts).permute({1, 0, 2})
                               : torch::empty({scores.size(2), scores.size(0), kTopK}, opts);
    if (scores.size(0) == 0 || scores.size(2) == 0)
    {
        return out;
    }
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    switch (variant)
    {
    case 1: launch<16, 2, 16>(scores, nValidBlocks, out, initBlocks, localBlocks, headMajorOutput, stream); break;
    case 2: launch<32, 1, 8>(scores, nValidBlocks, out, initBlocks, localBlocks, headMajorOutput, stream); break;
    case 3: launch<32, 1, 16>(scores, nValidBlocks, out, initBlocks, localBlocks, headMajorOutput, stream); break;
    case 4: launch<8, 4, 8>(scores, nValidBlocks, out, initBlocks, localBlocks, headMajorOutput, stream); break;
    default: launch<16, 2, 8>(scores, nValidBlocks, out, initBlocks, localBlocks, headMajorOutput, stream); break;
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("select_blocks_ctx", &selectBlocksCtx, "MiniMax-M3 context top-k block select (query-contiguous)");
}
