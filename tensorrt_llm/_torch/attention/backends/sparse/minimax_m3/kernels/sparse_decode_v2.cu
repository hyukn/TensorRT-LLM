// Block-sparse GQA decode attention over a paged NVFP4 KV cache (MiniMax-M3).
//
// One CTA owns one (request, KV head, split). It builds the union of the
// top-k block lists of the request's decode_query_len query tokens (Eagle
// verify tokens select heavily overlapping sets), takes a contiguous slice of
// that union, and streams each selected 128-token page into shared memory once
// with 1-D TMA bulk copies (K/V nibbles + E4M3 block scales, 18 KB per page).
// Consumer warps each own a 16-row tile of (query, head) rows; a page is
// consumed only by the tiles whose queries selected it. QK and PV run on
// mma.sync m16n8k16 (fp16 in, fp32 accumulate); K/V are dequantized from E2M1
// straight into MMA fragments (cvt.rn.f16x2.e2m1x2, exact) using permutations
// of the head-dim / token axes that make every fragment register a single
// packed byte (K) or a nibble interleave of two tokens (V).
//
// Splits are merged by the last CTA of a (request, head) to finish (atomic
// ticket, self-resetting), so the whole op is a single launch. Partials are
// stored as fp16 *before* the per-tensor V scale, which bounds them by
// 6 * 448 and makes fp16 safe.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sdv2
{

constexpr int kBlk = 128;                              // tokens per sparse block == page
constexpr int kKvBytes = kBlk * 128 / 2;               // packed E2M1 bytes per page/head
constexpr int kScBytes = kBlk * 128 / 16;              // E4M3 scale bytes per page/head
constexpr int kStageBytes = 2 * (kKvBytes + kScBytes); // 18432
constexpr int kMaxCand = 64;
// Candidates whose pages the prologue prefetches into L2. Off: the bulk prefetches queue in front of the
// CTA's own page loads on the SM's TMA unit (64 candidates x 4 copies per CTA at nsplit 1) and the
// prologue waited on them; without them the kernel is 4-6 % faster at 8-40 requests (bitwise the same).
#ifndef SDV2_PREFETCH_CANDS
#define SDV2_PREFETCH_CANDS 0
#endif
constexpr float kNegInf = -__builtin_huge_valf();

struct Params
{
    void const* q;
    int64_t q_sn, q_sh; // element strides; stride over head_dim must be 1
    uint8_t const* k;
    uint8_t const* v;
    int64_t k_sb, k_sh, v_sb, v_sh; // byte strides
    uint8_t const* ks;
    uint8_t const* vs;
    int64_t ks_sb, ks_sh, vs_sb, vs_sh;
    float const* k_gs;
    float const* v_gs;
    int const* topk;
    int64_t t_sh, t_sn, t_sk;
    int const* bt;
    int64_t bt_sb;
    int bt_cols;
    int const* seq_lens;
    __nv_bfloat16* out;
    int64_t o_sn, o_sh;
    __half* part_o;  // [groups][nsplit][rows_pad][128]
    float2* part_ml; // [groups][nsplit][rows_pad]  (m in log2 domain, l)
    int* counters;   // [groups]
    long long* dbg;  // optional per-CTA phase timestamps (globaltimer ns), 8 per CTA
    int num_kvh, group, dql, topk_n, nsplit, ntiles, wsub, stages;
    float sm_scale_log2e;
    int q_fp8;
    int kv_fused;      // V nibbles of a (page, head) directly follow K's: one 16 KB copy
    int sc_fused;      // same for the two scale blocks: one 2 KB copy
    int cluster_merge; // 1: splits form a cluster and merge over DSMEM; 0: global partials + ticket
    // Stream-K schedule only (sparse_decode_v2_sk): per-group unions from the prep kernel.
    int* u_blk;  // [groups][kMaxCand] block ids, first-occurrence order
    int* u_page; // [groups][kMaxCand] KV pool page ids
    int* u_mask; // [groups][kMaxCand] per-query membership bits
    int* u_cnt;  // [groups] union sizes
    int* u_wcum; // [groups][kMaxCand] inclusive prefix of the page weights within the group
    int* u_wsum; // [groups] total page weight of the group
    int ngroups;
};

// ------------------------------------------------------------------ PTX bits
__device__ __forceinline__ uint32_t smem_u32(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1 bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ bool mbar_test(uint64_t* bar, uint32_t parity)
{
    uint32_t ok;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}"
        : "=r"(ok)
        : "r"(smem_u32(bar)), "r"(parity)
        : "memory");
    return ok != 0;
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

__device__ __forceinline__ long long gtimer()
{
    long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

#define SDV2_STAMP(k)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (p.dbg != nullptr && tid == 0)                                                                              \
            p.dbg[blockIdx.x * 24 + (k)] = gtimer();                                                                   \
    } while (0)

__device__ __forceinline__ void prefetch_l2(void const* src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void griddep_wait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void griddep_launch()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// Four packed E2M1 pairs (one 32-bit word) -> four f16x2 (low nibble -> low half).
__device__ __forceinline__ void e2m1x8_to_h2x4(uint32_t w, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3)
{
    asm("{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t"
        "}"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(w));
}

// Two E4M3 bytes (low byte -> low half) -> f16x2.
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t two_bytes)
{
    uint32_t out;
    uint16_t s = static_cast<uint16_t>(two_bytes & 0xffffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(s));
    return out;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi)
{
    uint32_t d;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
    return d;
}

__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi)
{
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
    return d;
}

__device__ __forceinline__ float ex2(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t const* a, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldcg_u4(void const* p)
{
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float2 ldcg_f2(void const* p)
{
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

// bf16x2 -> f16x2 (exact for |x| in the fp16 normal range).
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t x)
{
    float const lo = __uint_as_float(x << 16);
    float const hi = __uint_as_float(x & 0xffff0000u);
    return pack_h2(lo, hi);
}

// Q row fragment: 32 dims [32t, 32t+32) of one row -> 16 f16x2 words, word i = dims 32t+2i, +1.
__device__ __forceinline__ void load_q_row(Params const& p, void const* row_ptr, bool valid, uint32_t (&w)[16])
{
    if (!valid)
    {
#pragma unroll
        for (int i = 0; i < 16; ++i)
            w[i] = 0u;
        return;
    }
    if (p.q_fp8)
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(row_ptr);
        const uint4 a = src[0], b = src[1];
        const uint32_t raw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            w[2 * i] = e4m3x2_to_h2(raw[i] & 0xffffu);
            w[2 * i + 1] = e4m3x2_to_h2(raw[i] >> 16);
        }
    }
    else
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(row_ptr);
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const uint4 a = src[c];
            w[4 * c + 0] = bf2_to_h2(a.x);
            w[4 * c + 1] = bf2_to_h2(a.y);
            w[4 * c + 2] = bf2_to_h2(a.z);
            w[4 * c + 3] = bf2_to_h2(a.w);
        }
    }
}

__device__ __forceinline__ uint32_t cluster_ctarank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_all()
{
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t dsmem_addr(void const* local, uint32_t rank)
{
    uint32_t out;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(smem_u32(local)), "r"(rank));
    return out;
}

__device__ __forceinline__ float4 ld_dsmem_f4(uint32_t addr)
{
    float4 v;
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(addr));
    return v;
}

__device__ __forceinline__ float2 ld_dsmem_f2(uint32_t addr)
{
    float2 v;
    asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr));
    return v;
}

__device__ __forceinline__ void named_bar(int id, int nthreads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

__device__ __forceinline__ int atom_add_acq_rel_gpu(int* ptr, int v)
{
    int old;
    asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(v) : "memory");
    return old;
}

// WPT warps cooperate on one 16-row (query, head) tile: QK is split over the
// page's tokens (each warp kNQ n-tiles of 8 tokens), PV over head_dim (each
// warp kNP n-tiles of 8 dims). The row max and P go through shared memory.
template <int WPT>
struct TileCfg
{
    static constexpr int kNQ = 16 / WPT;  // QK n-tiles (8 tokens) per warp
    static constexpr int kKS = 8 / WPT;   // k-steps (16 tokens) per warp
    static constexpr int kNP = 16 / WPT;  // PV n-tiles (8 dims) per warp
    static constexpr int kDW = 128 / WPT; // head dims per warp
    static constexpr int kBPT = kDW / 16; // packed V bytes per token per thread
};

constexpr int kPfragBytes = 8 * 32 * 16; // one tile's P fragments for one page: 8 k-steps x 32 lanes x uint4

struct SmemLayout
{
    int region, pfrag, red, bars, ints, total;

    __host__ __device__ SmemLayout(int stages, int ntiles, int wpt, int nsplit, int rows)
    {
        region = stages * kStageBytes;
        int const merge = ntiles * 16 * (128 + 2) * 4; // fp32 partial O + (m, l) per row
        if (merge > region)
            region = merge;
        pfrag = region;
        red = pfrag + ntiles * 2 * kPfragBytes;
        bars = red + ntiles * 2 * 16 * wpt * 4;
        bars = (bars + 15) & ~15;
        ints = bars + 2 * stages * 8;
        total = ints + (8 * kMaxCand + 16) * 4;
    }
};

// Issue the TMA loads of one page into one ring stage (2 to 4 bulk copies).
__device__ __forceinline__ void issue_page(Params const& p, int64_t pg, int h, uint8_t* sb, uint64_t* bar)
{
    mbar_arrive_expect_tx(bar, kStageBytes);
    uint8_t const* kp = p.k + pg * p.k_sb + h * p.k_sh;
    uint8_t const* ksp = p.ks + pg * p.ks_sb + h * p.ks_sh;
    if (p.kv_fused)
    {
        bulk_g2s(sb, kp, 2 * kKvBytes, bar);
    }
    else
    {
        bulk_g2s(sb, kp, kKvBytes, bar);
        bulk_g2s(sb + kKvBytes, p.v + pg * p.v_sb + h * p.v_sh, kKvBytes, bar);
    }
    if (p.sc_fused)
    {
        bulk_g2s(sb + 2 * kKvBytes, ksp, 2 * kScBytes, bar);
    }
    else
    {
        bulk_g2s(sb + 2 * kKvBytes, ksp, kScBytes, bar);
        bulk_g2s(sb + 2 * kKvBytes + kScBytes, p.vs + pg * p.vs_sb + h * p.vs_sh, kScBytes, bar);
    }
}

// ------------------------------------------------------------------ kernel
template <int WPT, int kMaxThreads, int kMinBlocks>
__global__ void __launch_bounds__(kMaxThreads, kMinBlocks) sparse_decode_v2_kernel(const __grid_constant__ Params p)
{
    using C = TileCfg<WPT>;
    extern __shared__ __align__(1024) uint8_t smem[];
    int const tid = threadIdx.x;
    int const warp = tid >> 5, lane = tid & 31;
    int const g = lane >> 2, t = lane & 3;
    int const S = p.stages;
    int const nwarps = p.ntiles * WPT;

    int cta = blockIdx.x;
    int const split = cta % p.nsplit;
    cta /= p.nsplit;
    int const grp = cta; // r * num_kvh + h
    int const h = cta % p.num_kvh;
    int const r = cta / p.num_kvh;
    int const rows_pad = p.ntiles * 16;
    int const R = p.dql * p.group;

    const SmemLayout L(S, p.ntiles, WPT, p.nsplit, R);
    uint8_t* stage_base = smem;
    uint4* pfrag = reinterpret_cast<uint4*>(smem + L.pfrag);
    float* red = reinterpret_cast<float*>(smem + L.red);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + L.bars);
    uint64_t* empty = full + S;
    int* s_cand = reinterpret_cast<int*>(smem + L.ints);
    int* s_cpage = s_cand + kMaxCand;
    int* s_ublk = s_cpage + kMaxCand;
    int* s_umask = s_ublk + kMaxCand;
    int* s_upage = s_umask + kMaxCand;
    int* s_cmask = s_upage + kMaxCand;
    int* s_cearly = s_cmask + kMaxCand;
    int* s_misc = s_cearly + kMaxCand;

    if (tid == 0)
    {
        for (int s = 0; s < S; ++s)
        {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], nwarps);
        }
        s_misc[0] = s_misc[1] = 0;
        fence_barrier_init();
    }

    // Tile geometry (independent of any device data).
    int const tile = warp / WPT;
    int const qw = warp - tile * WPT;
    int const gshift = __ffs(p.group) - 1; // group is 8 or 16
    int const f0 = tile * 16 + g, f1 = f0 + 8;
    int const j0 = f0 >> gshift, j1 = f1 >> gshift;
    int const hh0 = f0 & (p.group - 1), hh1 = f1 & (p.group - 1);
    bool const rv0 = j0 < p.dql, rv1 = j1 < p.dql;
    int tmask = 0;
    {
        int const qa = (tile * 16) >> gshift;
        int const qb = min((tile * 16 + 15) >> gshift, p.dql - 1);
        for (int qq = qa; qq <= qb; ++qq)
            tmask |= 1 << qq;
    }

    SDV2_STAMP(0);
    griddep_wait();
    SDV2_STAMP(1);

    int const seq_len = p.seq_lens[r];
    int const ncand = p.dql * p.topk_n;

    // ---- candidates: every (query, top-k slot) entry, with its page prefetched
    if (tid < kMaxCand)
    {
        int v = -1, page = 0;
        if (tid < ncand)
        {
            int j = 0, pp = tid;
            while (pp >= p.topk_n)
            {
                pp -= p.topk_n;
                ++j;
            }
            int const raw = p.topk[(int64_t) h * p.t_sh + (int64_t) (r * p.dql + j) * p.t_sn + (int64_t) pp * p.t_sk];
            bool const in_table = raw >= 0 && raw < p.bt_cols;
            if (in_table)
                page = p.bt[(int64_t) r * p.bt_sb + raw];
            int const kv_len = max(seq_len - p.dql + j + 1, 0);
            int const real = min(p.topk_n, (kv_len + kBlk - 1) / kBlk);
            if (pp < real && in_table)
                v = raw;
            // Warm L2 with this candidate's page before the dedup finishes; each
            // candidate index is prefetched by one split (duplicates are harmless).
            if (v >= 0 && tid < SDV2_PREFETCH_CANDS && tid >= split && (tid - split) % p.nsplit == 0)
            {
                const int64_t pg = page;
                prefetch_l2(p.k + pg * p.k_sb + h * p.k_sh, kKvBytes);
                prefetch_l2(p.v + pg * p.v_sb + h * p.v_sh, kKvBytes);
                prefetch_l2(p.ks + pg * p.ks_sb + h * p.ks_sh, kScBytes);
                prefetch_l2(p.vs + pg * p.vs_sb + h * p.vs_sh, kScBytes);
            }
        }
        s_cand[tid] = v;
        s_cpage[tid] = page;
        s_cmask[tid] = 0;
        s_cearly[tid] = 0;
    }
    SDV2_STAMP(8);

    // Q fragments (overlaps the candidate loads). Dims are permuted so that a
    // thread's K fragment for k-step j is bytes 16t+2j, 16t+2j+1 of a K row.
    uint32_t qf[8][4];
    {
        uint32_t w0[16], w1[16];
        const int64_t tok0 = (int64_t) (r * p.dql + j0) * p.q_sn + (int64_t) (h * p.group + hh0) * p.q_sh + 32 * t;
        const int64_t tok1 = (int64_t) (r * p.dql + j1) * p.q_sn + (int64_t) (h * p.group + hh1) * p.q_sh + 32 * t;
        int const esz = p.q_fp8 ? 1 : 2;
        load_q_row(p, static_cast<uint8_t const*>(p.q) + tok0 * esz, rv0, w0);
        load_q_row(p, static_cast<uint8_t const*>(p.q) + tok1 * esz, rv1, w1);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            qf[j][0] = w0[2 * j];
            qf[j][1] = w1[2 * j];
            qf[j][2] = w0[2 * j + 1];
            qf[j][3] = w1[2 * j + 1];
        }
    }
    float const sl2 = p.sm_scale_log2e * (*p.k_gs);
    float const v_gs = *p.v_gs;
    SDV2_STAMP(9);
    __syncthreads();
    SDV2_STAMP(10);

    // ---- union with per-query membership masks, in first-occurrence order.
    // One thread per (candidate c, list j): is c's page in list j, and does it
    // occur there before position c? (topk_n % 4 == 0 is checked on the host.)
    for (int pr = tid; pr < kMaxCand * p.dql; pr += blockDim.x)
    {
        int const c = pr & (kMaxCand - 1), j = pr >> 6;
        int const v = c < ncand ? s_cand[c] : -1;
        if (v >= 0)
        {
            int const base = j * p.topk_n;
            int4 const* c4 = reinterpret_cast<int4 const*>(s_cand + base);
            bool hit = false, early = false;
            for (int i4 = 0; i4 < (p.topk_n >> 2); ++i4)
            {
                const int4 cc = c4[i4];
                int const i = base + 4 * i4;
                bool const e0 = cc.x == v, e1 = cc.y == v, e2 = cc.z == v, e3 = cc.w == v;
                hit |= e0 | e1 | e2 | e3;
                early |= (e0 && i < c) || (e1 && i + 1 < c) || (e2 && i + 2 < c) || (e3 && i + 3 < c);
            }
            if (hit)
                atomicOr(&s_cmask[c], 1 << j);
            if (early)
                s_cearly[c] = 1;
        }
    }
    __syncthreads();
    int my_v = -1, my_mask = 0;
    bool my_first = false;
    unsigned my_bal = 0;
    if (tid < kMaxCand)
    {
        my_v = s_cand[tid];
        my_first = my_v >= 0 && s_cearly[tid] == 0;
        my_mask = s_cmask[tid];
        my_bal = __ballot_sync(0xffffffffu, my_first);
        if (lane == 0)
            s_misc[warp] = __popc(my_bal);
    }
    SDV2_STAMP(11);
    __syncthreads();
    if (tid < kMaxCand && my_first)
    {
        int const pos = (warp == 1 ? s_misc[0] : 0) + __popc(my_bal & ((1u << lane) - 1u));
        s_ublk[pos] = my_v;
        s_umask[pos] = my_mask;
        s_upage[pos] = s_cpage[tid];
    }
    __syncthreads();
    int const U = s_misc[0] + s_misc[1];
    // Round-robin over the union (query 0's pages first, then the later queries'
    // extra pages) balances both page count and active tiles across splits.
    int const n = split < U ? (U - split + p.nsplit - 1) / p.nsplit : 0;
    SDV2_STAMP(2);
    if (p.dbg != nullptr && tid == 0)
        p.dbg[blockIdx.x * 24 + 16] = clock64();

    // Thread 0 doubles as the TMA producer. Before consuming block i it issues
    // every block j < i + S whose stage is free: it blocks only for j == i (whose
    // stage was released by every warp at block i - S) and polls for the rest.
    bool const producer = tid == 0;
    // Initial fill: lane i of warp 0 issues page i into stage i, in parallel.
    int const nfill = min(n, S);
    if (warp == 0 && lane < nfill)
    {
        issue_page(p, s_upage[split + lane * p.nsplit], h, stage_base + lane * kStageBytes, &full[lane]);
    }
    // Refills (only when a CTA has more pages than stages) are thread 0's job.
    int issued = nfill;                       // next page to issue
    int issue_stage = nfill == S ? 0 : nfill; // its stage
    int iuse = nfill == S ? 1 : 0;            // fills of that stage so far

    int const kvlen0 = rv0 ? max(seq_len - p.dql + j0 + 1, 0) : 0;
    int const kvlen1 = rv1 ? max(seq_len - p.dql + j1 + 1, 0) : 0;
    int const t_odd = t & 1;
    int const g_swz = (g & 1) ^ ((g >> 1) & 1);
    const uint32_t vsel = t_odd ? 0x2301u : 0x3210u;
    int const bar_id = 1 + tile;
    int const bar_n = 32 * WPT;
    // PV: this warp owns dims [qw*kDW, (qw+1)*kDW); thread g owns kDW/8 of them.
    int const v_byte = qw * (C::kDW / 2) + C::kBPT * g;
    int const v_group = (qw * C::kDW + (C::kDW / 8) * g) >> 4;

    float o[C::kNP][4];
#pragma unroll
    for (int c = 0; c < C::kNP; ++c)
        o[c][0] = o[c][1] = o[c][2] = o[c][3] = 0.f;
    float m0 = kNegInf, m1 = kNegInf, l0 = 0.f, l1 = 0.f;
    int used = 0;
    long long c_wait = 0, c_qk = 0, c_sm = 0, c_pv = 0, c_tmp = 0, c_pump = 0, c_n = 0, c_act = 0;
#define SDV2_CLK(acc)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (p.dbg != nullptr)                                                                                          \
        {                                                                                                              \
            const long long now_ = clock64();                                                                          \
            acc += now_ - c_tmp;                                                                                       \
            c_tmp = now_;                                                                                              \
        }                                                                                                              \
    } while (0)

    int st = 0;
    uint32_t ph = 0;
    for (int i = 0; i < n; ++i)
    {
        if (p.dbg != nullptr)
            c_tmp = clock64();
        if (producer)
        {
            while (issued < n && issued < i + S)
            {
                // Every page issued here reuses a stage (the initial fill took the rest).
                const uint32_t par = (iuse - 1) & 1;
                if (issued <= i)
                {
                    mbar_wait(&empty[issue_stage], par);
                }
                else if (!mbar_test(&empty[issue_stage], par))
                {
                    break;
                }
                issue_page(p, s_upage[split + issued * p.nsplit], h, stage_base + issue_stage * kStageBytes,
                    &full[issue_stage]);
                ++issued;
                if (++issue_stage == S)
                {
                    issue_stage = 0;
                    ++iuse;
                }
            }
        }
        SDV2_CLK(c_pump);
        ++c_n;
        mbar_wait(&full[st], ph);
        SDV2_CLK(c_wait);
        int const uidx = split + i * p.nsplit;
        int const bm = s_umask[uidx];
        if (bm & tmask)
        {
            ++c_act;
            int const buf = used & 1;
            ++used;
            uint8_t const* kq = stage_base + st * kStageBytes;
            uint8_t const* vq = kq + kKvBytes;
            uint8_t const* ksc = kq + 2 * kKvBytes;
            uint8_t const* vsc = ksc + kScBytes;
            int const base_tok = s_ublk[uidx] * kBlk;
            int const lim0 = (rv0 && ((bm >> j0) & 1)) ? kvlen0 - base_tok : 0;
            int const lim1 = (rv1 && ((bm >> j1) & 1)) ? kvlen1 - base_tok : 0;
            bool const need_mask = __any_sync(0xffffffffu, lim0 < kBlk || lim1 < kBlk);
            int const vlim = seq_len - base_tok; // tokens at/after seq_len hold no data

            // ---------------- S = Q K^T for this warp's tokens
            float s[C::kNQ][4];
#pragma unroll
            for (int ii = 0; ii < C::kNQ; ++ii)
            {
                int const it = qw * C::kNQ + ii;
                int const tok = 16 * (it >> 1) + 4 * (g >> 1) + 2 * (it & 1) + g_swz;
                const uint4 kd = *reinterpret_cast<uint4 const*>(kq + tok * 64 + 16 * t);
                const uint32_t k2 = *reinterpret_cast<uint16_t const*>(ksc + tok * 8 + 2 * t);
                const uint32_t sc = e4m3x2_to_h2(k2);
                const uint32_t s0 = __byte_perm(sc, 0, 0x1010);
                const uint32_t s1 = __byte_perm(sc, 0, 0x3232);
                uint32_t b[16];
                e2m1x8_to_h2x4(kd.x, b[0], b[1], b[2], b[3]);
                e2m1x8_to_h2x4(kd.y, b[4], b[5], b[6], b[7]);
                e2m1x8_to_h2x4(kd.z, b[8], b[9], b[10], b[11]);
                e2m1x8_to_h2x4(kd.w, b[12], b[13], b[14], b[15]);
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    b[q] = hmul2(b[q], s0);
#pragma unroll
                for (int q = 8; q < 16; ++q)
                    b[q] = hmul2(b[q], s1);
                s[ii][0] = s[ii][1] = s[ii][2] = s[ii][3] = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    mma16816(s[ii], qf[j], b[2 * j], b[2 * j + 1]);
            }

            SDV2_CLK(c_qk);
            // ---------------- softmax: block row max across the tile's warps
#pragma unroll
            for (int ii = 0; ii < C::kNQ; ++ii)
            {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    s[ii][c] *= sl2;
            }
            if (need_mask)
            {
#pragma unroll
                for (int ii = 0; ii < C::kNQ; ++ii)
                {
                    int const it = qw * C::kNQ + ii;
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        int const tok = 16 * (it >> 1) + 4 * t + 2 * (it & 1) + ((c & 1) ^ t_odd);
                        if (tok >= (c < 2 ? lim0 : lim1))
                            s[ii][c] = kNegInf;
                    }
                }
            }
            float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
            for (int ii = 0; ii < C::kNQ; ++ii)
            {
                mx0 = fmaxf(mx0, fmaxf(s[ii][0], s[ii][1]));
                mx1 = fmaxf(mx1, fmaxf(s[ii][2], s[ii][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            float* rb = red + ((tile * 2 + buf) * 16) * WPT;
            if (WPT > 1)
            {
                if (t == 0)
                {
                    rb[g * WPT + qw] = mx0;
                    rb[(g + 8) * WPT + qw] = mx1;
                }
                named_bar(bar_id, bar_n);
#pragma unroll
                for (int w = 0; w < WPT; ++w)
                {
                    mx0 = fmaxf(mx0, rb[g * WPT + w]);
                    mx1 = fmaxf(mx1, rb[(g + 8) * WPT + w]);
                }
            }
            float const mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            float const ms0 = mn0 == kNegInf ? 0.f : mn0;
            float const ms1 = mn1 == kNegInf ? 0.f : mn1;
            float const al0 = ex2(m0 - ms0), al1 = ex2(m1 - ms1);
            m0 = mn0;
            m1 = mn1;
            float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
            for (int ii = 0; ii < C::kNQ; ++ii)
            {
                s[ii][0] = ex2(s[ii][0] - ms0);
                s[ii][1] = ex2(s[ii][1] - ms0);
                s[ii][2] = ex2(s[ii][2] - ms1);
                s[ii][3] = ex2(s[ii][3] - ms1);
                rs0 += s[ii][0] + s[ii][1];
                rs1 += s[ii][2] + s[ii][3];
            }
            l0 = l0 * al0 + rs0;
            l1 = l1 * al1 + rs1;
#pragma unroll
            for (int c = 0; c < C::kNP; ++c)
            {
                o[c][0] *= al0;
                o[c][1] *= al0;
                o[c][2] *= al1;
                o[c][3] *= al1;
            }
            // P fragments of this warp's k-steps -> shared memory, fragment order.
            uint4* pf = pfrag + (tile * 2 + buf) * (8 * 32);
#pragma unroll
            for (int ks = 0; ks < C::kKS; ++ks)
            {
                int const e0 = 2 * ks, e1 = 2 * ks + 1;
                pf[(qw * C::kKS + ks) * 32 + lane] = make_uint4(pack_h2(s[e0][0], s[e0][1]),
                    pack_h2(s[e0][2], s[e0][3]), pack_h2(s[e1][0], s[e1][1]), pack_h2(s[e1][2], s[e1][3]));
            }
            if (WPT > 1)
                named_bar(bar_id, bar_n);
            SDV2_CLK(c_sm);

            // ---------------- O[:, my dims] += P V
#pragma unroll
            for (int sidx = 0; sidx < 8; ++sidx)
            {
                const uint4 pq = pf[sidx * 32 + lane];
                const uint32_t pa[4] = {pq.x, pq.y, pq.z, pq.w};
                int const row_a = 16 * sidx + 4 * t;
                uint8_t const* vrow = vq + v_byte;
                uint32_t vs4 = *reinterpret_cast<uint32_t const*>(vsc + (4 * sidx + t) * 32 + 4 * v_group);
                vs4 = __byte_perm(vs4, 0, vsel);
                if (vlim < kBlk)
                {
                    uint32_t keep = 0;
                    keep |= (row_a + t_odd < vlim) ? 0x000000ffu : 0u;
                    keep |= (row_a + 1 - t_odd < vlim) ? 0x0000ff00u : 0u;
                    keep |= (row_a + 2 + t_odd < vlim) ? 0x00ff0000u : 0u;
                    keep |= (row_a + 3 - t_odd < vlim) ? 0xff000000u : 0u;
                    vs4 &= keep;
                }
                const uint32_t sab = e4m3x2_to_h2(vs4 & 0xffffu);
                const uint32_t scd = e4m3x2_to_h2(vs4 >> 16);
                uint32_t b01[C::kNP], b23[C::kNP];
                if constexpr (C::kBPT == 2)
                {
                    const uint32_t wa = *reinterpret_cast<uint16_t const*>(vrow + (row_a + t_odd) * 64);
                    const uint32_t wb = *reinterpret_cast<uint16_t const*>(vrow + (row_a + 1 - t_odd) * 64);
                    const uint32_t wc = *reinterpret_cast<uint16_t const*>(vrow + (row_a + 2 + t_odd) * 64);
                    const uint32_t wd = *reinterpret_cast<uint16_t const*>(vrow + (row_a + 3 - t_odd) * 64);
                    const uint32_t w1 = wa | (wc << 16), w2 = wb | (wd << 16);
                    const uint32_t x = (w1 & 0x0F0F0F0Fu) | ((w2 << 4) & 0xF0F0F0F0u);
                    const uint32_t y = ((w1 >> 4) & 0x0F0F0F0Fu) | (w2 & 0xF0F0F0F0u);
                    e2m1x8_to_h2x4(x, b01[0], b01[2], b23[0], b23[2]);
                    e2m1x8_to_h2x4(y, b01[1], b01[3], b23[1], b23[3]);
                }
                else
                {
#pragma unroll
                    for (int hw = 0; hw < C::kBPT / 4; ++hw)
                    {
                        const uint32_t wa = *reinterpret_cast<uint32_t const*>(vrow + (row_a + t_odd) * 64 + 4 * hw);
                        const uint32_t wb
                            = *reinterpret_cast<uint32_t const*>(vrow + (row_a + 1 - t_odd) * 64 + 4 * hw);
                        const uint32_t wc
                            = *reinterpret_cast<uint32_t const*>(vrow + (row_a + 2 + t_odd) * 64 + 4 * hw);
                        const uint32_t wd
                            = *reinterpret_cast<uint32_t const*>(vrow + (row_a + 3 - t_odd) * 64 + 4 * hw);
                        const uint32_t x_ab = (wa & 0x0F0F0F0Fu) | ((wb << 4) & 0xF0F0F0F0u);
                        const uint32_t y_ab = ((wa >> 4) & 0x0F0F0F0Fu) | (wb & 0xF0F0F0F0u);
                        const uint32_t x_cd = (wc & 0x0F0F0F0Fu) | ((wd << 4) & 0xF0F0F0F0u);
                        const uint32_t y_cd = ((wc >> 4) & 0x0F0F0F0Fu) | (wd & 0xF0F0F0F0u);
                        int const bb = 8 * hw;
                        e2m1x8_to_h2x4(x_ab, b01[bb + 0], b01[bb + 2], b01[bb + 4], b01[bb + 6]);
                        e2m1x8_to_h2x4(y_ab, b01[bb + 1], b01[bb + 3], b01[bb + 5], b01[bb + 7]);
                        e2m1x8_to_h2x4(x_cd, b23[bb + 0], b23[bb + 2], b23[bb + 4], b23[bb + 6]);
                        e2m1x8_to_h2x4(y_cd, b23[bb + 1], b23[bb + 3], b23[bb + 5], b23[bb + 7]);
                    }
                }
#pragma unroll
                for (int c = 0; c < C::kNP; ++c)
                {
                    b01[c] = hmul2(b01[c], sab);
                    b23[c] = hmul2(b23[c], scd);
                }
#pragma unroll
                for (int c = 0; c < C::kNP; ++c)
                    mma16816(o[c], pa, b01[c], b23[c]);
            }
            if (p.dbg != nullptr)
            {
                // Make the accumulators ready before reading the clock.
                float z = 0.f;
#pragma unroll
                for (int c = 0; c < C::kNP; ++c)
                    z += o[c][0];
                if (z == 12345.f)
                    c_pv += 1;
            }
            SDV2_CLK(c_pv);
        }
        __syncwarp();
        if (lane == 0)
            mbar_arrive(&empty[st]);
        if (++st == S)
        {
            st = 0;
            ph ^= 1;
        }
    }
    SDV2_STAMP(3);
    if (p.dbg != nullptr && tid == 0)
        p.dbg[blockIdx.x * 24 + 17] = clock64();
    if (p.dbg != nullptr && lane == 0 && warp == 0)
    {
        p.dbg[blockIdx.x * 24 + 12] = c_wait;
        p.dbg[blockIdx.x * 24 + 13] = c_qk;
        p.dbg[blockIdx.x * 24 + 14] = c_sm;
        p.dbg[blockIdx.x * 24 + 15] = c_pv;
        p.dbg[blockIdx.x * 24 + 18] = c_pump;
        p.dbg[blockIdx.x * 24 + 19] = c_n;
        p.dbg[blockIdx.x * 24 + 20] = c_act;
    }

    // ---------------- row sums across the tile's warps (m is already shared)
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (WPT > 1)
    {
        float* rb = red + ((tile * 2 + (used & 1)) * 16) * WPT;
        if (t == 0)
        {
            rb[g * WPT + qw] = l0;
            rb[(g + 8) * WPT + qw] = l1;
        }
        named_bar(bar_id, bar_n);
        l0 = 0.f;
        l1 = 0.f;
#pragma unroll
        for (int w = 0; w < WPT; ++w)
        {
            l0 += rb[g * WPT + w];
            l1 += rb[(g + 8) * WPT + w];
        }
    }

    // Thread (g, t) of warp qw holds rows g, g+8, dims qw*kDW + 2t*kNP + [0, 2*kNP).
    int const dim0 = qw * C::kDW + 2 * t * C::kNP;
    if (p.nsplit == 1)
    {
        float const sc0 = l0 > 0.f ? v_gs / l0 : 0.f;
        float const sc1 = l1 > 0.f ? v_gs / l1 : 0.f;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
        {
            bool const rv = rr ? rv1 : rv0;
            if (!rv)
                continue;
            float const sc = rr ? sc1 : sc0;
            uint32_t w[C::kNP];
#pragma unroll
            for (int c = 0; c < C::kNP / 2; ++c)
            {
                w[c] = pack_bf2(o[2 * c][2 * rr] * sc, o[2 * c + 1][2 * rr] * sc);
                w[C::kNP / 2 + c] = pack_bf2(o[2 * c][2 * rr + 1] * sc, o[2 * c + 1][2 * rr + 1] * sc);
            }
            int const jj = rr ? j1 : j0, hh = rr ? hh1 : hh0;
            uint4* dst = reinterpret_cast<uint4*>(
                p.out + (int64_t) (r * p.dql + jj) * p.o_sn + (int64_t) (h * p.group + hh) * p.o_sh + dim0);
#pragma unroll
            for (int c = 0; c < C::kNP / 4; ++c)
                dst[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
        }
        griddep_launch();
        return;
    }

    if (!p.cluster_merge)
    {
        // ---------------- split-K, global flavour: publish an fp16 partial (pre V-scale);
        // the last CTA of the (request, head) to take an acq_rel ticket merges.
        const int64_t part_row0 = ((int64_t) grp * p.nsplit + split) * rows_pad;
        {
            float const sc0 = l0 > 0.f ? 1.f / l0 : 0.f;
            float const sc1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                bool const rv = rr ? rv1 : rv0;
                float const ll = rr ? l1 : l0;
                if (!rv || !(ll > 0.f))
                    continue;
                float const sc = rr ? sc1 : sc0;
                uint32_t w[C::kNP];
#pragma unroll
                for (int c = 0; c < C::kNP / 2; ++c)
                {
                    w[c] = pack_h2(o[2 * c][2 * rr] * sc, o[2 * c + 1][2 * rr] * sc);
                    w[C::kNP / 2 + c] = pack_h2(o[2 * c][2 * rr + 1] * sc, o[2 * c + 1][2 * rr + 1] * sc);
                }
                int const f = rr ? f1 : f0;
                uint4* dst = reinterpret_cast<uint4*>(p.part_o + (part_row0 + f) * 128 + dim0);
#pragma unroll
                for (int c = 0; c < C::kNP / 4; ++c)
                    dst[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
            }
            if (qw == 0 && t == 0)
            {
                if (rv0)
                    p.part_ml[part_row0 + f0] = make_float2(m0, l0);
                if (rv1)
                    p.part_ml[part_row0 + f1] = make_float2(m1, l1);
            }
        }
        __syncthreads();
        SDV2_STAMP(4);
        if (tid == 0)
        {
            // Release publishes this CTA's partial (cumulative over the barrier);
            // acquire makes the other CTAs' partials visible to the last one.
            int const prev = atom_add_acq_rel_gpu(&p.counters[grp], 1);
            int const last = prev == p.nsplit - 1;
            if (last)
                p.counters[grp] = 0; // self-reset for the next launch / graph replay
            s_misc[2] = last;
        }
        __syncthreads();
        SDV2_STAMP(5);
        if (!s_misc[2])
        {
            griddep_launch();
            return;
        }

        // Last CTA: every thread merges exactly the fragment it owns (rows f0/f1,
        // dims dim0..), reading every split's (m, l) and fp16 partial eight splits
        // at a time. All splits, including this CTA's own, go through the published
        // partials in split order, so the result does not depend on arrival order.
        int const ns = p.nsplit;
        constexpr int kMergeChunk = 4; // splits in flight per round (8 spills at 128 regs and is slower)
        float M[2] = {kNegInf, kNegInf}, W[2] = {0.f, 0.f};
        float A[2][2 * C::kNP];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
        {
#pragma unroll
            for (int e = 0; e < 2 * C::kNP; ++e)
                A[rr][e] = 0.f;
        }
        const int64_t mrow0 = (int64_t) grp * ns * rows_pad + f0;
        for (int sp0 = 0; sp0 < ns; sp0 += kMergeChunk)
        {
            float2 ml[2][kMergeChunk];
            uint4 x[2][kMergeChunk][C::kNP / 4];
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
#pragma unroll
                for (int u = 0; u < kMergeChunk; ++u)
                {
                    int const sp = sp0 + u;
                    ml[rr][u] = make_float2(0.f, 0.f);
                    if ((rr ? rv1 : rv0) && sp < ns)
                    {
                        const int64_t row = mrow0 + (int64_t) sp * rows_pad + 8 * rr;
                        ml[rr][u] = p.part_ml[row];
#pragma unroll
                        for (int q = 0; q < C::kNP / 4; ++q)
                            x[rr][u][q] = reinterpret_cast<uint4 const*>(p.part_o + row * 128 + dim0)[q];
                    }
                }
            }
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
#pragma unroll
                for (int u = 0; u < kMergeChunk; ++u)
                {
                    if (ml[rr][u].y > 0.f)
                    {
                        float const mn = fmaxf(M[rr], ml[rr][u].x);
                        float const a = ex2(M[rr] - mn);
                        float const b = ml[rr][u].y * ex2(ml[rr][u].x - mn);
                        W[rr] = W[rr] * a + b;
#pragma unroll
                        for (int q = 0; q < C::kNP / 4; ++q)
                        {
                            const uint32_t xs[4] = {x[rr][u][q].x, x[rr][u][q].y, x[rr][u][q].z, x[rr][u][q].w};
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                            {
                                const float2 fv = __half22float2(*reinterpret_cast<__half2 const*>(&xs[e]));
                                A[rr][8 * q + 2 * e] = A[rr][8 * q + 2 * e] * a + b * fv.x;
                                A[rr][8 * q + 2 * e + 1] = A[rr][8 * q + 2 * e + 1] * a + b * fv.y;
                            }
                        }
                        M[rr] = mn;
                    }
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
        {
            if (!(rr ? rv1 : rv0))
                continue;
            float const sc = W[rr] > 0.f ? v_gs / W[rr] : 0.f;
            uint32_t w[C::kNP];
#pragma unroll
            for (int c = 0; c < C::kNP; ++c)
                w[c] = pack_bf2(A[rr][2 * c] * sc, A[rr][2 * c + 1] * sc);
            int const jj = rr ? j1 : j0, hh = rr ? hh1 : hh0;
            uint4* dst = reinterpret_cast<uint4*>(
                p.out + (int64_t) (r * p.dql + jj) * p.o_sn + (int64_t) (h * p.group + hh) * p.o_sh + dim0);
#pragma unroll
            for (int c = 0; c < C::kNP / 4; ++c)
                dst[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
        }
        SDV2_STAMP(7);
        griddep_launch();
        return;
    }

    // ---------------- split-K over a thread-block cluster: every CTA of the
    // (request, head) parks its fp32 partial in its own shared memory, then
    // merges a 1/nsplit share of the rows by reading all peers through DSMEM.
    // Deterministic (fixed split order), no global scratch, no atomics.
    float* sp_o = reinterpret_cast<float*>(smem);                         // [rows_pad][128]
    float2* sp_ml = reinterpret_cast<float2*>(smem + rows_pad * 128 * 4); // [rows_pad]
    __syncthreads();                                                      // the ring is no longer read by anyone
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
    {
        int const f = rr ? f1 : f0;
        float4* dst = reinterpret_cast<float4*>(sp_o + f * 128 + dim0);
#pragma unroll
        for (int c = 0; c < C::kNP / 4; ++c)
        {
            dst[c] = make_float4(o[4 * c][2 * rr], o[4 * c + 1][2 * rr], o[4 * c + 2][2 * rr], o[4 * c + 3][2 * rr]);
            dst[C::kNP / 4 + c] = make_float4(
                o[4 * c][2 * rr + 1], o[4 * c + 1][2 * rr + 1], o[4 * c + 2][2 * rr + 1], o[4 * c + 3][2 * rr + 1]);
        }
    }
    if (qw == 0 && t == 0)
    {
        sp_ml[f0] = make_float2(l0 > 0.f ? m0 : kNegInf, l0);
        sp_ml[f1] = make_float2(l1 > 0.f ? m1 : kNegInf, l1);
    }
    SDV2_STAMP(4);
    cluster_sync_all();
    SDV2_STAMP(5);
    if (p.dbg != nullptr && tid == 0)
        p.dbg[blockIdx.x * 24 + 21] = clock64();
    {
        int const ns = p.nsplit;
        int const nitems = R * 16; // (row, 8-dim chunk)
        int const per = (nitems + ns - 1) / ns;
        for (int k = tid; k < per && split * per + k < nitems; k += blockDim.x)
        {
            int const item = split * per + k;
            int const f = item >> 4, ch = item & 15;
            const uint32_t o_loc = smem_u32(sp_o + f * 128 + ch * 8);
            const uint32_t ml_loc = smem_u32(sp_ml + f);
            float M = kNegInf, W = 0.f;
            float A[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int q0 = 0; q0 < ns; q0 += 4)
            {
                float2 ml[4];
                float4 xa[4], xb[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    if (q0 + u < ns)
                    {
                        uint32_t a_o, a_ml;
                        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a_o) : "r"(o_loc), "r"(q0 + u));
                        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a_ml) : "r"(ml_loc), "r"(q0 + u));
                        ml[u] = ld_dsmem_f2(a_ml);
                        xa[u] = ld_dsmem_f4(a_o);
                        xb[u] = ld_dsmem_f4(a_o + 16);
                    }
                    else
                    {
                        ml[u] = make_float2(kNegInf, 0.f);
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    if (ml[u].y > 0.f)
                    {
                        float const mn = fmaxf(M, ml[u].x);
                        float const a = ex2(M - mn), b = ex2(ml[u].x - mn);
                        W = W * a + ml[u].y * b;
                        A[0] = A[0] * a + xa[u].x * b;
                        A[1] = A[1] * a + xa[u].y * b;
                        A[2] = A[2] * a + xa[u].z * b;
                        A[3] = A[3] * a + xa[u].w * b;
                        A[4] = A[4] * a + xb[u].x * b;
                        A[5] = A[5] * a + xb[u].y * b;
                        A[6] = A[6] * a + xb[u].z * b;
                        A[7] = A[7] * a + xb[u].w * b;
                        M = mn;
                    }
                }
            }
            float const sc = W > 0.f ? v_gs / W : 0.f;
            int const jq = f >> gshift, hq = f & (p.group - 1);
            uint4* dst = reinterpret_cast<uint4*>(
                p.out + (int64_t) (r * p.dql + jq) * p.o_sn + (int64_t) (h * p.group + hq) * p.o_sh + ch * 8);
            *dst = make_uint4(pack_bf2(A[0] * sc, A[1] * sc), pack_bf2(A[2] * sc, A[3] * sc),
                pack_bf2(A[4] * sc, A[5] * sc), pack_bf2(A[6] * sc, A[7] * sc));
        }
    }
    SDV2_STAMP(6);
    if (p.dbg != nullptr && tid == 0)
        p.dbg[blockIdx.x * 24 + 22] = clock64();
    // Peers may still be reading this CTA's shared memory.
    cluster_sync_all();
    SDV2_STAMP(7);
    griddep_launch();
}

// =================================================================== tcgen05 (UMMA) variant
// One M=64 UMMA covers every (query, head) row of the CTA, so each page is
// dequantized once into 128B-swizzled K-major shared memory (K as [token][dim],
// V transposed as [dim][token]) and shared by all queries. S = Q K^T and
// O += P V accumulate in TMEM (columns 0-127 and 128-255); 16 warps run the
// dequant and the softmax, thread 0 issues the MMAs.
namespace umma
{
__device__ __forceinline__ int sw128(int r, int k, int rows)
{ // byte offset of fp16 (r, k), K = 128
    int const kk = k & 63;
    return (k >> 6) * (rows * 128) + (r >> 3) * 1024 + (r & 7) * 128 + ((((kk >> 3)) ^ (r & 7)) << 4) + ((kk & 7) << 1);
}

__device__ __forceinline__ uint64_t desc(uint32_t saddr)
{
    return (uint64_t) ((saddr >> 4) & 0x3FFF) | ((uint64_t) 1 << 16) | ((uint64_t) (1024 >> 4) << 32)
        | ((uint64_t) 1 << 46) | ((uint64_t) 2 << 61);
}

__device__ __forceinline__ void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(acc));
}

__device__ __forceinline__ void commit(uint64_t* bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
                 : "memory");
}

__device__ __forceinline__ void fence_before()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_after()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void ld16(uint32_t taddr, float (&v)[16])
{
    uint32_t r[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
        v[i] = __uint_as_float(r[i]);
}

__device__ __forceinline__ void st16(uint32_t taddr, float const (&v)[16])
{
    asm volatile(
        "tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
            taddr),
        "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
        "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])), "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])),
        "r"(__float_as_uint(v[8])), "r"(__float_as_uint(v[9])), "r"(__float_as_uint(v[10])),
        "r"(__float_as_uint(v[11])), "r"(__float_as_uint(v[12])), "r"(__float_as_uint(v[13])),
        "r"(__float_as_uint(v[14])), "r"(__float_as_uint(v[15]))
        : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

constexpr int kRows = 64;
constexpr uint32_t kIdesc
    = (1u << 4) | ((128u >> 3) << 17) | ((uint32_t) (kRows >> 4) << 24); // f32 += f16*f16, M64 N128

struct Layout
{
    int ring, qs, kd, vd, ps, red, bars, ints, total;

    __host__ __device__ Layout(int stages)
    {
        ring = stages * kStageBytes;
        int const epi = kRows * 128 * 4 + kRows * 8; // fp32 O + (m, l) staging for the epilogue
        if (epi > ring)
            ring = epi;
        ring = (ring + 1023) & ~1023;
        qs = ring;              // Q [64][128] fp16, SW128 K-major
        kd = qs + kRows * 256;  // K [128 tok][128 dim]
        vd = kd + 128 * 256;    // V^T [128 dim][128 tok]
        ps = vd + 128 * 256;    // P [64][128 tok]
        red = ps + kRows * 256; // [2][64][4] row max + [64][4] row sum
        bars = red + 3 * kRows * 4 * 4;
        ints = bars + (2 * stages + 2) * 8 + 16;
        total = ints + (8 * kMaxCand + 16) * 4 + 1024; // + alignment slack
    }
};
} // namespace umma

__global__ void __launch_bounds__(512, 1) sparse_decode_umma_kernel(const __grid_constant__ Params p)
{
    extern __shared__ uint8_t smem_raw[];
    // 1024-B alignment for the swizzled UMMA operands.
    uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    int const tid = threadIdx.x;
    int const warp = tid >> 5, lane = tid & 31;
    int const g = lane >> 2, t = lane & 3;
    int const S = p.stages;
    constexpr int kWarps = 16;

    int cta = blockIdx.x;
    int const split = cta % p.nsplit;
    cta /= p.nsplit;
    int const grp = cta;
    int const h = cta % p.num_kvh;
    int const r = cta / p.num_kvh;
    int const R = p.dql * p.group;

    const umma::Layout L(S);
    uint8_t* stage_base = smem;
    uint8_t* Qs = smem + L.qs;
    uint8_t* Kd = smem + L.kd;
    uint8_t* Vd = smem + L.vd;
    uint8_t* Ps = smem + L.ps;
    float* red = reinterpret_cast<float*>(smem + L.red);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + L.bars);
    uint64_t* empty = full + S;
    uint64_t* bar_s = empty + S;
    uint64_t* bar_o = bar_s + 1;
    uint32_t* s_tmem = reinterpret_cast<uint32_t*>(bar_o + 1);
    int* s_cand = reinterpret_cast<int*>(smem + L.ints);
    int* s_cpage = s_cand + kMaxCand;
    int* s_ublk = s_cpage + kMaxCand;
    int* s_umask = s_ublk + kMaxCand;
    int* s_upage = s_umask + kMaxCand;
    int* s_cmask = s_upage + kMaxCand;
    int* s_cearly = s_cmask + kMaxCand;
    int* s_misc = s_cearly + kMaxCand;

    if (tid == 0)
    {
        for (int s = 0; s < S; ++s)
        {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], kWarps);
        }
        mbar_init(bar_s, 1);
        mbar_init(bar_o, 1);
        s_misc[0] = s_misc[1] = 0;
        fence_barrier_init();
    }
    if (warp == 0)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(smem_u32(s_tmem)));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    // Rows of this thread in the M=64 TMEM layout: quadrant q = warp % 4 holds
    // rows 16q..16q+15; column group cg = warp / 4 holds columns 32cg..32cg+31.
    int const gshift = __ffs(p.group) - 1;
    int const q = warp & 3, cg = warp >> 2;
    int const f0 = 16 * q + g, f1 = f0 + 8;
    int const j0 = f0 >> gshift, j1 = f1 >> gshift;
    bool const rv0 = j0 < p.dql, rv1 = j1 < p.dql;

    SDV2_STAMP(0);
    griddep_wait();
    SDV2_STAMP(1);

    int const seq_len = p.seq_lens[r];
    int const ncand = p.dql * p.topk_n;
    if (tid < kMaxCand)
    {
        int v = -1, page = 0;
        if (tid < ncand)
        {
            int j = 0, pp = tid;
            while (pp >= p.topk_n)
            {
                pp -= p.topk_n;
                ++j;
            }
            int const raw = p.topk[(int64_t) h * p.t_sh + (int64_t) (r * p.dql + j) * p.t_sn + (int64_t) pp * p.t_sk];
            bool const in_table = raw >= 0 && raw < p.bt_cols;
            if (in_table)
                page = p.bt[(int64_t) r * p.bt_sb + raw];
            int const kv_len = max(seq_len - p.dql + j + 1, 0);
            int const real = min(p.topk_n, (kv_len + kBlk - 1) / kBlk);
            if (pp < real && in_table)
                v = raw;
        }
        s_cand[tid] = v;
        s_cpage[tid] = page;
        s_cmask[tid] = 0;
        s_cearly[tid] = 0;
    }
    // Q -> fp16 SW128 tile (rows >= dql*group are zero).
    for (int e = tid; e < umma::kRows * 16; e += blockDim.x)
    {
        int const f = e >> 4, ch = e & 15;
        int const jq = f >> gshift, hq = f & (p.group - 1);
        uint32_t w[4] = {0u, 0u, 0u, 0u};
        if (jq < p.dql)
        {
            const int64_t off = (int64_t) (r * p.dql + jq) * p.q_sn + (int64_t) (h * p.group + hq) * p.q_sh + ch * 8;
            if (p.q_fp8)
            {
                const uint2 x = *reinterpret_cast<uint2 const*>(static_cast<uint8_t const*>(p.q) + off);
                w[0] = e4m3x2_to_h2(x.x & 0xffffu);
                w[1] = e4m3x2_to_h2(x.x >> 16);
                w[2] = e4m3x2_to_h2(x.y & 0xffffu);
                w[3] = e4m3x2_to_h2(x.y >> 16);
            }
            else
            {
                const uint4 x = *reinterpret_cast<uint4 const*>(static_cast<uint8_t const*>(p.q) + off * 2);
                w[0] = bf2_to_h2(x.x);
                w[1] = bf2_to_h2(x.y);
                w[2] = bf2_to_h2(x.z);
                w[3] = bf2_to_h2(x.w);
            }
        }
        *reinterpret_cast<uint4*>(Qs + umma::sw128(f, ch * 8, umma::kRows)) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    float const sl2 = p.sm_scale_log2e * (*p.k_gs);
    float const v_gs = *p.v_gs;
    umma::fence_proxy_async();
    umma::fence_before();
    __syncthreads();
    umma::fence_after();
    const uint32_t tmem = *s_tmem;
    const uint32_t tmem_s = tmem, tmem_o = tmem + 128;

    for (int pr = tid; pr < kMaxCand * p.dql; pr += blockDim.x)
    {
        int const c = pr & (kMaxCand - 1), j = pr >> 6;
        int const v = c < ncand ? s_cand[c] : -1;
        if (v >= 0)
        {
            int const base = j * p.topk_n;
            int4 const* c4 = reinterpret_cast<int4 const*>(s_cand + base);
            bool hit = false, early = false;
            for (int i4 = 0; i4 < (p.topk_n >> 2); ++i4)
            {
                const int4 cc = c4[i4];
                int const i = base + 4 * i4;
                bool const e0 = cc.x == v, e1 = cc.y == v, e2 = cc.z == v, e3 = cc.w == v;
                hit |= e0 | e1 | e2 | e3;
                early |= (e0 && i < c) || (e1 && i + 1 < c) || (e2 && i + 2 < c) || (e3 && i + 3 < c);
            }
            if (hit)
                atomicOr(&s_cmask[c], 1 << j);
            if (early)
                s_cearly[c] = 1;
        }
    }
    __syncthreads();
    int my_v = -1, my_mask = 0;
    bool my_first = false;
    unsigned my_bal = 0;
    if (tid < kMaxCand)
    {
        my_v = s_cand[tid];
        my_first = my_v >= 0 && s_cearly[tid] == 0;
        my_mask = s_cmask[tid];
        my_bal = __ballot_sync(0xffffffffu, my_first);
        if (lane == 0)
            s_misc[warp] = __popc(my_bal);
    }
    __syncthreads();
    if (tid < kMaxCand && my_first)
    {
        int const pos = (warp == 1 ? s_misc[0] : 0) + __popc(my_bal & ((1u << lane) - 1u));
        s_ublk[pos] = my_v;
        s_umask[pos] = my_mask;
        s_upage[pos] = s_cpage[tid];
    }
    __syncthreads();
    int const U = s_misc[0] + s_misc[1];
    int const n = split < U ? (U - split + p.nsplit - 1) / p.nsplit : 0;
    SDV2_STAMP(2);

    int const nfill = min(n, S);
    if (warp == 0 && lane < nfill)
    {
        issue_page(p, s_upage[split + lane * p.nsplit], h, stage_base + lane * kStageBytes, &full[lane]);
    }
    int issued = nfill, issue_stage = nfill == S ? 0 : nfill, iuse = nfill == S ? 1 : 0;

    int const kvlen0 = rv0 ? max(seq_len - p.dql + j0 + 1, 0) : 0;
    int const kvlen1 = rv1 ? max(seq_len - p.dql + j1 + 1, 0) : 0;
    // K dequant role: token kn, dims [32 kq, 32 kq + 32).
    int const kn = tid >> 2, kq = tid & 3;
    // V dequant role: tokens [8 vo, 8 vo + 8), packed word vw (dims 8 vw..8 vw + 7), nibble half vh.
    int const vo = tid >> 5, vw = (tid >> 1) & 15, vh = tid & 1;

    float m0 = kNegInf, m1 = kNegInf, l0 = 0.f, l1 = 0.f;
    long long cc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    long long ct = 0;
#define UCLK(k)                                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (p.dbg != nullptr)                                                                                          \
        {                                                                                                              \
            const long long now_ = clock64();                                                                          \
            cc[k] += now_ - ct;                                                                                        \
            ct = now_;                                                                                                 \
        }                                                                                                              \
    } while (0)
    int st = 0;
    uint32_t ph = 0;
    for (int i = 0; i < n; ++i)
    {
        if (tid == 0)
        {
            while (issued < n && issued < i + S)
            {
                const uint32_t par = (iuse - 1) & 1;
                if (issued <= i)
                {
                    mbar_wait(&empty[issue_stage], par);
                }
                else if (!mbar_test(&empty[issue_stage], par))
                {
                    break;
                }
                issue_page(p, s_upage[split + issued * p.nsplit], h, stage_base + issue_stage * kStageBytes,
                    &full[issue_stage]);
                ++issued;
                if (++issue_stage == S)
                {
                    issue_stage = 0;
                    ++iuse;
                }
            }
        }
        if (p.dbg != nullptr)
            ct = clock64();
        mbar_wait(&full[st], ph);
        UCLK(0);
        int const uidx = split + i * p.nsplit;
        int const bm = s_umask[uidx];
        int const base_tok = s_ublk[uidx] * kBlk;
        uint8_t const* kq_p = stage_base + st * kStageBytes;
        uint8_t const* vq_p = kq_p + kKvBytes;
        uint8_t const* ksc = kq_p + 2 * kKvBytes;
        uint8_t const* vsc = ksc + kScBytes;
        int const vlim = seq_len - base_tok;

        // ---- K -> Kd (S(i-1) was consumed by every thread before the previous barrier)
        {
            const uint4 kd = *reinterpret_cast<uint4 const*>(kq_p + kn * 64 + 16 * kq);
            uint32_t k2 = *reinterpret_cast<uint16_t const*>(ksc + kn * 8 + 2 * kq);
            if (kn >= vlim)
                k2 = 0; // never-written slots: keep K finite
            const uint32_t sc = e4m3x2_to_h2(k2);
            const uint32_t s0 = __byte_perm(sc, 0, 0x1010), s1 = __byte_perm(sc, 0, 0x3232);
            uint32_t b[16];
            e2m1x8_to_h2x4(kd.x, b[0], b[1], b[2], b[3]);
            e2m1x8_to_h2x4(kd.y, b[4], b[5], b[6], b[7]);
            e2m1x8_to_h2x4(kd.z, b[8], b[9], b[10], b[11]);
            e2m1x8_to_h2x4(kd.w, b[12], b[13], b[14], b[15]);
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const uint32_t sc_c = c < 2 ? s0 : s1;
                *reinterpret_cast<uint4*>(Kd + umma::sw128(kn, 32 * kq + 8 * c, 128))
                    = make_uint4(hmul2(b[4 * c], sc_c), hmul2(b[4 * c + 1], sc_c), hmul2(b[4 * c + 2], sc_c),
                        hmul2(b[4 * c + 3], sc_c));
            }
        }
        umma::fence_proxy_async();
        umma::fence_before();
        __syncthreads();
        if (tid == 0)
        {
            umma::fence_after();
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const uint64_t a = umma::desc(smem_u32(Qs) + (j >> 2) * (umma::kRows * 128) + (j & 3) * 32);
                const uint64_t bd = umma::desc(smem_u32(Kd) + (j >> 2) * (128 * 128) + (j & 3) * 32);
                umma::mma(tmem_s, a, bd, umma::kIdesc, j > 0);
            }
            umma::commit(bar_s);
        }
        UCLK(1);

        // ---- V -> Vd (transposed); PV(i-1) must be done reading Vd and Ps.
        if (i > 0)
            mbar_wait(bar_o, (i - 1) & 1);
        UCLK(2);
        {
            uint32_t w[8];
#pragma unroll
            for (int k = 0; k < 8; ++k)
                w[k] = *reinterpret_cast<uint32_t const*>(vq_p + (8 * vo + k) * 64 + 4 * vw);
            int const gq = vw >> 1;
            uint32_t sl = *reinterpret_cast<uint32_t const*>(vsc + (2 * vo) * 32 + 4 * gq);
            uint32_t sh = *reinterpret_cast<uint32_t const*>(vsc + (2 * vo + 1) * 32 + 4 * gq);
            if (8 * vo + 8 > vlim)
            {
                uint32_t keep_l = 0, keep_h = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    keep_l |= (8 * vo + k < vlim) ? (0xffu << (8 * k)) : 0u;
                    keep_h |= (8 * vo + 4 + k < vlim) ? (0xffu << (8 * k)) : 0u;
                }
                sl &= keep_l;
                sh &= keep_h;
            }
            const uint32_t sp[4] = {
                e4m3x2_to_h2(sl & 0xffffu), e4m3x2_to_h2(sl >> 16), e4m3x2_to_h2(sh & 0xffffu), e4m3x2_to_h2(sh >> 16)};
            uint32_t out[4][4]; // [dim m][token pair]
#pragma unroll
            for (int pr2 = 0; pr2 < 4; ++pr2)
            {
                const uint32_t wa = w[2 * pr2], wb = w[2 * pr2 + 1];
                const uint32_t x = vh ? (((wa >> 4) & 0x0F0F0F0Fu) | (wb & 0xF0F0F0F0u))
                                      : ((wa & 0x0F0F0F0Fu) | ((wb << 4) & 0xF0F0F0F0u));
                uint32_t c0, c1, c2, c3;
                e2m1x8_to_h2x4(x, c0, c1, c2, c3);
                out[0][pr2] = hmul2(c0, sp[pr2]);
                out[1][pr2] = hmul2(c1, sp[pr2]);
                out[2][pr2] = hmul2(c2, sp[pr2]);
                out[3][pr2] = hmul2(c3, sp[pr2]);
            }
#pragma unroll
            for (int m = 0; m < 4; ++m)
            {
                int const dim = 8 * vw + 2 * m + vh;
                *reinterpret_cast<uint4*>(Vd + umma::sw128(dim, 8 * vo, 128))
                    = make_uint4(out[m][0], out[m][1], out[m][2], out[m][3]);
            }
        }
        umma::fence_proxy_async();
        // The packed page is fully consumed.
        __syncwarp();
        if (lane == 0)
            mbar_arrive(&empty[st]);
        if (++st == S)
        {
            st = 0;
            ph ^= 1;
        }

        UCLK(3);
        // ---- softmax on S (16x256b fragments: rows f0/f1, cols 32cg + 8c + 2t + {0,1})
        mbar_wait(bar_s, i & 1);
        umma::fence_after();
        UCLK(4);
        float s[16];
        umma::ld16(tmem_s + ((uint32_t) (32 * q) << 16) + 32 * cg, s);
        int const lim0 = (rv0 && ((bm >> j0) & 1)) ? kvlen0 - base_tok : 0;
        int const lim1 = (rv1 && ((bm >> j1) & 1)) ? kvlen1 - base_tok : 0;
        float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            int const col = 32 * cg + 8 * c + 2 * t;
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                float a = s[4 * c + e] * sl2, b2 = s[4 * c + 2 + e] * sl2;
                a = (col + e < lim0) ? a : kNegInf;
                b2 = (col + e < lim1) ? b2 : kNegInf;
                s[4 * c + e] = a;
                s[4 * c + 2 + e] = b2;
                mx0 = fmaxf(mx0, a);
                mx1 = fmaxf(mx1, b2);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float* rb = red + (i & 1) * (umma::kRows * 4);
        if (t == 0)
        {
            rb[f0 * 4 + cg] = mx0;
            rb[f1 * 4 + cg] = mx1;
        }
        named_bar(1 + q, 128);
        {
            const float4 a = *reinterpret_cast<float4 const*>(rb + f0 * 4);
            const float4 b2 = *reinterpret_cast<float4 const*>(rb + f1 * 4);
            mx0 = fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
            mx1 = fmaxf(fmaxf(b2.x, b2.y), fmaxf(b2.z, b2.w));
        }
        float const mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        float const ms0 = mn0 == kNegInf ? 0.f : mn0, ms1 = mn1 == kNegInf ? 0.f : mn1;
        float const al0 = ex2(m0 - ms0), al1 = ex2(m1 - ms1);
        m0 = mn0;
        m1 = mn1;
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                s[4 * c + e] = ex2(s[4 * c + e] - ms0);
                s[4 * c + 2 + e] = ex2(s[4 * c + 2 + e] - ms1);
                rs0 += s[4 * c + e];
                rs1 += s[4 * c + 2 + e];
            }
        }
        l0 = l0 * al0 + rs0;
        l1 = l1 * al1 + rs1;
        UCLK(5);
        // O *= alpha (O holds PV(0..i-1), complete since bar_o was waited above).
        if (i > 0 && __any_sync(0xffffffffu, al0 != 1.f || al1 != 1.f))
        {
            float o[16];
            const uint32_t oaddr = tmem_o + ((uint32_t) (32 * q) << 16) + 32 * cg;
            umma::ld16(oaddr, o);
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                o[4 * c] *= al0;
                o[4 * c + 1] *= al0;
                o[4 * c + 2] *= al1;
                o[4 * c + 3] *= al1;
            }
            umma::st16(oaddr, o);
        }
        UCLK(6);
        // P -> Ps (A operand of PV).
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            int const col = 32 * cg + 8 * c + 2 * t;
            *reinterpret_cast<uint32_t*>(Ps + umma::sw128(f0, col, umma::kRows)) = pack_h2(s[4 * c], s[4 * c + 1]);
            *reinterpret_cast<uint32_t*>(Ps + umma::sw128(f1, col, umma::kRows)) = pack_h2(s[4 * c + 2], s[4 * c + 3]);
        }
        umma::fence_proxy_async();
        umma::fence_before();
        __syncthreads();
        if (tid == 0)
        {
            umma::fence_after();
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const uint64_t a = umma::desc(smem_u32(Ps) + (j >> 2) * (umma::kRows * 128) + (j & 3) * 32);
                const uint64_t bd = umma::desc(smem_u32(Vd) + (j >> 2) * (128 * 128) + (j & 3) * 32);
                umma::mma(tmem_o, a, bd, umma::kIdesc, (i > 0 || j > 0) ? 1u : 0u);
            }
            umma::commit(bar_o);
        }
        UCLK(7);
    }
    SDV2_STAMP(3);
    if (p.dbg != nullptr && tid == 0)
    {
        for (int k = 0; k < 8; ++k)
            p.dbg[blockIdx.x * 24 + 12 + k] = cc[k];
        p.dbg[blockIdx.x * 24 + 20] = n;
    }

    // ---------------- epilogue: O (TMEM) -> fp32 staging in shared memory
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    float* rl = red + 2 * umma::kRows * 4;
    if (t == 0)
    {
        rl[f0 * 4 + cg] = l0;
        rl[f1 * 4 + cg] = l1;
    }
    float* so = reinterpret_cast<float*>(smem);                            // [64][128] fp32 (ring is free)
    float2* sml = reinterpret_cast<float2*>(smem + umma::kRows * 128 * 4); // [64] (m, l)
    if (n > 0)
        mbar_wait(bar_o, (n - 1) & 1);
    umma::fence_after();
    __syncthreads(); // ring no longer read; row sums published
    if (n > 0)
    {
        float o[16];
        umma::ld16(tmem_o + ((uint32_t) (32 * q) << 16) + 32 * cg, o);
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            int const col = 32 * cg + 8 * c + 2 * t;
            *reinterpret_cast<float2*>(so + f0 * 128 + col) = make_float2(o[4 * c], o[4 * c + 1]);
            *reinterpret_cast<float2*>(so + f1 * 128 + col) = make_float2(o[4 * c + 2], o[4 * c + 3]);
        }
    }
    if (cg == 0 && t == 0)
    {
        const float4 a = *reinterpret_cast<float4 const*>(rl + f0 * 4);
        const float4 b2 = *reinterpret_cast<float4 const*>(rl + f1 * 4);
        float const L0 = a.x + a.y + a.z + a.w, L1 = b2.x + b2.y + b2.z + b2.w;
        sml[f0] = make_float2(L0 > 0.f ? m0 : kNegInf, L0);
        sml[f1] = make_float2(L1 > 0.f ? m1 : kNegInf, L1);
    }
    umma::fence_before();
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem));

    int const rows_pad = umma::kRows;
    const int64_t part_row0 = ((int64_t) grp * p.nsplit + split) * rows_pad;
    for (int it = tid; it < R * 16; it += blockDim.x)
    {
        int const f = it >> 4, ch = it & 15;
        const float2 ml = sml[f];
        const float4 a = *reinterpret_cast<float4 const*>(so + f * 128 + ch * 8);
        const float4 b2 = *reinterpret_cast<float4 const*>(so + f * 128 + ch * 8 + 4);
        int const jq = f >> gshift, hq = f & (p.group - 1);
        if (p.nsplit == 1)
        {
            uint4* dst = reinterpret_cast<uint4*>(
                p.out + (int64_t) (r * p.dql + jq) * p.o_sn + (int64_t) (h * p.group + hq) * p.o_sh + ch * 8);
            if (n > 0 && ml.y > 0.f)
            {
                float const sc = v_gs / ml.y;
                *dst = make_uint4(pack_bf2(a.x * sc, a.y * sc), pack_bf2(a.z * sc, a.w * sc),
                    pack_bf2(b2.x * sc, b2.y * sc), pack_bf2(b2.z * sc, b2.w * sc));
            }
            else
            {
                *dst = make_uint4(0u, 0u, 0u, 0u); // empty row (zero-length / CUDA-graph padding): 0, never NaN
            }
        }
        else
        {
            if (n > 0 && ml.y > 0.f)
            {
                float const sc = 1.f / ml.y;
                *reinterpret_cast<uint4*>(p.part_o + (part_row0 + f) * 128 + ch * 8)
                    = make_uint4(pack_h2(a.x * sc, a.y * sc), pack_h2(a.z * sc, a.w * sc),
                        pack_h2(b2.x * sc, b2.y * sc), pack_h2(b2.z * sc, b2.w * sc));
            }
            if (ch == 0)
                p.part_ml[part_row0 + f] = n > 0 ? ml : make_float2(kNegInf, 0.f);
        }
    }
    if (p.nsplit == 1)
    {
        griddep_launch();
        return;
    }
    __syncthreads();
    SDV2_STAMP(4);
    if (tid == 0)
    {
        int const prev = atom_add_acq_rel_gpu(&p.counters[grp], 1);
        int const last = prev == p.nsplit - 1;
        if (last)
            p.counters[grp] = 0;
        s_misc[2] = last;
    }
    __syncthreads();
    SDV2_STAMP(5);
    if (!s_misc[2])
    {
        griddep_launch();
        return;
    }
    // Last CTA: each thread merges its (row, 8-dim chunk) items over all splits.
    int const ns = p.nsplit;
    for (int it = tid; it < R * 16; it += blockDim.x)
    {
        int const f = it >> 4, ch = it & 15;
        float M = kNegInf, W = 0.f;
        float A[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int sp0 = 0; sp0 < ns; sp0 += 8)
        {
            float2 ml[8];
            uint4 x[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
                ml[u] = make_float2(0.f, 0.f);
                if (sp0 + u < ns)
                {
                    const int64_t row = ((int64_t) grp * ns + sp0 + u) * rows_pad + f;
                    ml[u] = p.part_ml[row];
                    x[u] = *reinterpret_cast<uint4 const*>(p.part_o + row * 128 + ch * 8);
                }
            }
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
                if (ml[u].y > 0.f)
                {
                    float const mn = fmaxf(M, ml[u].x);
                    float const al = ex2(M - mn), b = ml[u].y * ex2(ml[u].x - mn);
                    W = W * al + b;
                    const uint32_t xs[4] = {x[u].x, x[u].y, x[u].z, x[u].w};
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const float2 fv = __half22float2(*reinterpret_cast<__half2 const*>(&xs[e]));
                        A[2 * e] = A[2 * e] * al + b * fv.x;
                        A[2 * e + 1] = A[2 * e + 1] * al + b * fv.y;
                    }
                    M = mn;
                }
            }
        }
        float const sc = W > 0.f ? v_gs / W : 0.f;
        int const jq = f >> gshift, hq = f & (p.group - 1);
        uint4* dst = reinterpret_cast<uint4*>(
            p.out + (int64_t) (r * p.dql + jq) * p.o_sn + (int64_t) (h * p.group + hq) * p.o_sh + ch * 8);
        *dst = make_uint4(pack_bf2(A[0] * sc, A[1] * sc), pack_bf2(A[2] * sc, A[3] * sc),
            pack_bf2(A[4] * sc, A[5] * sc), pack_bf2(A[6] * sc, A[7] * sc));
    }
    SDV2_STAMP(7);
    griddep_launch();
}

void launch_umma(Params const& p, int grid, int smem, bool pdl, cudaStream_t stream)
{
    static int configured = 0;
    if (smem > configured)
    {
        C10_CUDA_CHECK(
            cudaFuncSetAttribute(sparse_decode_umma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = smem;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(512);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, sparse_decode_umma_kernel, p));
}

// =================================================================== tcgen05, warp-specialized
// Warp 0: TMA producer + UMMA issuer (lane 0). Warps 1-7: dequantize K and V
// of page i into double-buffered UMMA operands. Warps 8-15 (two per TMEM lane
// quadrant = query tile): softmax of S[i & 1], lazy O rescale, P -> smem.
// mbarriers hand pages between roles so page i+1's dequant and QK overlap
// page i's softmax and PV.
namespace umma
{
struct Layout2
{
    int ring, qs, kd, vd, ps, red, bars, ints, total;

    __host__ __device__ Layout2(int stages)
    {
        ring = stages * kStageBytes;
        int const epi = kRows * 128 * 4 + kRows * 8;
        if (epi > ring)
            ring = epi;
        ring = (ring + 1023) & ~1023;
        qs = ring;
        kd = qs + kRows * 256;   // 2 x [128 tok][128 dim]
        vd = kd + 2 * 128 * 256; // 2 x [128 dim][128 tok]
        ps = vd + 2 * 128 * 256; // 2 x [64][128 tok]
        red = ps + 2 * kRows * 256;
        bars = red + 3 * kRows * 2 * 4;
        ints = bars + (2 * stages + 16) * 8 + 16;
        total = ints + (8 * kMaxCand + 16) * 4 + 1024;
    }
};

__device__ __forceinline__ void ld32(uint32_t taddr, float (&v)[32])
{
    uint32_t r[32];
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
        "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 32; ++i)
        v[i] = __uint_as_float(r[i]);
}

__device__ __forceinline__ void st32(uint32_t taddr, float const (&v)[32])
{
    asm volatile(
        "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
        "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
        "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
        "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])), "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])),
        "r"(__float_as_uint(v[8])), "r"(__float_as_uint(v[9])), "r"(__float_as_uint(v[10])),
        "r"(__float_as_uint(v[11])), "r"(__float_as_uint(v[12])), "r"(__float_as_uint(v[13])),
        "r"(__float_as_uint(v[14])), "r"(__float_as_uint(v[15])), "r"(__float_as_uint(v[16])),
        "r"(__float_as_uint(v[17])), "r"(__float_as_uint(v[18])), "r"(__float_as_uint(v[19])),
        "r"(__float_as_uint(v[20])), "r"(__float_as_uint(v[21])), "r"(__float_as_uint(v[22])),
        "r"(__float_as_uint(v[23])), "r"(__float_as_uint(v[24])), "r"(__float_as_uint(v[25])),
        "r"(__float_as_uint(v[26])), "r"(__float_as_uint(v[27])), "r"(__float_as_uint(v[28])),
        "r"(__float_as_uint(v[29])), "r"(__float_as_uint(v[30])), "r"(__float_as_uint(v[31]))
        : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

constexpr float kRescaleThreshold = 8.f; // log2 units: P stays <= 256, exact in fp16 math
} // namespace umma

__global__ void __launch_bounds__(512, 1) sparse_decode_umma2_kernel(const __grid_constant__ Params p)
{
    extern __shared__ uint8_t smem_raw[];
    // 1024-B alignment for the swizzled UMMA operands.
    uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    int const tid = threadIdx.x;
    int const warp = tid >> 5, lane = tid & 31;
    int const g = lane >> 2, t = lane & 3;
    int const S = p.stages;
    constexpr int kWarps = 16;

    int cta = blockIdx.x;
    int const split = cta % p.nsplit;
    cta /= p.nsplit;
    int const grp = cta;
    int const h = cta % p.num_kvh;
    int const r = cta / p.num_kvh;
    int const R = p.dql * p.group;

    const umma::Layout2 L(S);
    uint8_t* stage_base = smem;
    uint8_t* Qs = smem + L.qs;
    uint8_t* Kd = smem + L.kd;
    uint8_t* Vd = smem + L.vd;
    uint8_t* Ps = smem + L.ps;
    float* red = reinterpret_cast<float*>(smem + L.red);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + L.bars);
    uint64_t* empty = full + S;
    uint64_t* kd_full = empty + S;     // [2] dequant -> MMA
    uint64_t* vd_full = kd_full + 2;   // [2]
    uint64_t* kd_empty = vd_full + 2;  // [2] QK done -> dequant
    uint64_t* vd_empty = kd_empty + 2; // [2] PV done -> dequant / softmax (Vd and Ps free)
    uint64_t* s_full = vd_empty + 2;   // [2] QK done -> softmax
    uint64_t* s_empty = s_full + 2;    // [2] softmax read S -> MMA
    uint64_t* p_full = s_empty + 2;    // [2] softmax wrote P (+ rescaled O) -> MMA
    uint64_t* o_done = p_full + 2;     // PV(i) done -> softmax (O rescale) / epilogue
    uint32_t* s_tmem = reinterpret_cast<uint32_t*>(o_done + 1);
    int* s_cand = reinterpret_cast<int*>(smem + L.ints);
    int* s_cpage = s_cand + kMaxCand;
    int* s_ublk = s_cpage + kMaxCand;
    int* s_umask = s_ublk + kMaxCand;
    int* s_upage = s_umask + kMaxCand;
    int* s_cmask = s_upage + kMaxCand;
    int* s_cearly = s_cmask + kMaxCand;
    int* s_misc = s_cearly + kMaxCand;

    if (tid == 0)
    {
        for (int s = 0; s < S; ++s)
        {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 7);
        }
        for (int b = 0; b < 2; ++b)
        {
            mbar_init(&kd_full[b], 7);
            mbar_init(&vd_full[b], 7);
            mbar_init(&kd_empty[b], 1);
            mbar_init(&vd_empty[b], 1);
            mbar_init(&s_full[b], 1);
            mbar_init(&s_empty[b], 8);
            mbar_init(&p_full[b], 8);
        }
        mbar_init(o_done, 1);
        s_misc[0] = s_misc[1] = 0;
        fence_barrier_init();
    }
    if (warp == 0)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(smem_u32(s_tmem)));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    // Rows of this thread in the M=64 TMEM layout: quadrant q = warp % 4 holds
    // rows 16q..16q+15; column group cg = warp / 4 holds columns 32cg..32cg+31.
    int const gshift = __ffs(p.group) - 1;
    int const q = warp & 3, cg = warp >> 2;
    int const f0 = 16 * q + g, f1 = f0 + 8;
    int const j0 = f0 >> gshift, j1 = f1 >> gshift;
    bool const rv0 = j0 < p.dql, rv1 = j1 < p.dql;

    SDV2_STAMP(0);
    griddep_wait();
    SDV2_STAMP(1);

    int const seq_len = p.seq_lens[r];
    int const ncand = p.dql * p.topk_n;
    if (tid < kMaxCand)
    {
        int v = -1, page = 0;
        if (tid < ncand)
        {
            int j = 0, pp = tid;
            while (pp >= p.topk_n)
            {
                pp -= p.topk_n;
                ++j;
            }
            int const raw = p.topk[(int64_t) h * p.t_sh + (int64_t) (r * p.dql + j) * p.t_sn + (int64_t) pp * p.t_sk];
            bool const in_table = raw >= 0 && raw < p.bt_cols;
            if (in_table)
                page = p.bt[(int64_t) r * p.bt_sb + raw];
            int const kv_len = max(seq_len - p.dql + j + 1, 0);
            int const real = min(p.topk_n, (kv_len + kBlk - 1) / kBlk);
            if (pp < real && in_table)
                v = raw;
        }
        s_cand[tid] = v;
        s_cpage[tid] = page;
        s_cmask[tid] = 0;
        s_cearly[tid] = 0;
    }
    // Q -> fp16 SW128 tile (rows >= dql*group are zero).
    for (int e = tid; e < umma::kRows * 16; e += blockDim.x)
    {
        int const f = e >> 4, ch = e & 15;
        int const jq = f >> gshift, hq = f & (p.group - 1);
        uint32_t w[4] = {0u, 0u, 0u, 0u};
        if (jq < p.dql)
        {
            const int64_t off = (int64_t) (r * p.dql + jq) * p.q_sn + (int64_t) (h * p.group + hq) * p.q_sh + ch * 8;
            if (p.q_fp8)
            {
                const uint2 x = *reinterpret_cast<uint2 const*>(static_cast<uint8_t const*>(p.q) + off);
                w[0] = e4m3x2_to_h2(x.x & 0xffffu);
                w[1] = e4m3x2_to_h2(x.x >> 16);
                w[2] = e4m3x2_to_h2(x.y & 0xffffu);
                w[3] = e4m3x2_to_h2(x.y >> 16);
            }
            else
            {
                const uint4 x = *reinterpret_cast<uint4 const*>(static_cast<uint8_t const*>(p.q) + off * 2);
                w[0] = bf2_to_h2(x.x);
                w[1] = bf2_to_h2(x.y);
                w[2] = bf2_to_h2(x.z);
                w[3] = bf2_to_h2(x.w);
            }
        }
        *reinterpret_cast<uint4*>(Qs + umma::sw128(f, ch * 8, umma::kRows)) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    float const sl2 = p.sm_scale_log2e * (*p.k_gs);
    float const v_gs = *p.v_gs;
    umma::fence_proxy_async();
    umma::fence_before();
    __syncthreads();
    umma::fence_after();
    const uint32_t tmem = *s_tmem;
    const uint32_t tmem_o = tmem + 256; // S[b] at columns 128 b

    for (int pr = tid; pr < kMaxCand * p.dql; pr += blockDim.x)
    {
        int const c = pr & (kMaxCand - 1), j = pr >> 6;
        int const v = c < ncand ? s_cand[c] : -1;
        if (v >= 0)
        {
            int const base = j * p.topk_n;
            int4 const* c4 = reinterpret_cast<int4 const*>(s_cand + base);
            bool hit = false, early = false;
            for (int i4 = 0; i4 < (p.topk_n >> 2); ++i4)
            {
                const int4 cc = c4[i4];
                int const i = base + 4 * i4;
                bool const e0 = cc.x == v, e1 = cc.y == v, e2 = cc.z == v, e3 = cc.w == v;
                hit |= e0 | e1 | e2 | e3;
                early |= (e0 && i < c) || (e1 && i + 1 < c) || (e2 && i + 2 < c) || (e3 && i + 3 < c);
            }
            if (hit)
                atomicOr(&s_cmask[c], 1 << j);
            if (early)
                s_cearly[c] = 1;
        }
    }
    __syncthreads();
    int my_v = -1, my_mask = 0;
    bool my_first = false;
    unsigned my_bal = 0;
    if (tid < kMaxCand)
    {
        my_v = s_cand[tid];
        my_first = my_v >= 0 && s_cearly[tid] == 0;
        my_mask = s_cmask[tid];
        my_bal = __ballot_sync(0xffffffffu, my_first);
        if (lane == 0)
            s_misc[warp] = __popc(my_bal);
    }
    __syncthreads();
    if (tid < kMaxCand && my_first)
    {
        int const pos = (warp == 1 ? s_misc[0] : 0) + __popc(my_bal & ((1u << lane) - 1u));
        s_ublk[pos] = my_v;
        s_umask[pos] = my_mask;
        s_upage[pos] = s_cpage[tid];
    }
    __syncthreads();
    int const U = s_misc[0] + s_misc[1];
    int const n = split < U ? (U - split + p.nsplit - 1) / p.nsplit : 0;
    SDV2_STAMP(2);

    int const nfill = min(n, S);
    if (warp == 0 && lane < nfill)
    {
        issue_page(p, s_upage[split + lane * p.nsplit], h, stage_base + lane * kStageBytes, &full[lane]);
    }
    float m0 = kNegInf, m1 = kNegInf, l0 = 0.f, l1 = 0.f; // softmax warps: reference max and row sum
    long long rc[6] = {0, 0, 0, 0, 0, 0};
    long long rt = 0;
    bool const rec = p.dbg != nullptr && lane == 0 && (warp == 1 || warp == 8);
#define RCLK(k)                                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (rec)                                                                                                       \
        {                                                                                                              \
            const long long now_ = clock64();                                                                          \
            rc[k] += now_ - rt;                                                                                        \
            rt = now_;                                                                                                 \
        }                                                                                                              \
    } while (0)

    if (warp == 0)
    {
        // ================= producer / MMA issuer
        if (lane == 0)
        {
            int issued = nfill, issue_stage = nfill == S ? 0 : nfill, iuse = nfill == S ? 1 : 0;
            auto pump = [&](int need) { // issue pages < need (blocking), then opportunistically up to the ring size
                while (issued < n)
                {
                    const uint32_t par = (iuse - 1) & 1;
                    if (issued < need)
                    {
                        mbar_wait(&empty[issue_stage], par);
                    }
                    else if (!mbar_test(&empty[issue_stage], par))
                    {
                        break;
                    }
                    issue_page(p, s_upage[split + issued * p.nsplit], h, stage_base + issue_stage * kStageBytes,
                        &full[issue_stage]);
                    ++issued;
                    if (++issue_stage == S)
                    {
                        issue_stage = 0;
                        ++iuse;
                    }
                }
            };
            auto issue_pv = [&](int k)
            {
                const int b = k & 1;
                mbar_wait(&p_full[b], (k >> 1) & 1);
                mbar_wait(&vd_full[b], (k >> 1) & 1);
                umma::fence_after();
                uint8_t* pb = Ps + b * (umma::kRows * 256);
                uint8_t* vb = Vd + b * (128 * 256);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    const uint64_t a = umma::desc(smem_u32(pb) + (j >> 2) * (umma::kRows * 128) + (j & 3) * 32);
                    const uint64_t bd = umma::desc(smem_u32(vb) + (j >> 2) * (128 * 128) + (j & 3) * 32);
                    umma::mma(tmem_o, a, bd, umma::kIdesc, (k > 0 || j > 0) ? 1u : 0u);
                }
                umma::commit(&vd_empty[b]);
                umma::commit(o_done);
            };
            for (int i = 0; i < n; ++i)
            {
                int const b = i & 1;
                pump(i + 2); // dequant runs up to one page ahead
                mbar_wait(&kd_full[b], (i >> 1) & 1);
                if (i >= 2)
                    mbar_wait(&s_empty[b], ((i >> 1) - 1) & 1);
                umma::fence_after();
                uint8_t* kb = Kd + b * (128 * 256);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    const uint64_t a = umma::desc(smem_u32(Qs) + (j >> 2) * (umma::kRows * 128) + (j & 3) * 32);
                    const uint64_t bd = umma::desc(smem_u32(kb) + (j >> 2) * (128 * 128) + (j & 3) * 32);
                    umma::mma(tmem + 128 * b, a, bd, umma::kIdesc, j > 0 ? 1u : 0u);
                }
                umma::commit(&s_full[b]);
                umma::commit(&kd_empty[b]);
                if (i >= 1)
                    issue_pv(i - 1);
            }
            if (n >= 1)
                issue_pv(n - 1);
        }
        __syncwarp();
    }
    else if (warp < 8)
    {
        // ================= dequant warps (224 threads)
        int const dt = tid - 32;
        int st = 0;
        uint32_t ph = 0;
        for (int i = 0; i < n; ++i)
        {
            int const b = i & 1;
            if (rec)
                rt = clock64();
            mbar_wait(&full[st], ph);
            RCLK(0);
            int const base_tok = s_ublk[split + i * p.nsplit] * kBlk;
            int const vlim = seq_len - base_tok;
            uint8_t const* kq_p = stage_base + st * kStageBytes;
            uint8_t const* vq_p = kq_p + kKvBytes;
            uint8_t const* ksc = kq_p + 2 * kKvBytes;
            uint8_t const* vsc = ksc + kScBytes;
            if (i >= 2)
                mbar_wait(&kd_empty[b], ((i >> 1) - 1) & 1);
            RCLK(1);
            uint8_t* kb = Kd + b * (128 * 256);
#pragma unroll
            for (int rep = 0; rep < 3; ++rep)
            {
                int const u = dt + 224 * rep; // 512 units: token kn, dims [32 kq, 32 kq + 32)
                if (u < 512)
                {
                    int const kn = u >> 2, kq = u & 3;
                    const uint4 kd4 = *reinterpret_cast<uint4 const*>(kq_p + kn * 64 + 16 * kq);
                    uint32_t k2 = *reinterpret_cast<uint16_t const*>(ksc + kn * 8 + 2 * kq);
                    if (kn >= vlim)
                        k2 = 0; // never-written slots: keep K finite
                    const uint32_t sc = e4m3x2_to_h2(k2);
                    const uint32_t s0 = __byte_perm(sc, 0, 0x1010), s1 = __byte_perm(sc, 0, 0x3232);
                    const uint32_t ws[4] = {kd4.x, kd4.y, kd4.z, kd4.w};
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        uint32_t c0, c1, c2, c3;
                        e2m1x8_to_h2x4(ws[c], c0, c1, c2, c3);
                        const uint32_t sc_c = c < 2 ? s0 : s1;
                        *reinterpret_cast<uint4*>(kb + umma::sw128(kn, 32 * kq + 8 * c, 128))
                            = make_uint4(hmul2(c0, sc_c), hmul2(c1, sc_c), hmul2(c2, sc_c), hmul2(c3, sc_c));
                    }
                }
            }
            umma::fence_proxy_async();
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&kd_full[b]);
            RCLK(2);
            if (i >= 2)
                mbar_wait(&vd_empty[b], ((i >> 1) - 1) & 1);
            RCLK(3);
            uint8_t* vb = Vd + b * (128 * 256);
#pragma unroll
            for (int rep = 0; rep < 3; ++rep)
            {
                int const u = dt + 224 * rep;
                if (u >= 512)
                    break;
                int const vo = u >> 5, vw = (u >> 1) & 15, vh = u & 1;
                uint32_t w8[8];
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    w8[k] = *reinterpret_cast<uint32_t const*>(vq_p + (8 * vo + k) * 64 + 4 * vw);
                int const gq = vw >> 1;
                uint32_t sl = *reinterpret_cast<uint32_t const*>(vsc + (2 * vo) * 32 + 4 * gq);
                uint32_t sh = *reinterpret_cast<uint32_t const*>(vsc + (2 * vo + 1) * 32 + 4 * gq);
                if (8 * vo + 8 > vlim)
                {
                    uint32_t keep_l = 0, keep_h = 0;
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                    {
                        keep_l |= (8 * vo + k < vlim) ? (0xffu << (8 * k)) : 0u;
                        keep_h |= (8 * vo + 4 + k < vlim) ? (0xffu << (8 * k)) : 0u;
                    }
                    sl &= keep_l;
                    sh &= keep_h;
                }
                const uint32_t sp[4] = {e4m3x2_to_h2(sl & 0xffffu), e4m3x2_to_h2(sl >> 16), e4m3x2_to_h2(sh & 0xffffu),
                    e4m3x2_to_h2(sh >> 16)};
                uint32_t out[4][4];
#pragma unroll
                for (int pr2 = 0; pr2 < 4; ++pr2)
                {
                    const uint32_t wa = w8[2 * pr2], wb = w8[2 * pr2 + 1];
                    const uint32_t x = vh ? (((wa >> 4) & 0x0F0F0F0Fu) | (wb & 0xF0F0F0F0u))
                                          : ((wa & 0x0F0F0F0Fu) | ((wb << 4) & 0xF0F0F0F0u));
                    uint32_t c0, c1, c2, c3;
                    e2m1x8_to_h2x4(x, c0, c1, c2, c3);
                    out[0][pr2] = hmul2(c0, sp[pr2]);
                    out[1][pr2] = hmul2(c1, sp[pr2]);
                    out[2][pr2] = hmul2(c2, sp[pr2]);
                    out[3][pr2] = hmul2(c3, sp[pr2]);
                }
#pragma unroll
                for (int m = 0; m < 4; ++m)
                {
                    *reinterpret_cast<uint4*>(vb + umma::sw128(8 * vw + 2 * m + vh, 8 * vo, 128))
                        = make_uint4(out[m][0], out[m][1], out[m][2], out[m][3]);
                }
            }
            umma::fence_proxy_async();
            __syncwarp();
            if (lane == 0)
            {
                mbar_arrive(&vd_full[b]);
                mbar_arrive(&empty[st]);
            }
            RCLK(4);
            if (++st == S)
            {
                st = 0;
                ph ^= 1;
            }
        }
    }
    else
    {
        // ================= softmax warps: quadrant q (query tile), column half hh
        int const hh = (warp - 8) >> 2;
        int const kvlen0 = rv0 ? max(seq_len - p.dql + j0 + 1, 0) : 0;
        int const kvlen1 = rv1 ? max(seq_len - p.dql + j1 + 1, 0) : 0;
        const uint32_t lane_off = (uint32_t) (32 * q) << 16;
        for (int i = 0; i < n; ++i)
        {
            int const b = i & 1;
            int const uidx = split + i * p.nsplit;
            int const bm = s_umask[uidx];
            int const base_tok = s_ublk[uidx] * kBlk;
            if (rec)
                rt = clock64();
            mbar_wait(&s_full[b], (i >> 1) & 1);
            RCLK(0);
            umma::fence_after();
            float s[32];
            umma::ld32(tmem + 128 * b + lane_off + 64 * hh, s);
            umma::fence_before();
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&s_empty[b]);
            int const lim0 = (rv0 && ((bm >> j0) & 1)) ? kvlen0 - base_tok : 0;
            int const lim1 = (rv1 && ((bm >> j1) & 1)) ? kvlen1 - base_tok : 0;
            float mx0 = kNegInf, mx1 = kNegInf;
            if (__any_sync(0xffffffffu, lim0 < kBlk || lim1 < kBlk))
            {
#pragma unroll
                for (int c = 0; c < 8; ++c)
                {
                    int const col = 64 * hh + 8 * c + 2 * t;
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        if (col + e >= lim0)
                            s[4 * c + e] = kNegInf;
                        if (col + e >= lim1)
                            s[4 * c + 2 + e] = kNegInf;
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c)
            {
                mx0 = fmaxf(mx0, fmaxf(s[4 * c], s[4 * c + 1]));
                mx1 = fmaxf(mx1, fmaxf(s[4 * c + 2], s[4 * c + 3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            float* rb = red + b * (umma::kRows * 2);
            if (t == 0)
            {
                rb[f0 * 2 + hh] = mx0;
                rb[f1 * 2 + hh] = mx1;
            }
            named_bar(1 + q, 64);
            mx0 = fmaxf(rb[f0 * 2], rb[f0 * 2 + 1]) * sl2; // sl2 > 0: max commutes with the scale
            mx1 = fmaxf(rb[f1 * 2], rb[f1 * 2 + 1]) * sl2;
            // Lazy rescale: move the reference max only when it grows by > 2^8.
            bool const up0 = mx0 > m0 + umma::kRescaleThreshold, up1 = mx1 > m1 + umma::kRescaleThreshold;
            float const al0 = up0 ? ex2(m0 - mx0) : 1.f, al1 = up1 ? ex2(m1 - mx1) : 1.f;
            bool const o_rescale = __any_sync(0xffffffffu, (up0 && m0 != kNegInf) || (up1 && m1 != kNegInf));
            if (up0)
                m0 = mx0;
            if (up1)
                m1 = mx1;
            float const ms0 = m0 == kNegInf ? 0.f : m0, ms1 = m1 == kNegInf ? 0.f : m1;
            float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
            for (int c = 0; c < 8; ++c)
            {
#pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    s[4 * c + e] = ex2(fmaf(s[4 * c + e], sl2, -ms0));
                    s[4 * c + 2 + e] = ex2(fmaf(s[4 * c + 2 + e], sl2, -ms1));
                    rs0 += s[4 * c + e];
                    rs1 += s[4 * c + 2 + e];
                }
            }
            l0 = l0 * al0 + rs0;
            l1 = l1 * al1 + rs1;
            RCLK(1);
            if (o_rescale && i > 0)
            {
                mbar_wait(o_done, (i - 1) & 1); // O holds exactly PV(0..i-1)
                umma::fence_after();
                float o[32];
                const uint32_t oaddr = tmem_o + lane_off + 64 * hh;
                umma::ld32(oaddr, o);
#pragma unroll
                for (int c = 0; c < 8; ++c)
                {
                    o[4 * c] *= al0;
                    o[4 * c + 1] *= al0;
                    o[4 * c + 2] *= al1;
                    o[4 * c + 3] *= al1;
                }
                umma::st32(oaddr, o);
            }
            RCLK(2);
            // Ps[b] is free once PV(i-2) completed.
            if (i >= 2)
                mbar_wait(&vd_empty[b], ((i >> 1) - 1) & 1);
            RCLK(3);
            uint8_t* pb = Ps + b * (umma::kRows * 256);
#pragma unroll
            for (int c = 0; c < 8; ++c)
            {
                int const col = 64 * hh + 8 * c + 2 * t;
                *reinterpret_cast<uint32_t*>(pb + umma::sw128(f0, col, umma::kRows)) = pack_h2(s[4 * c], s[4 * c + 1]);
                *reinterpret_cast<uint32_t*>(pb + umma::sw128(f1, col, umma::kRows))
                    = pack_h2(s[4 * c + 2], s[4 * c + 3]);
            }
            umma::fence_proxy_async();
            umma::fence_before();
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&p_full[b]);
            RCLK(4);
        }
    }
    SDV2_STAMP(3);
    if (rec)
    {
        int const base = blockIdx.x * 24 + (warp == 1 ? 12 : 18);
        for (int k = 0; k < 5; ++k)
            p.dbg[base + k] = rc[k];
    }
    if (p.dbg != nullptr && tid == 0)
        p.dbg[blockIdx.x * 24 + 17] = n;

    // ---------------- epilogue: softmax warps stage O and row sums in shared memory
    float* rl = red + 2 * umma::kRows * 2; // [64][2]
    float* so = reinterpret_cast<float*>(smem);
    float2* sml = reinterpret_cast<float2*>(smem + umma::kRows * 128 * 4);
    if (warp >= 8)
    {
        int const hh = (warp - 8) >> 2;
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        if (t == 0)
        {
            rl[f0 * 2 + hh] = l0;
            rl[f1 * 2 + hh] = l1;
        }
        if (n > 0)
        {
            mbar_wait(o_done, (n - 1) & 1);
            umma::fence_after();
            float o[32];
            umma::ld32(tmem_o + ((uint32_t) (32 * q) << 16) + 64 * hh, o);
#pragma unroll
            for (int c = 0; c < 8; ++c)
            {
                int const col = 64 * hh + 8 * c + 2 * t;
                *reinterpret_cast<float2*>(so + f0 * 128 + col) = make_float2(o[4 * c], o[4 * c + 1]);
                *reinterpret_cast<float2*>(so + f1 * 128 + col) = make_float2(o[4 * c + 2], o[4 * c + 3]);
            }
        }
        named_bar(1 + q, 64);
        if (hh == 0 && t == 0)
        {
            float const L0 = rl[f0 * 2] + rl[f0 * 2 + 1], L1 = rl[f1 * 2] + rl[f1 * 2 + 1];
            sml[f0] = make_float2(L0 > 0.f ? m0 : kNegInf, L0);
            sml[f1] = make_float2(L1 > 0.f ? m1 : kNegInf, L1);
        }
    }
    umma::fence_before();
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));

    int const rows_pad = umma::kRows;
    const int64_t part_row0 = ((int64_t) grp * p.nsplit + split) * rows_pad;
    for (int it = tid; it < R * 16; it += blockDim.x)
    {
        int const f = it >> 4, ch = it & 15;
        const float2 ml = sml[f];
        const float4 a = *reinterpret_cast<float4 const*>(so + f * 128 + ch * 8);
        const float4 b2 = *reinterpret_cast<float4 const*>(so + f * 128 + ch * 8 + 4);
        int const jq = f >> gshift, hq = f & (p.group - 1);
        if (p.nsplit == 1)
        {
            uint4* dst = reinterpret_cast<uint4*>(
                p.out + (int64_t) (r * p.dql + jq) * p.o_sn + (int64_t) (h * p.group + hq) * p.o_sh + ch * 8);
            if (n > 0 && ml.y > 0.f)
            {
                float const sc = v_gs / ml.y;
                *dst = make_uint4(pack_bf2(a.x * sc, a.y * sc), pack_bf2(a.z * sc, a.w * sc),
                    pack_bf2(b2.x * sc, b2.y * sc), pack_bf2(b2.z * sc, b2.w * sc));
            }
            else
            {
                *dst = make_uint4(0u, 0u, 0u, 0u); // empty row (zero-length / CUDA-graph padding): 0, never NaN
            }
        }
        else
        {
            if (n > 0 && ml.y > 0.f)
            {
                float const sc = 1.f / ml.y;
                *reinterpret_cast<uint4*>(p.part_o + (part_row0 + f) * 128 + ch * 8)
                    = make_uint4(pack_h2(a.x * sc, a.y * sc), pack_h2(a.z * sc, a.w * sc),
                        pack_h2(b2.x * sc, b2.y * sc), pack_h2(b2.z * sc, b2.w * sc));
            }
            if (ch == 0)
                p.part_ml[part_row0 + f] = n > 0 ? ml : make_float2(kNegInf, 0.f);
        }
    }
    if (p.nsplit == 1)
    {
        griddep_launch();
        return;
    }
    __syncthreads();
    SDV2_STAMP(4);
    if (tid == 0)
    {
        int const prev = atom_add_acq_rel_gpu(&p.counters[grp], 1);
        int const last = prev == p.nsplit - 1;
        if (last)
            p.counters[grp] = 0;
        s_misc[2] = last;
    }
    __syncthreads();
    SDV2_STAMP(5);
    if (!s_misc[2])
    {
        griddep_launch();
        return;
    }
    // Last CTA: each thread merges its (row, 8-dim chunk) items over all splits.
    int const ns = p.nsplit;
    for (int it = tid; it < R * 16; it += blockDim.x)
    {
        int const f = it >> 4, ch = it & 15;
        float M = kNegInf, W = 0.f;
        float A[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int sp0 = 0; sp0 < ns; sp0 += 8)
        {
            float2 ml[8];
            uint4 x[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
                ml[u] = make_float2(0.f, 0.f);
                if (sp0 + u < ns)
                {
                    const int64_t row = ((int64_t) grp * ns + sp0 + u) * rows_pad + f;
                    ml[u] = p.part_ml[row];
                    x[u] = *reinterpret_cast<uint4 const*>(p.part_o + row * 128 + ch * 8);
                }
            }
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
                if (ml[u].y > 0.f)
                {
                    float const mn = fmaxf(M, ml[u].x);
                    float const al = ex2(M - mn), b = ml[u].y * ex2(ml[u].x - mn);
                    W = W * al + b;
                    const uint32_t xs[4] = {x[u].x, x[u].y, x[u].z, x[u].w};
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const float2 fv = __half22float2(*reinterpret_cast<__half2 const*>(&xs[e]));
                        A[2 * e] = A[2 * e] * al + b * fv.x;
                        A[2 * e + 1] = A[2 * e + 1] * al + b * fv.y;
                    }
                    M = mn;
                }
            }
        }
        float const sc = W > 0.f ? v_gs / W : 0.f;
        int const jq = f >> gshift, hq = f & (p.group - 1);
        uint4* dst = reinterpret_cast<uint4*>(
            p.out + (int64_t) (r * p.dql + jq) * p.o_sn + (int64_t) (h * p.group + hq) * p.o_sh + ch * 8);
        *dst = make_uint4(pack_bf2(A[0] * sc, A[1] * sc), pack_bf2(A[2] * sc, A[3] * sc),
            pack_bf2(A[4] * sc, A[5] * sc), pack_bf2(A[6] * sc, A[7] * sc));
    }
    SDV2_STAMP(7);
    griddep_launch();
}

void launch_umma2(Params const& p, int grid, int smem, bool pdl, cudaStream_t stream)
{
    static int configured = 0;
    if (smem > configured)
    {
        C10_CUDA_CHECK(
            cudaFuncSetAttribute(sparse_decode_umma2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = smem;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(512);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, sparse_decode_umma2_kernel, p));
}

// =================================================================== stream-K schedule
// The one-CTA-per-(request, KV head) grid leaves SMs idle whenever the groups do not fill whole waves
// (26 requests x 4 KV heads = 104 groups on 152 SMs), and every static split trades that against a
// wave tail. Stream-K instead lays the union pages of all groups end to end and gives each of a
// fixed number of persistent CTAs (one per SM) an equal contiguous share. A CTA walks its share
// group by group ("segments"): a segment that covers a whole group writes the output directly,
// otherwise it publishes a partial that the group's last segment to finish merges, in segment
// order, exactly as the split-K path merges splits. The per-group unions come from a small prep
// kernel (same candidate / first-occurrence code as the one-shot kernel), so the work per page and
// the per-page math are unchanged.

constexpr int kMaxSkGroups = 1024;
// Pages (and segments) one stream-K CTA can hold. With at most 2 groups per CTA of the grid and a
// share of at most 2 x 64 pages x 24 / 9 per CTA, a share has at most ~342 pages.
constexpr int kSkMaxPages = 384;
// Work model for balancing (stream-K timeline at 26 requests x 4 KV heads: a group's head pages, selected
// by ~3.5 query tiles, cost ~2.55 us, its tail pages, ~1.3 tiles, ~1.25 us): ~0.5 us per page plus ~0.6 us
// per query tile that selected it, i.e. weight 4 + 5 x tiles.
#ifndef SDV2_SK_PAGE_BASE
#define SDV2_SK_PAGE_BASE 4
#endif
#ifndef SDV2_SK_PER_TILE
#define SDV2_SK_PER_TILE 5
#endif
constexpr int kSkPageBase = SDV2_SK_PAGE_BASE;
constexpr int kSkPerTile = SDV2_SK_PER_TILE;
constexpr int kSkMaxPageWeight = kSkPageBase + kSkPerTile * 4; // dql <= 4

// Prep: one CTA per (request, KV head) group builds its union (first-occurrence order, per-query
// membership masks) into p.u_blk / u_page / u_mask, and its size into p.u_cnt. A group with no
// valid page gets zero output rows here, since no stream-K segment will cover it.
__global__ void __launch_bounds__(128) sparse_decode_v2_union_kernel(const __grid_constant__ Params p)
{
    __shared__ int s_cand[kMaxCand], s_cpage[kMaxCand], s_cmask[kMaxCand], s_cearly[kMaxCand], s_misc[4];
    int const tid = threadIdx.x;
    int const warp = tid >> 5, lane = tid & 31;
    int const grp = blockIdx.x;
    int const h = grp % p.num_kvh;
    int const r = grp / p.num_kvh;
    if (tid < 4)
        s_misc[tid] = 0;
    // Let the stream-K kernel launch now: its CTAs set up and then wait (griddepcontrol.wait) for this
    // grid to complete. Every CTA of this small grid is resident at once, so they cannot be starved.
    griddep_launch();
    griddep_wait();
    int const seq_len = p.seq_lens[r];
    int const ncand = p.dql * p.topk_n;
    if (tid < kMaxCand)
    {
        int v = -1, page = 0;
        if (tid < ncand)
        {
            int j = 0, pp = tid;
            while (pp >= p.topk_n)
            {
                pp -= p.topk_n;
                ++j;
            }
            int const raw = p.topk[(int64_t) h * p.t_sh + (int64_t) (r * p.dql + j) * p.t_sn + (int64_t) pp * p.t_sk];
            bool const in_table = raw >= 0 && raw < p.bt_cols;
            if (in_table)
                page = p.bt[(int64_t) r * p.bt_sb + raw];
            int const kv_len = max(seq_len - p.dql + j + 1, 0);
            int const real = min(p.topk_n, (kv_len + kBlk - 1) / kBlk);
            if (pp < real && in_table)
                v = raw;
            if (v >= 0 && tid < SDV2_PREFETCH_CANDS)
            {
                // Warm L2 with the page before the stream-K kernel streams it (duplicates are harmless).
                const int64_t pg = page;
                prefetch_l2(p.k + pg * p.k_sb + h * p.k_sh, kKvBytes);
                prefetch_l2(p.v + pg * p.v_sb + h * p.v_sh, kKvBytes);
                prefetch_l2(p.ks + pg * p.ks_sb + h * p.ks_sh, kScBytes);
                prefetch_l2(p.vs + pg * p.vs_sb + h * p.vs_sh, kScBytes);
            }
        }
        s_cand[tid] = v;
        s_cpage[tid] = page;
        s_cmask[tid] = 0;
        s_cearly[tid] = 0;
    }
    __syncthreads();
    // Membership masks and first occurrence in the slot-major walk (slot 0 of every query, then slot 1,
    // ...): candidate c = (query j, slot k) has order key k * dql + j. Interleaving the queries spreads
    // every query tile's pages evenly over the union, so a contiguous stream-K share of it gives each
    // tile a proportional part of the work (a query-major union puts each query's own pages in one run,
    // which one tile then works through alone).
    for (int pr = tid; pr < kMaxCand * p.dql; pr += blockDim.x)
    {
        int const c = pr & (kMaxCand - 1), j = pr >> 6;
        int const v = c < ncand ? s_cand[c] : -1;
        if (v >= 0)
        {
            int const base = j * p.topk_n;
            int const jc = c / p.topk_n;
            int const key_c = (c - jc * p.topk_n) * p.dql + jc;
            int4 const* c4 = reinterpret_cast<int4 const*>(s_cand + base);
            bool hit = false, early = false;
            for (int i4 = 0; i4 < (p.topk_n >> 2); ++i4)
            {
                const int4 cc = c4[i4];
                int const key = 4 * i4 * p.dql + j; // order key of slot 4 * i4 of list j
                bool const e0 = cc.x == v, e1 = cc.y == v, e2 = cc.z == v, e3 = cc.w == v;
                hit |= e0 | e1 | e2 | e3;
                early |= (e0 && key < key_c) || (e1 && key + p.dql < key_c) || (e2 && key + 2 * p.dql < key_c)
                    || (e3 && key + 3 * p.dql < key_c);
            }
            if (hit)
                atomicOr(&s_cmask[c], 1 << j);
            if (early)
                s_cearly[c] = 1;
        }
    }
    __syncthreads();
    int my_v = -1, my_mask = 0;
    bool my_first = false;
    unsigned my_bal = 0;
    // Thread tid takes the candidate of order key tid, so the compaction lays the union out in that order.
    int const my_c = tid < ncand ? (tid % p.dql) * p.topk_n + tid / p.dql : tid;
    if (tid < kMaxCand)
    {
        my_v = s_cand[my_c];
        my_first = my_v >= 0 && s_cearly[my_c] == 0;
        my_mask = s_cmask[my_c];
        my_bal = __ballot_sync(0xffffffffu, my_first);
        if (lane == 0)
            s_misc[warp] = __popc(my_bal);
    }
    __syncthreads();
    int* ub = p.u_blk + (int64_t) grp * kMaxCand;
    int* up = p.u_page + (int64_t) grp * kMaxCand;
    int* um = p.u_mask + (int64_t) grp * kMaxCand;
    __shared__ int s_w[kMaxCand];
    if (tid < kMaxCand && my_first)
    {
        int const pos = (warp == 1 ? s_misc[0] : 0) + __popc(my_bal & ((1u << lane) - 1u));
        ub[pos] = my_v;
        um[pos] = my_mask;
        up[pos] = s_cpage[my_c];
        s_w[pos] = kSkPageBase + kSkPerTile * __popc(my_mask);
    }
    int const U = s_misc[0] + s_misc[1];
    __syncthreads();
    // Inclusive prefix of the page weights (warps 0-1 cover the 64 slots).
    if (tid < kMaxCand)
    {
        int w = tid < U ? s_w[tid] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int const y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o)
                w += y;
        }
        s_cearly[tid] = w; // per-warp inclusive prefix (s_cearly is free now)
    }
    __syncthreads();
    if (tid < kMaxCand)
    {
        int const w = s_cearly[tid] + (warp == 1 ? s_cearly[31] : 0);
        if (tid < U)
            p.u_wcum[(int64_t) grp * kMaxCand + tid] = w;
        if (tid == kMaxCand - 1)
        {
            p.u_wsum[grp] = w;
            p.u_cnt[grp] = U;
        }
    }
    if (U == 0)
    {
        // No page for any query of this group: its rows are zero (as the one-shot kernel writes them).
        int const R = p.dql * p.group;
        for (int e = tid; e < R * 16; e += blockDim.x)
        {
            int const row = e >> 4, ch = e & 15;
            int const jj = row / p.group, hh = row - jj * p.group;
            uint4* dst = reinterpret_cast<uint4*>(
                p.out + (int64_t) (r * p.dql + jj) * p.o_sn + (int64_t) (h * p.group + hh) * p.o_sh);
            dst[ch] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    griddep_launch();
}

// Owner CTA of union position x when P positions are cut into G equal contiguous shares.
__device__ __forceinline__ int sk_owner(int64_t x, int64_t P, int G)
{
    return static_cast<int>(((x + 1) * G - 1) / P);
}

template <int WPT, int kMaxThreads, int kMinBlocks>
__global__ void __launch_bounds__(kMaxThreads, kMinBlocks) sparse_decode_v2_sk_kernel(const __grid_constant__ Params p)
{
    using C = TileCfg<WPT>;
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int s_goff[kMaxSkGroups + 1];
    // This CTA's pages (its share of every group it touches, concatenated) and its segments.
    __shared__ int s_ublk[kSkMaxPages], s_umask[kSkMaxPages], s_upage[kSkMaxPages];
    __shared__ short s_uhead[kSkMaxPages];
    // Per segment: group, pages, first owner CTA, segments of the group, first page in the group's union,
    // first page in the CTA's list.
    __shared__ int s_seg[kSkMaxPages][6];
    __shared__ int s_npages, s_nsegs, s_last, s_ga, s_ng;
    int const tid = threadIdx.x;
    int const warp = tid >> 5, lane = tid & 31;
    int const g = lane >> 2, t = lane & 3;
    int const S = p.stages;
    int const nwarps = p.ntiles * WPT;
    int const rows_pad = p.ntiles * 16;
    int const R = p.dql * p.group;

    const SmemLayout L(S, p.ntiles, WPT, 1, R);
    uint8_t* stage_base = smem;
    uint4* pfrag = reinterpret_cast<uint4*>(smem + L.pfrag);
    float* red = reinterpret_cast<float*>(smem + L.red);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + L.bars);
    uint64_t* empty = full + S;

    if (tid == 0)
    {
        for (int s = 0; s < S; ++s)
        {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], nwarps);
        }
        fence_barrier_init();
    }

    // Tile geometry (independent of any device data).
    int const tile = warp / WPT;
    int const qw = warp - tile * WPT;
    int const gshift = __ffs(p.group) - 1; // group is 8 or 16
    int const f0 = tile * 16 + g, f1 = f0 + 8;
    int const j0 = f0 >> gshift, j1 = f1 >> gshift;
    int const hh0 = f0 & (p.group - 1), hh1 = f1 & (p.group - 1);
    bool const rv0 = j0 < p.dql, rv1 = j1 < p.dql;
    int tmask = 0;
    {
        int const qa = (tile * 16) >> gshift;
        int const qb = min((tile * 16 + 15) >> gshift, p.dql - 1);
        for (int qq = qa; qq <= qb; ++qq)
            tmask |= 1 << qq;
    }
    int const t_odd = t & 1;
    int const g_swz = (g & 1) ^ ((g >> 1) & 1);
    const uint32_t vsel = t_odd ? 0x2301u : 0x3210u;
    int const bar_id = 1 + tile;
    int const bar_n = 32 * WPT;
    int const v_byte = qw * (C::kDW / 2) + C::kBPT * g;
    int const v_group = (qw * C::kDW + (C::kDW / 8) * g) >> 4;
    int const dim0 = qw * C::kDW + 2 * t * C::kNP;
#define SK_STAMP(k)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (p.dbg != nullptr && tid == 0 && (k) < 24)                                                                  \
            p.dbg[blockIdx.x * 24 + (k)] = gtimer();                                                                   \
    } while (0)
    SK_STAMP(0);

    griddep_wait();

    // Exclusive prefix of the group weights (warp 0, 32 groups per step).
    int const ng = p.ngroups;
    if (warp == 0)
    {
        int carry = 0;
        if (lane == 0)
            s_goff[0] = 0;
        for (int b0 = 0; b0 < ng; b0 += 32)
        {
            int const gg = b0 + lane;
            int v = gg < ng ? p.u_wsum[gg] : 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                int const y = __shfl_up_sync(0xffffffffu, v, o);
                if (lane >= o)
                    v += y;
            }
            if (gg < ng)
                s_goff[gg + 1] = carry + v;
            carry += __shfl_sync(0xffffffffu, v, 31);
        }
    }
    __syncthreads();
    // Positions are page weights: page i of group gg spans [off + wstart(i), off + wcum(i)), and belongs
    // to the CTA whose share holds its start. Each share is at least one maximal page weight wide, so
    // every working CTA owns at least one page and the owners of a group's pages are consecutive CTAs
    // (its segments' partial slots are first_owner + gg ... first_owner + nseg - 1 + gg).
    int64_t const P = s_goff[ng];
    int const G = static_cast<int>(min((int64_t) gridDim.x, max(P / kSkMaxPageWeight, (int64_t) 1)));
    int const cta = blockIdx.x;

    // The CTA's segments: thread 0 finds the groups its share touches, one warp per group finds the
    // pages whose weight start lies in the share, thread 0 compacts them into page offsets, and every
    // thread gathers the pages.
    int const lo = cta < G ? static_cast<int>((int64_t) cta * P / G) : 0;
    int const hi = cta < G ? static_cast<int>((int64_t) (cta + 1) * P / G) : 0;
    if (tid == 0)
    {
        int ga = 0, gb = ng; // first group whose pages reach past lo
        while (ga < gb)
        {
            int const m = (ga + gb) >> 1;
            if (s_goff[m + 1] > lo)
                gb = m;
            else
                ga = m + 1;
        }
        int gz = ga, gc = ng; // first group (>= ga) that starts at or after hi
        while (gz < gc)
        {
            int const m = (gz + gc) >> 1;
            if (s_goff[m] >= hi)
                gc = m;
            else
                gz = m + 1;
        }
        s_ga = ga;
        s_ng = lo < hi ? min(gz - ga, kSkMaxPages) : 0;
    }
    __syncthreads();
    for (int x = warp; x < s_ng; x += nwarps)
    {
        int const gg = s_ga + x;
        int const off = s_goff[gg];
        int const U = p.u_cnt[gg];
        int const rel_lo = max(lo, off) - off;
        int const rel_hi = min(hi, s_goff[gg + 1]) - off;
        int a = 0, b = 0, last_start = 0;
        for (int i0 = 0; i0 < U; i0 += 32)
        {
            int const i = i0 + lane;
            int ws = 0;
            if (i < U && i > 0)
                ws = p.u_wcum[(int64_t) gg * kMaxCand + i - 1];
            a += __popc(__ballot_sync(0xffffffffu, i < U && ws < rel_lo));
            b += __popc(__ballot_sync(0xffffffffu, i < U && ws < rel_hi));
            int const li = U - 1 - i0; // lane holding page U - 1 in this chunk
            if (li >= 0 && li < 32)
                last_start = __shfl_sync(0xffffffffu, ws, li);
        }
        if (lane == 0)
        {
            int const first_owner = U > 0 ? sk_owner(off, P, G) : 0;
            s_seg[x][0] = gg;
            s_seg[x][1] = b - a;
            s_seg[x][2] = first_owner;
            s_seg[x][3] = U > 0 ? sk_owner(off + last_start, P, G) - first_owner + 1 : 0;
            s_seg[x][4] = a;
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        int npg = 0, nsg = 0;
        for (int x = 0; x < s_ng; ++x)
        {
            int const n = s_seg[x][1];
            if (n <= 0)
                continue;
#pragma unroll
            for (int f = 0; f < 5; ++f)
                s_seg[nsg][f] = s_seg[x][f]; // in place: nsg <= x
            s_seg[nsg][5] = npg;
            npg += n;
            ++nsg;
        }
        s_npages = npg;
        s_nsegs = nsg;
    }
    __syncthreads();
    for (int sg = 0; sg < s_nsegs; ++sg)
    {
        int const gg = s_seg[sg][0];
        int const n = s_seg[sg][1];
        int64_t const src0 = (int64_t) gg * kMaxCand + s_seg[sg][4];
        int const dst0 = s_seg[sg][5];
        for (int i = tid; i < n; i += blockDim.x)
        {
            s_ublk[dst0 + i] = p.u_blk[src0 + i];
            s_umask[dst0 + i] = p.u_mask[src0 + i];
            s_upage[dst0 + i] = p.u_page[src0 + i];
            s_uhead[dst0 + i] = static_cast<short>(gg % p.num_kvh);
        }
    }
    __syncthreads();
    SK_STAMP(1);
    int const K = s_npages;
    int const NS = s_nsegs;

    // The ring runs over the CTA's whole page list, across segment boundaries: while the tiles finish
    // one segment (epilogue, partial, merge) the next segment's first pages are already in flight.
    int const nfill = min(K, S);
    if (warp == 0 && lane < nfill)
        issue_page(p, s_upage[lane], s_uhead[lane], stage_base + lane * kStageBytes, &full[lane]);
    int issued = nfill;
    int pst = nfill == S ? 0 : nfill; // producer: stage of the next page to issue
    int puse = nfill == S ? 1 : 0;    // ... and how often that stage was filled before
    int st = 0;
    uint32_t ph = 0;
    int used = 0; // P / reduction buffer parity, running across segments
    float const sl2 = p.sm_scale_log2e * (*p.k_gs);
    float const v_gs = *p.v_gs;
    bool const producer = tid == 0;
    long long c_wait = 0, c_qk = 0, c_sm = 0, c_pv = 0, c_tmp = 0;
    int c_act = 0;
    (void) c_act;

    int k = 0; // index in the CTA's page list
    for (int sg = 0; sg < NS; ++sg)
    {
        int const gg = s_seg[sg][0];
        int const n = s_seg[sg][1];
        int const h = gg % p.num_kvh;
        int const r = gg / p.num_kvh;
        int const seq_len = p.seq_lens[r];
        uint32_t qf[8][4];
        {
            uint32_t w0[16], w1[16];
            const int64_t tok0 = (int64_t) (r * p.dql + j0) * p.q_sn + (int64_t) (h * p.group + hh0) * p.q_sh + 32 * t;
            const int64_t tok1 = (int64_t) (r * p.dql + j1) * p.q_sn + (int64_t) (h * p.group + hh1) * p.q_sh + 32 * t;
            int const esz = p.q_fp8 ? 1 : 2;
            load_q_row(p, static_cast<uint8_t const*>(p.q) + tok0 * esz, rv0, w0);
            load_q_row(p, static_cast<uint8_t const*>(p.q) + tok1 * esz, rv1, w1);
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                qf[j][0] = w0[2 * j];
                qf[j][1] = w1[2 * j];
                qf[j][2] = w0[2 * j + 1];
                qf[j][3] = w1[2 * j + 1];
            }
        }
        int const kvlen0 = rv0 ? max(seq_len - p.dql + j0 + 1, 0) : 0;
        int const kvlen1 = rv1 ? max(seq_len - p.dql + j1 + 1, 0) : 0;

        float o[C::kNP][4];
#pragma unroll
        for (int c = 0; c < C::kNP; ++c)
            o[c][0] = o[c][1] = o[c][2] = o[c][3] = 0.f;
        float m0 = kNegInf, m1 = kNegInf, l0 = 0.f, l1 = 0.f;

        for (int i = 0; i < n; ++i, ++k)
        {
            if (producer)
            {
                while (issued < K && issued < k + S)
                {
                    // issued >= nfill == S here, so the stage has been filled before (puse >= 1).
                    uint32_t const par = (puse - 1) & 1;
                    if (issued <= k)
                    {
                        mbar_wait(&empty[pst], par);
                    }
                    else if (!mbar_test(&empty[pst], par))
                    {
                        break;
                    }
                    issue_page(p, s_upage[issued], s_uhead[issued], stage_base + pst * kStageBytes, &full[pst]);
                    ++issued;
                    if (++pst == S)
                    {
                        pst = 0;
                        ++puse;
                    }
                }
            }
            mbar_wait(&full[st], ph);
            SDV2_CLK(c_wait);
            int const uidx = k;
            int const bm = s_umask[uidx];
            if (bm & tmask)
            {
                ++c_act;
                int const buf = used & 1;
                ++used;
                uint8_t const* kq = stage_base + st * kStageBytes;
                uint8_t const* vq = kq + kKvBytes;
                uint8_t const* ksc = kq + 2 * kKvBytes;
                uint8_t const* vsc = ksc + kScBytes;
                int const base_tok = s_ublk[uidx] * kBlk;
                int const lim0 = (rv0 && ((bm >> j0) & 1)) ? kvlen0 - base_tok : 0;
                int const lim1 = (rv1 && ((bm >> j1) & 1)) ? kvlen1 - base_tok : 0;
                bool const need_mask = __any_sync(0xffffffffu, lim0 < kBlk || lim1 < kBlk);
                int const vlim = seq_len - base_tok; // tokens at/after seq_len hold no data

                // ---------------- S = Q K^T for this warp's tokens
                float s[C::kNQ][4];
#pragma unroll
                for (int ii = 0; ii < C::kNQ; ++ii)
                {
                    int const it = qw * C::kNQ + ii;
                    int const tok = 16 * (it >> 1) + 4 * (g >> 1) + 2 * (it & 1) + g_swz;
                    const uint4 kd = *reinterpret_cast<uint4 const*>(kq + tok * 64 + 16 * t);
                    const uint32_t k2 = *reinterpret_cast<uint16_t const*>(ksc + tok * 8 + 2 * t);
                    const uint32_t sc = e4m3x2_to_h2(k2);
                    const uint32_t s0 = __byte_perm(sc, 0, 0x1010);
                    const uint32_t s1 = __byte_perm(sc, 0, 0x3232);
                    uint32_t b[16];
                    e2m1x8_to_h2x4(kd.x, b[0], b[1], b[2], b[3]);
                    e2m1x8_to_h2x4(kd.y, b[4], b[5], b[6], b[7]);
                    e2m1x8_to_h2x4(kd.z, b[8], b[9], b[10], b[11]);
                    e2m1x8_to_h2x4(kd.w, b[12], b[13], b[14], b[15]);
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                        b[q] = hmul2(b[q], s0);
#pragma unroll
                    for (int q = 8; q < 16; ++q)
                        b[q] = hmul2(b[q], s1);
                    s[ii][0] = s[ii][1] = s[ii][2] = s[ii][3] = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        mma16816(s[ii], qf[j], b[2 * j], b[2 * j + 1]);
                }

                SDV2_CLK(c_qk);
                // ---------------- softmax: block row max across the tile's warps
#pragma unroll
                for (int ii = 0; ii < C::kNQ; ++ii)
                {
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        s[ii][c] *= sl2;
                }
                if (need_mask)
                {
#pragma unroll
                    for (int ii = 0; ii < C::kNQ; ++ii)
                    {
                        int const it = qw * C::kNQ + ii;
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                        {
                            int const tok = 16 * (it >> 1) + 4 * t + 2 * (it & 1) + ((c & 1) ^ t_odd);
                            if (tok >= (c < 2 ? lim0 : lim1))
                                s[ii][c] = kNegInf;
                        }
                    }
                }
                float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
                for (int ii = 0; ii < C::kNQ; ++ii)
                {
                    mx0 = fmaxf(mx0, fmaxf(s[ii][0], s[ii][1]));
                    mx1 = fmaxf(mx1, fmaxf(s[ii][2], s[ii][3]));
                }
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
                float* rb = red + ((tile * 2 + buf) * 16) * WPT;
                if (WPT > 1)
                {
                    if (t == 0)
                    {
                        rb[g * WPT + qw] = mx0;
                        rb[(g + 8) * WPT + qw] = mx1;
                    }
                    named_bar(bar_id, bar_n);
#pragma unroll
                    for (int w = 0; w < WPT; ++w)
                    {
                        mx0 = fmaxf(mx0, rb[g * WPT + w]);
                        mx1 = fmaxf(mx1, rb[(g + 8) * WPT + w]);
                    }
                }
                float const mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
                float const ms0 = mn0 == kNegInf ? 0.f : mn0;
                float const ms1 = mn1 == kNegInf ? 0.f : mn1;
                float const al0 = ex2(m0 - ms0), al1 = ex2(m1 - ms1);
                m0 = mn0;
                m1 = mn1;
                float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
                for (int ii = 0; ii < C::kNQ; ++ii)
                {
                    s[ii][0] = ex2(s[ii][0] - ms0);
                    s[ii][1] = ex2(s[ii][1] - ms0);
                    s[ii][2] = ex2(s[ii][2] - ms1);
                    s[ii][3] = ex2(s[ii][3] - ms1);
                    rs0 += s[ii][0] + s[ii][1];
                    rs1 += s[ii][2] + s[ii][3];
                }
                l0 = l0 * al0 + rs0;
                l1 = l1 * al1 + rs1;
#pragma unroll
                for (int c = 0; c < C::kNP; ++c)
                {
                    o[c][0] *= al0;
                    o[c][1] *= al0;
                    o[c][2] *= al1;
                    o[c][3] *= al1;
                }
                // P fragments of this warp's k-steps -> shared memory, fragment order.
                uint4* pf = pfrag + (tile * 2 + buf) * (8 * 32);
#pragma unroll
                for (int ks = 0; ks < C::kKS; ++ks)
                {
                    int const e0 = 2 * ks, e1 = 2 * ks + 1;
                    pf[(qw * C::kKS + ks) * 32 + lane] = make_uint4(pack_h2(s[e0][0], s[e0][1]),
                        pack_h2(s[e0][2], s[e0][3]), pack_h2(s[e1][0], s[e1][1]), pack_h2(s[e1][2], s[e1][3]));
                }
                if (WPT > 1)
                    named_bar(bar_id, bar_n);
                SDV2_CLK(c_sm);

                // ---------------- O[:, my dims] += P V
#pragma unroll
                for (int sidx = 0; sidx < 8; ++sidx)
                {
                    const uint4 pq = pf[sidx * 32 + lane];
                    const uint32_t pa[4] = {pq.x, pq.y, pq.z, pq.w};
                    int const row_a = 16 * sidx + 4 * t;
                    uint8_t const* vrow = vq + v_byte;
                    uint32_t vs4 = *reinterpret_cast<uint32_t const*>(vsc + (4 * sidx + t) * 32 + 4 * v_group);
                    vs4 = __byte_perm(vs4, 0, vsel);
                    if (vlim < kBlk)
                    {
                        uint32_t keep = 0;
                        keep |= (row_a + t_odd < vlim) ? 0x000000ffu : 0u;
                        keep |= (row_a + 1 - t_odd < vlim) ? 0x0000ff00u : 0u;
                        keep |= (row_a + 2 + t_odd < vlim) ? 0x00ff0000u : 0u;
                        keep |= (row_a + 3 - t_odd < vlim) ? 0xff000000u : 0u;
                        vs4 &= keep;
                    }
                    const uint32_t sab = e4m3x2_to_h2(vs4 & 0xffffu);
                    const uint32_t scd = e4m3x2_to_h2(vs4 >> 16);
                    uint32_t b01[C::kNP], b23[C::kNP];
                    if constexpr (C::kBPT == 2)
                    {
                        const uint32_t wa = *reinterpret_cast<uint16_t const*>(vrow + (row_a + t_odd) * 64);
                        const uint32_t wb = *reinterpret_cast<uint16_t const*>(vrow + (row_a + 1 - t_odd) * 64);
                        const uint32_t wc = *reinterpret_cast<uint16_t const*>(vrow + (row_a + 2 + t_odd) * 64);
                        const uint32_t wd = *reinterpret_cast<uint16_t const*>(vrow + (row_a + 3 - t_odd) * 64);
                        const uint32_t w1 = wa | (wc << 16), w2 = wb | (wd << 16);
                        const uint32_t x = (w1 & 0x0F0F0F0Fu) | ((w2 << 4) & 0xF0F0F0F0u);
                        const uint32_t y = ((w1 >> 4) & 0x0F0F0F0Fu) | (w2 & 0xF0F0F0F0u);
                        e2m1x8_to_h2x4(x, b01[0], b01[2], b23[0], b23[2]);
                        e2m1x8_to_h2x4(y, b01[1], b01[3], b23[1], b23[3]);
                    }
                    else
                    {
#pragma unroll
                        for (int hw = 0; hw < C::kBPT / 4; ++hw)
                        {
                            const uint32_t wa
                                = *reinterpret_cast<uint32_t const*>(vrow + (row_a + t_odd) * 64 + 4 * hw);
                            const uint32_t wb
                                = *reinterpret_cast<uint32_t const*>(vrow + (row_a + 1 - t_odd) * 64 + 4 * hw);
                            const uint32_t wc
                                = *reinterpret_cast<uint32_t const*>(vrow + (row_a + 2 + t_odd) * 64 + 4 * hw);
                            const uint32_t wd
                                = *reinterpret_cast<uint32_t const*>(vrow + (row_a + 3 - t_odd) * 64 + 4 * hw);
                            const uint32_t x_ab = (wa & 0x0F0F0F0Fu) | ((wb << 4) & 0xF0F0F0F0u);
                            const uint32_t y_ab = ((wa >> 4) & 0x0F0F0F0Fu) | (wb & 0xF0F0F0F0u);
                            const uint32_t x_cd = (wc & 0x0F0F0F0Fu) | ((wd << 4) & 0xF0F0F0F0u);
                            const uint32_t y_cd = ((wc >> 4) & 0x0F0F0F0Fu) | (wd & 0xF0F0F0F0u);
                            int const bb = 8 * hw;
                            e2m1x8_to_h2x4(x_ab, b01[bb + 0], b01[bb + 2], b01[bb + 4], b01[bb + 6]);
                            e2m1x8_to_h2x4(y_ab, b01[bb + 1], b01[bb + 3], b01[bb + 5], b01[bb + 7]);
                            e2m1x8_to_h2x4(x_cd, b23[bb + 0], b23[bb + 2], b23[bb + 4], b23[bb + 6]);
                            e2m1x8_to_h2x4(y_cd, b23[bb + 1], b23[bb + 3], b23[bb + 5], b23[bb + 7]);
                        }
                    }
#pragma unroll
                    for (int c = 0; c < C::kNP; ++c)
                    {
                        b01[c] = hmul2(b01[c], sab);
                        b23[c] = hmul2(b23[c], scd);
                    }
#pragma unroll
                    for (int c = 0; c < C::kNP; ++c)
                        mma16816(o[c], pa, b01[c], b23[c]);
                }
                if (p.dbg != nullptr)
                {
                    // Make the accumulators ready before reading the clock.
                    float z = 0.f;
#pragma unroll
                    for (int c = 0; c < C::kNP; ++c)
                        z += o[c][0];
                    if (z == 12345.f)
                        c_pv += 1;
                }
                SDV2_CLK(c_pv);
            }
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&empty[st]);
            if (++st == S)
            {
                st = 0;
                ph ^= 1;
            }
        }
        int const first_owner = s_seg[sg][2];
        int const nseg = s_seg[sg][3];
        // ---------------- row sums across the tile's warps (m is already shared)
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        if (WPT > 1)
        {
            float* rb = red + ((tile * 2 + (used & 1)) * 16) * WPT;
            if (t == 0)
            {
                rb[g * WPT + qw] = l0;
                rb[(g + 8) * WPT + qw] = l1;
            }
            named_bar(bar_id, bar_n);
            l0 = 0.f;
            l1 = 0.f;
#pragma unroll
            for (int w = 0; w < WPT; ++w)
            {
                l0 += rb[g * WPT + w];
                l1 += rb[(g + 8) * WPT + w];
            }
        }

        if (nseg == 1)
        {
            // The whole group is this segment: write the output (as the one-split kernel does).
            float const sc0 = l0 > 0.f ? v_gs / l0 : 0.f;
            float const sc1 = l1 > 0.f ? v_gs / l1 : 0.f;
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                bool const rv = rr ? rv1 : rv0;
                if (!rv)
                    continue;
                float const sc = rr ? sc1 : sc0;
                uint32_t w[C::kNP];
#pragma unroll
                for (int c = 0; c < C::kNP / 2; ++c)
                {
                    w[c] = pack_bf2(o[2 * c][2 * rr] * sc, o[2 * c + 1][2 * rr] * sc);
                    w[C::kNP / 2 + c] = pack_bf2(o[2 * c][2 * rr + 1] * sc, o[2 * c + 1][2 * rr + 1] * sc);
                }
                int const jj = rr ? j1 : j0, hh = rr ? hh1 : hh0;
                uint4* dst = reinterpret_cast<uint4*>(
                    p.out + (int64_t) (r * p.dql + jj) * p.o_sn + (int64_t) (h * p.group + hh) * p.o_sh + dim0);
#pragma unroll
                for (int c = 0; c < C::kNP / 4; ++c)
                    dst[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
            }
        }
        else
        {
            // Publish an fp16 partial (pre V-scale) in segment slot cta + gg (segments of a group are
            // consecutive CTAs, so the slots of group gg are first_owner + gg ... + nseg - 1).
            const int64_t part_row0 = (int64_t) (cta + gg) * rows_pad;
            {
                float const sc0 = l0 > 0.f ? 1.f / l0 : 0.f;
                float const sc1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    bool const rv = rr ? rv1 : rv0;
                    float const ll = rr ? l1 : l0;
                    if (!rv)
                        continue;
                    int const f = rr ? f1 : f0;
                    if (!(ll > 0.f))
                    {
                        if (qw == 0 && t == 0)
                            p.part_ml[part_row0 + f] = make_float2(kNegInf, 0.f);
                        continue;
                    }
                    float const sc = rr ? sc1 : sc0;
                    uint32_t w[C::kNP];
#pragma unroll
                    for (int c = 0; c < C::kNP / 2; ++c)
                    {
                        w[c] = pack_h2(o[2 * c][2 * rr] * sc, o[2 * c + 1][2 * rr] * sc);
                        w[C::kNP / 2 + c] = pack_h2(o[2 * c][2 * rr + 1] * sc, o[2 * c + 1][2 * rr + 1] * sc);
                    }
                    uint4* dst = reinterpret_cast<uint4*>(p.part_o + (part_row0 + f) * 128 + dim0);
#pragma unroll
                    for (int c = 0; c < C::kNP / 4; ++c)
                        dst[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
                    if (qw == 0 && t == 0)
                        p.part_ml[part_row0 + f] = make_float2(rr ? m1 : m0, ll);
                }
            }
            __syncthreads();
            if (tid == 0)
            {
                // Release publishes this segment's partial (cumulative over the barrier); acquire
                // makes the group's other partials visible to the last segment.
                int const prev = atom_add_acq_rel_gpu(&p.counters[gg], 1);
                int const last = prev == nseg - 1;
                if (last)
                    p.counters[gg] = 0; // self-reset for the next launch / graph replay
                s_last = last;
            }
            __syncthreads();
            if (s_last)
            {
                // Last segment: every thread merges the fragment it owns over the group's segments in
                // order, so the result does not depend on which CTA finishes last.
                constexpr int kMergeChunk = 4;
                float M[2] = {kNegInf, kNegInf}, W[2] = {0.f, 0.f};
                float A[2][2 * C::kNP];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
#pragma unroll
                    for (int e = 0; e < 2 * C::kNP; ++e)
                        A[rr][e] = 0.f;
                }
                const int64_t mrow0 = (int64_t) (first_owner + gg) * rows_pad + f0;
                for (int sp0 = 0; sp0 < nseg; sp0 += kMergeChunk)
                {
                    float2 ml[2][kMergeChunk];
                    uint4 x[2][kMergeChunk][C::kNP / 4];
#pragma unroll
                    for (int rr = 0; rr < 2; ++rr)
                    {
#pragma unroll
                        for (int u = 0; u < kMergeChunk; ++u)
                        {
                            int const sp = sp0 + u;
                            ml[rr][u] = make_float2(0.f, 0.f);
                            if ((rr ? rv1 : rv0) && sp < nseg)
                            {
                                const int64_t row = mrow0 + (int64_t) sp * rows_pad + 8 * rr;
                                ml[rr][u] = p.part_ml[row];
                                if (ml[rr][u].y > 0.f)
                                {
#pragma unroll
                                    for (int q = 0; q < C::kNP / 4; ++q)
                                        x[rr][u][q] = reinterpret_cast<uint4 const*>(p.part_o + row * 128 + dim0)[q];
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int rr = 0; rr < 2; ++rr)
                    {
#pragma unroll
                        for (int u = 0; u < kMergeChunk; ++u)
                        {
                            if (ml[rr][u].y > 0.f)
                            {
                                float const mn = fmaxf(M[rr], ml[rr][u].x);
                                float const a2 = ex2(M[rr] - mn);
                                float const b2 = ml[rr][u].y * ex2(ml[rr][u].x - mn);
                                W[rr] = W[rr] * a2 + b2;
#pragma unroll
                                for (int q = 0; q < C::kNP / 4; ++q)
                                {
                                    const uint32_t xs[4] = {x[rr][u][q].x, x[rr][u][q].y, x[rr][u][q].z, x[rr][u][q].w};
#pragma unroll
                                    for (int e = 0; e < 4; ++e)
                                    {
                                        const float2 fv = __half22float2(*reinterpret_cast<__half2 const*>(&xs[e]));
                                        A[rr][8 * q + 2 * e] = A[rr][8 * q + 2 * e] * a2 + b2 * fv.x;
                                        A[rr][8 * q + 2 * e + 1] = A[rr][8 * q + 2 * e + 1] * a2 + b2 * fv.y;
                                    }
                                }
                                M[rr] = mn;
                            }
                        }
                    }
                }
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    if (!(rr ? rv1 : rv0))
                        continue;
                    float const sc = W[rr] > 0.f ? v_gs / W[rr] : 0.f;
                    uint32_t w[C::kNP];
#pragma unroll
                    for (int c = 0; c < C::kNP; ++c)
                        w[c] = pack_bf2(A[rr][2 * c] * sc, A[rr][2 * c + 1] * sc);
                    int const jj = rr ? j1 : j0, hh = rr ? hh1 : hh0;
                    uint4* dst = reinterpret_cast<uint4*>(
                        p.out + (int64_t) (r * p.dql + jj) * p.o_sn + (int64_t) (h * p.group + hh) * p.o_sh + dim0);
#pragma unroll
                    for (int c = 0; c < C::kNP / 4; ++c)
                        dst[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
                }
            }
        }
    }
    SK_STAMP(23);
    griddep_launch();
}

template <int WPT, int kMaxThreads, int kMinBlocks>
void launch_sk_impl(Params const& p, int grid, int threads, int smem, bool pdl, cudaStream_t stream)
{
    // Prep: per-group unions (128 threads per group).
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(p.ngroups);
        cfg.blockDim = dim3(128);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        int na = 0;
        if (pdl)
        {
            attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[na].val.programmaticStreamSerializationAllowed = 1;
            ++na;
        }
        cfg.attrs = attr;
        cfg.numAttrs = na;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, sparse_decode_v2_union_kernel, p));
    }
    auto kern = sparse_decode_v2_sk_kernel<WPT, kMaxThreads, kMinBlocks>;
    static int configured = 0;
    if (smem > configured)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = smem;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    int na = 0;
    if (pdl)
    {
        attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
}

template <int WPT, int kMaxThreads, int kMinBlocks>
void launch_impl(Params const& p, int grid, int threads, int smem, bool pdl, cudaStream_t stream)
{
    auto kern = sparse_decode_v2_kernel<WPT, kMaxThreads, kMinBlocks>;
    static int configured = 0;
    if (smem > configured)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = smem;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    static bool nonportable = false;
    if (!nonportable && p.nsplit > 8 && p.cluster_merge)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        nonportable = true;
    }
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (p.nsplit > 1 && p.cluster_merge)
    {
        attr[na].id = cudaLaunchAttributeClusterDimension;
        attr[na].val.clusterDim.x = p.nsplit;
        attr[na].val.clusterDim.y = 1;
        attr[na].val.clusterDim.z = 1;
        ++na;
    }
    if (pdl)
    {
        attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
}

} // namespace sdv2

void sparse_decode_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor ks, torch::Tensor vs,
    torch::Tensor k_gs, torch::Tensor v_gs, torch::Tensor topk, torch::Tensor bt, torch::Tensor seq_lens,
    torch::Tensor out, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor dbg,
    int64_t dql, int64_t nsplit, int64_t ntiles, int64_t wpt, int64_t stages, double sm_scale_log2e, bool pdl,
    bool cluster_merge)
{
    using namespace sdv2;
    const at::cuda::CUDAGuard guard(q.device());
    Params p{};
    p.q = q.data_ptr();
    p.q_sn = q.stride(0);
    p.q_sh = q.stride(1);
    p.k = k.data_ptr<uint8_t>();
    p.v = v.data_ptr<uint8_t>();
    p.k_sb = k.stride(0);
    p.k_sh = k.stride(1);
    p.v_sb = v.stride(0);
    p.v_sh = v.stride(1);
    p.ks = ks.data_ptr<uint8_t>();
    p.vs = vs.data_ptr<uint8_t>();
    p.ks_sb = ks.stride(0);
    p.ks_sh = ks.stride(1);
    p.vs_sb = vs.stride(0);
    p.vs_sh = vs.stride(1);
    p.k_gs = k_gs.data_ptr<float>();
    p.v_gs = v_gs.data_ptr<float>();
    p.topk = topk.data_ptr<int>();
    p.t_sh = topk.stride(0);
    p.t_sn = topk.stride(1);
    p.t_sk = topk.stride(2);
    p.bt = bt.data_ptr<int>();
    p.bt_sb = bt.stride(0);
    p.bt_cols = static_cast<int>(bt.size(1));
    p.seq_lens = seq_lens.data_ptr<int>();
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.o_sn = out.stride(0);
    p.o_sh = out.stride(1);
    p.part_o = reinterpret_cast<__half*>(part_o.data_ptr());
    p.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr());
    p.counters = counters.data_ptr<int>();
    p.dbg = dbg.numel() > 0 ? reinterpret_cast<long long*>(dbg.data_ptr()) : nullptr;
    int const num_reqs = static_cast<int>(seq_lens.size(0));
    p.num_kvh = static_cast<int>(k.size(1));
    p.group = static_cast<int>(q.size(1) / k.size(1));
    p.dql = static_cast<int>(dql);
    p.topk_n = static_cast<int>(topk.size(2));
    p.nsplit = static_cast<int>(nsplit);
    p.ntiles = static_cast<int>(ntiles);
    p.wsub = static_cast<int>(wpt);
    p.stages = static_cast<int>(stages);
    p.sm_scale_log2e = static_cast<float>(sm_scale_log2e);
    p.q_fp8 = q.scalar_type() == at::ScalarType::Float8_e4m3fn ? 1 : 0;
    p.cluster_merge = cluster_merge ? 1 : 0;
    p.kv_fused = (p.v == p.k + kKvBytes && p.v_sb == p.k_sb && p.v_sh == p.k_sh) ? 1 : 0;
    p.sc_fused = (p.vs == p.ks + kScBytes && p.vs_sb == p.ks_sb && p.vs_sh == p.ks_sh) ? 1 : 0;

    int const threads = wpt <= 0 ? 512 : p.ntiles * static_cast<int>(wpt) * 32;
    int const grid = num_reqs * p.num_kvh * p.nsplit;
    int const smem = SmemLayout(p.stages, p.ntiles, static_cast<int>(wpt), p.nsplit, p.dql * p.group).total;
    if (grid == 0)
        return;
    TORCH_CHECK(p.dql * p.topk_n <= kMaxCand && threads >= 64 && p.topk_n % 4 == 0 && (p.group == 8 || p.group == 16),
        "sparse_decode_v2: unsupported shape");
    TORCH_CHECK(wpt <= 0 || p.dql * p.group * 16 <= 8 * threads, "sparse_decode_v2: merge needs more threads");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (wpt == -1)
    {
        TORCH_CHECK(
            p.dql * p.group <= umma::kRows && p.stages <= 2, "sparse_decode_v2 (umma2): at most 64 rows, 2 stages");
        launch_umma2(p, grid, umma::Layout2(p.stages).total, pdl, stream);
        return;
    }
    if (wpt == 0)
    {
        TORCH_CHECK(p.dql * p.group <= umma::kRows, "sparse_decode_v2 (umma): at most 64 rows");
        launch_umma(p, grid, umma::Layout(p.stages).total, pdl, stream);
        return;
    }
    if (wpt == 4)
    {
        launch_impl<4, 512, 1>(p, grid, threads, smem, pdl, stream);
    }
    else if (wpt == 2)
    {
        launch_impl<2, 256, 1>(p, grid, threads, smem, pdl, stream);
    }
    else
    {
        TORCH_CHECK(false, "sparse_decode_v2: wpt must be 2 or 4");
    }
}

// Stream-K schedule (see sparse_decode_v2_sk_kernel): prep kernel for the per-group unions, then
// `grid` persistent CTAs. part_o / part_ml need (grid + groups) * rows_pad rows, u_blk / u_page /
// u_mask / u_wcum groups * 64 ints, u_cnt / u_wsum groups ints.
void sparse_decode_v2_sk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor ks, torch::Tensor vs,
    torch::Tensor k_gs, torch::Tensor v_gs, torch::Tensor topk, torch::Tensor bt, torch::Tensor seq_lens,
    torch::Tensor out, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor u_blk,
    torch::Tensor u_page, torch::Tensor u_mask, torch::Tensor u_cnt, torch::Tensor u_wcum, torch::Tensor u_wsum,
    torch::Tensor dbg, int64_t dql, int64_t ntiles, int64_t wpt, int64_t stages, int64_t grid, double sm_scale_log2e,
    bool pdl)
{
    using namespace sdv2;
    const at::cuda::CUDAGuard guard(q.device());
    Params p{};
    p.q = q.data_ptr();
    p.q_sn = q.stride(0);
    p.q_sh = q.stride(1);
    p.k = k.data_ptr<uint8_t>();
    p.v = v.data_ptr<uint8_t>();
    p.k_sb = k.stride(0);
    p.k_sh = k.stride(1);
    p.v_sb = v.stride(0);
    p.v_sh = v.stride(1);
    p.ks = ks.data_ptr<uint8_t>();
    p.vs = vs.data_ptr<uint8_t>();
    p.ks_sb = ks.stride(0);
    p.ks_sh = ks.stride(1);
    p.vs_sb = vs.stride(0);
    p.vs_sh = vs.stride(1);
    p.k_gs = k_gs.data_ptr<float>();
    p.v_gs = v_gs.data_ptr<float>();
    p.topk = topk.data_ptr<int>();
    p.t_sh = topk.stride(0);
    p.t_sn = topk.stride(1);
    p.t_sk = topk.stride(2);
    p.bt = bt.data_ptr<int>();
    p.bt_sb = bt.stride(0);
    p.bt_cols = static_cast<int>(bt.size(1));
    p.seq_lens = seq_lens.data_ptr<int>();
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.o_sn = out.stride(0);
    p.o_sh = out.stride(1);
    p.part_o = reinterpret_cast<__half*>(part_o.data_ptr());
    p.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr());
    p.counters = counters.data_ptr<int>();
    p.dbg = dbg.numel() > 0 ? reinterpret_cast<long long*>(dbg.data_ptr()) : nullptr;
    int const num_reqs = static_cast<int>(seq_lens.size(0));
    p.num_kvh = static_cast<int>(k.size(1));
    p.group = static_cast<int>(q.size(1) / k.size(1));
    p.dql = static_cast<int>(dql);
    p.topk_n = static_cast<int>(topk.size(2));
    p.nsplit = 1;
    p.ntiles = static_cast<int>(ntiles);
    p.wsub = static_cast<int>(wpt);
    p.stages = static_cast<int>(stages);
    p.sm_scale_log2e = static_cast<float>(sm_scale_log2e);
    p.q_fp8 = q.scalar_type() == at::ScalarType::Float8_e4m3fn ? 1 : 0;
    p.cluster_merge = 0;
    p.kv_fused = (p.v == p.k + kKvBytes && p.v_sb == p.k_sb && p.v_sh == p.k_sh) ? 1 : 0;
    p.sc_fused = (p.vs == p.ks + kScBytes && p.vs_sb == p.ks_sb && p.vs_sh == p.ks_sh) ? 1 : 0;
    p.u_blk = u_blk.data_ptr<int>();
    p.u_page = u_page.data_ptr<int>();
    p.u_mask = u_mask.data_ptr<int>();
    p.u_cnt = u_cnt.data_ptr<int>();
    p.u_wcum = u_wcum.data_ptr<int>();
    p.u_wsum = u_wsum.data_ptr<int>();
    p.ngroups = num_reqs * p.num_kvh;

    int const threads = p.ntiles * static_cast<int>(wpt) * 32;
    int const smem = SmemLayout(p.stages, p.ntiles, static_cast<int>(wpt), 1, p.dql * p.group).total;
    if (p.ngroups == 0)
        return;
    TORCH_CHECK(p.dql * p.topk_n <= kMaxCand && threads >= 64 && p.topk_n % 4 == 0 && (p.group == 8 || p.group == 16),
        "sparse_decode_v2_sk: unsupported shape");
    // A CTA's share holds at most kSkMaxPages pages when groups <= 2 x grid (see kSkMaxPages), or trivially
    // when all groups together have no more pages than that.
    TORCH_CHECK(p.ngroups <= kMaxSkGroups && (p.ngroups <= 2 * grid || p.ngroups * kMaxCand <= kSkMaxPages),
        "sparse_decode_v2_sk: too many (request, KV head) groups for the grid");
    TORCH_CHECK(grid >= 1, "sparse_decode_v2_sk: grid must be positive");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (wpt == 4)
    {
        launch_sk_impl<4, 512, 1>(p, static_cast<int>(grid), threads, smem, pdl, stream);
    }
    else if (wpt == 2)
    {
        launch_sk_impl<2, 256, 1>(p, static_cast<int>(grid), threads, smem, pdl, stream);
    }
    else
    {
        TORCH_CHECK(false, "sparse_decode_v2_sk: wpt must be 2 or 4");
    }
}

// How many clusters of `cluster` CTAs of the given geometry can be co-resident.
int64_t max_active_clusters(int64_t wpt, int64_t ntiles, int64_t stages, int64_t nsplit, int64_t rows, int64_t cluster)
{
    using namespace sdv2;
    int const threads = static_cast<int>(ntiles * wpt * 32);
    int const smem = SmemLayout(static_cast<int>(stages), static_cast<int>(ntiles), static_cast<int>(wpt),
        static_cast<int>(nsplit), static_cast<int>(rows))
                         .total;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(cluster) * 64);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = static_cast<unsigned>(cluster);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = -1;
    if (wpt == 4)
    {
        auto kern = sparse_decode_v2_kernel<4, 512, 1>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, kern, &cfg));
    }
    else
    {
        auto kern = sparse_decode_v2_kernel<2, 256, 1>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, kern, &cfg));
    }
    return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_decode_v2", &sparse_decode_v2);
    m.def("sparse_decode_v2_sk", &sparse_decode_v2_sk);
    m.def("max_active_clusters", &max_active_clusters);
}
