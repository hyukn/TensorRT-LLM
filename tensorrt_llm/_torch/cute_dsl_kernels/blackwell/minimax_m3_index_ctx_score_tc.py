# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tcgen05 (UMMA, FP8 kind::f8f6f4) MiniMax-M3 index scorer for context (prefill) rows, SM100 family.

Replaces the fmha_sm100 output_maxscore proxy pass for context rows: same [heads, k_tiles, tokens]
raw Q.K per-block max the block selector consumes (bitwise equal to the proxy), but one CTA per
(128-row q tile, KV split) instead of one CTA walking the whole KV, so short prefills over long
cached prefixes no longer cost ~3.9 ns per cached token per layer.

Same contract, grid and varlen tile lookup as ctx_scorer.py (v1, mma.sync), but the score GEMM
runs on the 5th-gen tensor cores with native FP8 operands:

    A = Q tile: 128 rows of idx_q viewed as [total_q * H, 128]  (TQ = 128 / H tokens x H heads)
    B = one index-K page [128 keys, 128 dims] (K-major), page from block_table
    D = [128 q rows (TMEM lanes), 128 keys (TMEM columns)] fp32, double-buffered in TMEM

Each epilogue thread owns one q row (TMEM lane) and max-reduces its 128 key columns in registers
(causal mask only on blocks reaching past the tile's first query), then stores
score[h, t, block] for its row. No cross-thread reduction is needed.

Warps: 0-3 epilogue (TMEM lanes 0-127), 4 TMA producer (Q once, K pages through a
num_kv_stages ring), 5 MMA issuer.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from .cute_ptx_utils import TORCH_TO_CUTE_DTYPE
from .utils import TRTLLM_ENABLE_PDL, griddepcontrol_wait

__all__ = [
    "CtxIndexScoreTcKernel",
    "ctx_index_score_tc",
    "ctx_plan_tc",
    "tile_blocks",
    "pick_split_k_tc",
]


@dsl_user_op
def fmax3(a, b, c, *, loc=None, ip=None):
    """Three-input max.f32 (PTX 8.8+, sm_100+): halves the FMNMX count of the row reduction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "max.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


class CtxIndexScoreTcKernel:
    BLOCK_K = 128
    BLOCK_M = 128

    def __init__(
        self,
        num_heads: int,
        head_dim: int = 128,
        num_kv_stages: int = 4,
        num_acc_stages: int = 2,
        epi_n: int = 128,
        use_fmax3: bool = True,
    ):
        assert 128 % num_heads == 0
        self.num_heads = num_heads
        self.tile_q = 128 // num_heads
        self.head_dim = head_dim
        self.num_kv_stages = num_kv_stages
        self.num_acc_stages = num_acc_stages
        self.EPI_N = epi_n
        self.use_fmax3 = use_fmax3
        self.mma_tiler = (self.BLOCK_M, self.BLOCK_K, head_dim)
        self.tma_warp = 4
        self.mma_warp = 5
        self.threads = 32 * 6

    def _make_tiled_mma(self, ab_dtype):
        return sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q * H, head_dim, 1), K-major
        mK: cute.Tensor,  # (128 keys, head_dim, num_pages), K-major, page stride arbitrary
        block_table: cute.Tensor,  # [batch, max_pages]
        score: cute.Tensor,  # [H, total_q, max_blocks] any strides
        cu_seqlens_q: cute.Tensor,  # [batch + 1]
        seq_lens: cute.Tensor,  # [batch]
        num_tiles: Int32,
        split_k: Int32,  # grid y = max splits per tile
        blocks_per_cta: Int32,  # a tile with n_blk blocks uses min(split_k, ceil(n_blk / blocks_per_cta)) splits
        stream: CUstream,
    ):
        ab_dtype = mQ.element_type
        tiled_mma = self._make_tiled_mma(ab_dtype)
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, ab_dtype, 1
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, ab_dtype, self.num_kv_stages
        )
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        self.q_bytes = cute.size_in_bytes(ab_dtype, a_smem_layout)
        self.k_bytes = cute.size_in_bytes(ab_dtype, b_smem_layout)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stages))
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            block_table,
            score,
            cu_seqlens_q,
            seq_lens,
            blocks_per_cta,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ).launch(
            grid=(num_tiles, split_k, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        block_table: cute.Tensor,
        score: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        seq_lens: cute.Tensor,
        blocks_per_cta: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = cute.arch.lane_idx()
        tile_id, split_id, _ = cute.arch.block_idx()
        _, split_max, _ = cute.arch.grid_dim()

        H = self.num_heads
        TQ = self.tile_q
        BLOCK_K = self.BLOCK_K

        if warp_idx == self.tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        q_mbar = storage.q_mbar.data_ptr()

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_mbar, 1)
            cute.arch.mbarrier_init_fence()

        kv_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.k_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.num_acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )

        sA = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        cute.arch.sync_threads()

        griddepcontrol_wait()

        # ---- varlen tile lookup (same as v1) ----
        batch = seq_lens.shape[0]
        req = Int32(0)
        excl = Int32(0)
        carry = Int32(0)
        for c in range(0, batch, 32):
            b = c + lane_id
            in_range = b < batch
            nt = Int32(0)
            if in_range:
                qlen = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
                nt = (qlen + (TQ - 1)) // TQ
            incl = nt
            for i in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_up(incl, offset=1 << i)
                incl = incl + (other if lane_id >= (1 << i) else Int32(0))
            incl = incl + carry
            done = in_range and incl <= tile_id
            req = req + cute.arch.warp_reduction_sum(Int32(1) if done else Int32(0))
            excl = excl + cute.arch.warp_reduction_sum(nt if done else Int32(0))
            carry = cute.arch.shuffle_sync(incl, 31)

        n_blk = Int32(0)
        q_begin = Int32(0)
        q_count = Int32(0)
        pos0 = Int32(0)
        if req < batch:
            q_lo = cu_seqlens_q[req]
            q_hi = cu_seqlens_q[req + 1]
            local_tile = tile_id - excl
            q_begin = q_lo + local_tile * TQ
            q_count = cutlass.min(Int32(TQ), q_hi - q_begin)
            seqlen = seq_lens[req]
            pos0 = seqlen - (q_hi - q_lo) + local_tile * TQ
            last_pos = pos0 + q_count - 1
            if last_pos >= 0:
                n_blk = last_pos // BLOCK_K + 1

        # Per-tile split count: long requests get many CTAs, short ones few, so every CTA walks
        # about blocks_per_cta blocks (balances varlen batches); grid y is only an upper bound.
        split_k = cutlass.min(
            split_max, cutlass.max(Int32(1), (n_blk + blocks_per_cta - 1) // blocks_per_cta)
        )
        if split_id < split_k and split_id < n_blk:
            thr_mma = tiled_mma.get_slice(0)
            # A: Q rows [q_begin * H, q_begin * H + 128) (TMA zero-fills past the end)
            gA = cute.local_tile(
                cute.domain_offset((q_begin * H, 0, 0), mA),
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            tCgA = thr_mma.partition_A(gA)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            gB = cute.local_tile(
                mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            tCgB = thr_mma.partition_B(gB)
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stages))

            if warp_idx == self.tma_warp:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_mbar, self.q_bytes)
                cute.copy(tma_atom_a, tAgA[(None, 0, 0, 0)], tAsA[(None, 0)], tma_bar_ptr=q_mbar)
                kv_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_kv_stages
                )
                # Page ids are fetched 32 at a time (one per lane) and broadcast by shuffle, so the
                # TMA issue loop does not wait on a dependent block_table load per block (measured
                # neutral on the benchmark shapes; kept since it can only shorten the issue loop).
                n_mine = (n_blk - split_id + split_k - 1) // split_k
                my_page = Int32(0)
                for i in range(n_mine):
                    if i % 32 == 0:
                        my_blk = split_id + (i + lane_id) * split_k
                        my_page = Int32(0)
                        if my_blk < n_blk:
                            my_page = block_table[req, my_blk]
                    page_id = cute.arch.shuffle_sync(my_page, i % 32)
                    kv_pipeline.producer_acquire(kv_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, 0, 0, page_id)],
                        tBsB[(None, kv_state.index)],
                        tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_state),
                    )
                    kv_state.advance()
                kv_pipeline.producer_tail(kv_state)

            elif warp_idx == self.mma_warp:
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(Float32)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                cute.arch.mbarrier_wait(q_mbar, 0)
                kv_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_kv_stages
                )
                acc_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stages
                )
                num_kblocks = cute.size(tCrA, mode=[2])
                for block_id in range(split_id, n_blk, split_k):
                    acc_pipeline.producer_acquire(acc_state)
                    kv_pipeline.consumer_wait(kv_state)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range(num_kblocks, unroll_full=True):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kb, 0)],
                            tCrB[(None, None, kb, kv_state.index)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_pipeline.consumer_release(kv_state)
                    acc_pipeline.producer_commit(acc_state)
                    kv_state.advance()
                    acc_state.advance()
                acc_pipeline.producer_tail(acc_state)

            elif warp_idx < 4:
                tmem.allocate(self.num_tmem_cols)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(Float32)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                EPI_N = self.EPI_N
                rep = {
                    16: tcgen05.Repetition.x16,
                    32: tcgen05.Repetition.x32,
                    64: tcgen05.Repetition.x64,
                    128: tcgen05.Repetition.x128,
                }[EPI_N]
                copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), Float32)
                tAcc_epi = cute.flat_divide(
                    tCtAcc_base[((None, None), 0, 0, None)], (self.BLOCK_M, EPI_N)
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                cAcc = cute.make_identity_tensor((self.BLOCK_M, BLOCK_K))
                tTR_cAcc = thr_copy_t2r.partition_D(cute.flat_divide(cAcc, (self.BLOCK_M, EPI_N)))
                tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[(None, None, None, 0, 0)].shape, Float32)
                n_elem = cute.size(tTR_rAcc)

                row = tTR_cAcc[0][0]
                head = row % H
                tok = row // H
                q_pos = pos0 + tok
                store_ok = tok < q_count
                t_store = q_begin + tok

                acc_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stages
                )
                for block_id in range(split_id, n_blk, split_k):
                    acc_pipeline.consumer_wait(acc_state)
                    need_mask = block_id * BLOCK_K + (BLOCK_K - 1) > pos0
                    m0 = Float32(float("-inf"))
                    m1 = Float32(float("-inf"))
                    m2 = Float32(float("-inf"))
                    m3 = Float32(float("-inf"))
                    n_sub = BLOCK_K // EPI_N
                    for s in cutlass.range_constexpr(n_sub):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, 0, s, acc_state.index)],
                            tTR_rAcc,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(s == n_sub - 1):
                            # Everything of this accumulator is in registers: hand it back to the MMA
                            # warp before reducing so the next block's UMMA overlaps the reduction.
                            acc_pipeline.consumer_release(acc_state)
                        if need_mask:
                            for i in cutlass.range_constexpr(n_elem):
                                k_pos = (
                                    block_id * BLOCK_K + tTR_cAcc[(None, None, None, 0, s)][i][1]
                                )
                                tTR_rAcc[i] = (
                                    tTR_rAcc[i] if q_pos >= k_pos else Float32(float("-inf"))
                                )
                        if cutlass.const_expr(self.use_fmax3):
                            for i in cutlass.range_constexpr(0, n_elem, 8):
                                m0 = fmax3(m0, tTR_rAcc[i], tTR_rAcc[i + 1])
                                m1 = fmax3(m1, tTR_rAcc[i + 2], tTR_rAcc[i + 3])
                                m2 = fmax3(m2, tTR_rAcc[i + 4], tTR_rAcc[i + 5])
                                m3 = fmax3(m3, tTR_rAcc[i + 6], tTR_rAcc[i + 7])
                        else:
                            for i in cutlass.range_constexpr(0, n_elem, 4):
                                m0 = cute.arch.fmax(m0, tTR_rAcc[i])
                                m1 = cute.arch.fmax(m1, tTR_rAcc[i + 1])
                                m2 = cute.arch.fmax(m2, tTR_rAcc[i + 2])
                                m3 = cute.arch.fmax(m3, tTR_rAcc[i + 3])
                    acc_state.advance()
                    mx = cute.arch.fmax(cute.arch.fmax(m0, m1), cute.arch.fmax(m2, m3))
                    if store_ok:
                        score[head, t_store, block_id] = mx

                tmem.relinquish_alloc_permit()
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                tmem.free(tmem_ptr)


# ----------------------------------------------------------------------------------------
_CACHE = {}


def _compile(q_dtype, num_heads, head_dim, num_kv_stages, num_acc_stages, epi_n, use_fmax3):
    key = (q_dtype, num_heads, head_dim, num_kv_stages, num_acc_stages, epi_n, use_fmax3)
    fn = _CACHE.get(key)
    if fn is not None:
        return fn
    cdt = TORCH_TO_CUTE_DTYPE[q_dtype]
    q_fake = cute.runtime.make_fake_tensor(
        cdt, (cute.sym_int(), head_dim, 1), stride=(head_dim, 1, cute.sym_int64(divisibility=16))
    )
    k_fake = cute.runtime.make_fake_tensor(
        cdt, (128, head_dim, cute.sym_int()), stride=(head_dim, 1, cute.sym_int64(divisibility=16))
    )
    sym_b = cute.sym_int()
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_b, cute.sym_int()), stride_order=(1, 0)
    )
    score_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (num_heads, cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
    )
    cu_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(),), stride_order=(0,)
    )
    sl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_b,), stride_order=(0,))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kern = CtxIndexScoreTcKernel(
        num_heads,
        head_dim,
        num_kv_stages=num_kv_stages,
        num_acc_stages=num_acc_stages,
        epi_n=epi_n,
        use_fmax3=use_fmax3,
    )
    fn = cute.compile(
        kern,
        q_fake,
        k_fake,
        bt_fake,
        score_fake,
        cu_fake,
        sl_fake,
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )
    _CACHE[key] = fn
    return fn


_NUM_SMS = None


@functools.lru_cache(maxsize=4096)
def pick_split_k_tc(
    num_tiles: int, max_blocks: int, ctas_per_sm: int = 1, overhead_blocks: int = 6
) -> int:
    """Wave-quantization-aware split from host-known sizes only (graph-safe).

    With the x128 TMEM epilogue the kernel runs 1 CTA/SM (registers), so a grid of
    num_tiles * split CTAs runs in ceil(ctas / slots) waves, each as long as one CTA's
    ceil(max_blocks / split) blocks plus a fixed prologue (~overhead_blocks). Candidates are
    the largest split that fits each wave count; pick the cheapest. Tuned on GB300
    (bench_ctx_scorer.py --tune-tc).
    """
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    slots = _NUM_SMS * ctas_per_sm
    tiles = max(1, num_tiles)
    best = (float("inf"), 1)
    for waves in range(1, 65):
        split = min(max_blocks, max(1, (waves * slots) // tiles))
        w = -(-tiles * split // slots)
        cost = w * (-(-max_blocks // split) + overhead_blocks)
        if cost < best[0]:
            best = (cost, split)
        if split >= max_blocks:
            break
    return int(best[1])


def tile_blocks(qo_lens_cpu, kv_lens_cpu, num_heads: int) -> torch.Tensor:
    """Host: per-q-tile causal block count n_blk (what each tile's CTAs walk), int64 [num_tiles]."""
    tq = 128 // num_heads
    out = []
    for qo, kv in zip(qo_lens_cpu.tolist(), kv_lens_cpu.tolist()):
        if qo <= 0:
            continue
        ends = torch.arange(1, -(-qo // tq) + 1, dtype=torch.int64) * tq
        last = (kv - qo) + torch.clamp(ends, max=qo) - 1
        out.append((last // 128 + 1).clamp(min=0))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.int64)


def ctx_plan_tc(qo_lens_cpu, kv_lens_cpu, num_heads: int, *, overhead_blocks: int = 6):
    """Host-side launch plan (split_max, blocks_per_cta) from the step's host lengths (the same
    qo_lens_cpu / kv_lens_cpu the fmha proxy plan uses; compute once per step, reuse per layer).

    Tile t runs min(split_max, ceil(n_blk_t / bpc)) CTAs. Cost model (1 CTA/SM):
    ceil(num_ctas / SMs) waves x (bpc + overhead). Picks the cheapest bpc in 4..256, which
    lands the CTA count just under a wave multiple and balances varlen batches.
    """
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    nb = tile_blocks(qo_lens_cpu, kv_lens_cpu, num_heads)
    nb = nb[nb > 0]
    if nb.numel() == 0:
        return 1, 1
    cand = torch.arange(4, 257, dtype=torch.int64)
    per_tile = -(-nb[None, :] // cand[:, None])  # [cand, tiles] CTAs per tile
    ctas = per_tile.sum(1)
    longest = (-(-nb[None, :] // per_tile)).max(1).values  # longest CTA (blocks)
    waves = -(-ctas // _NUM_SMS)
    cost = waves * (longest + overhead_blocks)
    i = int(torch.argmin(cost))
    bpc = int(cand[i])
    split_max = int(per_tile[i].max())
    return split_max, bpc


def ctx_index_score_tc(
    idx_q,
    index_k_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    score,
    *,
    split_k=None,
    num_kv_stages=4,
    num_acc_stages=2,
    epi_n=128,
    use_fmax3=True,
    max_blocks_hint=None,
    plan=None,
):
    """Same contract as ctx_scorer.ctx_index_score (v1). FP8 E4M3 only; idx_q must be
    [total_q, H, 128] with the (H, D) part contiguous so rows flatten to [total_q * H, 128].
    Launch geometry depends only on shapes and optional host hints (never on device data):
      max_blocks_hint  - max ceil(kv_len / 128) of the step (else block_table width);
      plan             - ctx_plan_tc(qo_lens_cpu, kv_lens_cpu, H) = (split_max, blocks_per_cta):
                         per-tile split counts with ~equal blocks per CTA (balances varlen
                         batches, avoids wave cliffs). Recommended in production.
    Correctness never depends on the hints (graph replays with other lengths stay correct)."""
    assert idx_q.dtype == torch.float8_e4m3fn and index_k_cache.dtype == torch.float8_e4m3fn
    if index_k_cache.dim() == 4:
        index_k_cache = index_k_cache.squeeze(1)
    total_q, H, D = idx_q.shape
    batch = int(seq_lens.shape[0])
    if total_q == 0 or batch == 0:
        return
    assert idx_q.stride(2) == 1 and idx_q.stride(1) == D and idx_q.stride(0) == H * D
    rows = total_q * H
    q3 = torch.as_strided(idx_q, (rows, D, 1), (D, 1, max(16, rows * D)))
    k3 = index_k_cache.permute(1, 2, 0)  # (128 keys, D, pages), page stride arbitrary
    tq = 128 // H
    n_tiles = (total_q + batch * (tq - 1)) // tq
    max_blocks = int(max_blocks_hint or block_table.shape[1])
    if split_k is not None:
        bpc = 1  # explicit uniform split (tests / tuning)
    elif plan is not None:
        split_k, bpc = plan
    else:
        split_k = pick_split_k_tc(n_tiles, max_blocks)
        bpc = -(-max_blocks // split_k)
    fn = _compile(idx_q.dtype, H, D, num_kv_stages, num_acc_stages, epi_n, use_fmax3)
    fn(q3, k3, block_table, score, cu_seqlens_q, seq_lens, n_tiles, split_k, bpc)
