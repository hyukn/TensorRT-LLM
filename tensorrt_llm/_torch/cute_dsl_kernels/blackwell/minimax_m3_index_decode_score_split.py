# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
"""CuTe DSL MiniMax-M3 index decode scorer, split across tensor-parallel ranks (Blackwell SM100).

Derived from minimax_m3_index_decode_score.py (vendored from vLLM, Apache-2.0; see that file for
the upstream link and its differences from upstream). The per-block math is the same instruction
sequence, so each score is bitwise identical to the unsplit kernel's.

Under tensor parallelism every rank holds the whole MQA index-K but only its own index heads, so
the unsplit kernel reads the full index-K once per rank. Here rank r of W scores every index head
on blocks r, r + W, r + 2W, ... (each index-K block is read by exactly one rank) and writes head
h's scores into score<h>, which the caller points at the owning rank's buffer (peer memory). After
a cross-rank barrier every rank holds its own heads' scores for all blocks.

Scheduling. A (request, split) grid of CTAs a few blocks long each leaves the GPU draining at
partial occupancy for a whole CTA lifetime at the end, and re-pays the CTA prologue (lengths,
page ids, Q) every few blocks. Instead the grid is persistent -- split_k CTAs, CTAS_PER_SM per
SM -- and the work is balanced explicitly. The rank's blocks of all requests form one list,
request-major (request b's j-th entry is block r + j * W). CTA c streams an equal static share of
the first (100 - DYN_PCT)% of the list, then claims DYN_CHUNK blocks at a time from the rest
through an atomic counter (work_counter[0], low 32 bits): memory bandwidth is not shared evenly
between SMs, or between the CTAs of one SM, so a purely static split leaves the kernel waiting for
its slowest CTA. Every CTA counts itself finished in the high 32 bits, and the last one zeroes the
counter for the next launch (graph replays included). Launches sharing a counter must not overlap;
on one stream they cannot, since a PDL launch only reads the counter after griddepcontrol.wait.
Small launches (under DYN_MIN_PER_CTA blocks per CTA) stay fully static.

Within a CTA the TMA warp walks the CTA's slices and pushes ring items -- a request's Q tile, then
its K tiles -- with a small SMEM descriptor per stage (request, block id, length), so the four MMA
warps just follow the ring. Each MMA warp writes the scores of one index head.
"""

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Int64, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils.smem_allocator import SmemAllocator

from .cute_ptx_utils import EVICT_FIRST, fp8x4_to_fp16x4, mma_sync, simple_tma_copy
from .utils import TRTLLM_ENABLE_PDL, griddepcontrol_wait

__all__ = ["IndexDecodeScoreSplitKernel"]


@cute.jit
def _fp8_to_f16_mma_fragments(src: cute.Tensor):
    """Split an FP8 ldmatrix fragment into the two FP16 k-fragments MMA wants."""
    src_elems = cute.size(src)
    src_u32 = cute.recast_tensor(src, Uint32)
    src_f16 = cute.make_rmem_tensor(src_elems, Float16)
    src_f16_u32 = cute.recast_tensor(src_f16, Uint32)
    # Packed conversion; faster and fewer SASS instructions than
    # src.load().to(Float16).
    for i in cutlass.range_constexpr(src_elems // 4):
        converted = fp8x4_to_fp16x4(src_u32[i])
        src_f16_u32[i * 2] = converted[0]
        src_f16_u32[i * 2 + 1] = converted[1]
    lower = cute.make_rmem_tensor(src_elems // 2, Float16)
    upper = cute.make_rmem_tensor(src_elems // 2, Float16)

    # FP8 ldmatrix yields four consecutive values along K. Split each group
    # into the lower two and upper two for the two FP16 MMA k-fragments.
    for i in cutlass.range_constexpr(src_elems // 2):
        lower[i] = src_f16[(i // 2) * 4 + i % 2]
        upper[i] = src_f16[(i // 2) * 4 + 2 + i % 2]
    return lower, upper


@cute.jit
def _warp_sum(x: Int32) -> Int32:
    for i in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=1 << i)
    return x


@cute.jit
def _warp_inclusive_sum(x: Int32, lane_id: Int32) -> Int32:
    for i in cutlass.range_constexpr(5):
        y = cute.arch.shuffle_sync_up(x, offset=1 << i)
        x = x + (y if lane_id >= (1 << i) else Int32(0))
    return x


class IndexDecodeScoreSplitKernel:
    """Per-block max index score for one decode step, rank-split, persistent grid.

    Grid is (split_k,), normally CTAS_PER_SM per SM; see the module docstring
    for how the rank's blocks are distributed. Every block's score is written
    directly, so no cross-CTA reduction is needed.
    """

    BLOCK_K = 128
    BAR_MMA = 1
    # Resident CTAs per SM the persistent grid is sized for (also the launch
    # bound that keeps registers and SMEM within that occupancy).
    CTAS_PER_SM = 4
    NUM_STAGES = 2
    # Share of the rank's blocks handed out dynamically, in chunks of DYN_CHUNK,
    # once each CTA has at least DYN_MIN_PER_CTA blocks.
    DYN_PCT = 15
    DYN_CHUNK = 1
    DYN_MIN_PER_CTA = 4
    # Requests whose lengths the TMA warp keeps in registers (in units of 32).
    SEQ_LEN_CHUNKS = 2
    # Descriptor block ids of the non-K ring items.
    ITEM_Q = -1
    ITEM_END = -2

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
        rank: int = 0,
        world: int = 1,
        *,
        num_stages: int = NUM_STAGES,
        dyn_pct: int = DYN_PCT,
        dyn_chunk: int = DYN_CHUNK,
    ):
        assert 0 <= rank < world
        assert num_heads <= 4, "one MMA warp writes each head's scores"
        assert 0 <= dyn_pct <= 100 and dyn_chunk >= 1
        self.rank = rank
        self.world = world
        self.dtype = dtype
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim
        self.num_stages = num_stages
        self.dyn_pct = dyn_pct
        self.dyn_chunk = dyn_chunk

    def _swizzle_elems(self) -> int:
        """Elements spanned by one 128-byte swizzle atom."""
        return 128 * 8 // self.dtype.width

    def _sq_layout(self):
        """Composed SMEM layout for the Q tile, shared by the descriptor and the kernel."""
        elems = self._swizzle_elems()
        head_dim = self.head_dim
        block_q = self.num_heads * self.max_decode_query_len
        layout = cute.make_layout(
            (self.max_decode_query_len, self.num_heads, (elems, head_dim // elems)),
            stride=(elems, self.max_decode_query_len * elems, (1, block_q * elems)),
        )
        return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, layout)

    def _sk_layout(self):
        """Composed SMEM layout for the pipelined K tiles."""
        elems = self._swizzle_elems()
        head_dim = self.head_dim
        block_k = self.BLOCK_K
        layout = cute.make_layout(
            (1, block_k, (elems, head_dim // elems), self.num_stages),
            stride=(0, elems, (1, block_k * elems), block_k * head_dim),
        )
        return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, layout)

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        score0: cute.Tensor,  # [bs * runtime_decode_query_len, max_pages] for head 0
        score1: cute.Tensor,  # ... head 1 (may alias score0 when num_heads < 2)
        score2: cute.Tensor,
        score3: cute.Tensor,
        seq_lens: cute.Tensor,  # [bs]
        work_counter: cute.Tensor,  # [1] int64, zero between launches
        stream: CUstream,
    ):
        num_heads = self.num_heads
        head_dim = self.head_dim
        MAX_DQL = self.max_decode_query_len
        assert num_heads * MAX_DQL <= 32

        batch = seq_lens.shape[0]
        decode_query_len = gQ.shape[0] // batch
        grid = (self.split_k, 1, 1)
        block = (32 * 5, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        elems = self._swizzle_elems()

        q_tma_atom, q_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, elems)),
            self._sq_layout(),
            (MAX_DQL, num_heads, head_dim),
        )
        k_tma_atom, k_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            self._sk_layout(),
            (1, self.BLOCK_K, head_dim),
        )

        self.kernel(
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            block_table,
            score0,
            score1,
            score2,
            score3,
            seq_lens,
            work_counter,
            decode_query_len,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.CTAS_PER_SM,
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def kernel(
        self,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        block_table: cute.Tensor,
        score0: cute.Tensor,
        score1: cute.Tensor,
        score2: cute.Tensor,
        score3: cute.Tensor,
        seq_lens: cute.Tensor,
        work_counter: cute.Tensor,
        decode_query_len,
    ):
        cta_id, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        BLOCK_K = self.BLOCK_K
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        EPI_Q = Q_TILES * MMA_N
        RANK = self.rank
        WORLD = self.world
        ITEM_Q = self.ITEM_Q
        ITEM_END = self.ITEM_END
        DYN = self.dyn_pct > 0

        sq_layout = self._sq_layout()
        sk_layout = self._sk_layout()

        smem = SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            sk_layout.outer,
            byte_alignment=128,
            swizzle=sk_layout.inner,
        )[0, None, None, None]
        q_tma_elems = self._swizzle_elems()
        epi_buffer = smem.allocate_tensor(Float32, cute.make_layout((EPI_Q, 4)))
        # Per-stage item descriptor: (request, block id or ITEM_*, request length).
        desc = smem.allocate_tensor(Int32, cute.make_layout((num_stages, 4), stride=(4, 1)))

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        # Barrier init and descriptor prefetch do not depend on the predecessor
        # grid, so they are the work PDL can overlap.
        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(q_tma_atom)
            cpasync.prefetch_descriptor(k_tma_atom)
        cute.arch.sync_threads()

        # No explicit launch_dependents: the only consumer is the block selector,
        # which reduces over every block this grid writes, so triggering it early
        # would leave it resident and spinning in its own wait. The implicit
        # trigger as the CTAs exit still gives it the launch overlap.
        griddepcontrol_wait()

        if warp_id == 4:
            # TMA warp. seq_lens must be read after the wait: it is a graph-stable
            # buffer that on_update_kv_lens patches every step, and the block counts
            # derived from it bound an unmasked block_table load and an unmasked
            # score store, so a stale length is an out-of-bounds write rather than
            # just a wrong answer.
            #
            # Request b owns n_b = ceil((ceil(len_b / BLOCK_K) - RANK) / WORLD) of
            # this rank's blocks. The first NCACHE * 32 lengths are loaded at once
            # (one memory round trip) and kept in registers; any further ones are
            # reloaded per walk.
            batch = seq_lens.shape[0]
            NCACHE = self.SEQ_LEN_CHUNKS
            sl_cache = []
            for c in cutlass.range_constexpr(NCACHE):
                bc = c * 32 + lane_id
                v = seq_lens[cutlass.min(bc, batch - 1)]
                sl_cache.append(v if bc < batch else Int32(0))
            total = Int32(0)
            for c in cutlass.range_constexpr(NCACHE):
                total = total + _warp_sum(
                    (cute.ceil_div(sl_cache[c], BLOCK_K) + (WORLD - 1 - RANK)) // WORLD
                )
            for b0 in range(NCACHE * 32, batch, 32):
                bf = b0 + lane_id
                cntf = Int32(0)
                if bf < batch:
                    cntf = (cute.ceil_div(seq_lens[bf], BLOCK_K) + (WORLD - 1 - RANK)) // WORLD
                total = total + _warp_sum(cntf)

            n_static = total
            use_dyn = Int32(0)
            if cutlass.const_expr(DYN):
                # Small launches stay fully static: a claim costs an atomic round
                # trip, which only pays off when each CTA has many blocks to stream.
                if total >= num_ctas * self.DYN_MIN_PER_CTA:
                    use_dyn = Int32(1)
                    n_static = total - Int32((Int64(total) * self.dyn_pct) // 100)
            range_lo = Int32((Int64(cta_id) * Int64(n_static)) // Int64(num_ctas))
            range_hi = Int32((Int64(cta_id + 1) * Int64(n_static)) // Int64(num_ctas))

            tma_stage = Int32(0)
            tma_parity = Int32(1)
            last_q = Int32(-1)
            walking = Int32(1)
            while walking != 0:
                # Walk the requests overlapping [range_lo, range_hi), 32 at a time.
                carry = Int32(0)
                for b0 in range(0, batch, 32):
                    b = b0 + lane_id
                    seqlen_lane = Int32(0)
                    if b0 < NCACHE * 32:
                        if cutlass.const_expr(NCACHE > 0):
                            seqlen_lane = sl_cache[NCACHE - 1]
                            for c in cutlass.range_constexpr(NCACHE - 1):
                                seqlen_lane = sl_cache[c] if b0 == c * 32 else seqlen_lane
                    elif b < batch:
                        seqlen_lane = seq_lens[b]
                    cnt = (cute.ceil_div(seqlen_lane, BLOCK_K) + (WORLD - 1 - RANK)) // WORLD
                    incl = _warp_inclusive_sum(cnt, lane_id)
                    excl = carry + incl - cnt
                    seg_lo = cutlass.max(excl, range_lo) - excl
                    seg_hi = cutlass.min(excl + cnt, range_hi) - excl
                    pending = cute.arch.vote_ballot_sync(seg_lo < seg_hi)
                    while pending != 0:
                        src = cute.arch.popc((pending & (-pending)) - 1)
                        pending = pending & (pending - 1)
                        batch_id = cute.arch.shuffle_sync(b, offset=src)
                        j_lo = cute.arch.shuffle_sync(seg_lo, offset=src)
                        j_hi = cute.arch.shuffle_sync(seg_hi, offset=src)
                        seqlen = cute.arch.shuffle_sync(seqlen_lane, offset=src)

                        if batch_id != last_q:
                            # The request's Q tile, unless the MMA warps hold it already.
                            gQ_tile = cute.local_tile(
                                cute.domain_offset(
                                    (batch_id * decode_query_len, 0, 0), q_tma_tensor
                                ),
                                tiler=(MAX_DQL, NUM_HEADS, head_dim),
                                coord=(0, 0, 0),
                            )
                            sQ_tma = cute.make_tensor(
                                sK[None, None, tma_stage].iterator, layout=sq_layout.outer
                            )
                            q_mbar = tma_full_mbar + tma_stage
                            cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                            with cute.arch.elect_one():
                                desc[tma_stage, 0] = batch_id
                                desc[tma_stage, 1] = Int32(ITEM_Q)
                                desc[tma_stage, 2] = seqlen
                                Q_size = BLOCK_Q * head_dim * (dtype.width // 8)
                                cute.arch.mbarrier_arrive_and_expect_tx(q_mbar, Q_size)
                            # TMA bounds-checks rows when the runtime decode_query_len
                            # is below MAX_DQL; padded Q columns are masked before the
                            # stores.
                            simple_tma_copy(q_tma_atom, gQ_tile, sQ_tma, q_mbar)
                            tma_stage = tma_stage + 1
                            if tma_stage == num_stages:
                                tma_stage = Int32(0)
                                tma_parity = tma_parity ^ 1
                            last_q = batch_id

                        # Page ids are fetched 32 at a time, one per lane, so the
                        # dependent block_table load is paid once per 32 tiles.
                        for j0 in range(j_lo, j_hi, 32):
                            page_lane = Int32(0)
                            if j0 + lane_id < j_hi:
                                page_lane = block_table[batch_id, RANK + (j0 + lane_id) * WORLD]
                            for i in range(cutlass.min(j_hi - j0, 32)):
                                page_id = cute.arch.shuffle_sync(page_lane, offset=i)
                                gK_tile = k_tma_tensor[page_id, None, None]
                                k_mbar = tma_full_mbar + tma_stage

                                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                                with cute.arch.elect_one():
                                    desc[tma_stage, 0] = batch_id
                                    desc[tma_stage, 1] = RANK + (j0 + i) * WORLD
                                    desc[tma_stage, 2] = seqlen
                                    K_size = BLOCK_K * head_dim * (dtype.width // 8)
                                    cute.arch.mbarrier_arrive_and_expect_tx(k_mbar, K_size)
                                simple_tma_copy(
                                    k_tma_atom,
                                    gK_tile,
                                    sK[None, None, tma_stage],
                                    k_mbar,
                                    cache_policy=EVICT_FIRST,
                                )
                                tma_stage = tma_stage + 1
                                if tma_stage == num_stages:
                                    tma_stage = Int32(0)
                                    tma_parity = tma_parity ^ 1
                    carry = carry + cute.arch.shuffle_sync(incl, offset=31)

                walking = Int32(0)
                if cutlass.const_expr(DYN):
                    if use_dyn != 0:
                        # Claim the next chunk of the dynamic share. The ring still
                        # holds num_stages items for the MMA warps, which hides most
                        # of the atomic's latency.
                        claimed = Int64(0)
                        if lane_id == 0:
                            claimed = cute.arch.atomic_add(
                                work_counter.iterator, Int64(self.dyn_chunk)
                            )
                        claimed = cute.arch.shuffle_sync(claimed, offset=0)
                        claim_lo = n_static + Int32(claimed & Int64(0xFFFFFFFF))
                        if claim_lo < total:
                            range_lo = claim_lo
                            range_hi = cutlass.min(claim_lo + self.dyn_chunk, total)
                            walking = Int32(1)

            # End of work: an item with no payload releases the MMA warps.
            cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
            with cute.arch.elect_one():
                desc[tma_stage, 1] = Int32(ITEM_END)
                cute.arch.mbarrier_arrive(tma_full_mbar + tma_stage)

            if cutlass.const_expr(DYN):
                # Count this CTA as finished (high 32 bits); the last one resets the
                # counter for the next launch. Every CTA's claims precede its finish
                # increment on the same address, so none can still be claiming.
                if lane_id == 0 and use_dyn != 0:
                    done = cute.arch.atomic_add(work_counter.iterator, Int64(1) << 32)
                    if (done >> 32) == Int64(num_ctas - 1):
                        cute.arch.atomic_exch(work_counter.iterator, Int64(0))

        else:
            # MMA warps; each handles K[32, head_dim] @ Q[BLOCK_Q, head_dim].T
            sK_warp = cute.local_tile(sK, (32, head_dim, num_stages), (warp_id, 0, 0))

            elems = 128 // dtype.width  # 16B
            MMA_K = 32 * 8 // dtype.width  # 32B

            # Pre-compute ldmatrix addresses.
            # sK loads a [16 x 16B] tile:
            #   ((16, (16B, 2), 1), (32 / 16, head_dim / 32B, num_stages))
            sK_ldsm = cute.zipped_divide(sK_warp, (16, cute.make_layout((elems, 2)), 1))
            # sK: (16B, (32 / 16, head_dim / 32B, num_stages))
            sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]

            ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
            ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

            rQ = cute.make_rmem_tensor(((elems // 2, 2), head_dim // (MMA_K * 2), Q_TILES), dtype)
            rK = cute.make_rmem_tensor((elems, 2, head_dim // MMA_K), dtype)
            rC = cute.make_rmem_tensor((4, 2, Q_TILES), Float32)
            rQ_f16 = cute.make_rmem_tensor((4, head_dim // MMA_K, Q_TILES, 2), Float16)

            # Warp h writes head h's scores, lane q the score of query q, so a
            # store is one warp-uniform address computation instead of a branch
            # over heads (and each score is stored once, not by every warp).
            st_base = score0.iterator.toint()
            st_t = Int64(score0.stride[0])
            st_b = Int64(score0.stride[1])
            if cutlass.const_expr(NUM_HEADS > 1):
                st_base = score1.iterator.toint() if warp_id == 1 else st_base
                st_t = Int64(score1.stride[0]) if warp_id == 1 else st_t
                st_b = Int64(score1.stride[1]) if warp_id == 1 else st_b
            if cutlass.const_expr(NUM_HEADS > 2):
                st_base = score2.iterator.toint() if warp_id == 2 else st_base
                st_t = Int64(score2.stride[0]) if warp_id == 2 else st_t
                st_b = Int64(score2.stride[1]) if warp_id == 2 else st_b
            if cutlass.const_expr(NUM_HEADS > 3):
                st_base = score3.iterator.toint() if warp_id == 3 else st_base
                st_t = Int64(score3.stride[0]) if warp_id == 3 else st_t
                st_b = Int64(score3.stride[1]) if warp_id == 3 else st_b
            store_lane = lane_id < MAX_DQL and lane_id < decode_query_len and warp_id < NUM_HEADS

            tma_stage = Int32(0)
            tma_parity = Int32(0)
            q_start = Int32(0)
            st_row = Int64(0)
            # The ring delivers Q items, runs of K items and one END item; K runs are
            # the inner loop so the per-request Q path stays out of it.
            if warp_id == 0:
                cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
            cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
            item_req = desc[tma_stage, 0]
            block_id = desc[tma_stage, 1]
            item_len = desc[tma_stage, 2]
            running = Int32(1)
            while running != 0:
                if block_id == ITEM_END:
                    running = Int32(0)
                else:
                    if block_id == ITEM_Q:
                        q_start = item_len - decode_query_len
                        # Address of score<h>[the request's query `lane`, block 0].
                        st_row = st_base + Int64(item_req * decode_query_len + lane_id) * st_t * 4

                        # TMA sees Q as (query, head, dim) while ldmatrix consumes a
                        # flattened Q column mode. The target profile keeps the rank-2
                        # view even for degenerate shapes such as MAX_DQL == 1.
                        sQ_tma = cute.make_tensor(
                            sK[None, None, tma_stage].iterator, layout=sq_layout.outer
                        )
                        sQ = cute.coalesce(
                            cute.group_modes(sQ_tma, 0, 2),
                            target_profile=(BLOCK_Q, (q_tma_elems, head_dim // q_tma_elems)),
                        )
                        # sQ loads an [8 x 32B] tile:
                        #   ((8, (16B, 4)), (BLOCK_Q / MMA_N, head_dim / 64B))
                        sQ_ldsm = cute.zipped_divide(sQ, (MMA_N, cute.make_layout((elems, 4))))
                        # sQ: (16B, (BLOCK_Q / MMA_N, head_dim / 64B))
                        sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]
                        for q in cutlass.range_constexpr(Q_TILES):
                            cute.copy(ldsm_atom, sQ_ldsm[None, (q, None)], rQ[None, None, q])
                        cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                        if cutlass.const_expr(dtype is Float8E4M3FN):
                            q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                            rQ_f16[None, None, None, 0].store(q_lower.load())
                            rQ_f16[None, None, None, 1].store(q_upper.load())
                        tma_stage = tma_stage + 1
                        if tma_stage == num_stages:
                            tma_stage = Int32(0)
                            tma_parity = tma_parity ^ 1
                        if warp_id == 0:
                            cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                        cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
                        item_req = desc[tma_stage, 0]
                        block_id = desc[tma_stage, 1]
                        item_len = desc[tma_stage, 2]
                    while block_id >= 0:
                        # The causal mask selects every element of a block that ends
                        # at or before the first query position, so only a request's
                        # tail blocks run the masked copy of the body.
                        need_mask = (
                            Int32(1) if block_id * BLOCK_K + (BLOCK_K - 1) > q_start else Int32(0)
                        )
                        for masked in cutlass.range_constexpr(2):
                            if need_mask == masked:
                                rC.fill(0.0)
                                for k in cutlass.range_constexpr(head_dim // MMA_K):
                                    cute.copy(
                                        ldsm_atom,
                                        sK_ldsm[None, (None, k, tma_stage)],
                                        rK[None, None, k],
                                    )
                                    for m in cutlass.range_constexpr(2):
                                        if cutlass.const_expr(dtype is Float8E4M3FN):
                                            rK_lower, rK_upper = _fp8_to_f16_mma_fragments(
                                                rK[None, m, k]
                                            )
                                            for n in cutlass.range_constexpr(Q_TILES):
                                                rC[None, m, n] = mma_sync(
                                                    rK_lower,
                                                    rQ_f16[None, k, n, 0],
                                                    rC[None, m, n],
                                                )
                                                rC[None, m, n] = mma_sync(
                                                    rK_upper,
                                                    rQ_f16[None, k, n, 1],
                                                    rC[None, m, n],
                                                )
                                        else:
                                            for n in cutlass.range_constexpr(Q_TILES):
                                                rC[None, m, n] = mma_sync(
                                                    rK[None, m, k],
                                                    rQ[(None, k % 2), k // 2, n],
                                                    rC[None, m, n],
                                                )

                                cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                                if cutlass.const_expr(masked == 1):
                                    k_start = block_id * BLOCK_K + warp_id * 32
                                    for q in cutlass.range_constexpr(Q_TILES):
                                        for i in cutlass.range_constexpr(4):
                                            for jj in cutlass.range_constexpr(2):
                                                col = q * 8 + (lane_id % 4) * 2 + jj
                                                q_local_pos = col % MAX_DQL
                                                q_pos = q_start + q_local_pos
                                                k_pos = k_start + i * 8 + lane_id // 4
                                                rC[q * 8 + i * 2 + jj] = (
                                                    rC[q * 8 + i * 2 + jj]
                                                    if q_pos >= k_pos
                                                    else float("-inf")
                                                )

                                for q in cutlass.range_constexpr(Q_TILES):
                                    # Thread-local reduction along the BLOCK_K dim.
                                    rScore = cute.make_rmem_tensor(2, Float32)
                                    rScore.fill(float("-inf"))
                                    for i in cutlass.range_constexpr(4):
                                        rScore[0] = cute.arch.fmax(rScore[0], rC[i * 2 + 0 + q * 8])
                                        rScore[1] = cute.arch.fmax(rScore[1], rC[i * 2 + 1 + q * 8])

                                    # Warp reduction among lanes 0, 4, 8, 12, ...
                                    for i in cutlass.range_constexpr(3):
                                        offset = 4 << i
                                        other0 = cute.arch.shuffle_sync_bfly(
                                            rScore[0], offset=offset, mask=-1, mask_and_clamp=31
                                        )
                                        other1 = cute.arch.shuffle_sync_bfly(
                                            rScore[1], offset=offset, mask=-1, mask_and_clamp=31
                                        )
                                        rScore[0] = cute.arch.fmax(rScore[0], other0)
                                        rScore[1] = cute.arch.fmax(rScore[1], other1)

                                    if lane_id * 2 < MMA_N:
                                        epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                                        epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                                cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                                if store_lane:
                                    o = warp_id * MAX_DQL + lane_id
                                    final_score = epi_buffer[o, 0]
                                    for i in cutlass.range_constexpr(1, 4):
                                        final_score = cute.arch.fmax(final_score, epi_buffer[o, i])
                                    st_ptr = cute.make_ptr(
                                        Float32,
                                        st_row + Int64(block_id) * st_b * 4,
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    cute.arch.store(st_ptr, final_score)
                        tma_stage = tma_stage + 1
                        if tma_stage == num_stages:
                            tma_stage = Int32(0)
                            tma_parity = tma_parity ^ 1
                        if warp_id == 0:
                            cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                        cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
                        item_req = desc[tma_stage, 0]
                        block_id = desc[tma_stage, 1]
                        item_len = desc[tma_stage, 2]
