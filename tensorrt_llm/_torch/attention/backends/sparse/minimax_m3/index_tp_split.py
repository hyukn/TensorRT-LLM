# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel split of the MiniMax-M3 index decode scoring.

The index-K cache is MQA: one head that every index head scores against. Under attention TP each
rank owns num_index_heads / tp of the index heads yet reads the whole index-K, so the decode
scoring pass (HBM bound) reads the same bytes on every rank. Here each rank instead scores every
index head on 1/tp of the KV blocks:

1. ``gather_q``: every rank copies its index-Q rows into every rank's gather buffer (slot =
   rank, token-major so the result is a plain [tokens, heads, dim] view) and passes flag barrier A.
2. The rank-split CuTe DSL scorer (``trtllm::cute_dsl_minimax_m3_index_decode_score_split``)
   scores all heads on blocks rank::tp and stores head h's scores straight into the owning
   rank's score buffer (peer memory over NVLink).
3. ``barrier``: flag barrier B. Afterwards every rank's score buffer holds its own heads' scores
   for every block -- bitwise what the unsplit scorer computes -- and block selection proceeds
   unchanged on the local buffer.

The buffers are single-buffered: a rank writes layer l+1's Q rows into a peer only after passing
barrier B of layer l, which the peer enters only after its scorer for layer l has consumed its
Q; and it writes layer l+1's scores only after barrier A of layer l+1, which the peer enters only
after its block selection for layer l has consumed its scores.

Replicated heads (TP wider than the index heads, e.g. TP8 with 4 index heads): consecutive
groups of ``replicas`` ranks hold the same heads (``index_head_range``). Every rank still scores
all heads on blocks rank::tp, but head h's scores go to the first rank of its group (the primary)
only, and only primaries contribute index-Q rows (one gather slot per group). The other ranks of
the group read the primary's score buffer in place (peer loads; block selection only reads it).
Every rank still signals every barrier, so the single-buffer argument above holds for the
replicas' reads as well: nobody writes layer l+1's scores before every rank -- replicas included --
has passed barrier A of layer l+1, i.e. finished its block selection for layer l.

Barriers use monotonically increasing epochs kept on the device, so they replay correctly in CUDA
graphs as long as every rank issues the same sequence of barriers (the split is chosen from the
batch shape, which is identical on all TP ranks). Peer signalling is a plain st.release.sys per
peer and the wait spins on ld.acquire.sys; both kernels launch without PDL, so they start only
after the preceding kernel (and its remote stores) has completed.

Enable with TRTLLM_M3_INDEX_TP_SPLIT=1; the split runs on pure-decode steps with at least
TRTLLM_M3_INDEX_TP_SPLIT_MIN_BATCH requests (default 12). Below that, two barriers cost more
than the saved index-K reads.

Local index-Q (TRTLLM_M3_INDEX_TP_SPLIT_LOCAL_Q=1, with the split enabled): every rank computes
all index-Q heads itself (the model replicates the index-Q projection in its fused QKV+index
GEMM instead of sharding it by KV group), so step 1 and barrier A disappear and each split call
is scorer + barrier B. The scorer reads the same Q bytes the gather would have delivered: the
projection input is identical on every rank (the output of the layer-boundary AllReduce) and
each Q column is the same dot product, so the scores stay bitwise those of the unsplit scorer.
Without barrier A, what keeps layer l+1's remote score stores out of a peer's buffer while its
block selection for layer l still reads it is the layer's tensor-parallel AllReduce in between:
a rank's layer-l attention AllReduce completes only once every peer has contributed its o_proj
output, which the peer computes from the attention output of its layer-l block selection; the
replicas' in-place reads of a primary's buffer are covered the same way. The split must
therefore only run under attention TP, as it does (never under attention DP).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

ENV_ENABLE = "TRTLLM_M3_INDEX_TP_SPLIT"
ENV_MIN_BATCH = "TRTLLM_M3_INDEX_TP_SPLIT_MIN_BATCH"
ENV_LOCAL_Q = "TRTLLM_M3_INDEX_TP_SPLIT_LOCAL_Q"
DEFAULT_MIN_BATCH = 12
# Flag slots per barrier: one per peer plus one dummy target for the masked self lane.
_FLAG_SLOTS = 64
_ALIGN = 4096
# Token capacity reserved up front (decode steps: max batch x (1 + draft_len)).
_MIN_TOKEN_CAPACITY = 1024


def tp_split_min_batch() -> Optional[int]:
    """Minimum decode batch for the split, or None when the split is disabled."""
    if os.environ.get(ENV_ENABLE, "0") != "1":
        return None
    return int(os.environ.get(ENV_MIN_BATCH, str(DEFAULT_MIN_BATCH)))


def tp_split_local_q_requested() -> bool:
    """Whether every rank should compute all index-Q heads (see the module docstring).

    Only meaningful with the split enabled; the model still checks the TP mapping and peer
    access before it replicates the index-Q projection.
    """
    return tp_split_min_batch() is not None and os.environ.get(ENV_LOCAL_Q, "0") == "1"


_PEER_ACCESS: Dict[Tuple[int, int, int], bool] = {}


def tp_split_peer_access(mapping) -> bool:
    """Whether every rank of the TP group can map this rank's peer buffers (same node, P2P).

    The workspace uses IpcMemory, so a TP group spanning nodes (e.g. TP8 over two 4-GPU trays)
    cannot split; every rank of such a group gets False, so they all keep the unsplit scorer.
    """
    key = (int(mapping.rank), int(mapping.tp_size), int(mapping.pp_rank))
    ok = _PEER_ACCESS.get(key)
    if ok is None:
        from tensorrt_llm._ipc_utils import can_access_peer

        ok = bool(can_access_peer(mapping))
        _PEER_ACCESS[key] = ok
        if not ok:
            logger.info(
                "MiniMax-M3 index TP split disabled: the TP group has no peer access "
                "(spans nodes); using the unsplit scorer."
            )
    return ok


def split_k_for_batch(batch: int) -> int:
    """Scorer grid size: 0 lets the op size its persistent grid to the GPU (one CTA per SM slot).

    The kernel balances the rank's blocks across its CTAs itself, so the grid no longer depends
    on the batch.
    """
    return 0


@triton.jit
def _st_release_sys(addr, val):
    return tl.inline_asm_elementwise(
        "st.release.sys.global.b32 [$1], $2; mov.u32 $0, 0;",
        "=r,l,r",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ld_acquire_sys(addr):
    return tl.inline_asm_elementwise(
        "ld.acquire.sys.global.b32 $0, [$1];",
        "=r,l",
        [addr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _signal_and_wait(peer_flag_ptrs, local_flags_addr, e, rank, WORLD: tl.constexpr):
    offs = tl.arange(0, WORLD)
    is_peer = offs != rank
    peer_base = tl.load(peer_flag_ptrs + offs)
    # The self lane stores into the dummy slot WORLD of its own flag array.
    dst = tl.where(is_peer, peer_base + rank * 4, peer_base + WORLD * 4)
    _st_release_sys(dst, tl.zeros([WORLD], tl.int32) + e)
    src = local_flags_addr + offs.to(tl.int64) * 4
    v = tl.where(is_peer, _ld_acquire_sys(src), e)
    while tl.min(v - e) < 0:  # wrap-safe epoch comparison
        v = tl.where(is_peer, _ld_acquire_sys(src), e)


@triton.jit
def _flag_barrier_kernel(peer_flag_ptrs, local_flags_addr, epoch_ptr, rank, WORLD: tl.constexpr):
    e = tl.load(epoch_ptr) + 1
    _signal_and_wait(peer_flag_ptrs, local_flags_addr, e, rank, WORLD)
    tl.store(epoch_ptr, e)


@triton.jit
def _gather_rows_barrier_kernel(
    src,
    n_rows,
    peer_dst_ptrs,
    peer_flag_ptrs,
    local_flags_addr,
    epoch_ptr,
    rank,
    slot,
    ROW_WORDS: tl.constexpr,
    WORLD: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # One program per destination rank p (grid = WORLD): copy this rank's rows into slot `slot` of
    # p's gather buffer (row t lands at row t * SLOTS + slot; n_rows is 0 on ranks that hold a
    # replicated copy of another rank's heads), then signal p. Only the program
    # whose destination is this rank waits for the other ranks' signals; the kernel completes
    # once it has, so the next kernel sees every rank's rows. Each program keeps its own epoch
    # counter (epoch_ptr[p]); all of them advance once per call on every rank.
    p = tl.program_id(0)
    e = tl.load(epoch_ptr + p) + 1
    dst = tl.load(peer_dst_ptrs + p).to(tl.pointer_type(tl.int32))
    cols = tl.arange(0, ROW_WORDS)
    for r0 in range(0, n_rows, BLOCK_ROWS):
        rows = r0 + tl.arange(0, BLOCK_ROWS)
        m = rows[:, None] < n_rows
        vals = tl.load(src + rows[:, None] * ROW_WORDS + cols[None, :], mask=m)
        tl.store(dst + (rows[:, None] * SLOTS + slot) * ROW_WORDS + cols[None, :], vals, mask=m)
    tl.debug_barrier()
    lanes = tl.arange(0, WORLD)
    peer_base = tl.load(peer_flag_ptrs + lanes)
    # Lane p of program p (p != rank) signals rank p; every other lane hits the dummy slot.
    signal = (lanes == p) & (p != rank)
    dst_flag = tl.where(signal, peer_base + rank * 4, tl.load(peer_flag_ptrs + rank) + WORLD * 4)
    _st_release_sys(dst_flag, tl.zeros([WORLD], tl.int32) + e)
    if p == rank:
        src_flag = local_flags_addr + lanes.to(tl.int64) * 4
        is_peer = lanes != rank
        v = tl.where(is_peer, _ld_acquire_sys(src_flag), e)
        while tl.min(v - e) < 0:  # wrap-safe epoch comparison
            v = tl.where(is_peer, _ld_acquire_sys(src_flag), e)
    tl.store(epoch_ptr + p, e)


def _wrap(ptr: int, numel: int, dtype: torch.dtype, device_id: int) -> torch.Tensor:
    from tensorrt_llm._dlpack_utils import pack_strided_memory

    nbytes = numel * torch.tensor([], dtype=dtype).element_size()
    return pack_strided_memory(ptr, nbytes, nbytes, 1, dtype, device_id).reshape(-1)


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class IndexTpSplitWorkspace:
    """Peer-accessible score/Q buffers and barrier flags of one TP group (one per process)."""

    _instances: Dict[Tuple[int, int, int, int], "IndexTpSplitWorkspace"] = {}

    @classmethod
    def get(
        cls,
        mapping,
        *,
        local_heads: int,
        k_tiles: int,
        q_row_bytes: int,
        max_tokens: int,
        replicas: int = 1,
    ) -> "IndexTpSplitWorkspace":
        """The process's workspace, (re)created if it cannot hold this geometry.

        CUDA graphs bake in the buffer addresses, and the per-graph attention metadata asks for
        token counts that grow with the captured batch size, so capacity is reserved up front
        for at least _MIN_TOKEN_CAPACITY tokens: within one executor the workspace is then
        created once and never replaced. It is only replaced when a later executor needs more
        (e.g. the final executor after the memory-profiling one, whose graphs are gone by then);
        that happens in an eager warmup forward, collectively on all ranks, since the geometry
        is identical on every TP rank. ``replicas`` consecutive ranks share the same heads.
        """
        key = (int(mapping.pp_rank), int(mapping.tp_rank), int(mapping.tp_size), int(replicas))
        ws = cls._instances.get(key)
        need_scores = int(local_heads) * int(k_tiles) * int(max_tokens)
        need_q = int(q_row_bytes) * int(max_tokens)
        if ws is not None and need_scores <= ws.score_numel and need_q <= ws.q_bytes_per_rank_slot:
            return ws
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MiniMax-M3 index TP split: the peer workspace must be (re)created by an eager "
                "forward before CUDA graph capture."
            )
        tokens = max(int(max_tokens), _MIN_TOKEN_CAPACITY)
        if ws is not None:
            logger.warning(
                "MiniMax-M3 index TP split: replacing the peer workspace "
                f"({ws.score_numel} -> {need_scores} scores); any CUDA graph captured against "
                "the old one must not be replayed."
            )
            # Every peer write into the old buffers completed before the barrier this rank last
            # passed, so after a local sync nothing references them any more.
            torch.cuda.synchronize()
            del cls._instances[key]
            del ws
        ws = cls(
            mapping,
            score_numel=int(local_heads) * int(k_tiles) * tokens,
            q_row_bytes=q_row_bytes,
            max_tokens=tokens,
            replicas=replicas,
        )
        cls._instances[key] = ws
        return ws

    def __init__(
        self, mapping, *, score_numel: int, q_row_bytes: int, max_tokens: int, replicas: int = 1
    ):
        from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer

        if not can_access_peer(mapping):
            raise RuntimeError("MiniMax-M3 index TP split needs peer access between all TP ranks.")
        self.rank = int(mapping.tp_rank)
        self.world = int(mapping.tp_size)
        self.replicas = int(replicas)
        if self.replicas < 1 or self.world % self.replicas != 0:
            raise ValueError(f"TP size {self.world} is not a multiple of replicas {replicas}")
        # Head groups: ranks [g * replicas, (g + 1) * replicas) hold group g's heads; the first of
        # them (the primary) owns the group's score buffer and contributes its index-Q rows.
        self.groups = self.world // self.replicas
        self.group = self.rank // self.replicas
        self.primary = self.group * self.replicas
        self.score_numel = int(score_numel)
        self.q_bytes_per_rank_slot = int(q_row_bytes) * int(max_tokens)
        score_bytes = _align(self.score_numel * 4)
        q_bytes = _align(self.q_bytes_per_rank_slot * self.world)
        flag_bytes = _align(2 * _FLAG_SLOTS * 4)
        self._q_off = score_bytes
        self._flag_off = score_bytes + q_bytes
        total = score_bytes + q_bytes + flag_bytes
        # Zero-initialised (flags start at epoch 0 on every rank).
        self._ipc = IpcMemory(mapping, total)
        dev = torch.cuda.current_device()
        ptrs = list(self._ipc.peer_ptrs)
        self.scores: List[torch.Tensor] = [
            _wrap(p, self.score_numel, torch.float32, dev) for p in ptrs
        ]
        self.q_gather = _wrap(
            ptrs[self.rank] + self._q_off, self.q_bytes_per_rank_slot * self.world, torch.uint8, dev
        )
        self.q_peer_ptrs = torch.tensor(
            [p + self._q_off for p in ptrs], dtype=torch.int64, device="cuda"
        )
        self.flag_a_ptrs = torch.tensor(
            [p + self._flag_off for p in ptrs], dtype=torch.int64, device="cuda"
        )
        self.flag_b_ptrs = torch.tensor(
            [p + self._flag_off + _FLAG_SLOTS * 4 for p in ptrs], dtype=torch.int64, device="cuda"
        )
        self.flag_a_local = ptrs[self.rank] + self._flag_off
        self.flag_b_local = ptrs[self.rank] + self._flag_off + _FLAG_SLOTS * 4
        # epochs[:world]: one counter per gather program (destination rank); epochs[world]: barrier B.
        self.epochs = torch.zeros(self.world + 1, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        logger.info(
            f"MiniMax-M3 index TP split workspace: rank {self.rank}/{self.world}, "
            f"{total / 2**20:.1f} MiB peer memory"
        )

    def gather_q(self, q_local: torch.Tensor, num_heads_total: int) -> torch.Tensor:
        """All-gather index-Q over the head groups; returns [tokens, num_heads_total, dim].

        ``q_local`` is this rank's [tokens, local_heads * dim] rows (contiguous). Group g's heads
        (from its primary rank) become heads [g * local_heads, (g + 1) * local_heads) of the
        result, matching the checkpoint's index-head order under TP sharding.
        """
        tokens, row = int(q_local.shape[0]), int(q_local.shape[1])
        row_bytes = row * q_local.element_size()
        if row_bytes % 4 != 0 or tokens * row_bytes > self.q_bytes_per_rank_slot:
            raise ValueError(
                f"index-Q rows of {row_bytes} bytes x {tokens} do not fit the gather slot"
            )
        words = q_local.contiguous().view(torch.int32).view(-1)
        _gather_rows_barrier_kernel[(self.world,)](
            words,
            tokens if self.rank == self.primary else 0,
            self.q_peer_ptrs,
            self.flag_a_ptrs,
            self.flag_a_local,
            self.epochs,
            self.rank,
            self.group,
            ROW_WORDS=row_bytes // 4,
            WORLD=self.world,
            SLOTS=self.groups,
            BLOCK_ROWS=64,
            num_warps=4,
        )
        dim = row // (num_heads_total // self.groups)
        return (
            self.q_gather[: tokens * self.groups * row_bytes]
            .view(q_local.dtype)
            .view(tokens, num_heads_total, dim)
        )

    def score_views(self, local_heads: int, k_tiles: int, tokens: int) -> List[torch.Tensor]:
        """Per global index head h, the [tokens, k_tiles] score view in its group primary's buffer."""
        numel = local_heads * k_tiles * tokens
        out = []
        for h in range(local_heads * self.groups):
            group, lh = divmod(h, local_heads)
            owner = group * self.replicas
            out.append(self.scores[owner][:numel].view(local_heads, k_tiles, tokens)[lh].t())
        return out

    def local_scores(self, local_heads: int, k_tiles: int, tokens: int) -> torch.Tensor:
        """This rank's [local_heads, k_tiles, tokens] score block, as block selection reads it.

        On a replica rank this is the group primary's buffer (peer memory, read in place).
        """
        numel = local_heads * k_tiles * tokens
        return self.scores[self.primary][:numel].view(local_heads, k_tiles, tokens)

    def barrier(self) -> None:
        _flag_barrier_kernel[(1,)](
            self.flag_b_ptrs,
            self.flag_b_local,
            self.epochs[self.world :],
            self.rank,
            WORLD=self.world,
            num_warps=1,
        )


def tp_split_score(
    ws: IndexTpSplitWorkspace,
    idx_q_local: torch.Tensor,
    idx_k_paged: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    decode_query_len: int,
    local_heads: int,
    k_tiles: int,
    q_all_heads: bool = False,
) -> torch.Tensor:
    """Run steps 1-3 and return this rank's [local_heads, k_tiles, tokens] max-score buffer.

    ``q_all_heads``: ``idx_q_local`` already holds every index head ([tokens, total heads, dim],
    any 16-element-aligned strides; local index-Q, see the module docstring), so step 1 and
    barrier A are skipped.
    """
    tokens = int(idx_q_local.shape[0])
    total_heads = local_heads * ws.groups
    if q_all_heads:
        if idx_q_local.dim() != 3 or int(idx_q_local.shape[1]) != total_heads:
            raise ValueError(
                f"local index-Q must be [tokens, {total_heads}, dim], got {tuple(idx_q_local.shape)}"
            )
        q_all = idx_q_local
    else:
        q_all = ws.gather_q(idx_q_local.reshape(tokens, -1), total_heads)
    outs = ws.score_views(local_heads, k_tiles, tokens)
    outs = outs + [outs[0]] * (4 - len(outs))
    batch = int(seq_lens_cuda.shape[0])
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score_split(
        q_all,
        idx_k_paged.squeeze(1),
        block_table,
        seq_lens_cuda,
        outs[0],
        outs[1],
        outs[2],
        outs[3],
        total_heads,
        decode_query_len,
        ws.rank,
        ws.world,
        split_k_for_batch(batch),
    )
    ws.barrier()
    return ws.local_scores(local_heads, k_tiles, tokens)


__all__ = [
    "IndexTpSplitWorkspace",
    "tp_split_peer_access",
    "split_k_for_batch",
    "tp_split_local_q_requested",
    "tp_split_min_batch",
    "tp_split_score",
]
