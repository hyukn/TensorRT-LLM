"""Faster MiniMax-M3 NVFP4 sparse decode attention (v2), dispatched from triton_sparse_decode.

`minimax_m3_sparse_attn_decode_v2` has the signature and semantics of
TensorRT-LLM's `minimax_m3_sparse_attn_decode`. On the NVFP4 path, for the
shapes it supports and wins on (see `select_v2_config`), it runs a single CUDA
kernel (sparse_decode_v2.cu, JIT-built with torch.utils.cpp_extension); every
other call is forwarded unchanged to the original Triton implementation.

The v2 kernel is CUDA-graph capturable: launch geometry depends only on
shapes, all per-step data (seq_lens, top-k, block table, global scales) is read
on device, the split-K scratch comes from TRT-LLM's persistent buffer arena and
the split-K arrival counters are a persistent zero-initialised buffer that the
kernel resets itself.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
_COUNTERS: dict[int, torch.Tensor] = {}
_COUNTER_CAPACITY = 1 << 16
_SM_COUNT: dict[int, int] = {}
_EMPTY_DBG: dict = {}

SPARSE_BLOCK_SIZE = 128
HEAD_DIM = 128
_FP8 = torch.float8_e4m3fn
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
_STAGE_BYTES = 18432
_UMMA_MAX_STAGES = 6  # 6 x 18 KB ring + 96 KB of UMMA operands fits the 227 KB of shared memory
_MAX_SMEM = 200 * 1024
_MAX_SPLIT = 16  # cluster merge needs <= 16 (non-portable cluster); keeps the merge cheap too


def _load_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        # A shared, persistent build dir (e.g. on Lustre) lets every rank and job reuse one build;
        # torch's cpp_extension file baton serialises concurrent builders.
        build_dir = os.environ.get(
            "TRTLLM_M3_SPARSE_DECODE_V2_BUILD_DIR",
            os.environ.get("SDV2_BUILD_DIR", os.path.join(_HERE, ".build")),
        )
        os.makedirs(build_dir, exist_ok=True)
        _EXT = load(
            name="sparse_decode_v2_ext",
            sources=[os.path.join(_HERE, "sparse_decode_v2.cu")],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-lineinfo",
                # Explicit arch flags also stop torch from adding non-"a" targets,
                # which lack cvt.rn.f16x2.e2m1x2.
                "-gencode=arch=compute_100a,code=sm_100a",
                "-gencode=arch=compute_103a,code=sm_103a",
            ]
            + (["-Xptxas=-v"] if os.environ.get("SDV2_PTXAS_VERBOSE") else []),
            build_directory=build_dir,
            verbose=bool(os.environ.get("SDV2_VERBOSE")),
        )
    return _EXT


def _old_impl():
    from .triton_sparse_decode import _minimax_m3_sparse_attn_decode_triton

    return _minimax_m3_sparse_attn_decode_triton


def _counters(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    buf = _COUNTERS.get(idx)
    if buf is None:
        buf = torch.zeros(_COUNTER_CAPACITY, dtype=torch.int32, device=device)
        _COUNTERS[idx] = buf
    return buf


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _scratch(shape, dtype, name):
    reserve = torch.cuda.is_current_stream_capturing()
    try:
        from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
    except ImportError:  # pragma: no cover - standalone use
        get_memory_buffers = None
    if get_memory_buffers is not None:
        return get_memory_buffers().get_buffer(
            list(shape), dtype, buffer_name=name, reserve_buffer=reserve
        )
    return torch.empty(shape, dtype=dtype, device="cuda")


def kernel_geometry(
    *,
    num_reqs: int,
    num_kv_heads: int,
    group: int,
    decode_query_len: int,
    topk: int,
    sm_count: int = 152,
    nsplit: Optional[int] = None,
) -> Optional[dict]:
    """Launch geometry the v2 kernel supports for a shape, or None if it cannot run it."""
    if group not in (8, 16) or not (1 <= decode_query_len <= 4) or topk > 16 or topk % 4:
        return None
    if decode_query_len * topk > 64:
        return None
    groups = num_reqs * num_kv_heads
    if groups == 0 or groups > _COUNTER_CAPACITY:
        return None
    ntiles = (decode_query_len * group + 15) // 16
    if nsplit is None:
        if ntiles >= 3:
            # 512-thread CTAs run one per SM: fill exactly one wave.
            nsplit = max(1, min(8, sm_count // groups))
            if nsplit == 1 and 3 * groups <= 2 * sm_count:
                # Between half a wave and ~2/3 of a wave of groups, one CTA per group leaves SMs
                # idle while each CTA walks all of its group's pages; three splits (two waves)
                # are faster: kv4 B20/22/24 57.6/57.6/57.8 -> 48.3/50.2/51.2 us; from B28 (112
                # groups) one split is best again (results/sweep_kvh4_full).
                nsplit = 3
        else:
            # Small CTAs co-reside; measured best on GB300 (results/sweep_kvh*_dql1).
            nsplit = 8 if groups <= 32 else (4 if groups <= 128 else 1)
    nsplit = max(1, min(int(nsplit), _MAX_SPLIT))
    stages = max(1, min(6, -(-(decode_query_len * topk) // nsplit)))
    return dict(nsplit=nsplit, ntiles=ntiles, wpt=4, stages=stages)


# Where v2 beats the old kernel on GB300 (bench_sparse_decode.py --sweep-nsplit, results/):
# minimum number of (request, KV head) groups per (decode_query_len, group).
_V2_MIN_GROUPS = {
    # kv1: B=16 1.40x, B=24 2.0x, B=32 1.78x, B=64 1.94x (B<=8 old wins);
    # kv4: B=8..64 1.68-1.92x (B=1 old wins).
    (4, 16): 16,
    # kv1: B=24/32 1.24x, B=64 1.55x (B<=16 old wins); kv4: B=8..64 1.19-1.54x.
    (1, 16): 24,
    # group 8, kv1 (results/sweep_g8): dql4 B=16..64 1.67-2.17x, dql2 B=16..64 1.15-1.69x,
    # dql1 B=32/64 1.26/1.52x (B=8 old wins everywhere, dql1 B=16 too).
    (4, 8): 16,
    (2, 8): 16,
    (1, 8): 32,
}
_V2_TUNED_CAPABILITIES = ((10, 3),)


def select_v2_config(
    *,
    num_reqs: int,
    num_kv_heads: int,
    group: int,
    decode_query_len: int,
    topk: int,
    sm_count: int = 152,
    capability: Optional[tuple] = None,
) -> Optional[dict]:
    """Shape-only dispatch: the v2 launch geometry where it is measured faster, else None (old kernel).

    nsplit: CTAs per (request, KV head), each taking a round-robin slice of the
    union of the request's top-k lists. ntiles: 16-row (query, head) tiles.
    wpt: warps cooperating on one tile. stages: TMA ring depth.
    """
    if capability is None:
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if capability not in _V2_TUNED_CAPABILITIES:
        return None
    min_groups = _V2_MIN_GROUPS.get((decode_query_len, group))
    if min_groups is None or num_reqs * num_kv_heads < min_groups:
        return None
    return kernel_geometry(
        num_reqs=num_reqs,
        num_kv_heads=num_kv_heads,
        group=group,
        decode_query_len=decode_query_len,
        topk=topk,
        sm_count=sm_count,
    )


def _v2_eligible(q, k_paged, v_paged, topk_idx, block_table, seq_lens, output, nvfp4_args) -> bool:
    if any(a is None for a in nvfp4_args):
        return False
    k_bs, v_bs, k_gs, v_gs = nvfp4_args
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) not in _SUPPORTED_CAPABILITIES:
        return False
    if q.dim() != 3 or q.shape[2] != HEAD_DIM or q.dtype not in (torch.bfloat16, _FP8):
        return False
    if output.dtype != torch.bfloat16 or tuple(output.shape) != tuple(q.shape):
        return False
    esz = q.element_size()
    if (
        q.stride(2) != 1
        or (q.stride(0) * esz) % 16
        or (q.stride(1) * esz) % 16
        or q.data_ptr() % 16
    ):
        return False
    if (
        output.stride(2) != 1
        or output.stride(0) % 8
        or output.stride(1) % 8
        or output.data_ptr() % 16
    ):
        return False
    for data in (k_paged, v_paged):
        if data.dim() != 4 or data.shape[2] != SPARSE_BLOCK_SIZE or data.shape[3] != HEAD_DIM // 2:
            return False
        if data.element_size() != 1 or data.stride(3) != 1 or data.stride(2) != HEAD_DIM // 2:
            return False
        if data.stride(0) % 16 or data.stride(1) % 16 or data.data_ptr() % 16:
            return False
    for sc in (k_bs, v_bs):
        if sc.dim() != 4 or tuple(sc.shape) != tuple(k_paged.shape[:3]) + (HEAD_DIM // 16,):
            return False
        if sc.element_size() != 1 or sc.stride(3) != 1 or sc.stride(2) != HEAD_DIM // 16:
            return False
        if sc.stride(0) % 16 or sc.stride(1) % 16 or sc.data_ptr() % 16:
            return False
    if tuple(v_paged.shape) != tuple(k_paged.shape):
        return False
    for gs in (k_gs, v_gs):
        if gs.dtype != torch.float32 or gs.numel() != 1:
            return False
    if topk_idx.dtype != torch.int32 or topk_idx.dim() != 3:
        return False
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        return False
    if seq_lens.dtype != torch.int32 or seq_lens.dim() != 1 or seq_lens.stride(0) != 1:
        return False
    num_kv_heads = int(k_paged.shape[1])
    if (
        q.shape[1] % num_kv_heads
        or topk_idx.shape[0] != num_kv_heads
        or topk_idx.shape[1] != q.shape[0]
    ):
        return False
    return True


@torch.no_grad()
def minimax_m3_sparse_attn_decode_v2(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    kv_scale: Optional[torch.Tensor] = None,
    k_block_scale: Optional[torch.Tensor] = None,
    v_block_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[torch.Tensor] = None,
    v_global_scale: Optional[torch.Tensor] = None,
    num_topk_chunks: Optional[int] = None,
    _force: Optional[str] = None,
    _config: Optional[dict] = None,
    _debug: Optional[torch.Tensor] = None,
) -> None:
    """Same contract as minimax_m3_sparse_attn_decode.

    num_topk_chunks (tests only) overrides the v2 split count; the result must
    not depend on it. `_force` ("old"/"v2") and `_config` exist for benchmarks.
    """
    nvfp4_args = (k_block_scale, v_block_scale, k_global_scale, v_global_scale)
    total_q, num_heads, _ = q.shape
    num_reqs = int(seq_lens.shape[0])
    use_v2 = _force != "old" and _v2_eligible(
        q, k_paged, v_paged, topk_idx, block_table, seq_lens, output, nvfp4_args
    )
    cfg = None
    if use_v2:
        if total_q != num_reqs * decode_query_len:
            raise ValueError(
                f"MiniMax-M3 sparse decode v2: total_q ({total_q}) must be batch ({num_reqs}) * "
                f"decode_query_len ({decode_query_len})."
            )
        num_kv_heads = int(k_paged.shape[1])
        shape = dict(
            num_reqs=num_reqs,
            num_kv_heads=num_kv_heads,
            group=num_heads // num_kv_heads,
            decode_query_len=decode_query_len,
            topk=int(topk_idx.shape[-1]),
            sm_count=_sm_count(q.device),
        )
        if _config is not None:
            cfg = _config
        elif _force == "v2":
            cfg = kernel_geometry(**shape)  # tests / benchmarks: run v2 even where old is faster
        else:
            cfg = select_v2_config(**shape, capability=torch.cuda.get_device_capability(q.device))
        if cfg is not None and num_topk_chunks is not None:
            # Test hook: the split count must not change the result (any value works for v2).
            cfg = kernel_geometry(**shape, nsplit=int(num_topk_chunks))
    if (
        cfg is not None
        and os.environ.get("SDV2_UMMA") in ("1", "2")
        and decode_query_len * (num_heads // int(k_paged.shape[1])) <= 64
    ):
        cfg = dict(cfg, umma=int(os.environ["SDV2_UMMA"]))
    if cfg is not None and cfg.get("umma"):
        cap = _UMMA_MAX_STAGES if int(cfg["umma"]) == 1 else 2
        cfg = dict(cfg, ntiles=4, stages=max(1, min(int(cfg["stages"]), cap)))
    if cfg is None:
        if _force == "v2":
            raise RuntimeError("v2 forced but the call is not eligible")
        _old_impl()(
            q,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens,
            sm_scale=sm_scale,
            output=output,
            decode_query_len=decode_query_len,
            kv_scale=kv_scale,
            k_block_scale=k_block_scale,
            v_block_scale=v_block_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
            num_topk_chunks=num_topk_chunks,
        )
        return

    ext = _load_ext()
    num_kv_heads = int(k_paged.shape[1])
    nsplit = int(cfg["nsplit"])
    ntiles = 4 if cfg.get("umma") else int(cfg["ntiles"])  # UMMA: one M=64 tile
    groups = num_reqs * num_kv_heads
    rows_pad = ntiles * 16
    cluster_merge = bool(cfg.get("cluster", 0)) and nsplit > 1
    if nsplit > 1 and not cluster_merge:
        part_o = _scratch(
            (groups * nsplit * rows_pad * HEAD_DIM,), torch.float16, "m3_sparse_decode_v2_part_o"
        )
        part_ml = _scratch(
            (groups * nsplit * rows_pad * 2,), torch.float32, "m3_sparse_decode_v2_part_ml"
        )
    else:
        # Cluster splits keep their partials in shared memory.
        part_o = output
        part_ml = output
    pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    ext.sparse_decode_v2(
        q,
        k_paged.view(torch.uint8),
        v_paged.view(torch.uint8),
        k_block_scale.view(torch.uint8),
        v_block_scale.view(torch.uint8),
        k_global_scale,
        v_global_scale,
        topk_idx,
        block_table,
        seq_lens,
        output,
        part_o,
        part_ml,
        _counters(q.device),
        _debug
        if _debug is not None
        else _EMPTY_DBG.setdefault(q.device, torch.empty(0, dtype=torch.int64, device=q.device)),
        int(decode_query_len),
        nsplit,
        ntiles,
        {1: 0, 2: -1}[int(cfg["umma"])] if cfg.get("umma") else int(cfg["wpt"]),
        int(cfg["stages"]),
        float(sm_scale) * math.log2(math.e),
        pdl,
        cluster_merge,
    )


__all__ = ["minimax_m3_sparse_attn_decode_v2", "select_v2_config"]
