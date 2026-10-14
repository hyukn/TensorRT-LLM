# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-k block selection for wide, query-contiguous MSA score tensors (context steps).

`select_blocks_ctx` returns exactly what torch.ops.trtllm.minimax_m3_select_blocks returns (same
blocks, same order, same -1 padding, same output layout) for the inputs it accepts; see
select_blocks_ctx.cu for why the result is bitwise identical and why it is faster: the packaged
histogram kernel walks each (query, head) row along the block axis, whose stride on the
[heads, blocks, tokens] layout the index scorers write is the token count, so every score read is
its own sector. This kernel reads 32 queries per 128-byte line instead.

Enabled with TRTLLM_M3_CTX_SELECT=1 (default off). The CUDA source is JIT-built once with
torch.utils.cpp_extension into TRTLLM_M3_CTX_SELECT_BUILD_DIR (default: next to this file).
"""

from __future__ import annotations

import functools
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
CTX_SELECT_ENV = "TRTLLM_M3_CTX_SELECT"
# Below this many (query, head) rows the packaged kernel is as fast (decode-sized calls).
_MIN_ROWS = int(os.environ.get("TRTLLM_M3_CTX_SELECT_MIN_ROWS", "256"))
# The packaged op takes its bitonic paths up to 128 blocks; only the histogram path is replaced.
_MIN_BLOCKS = 129
_MAX_BLOCKS = 65535
_VARIANT = int(os.environ.get("TRTLLM_M3_CTX_SELECT_VARIANT", "0"))


@functools.lru_cache(maxsize=1)
def ctx_select_enabled() -> bool:
    return os.environ.get(CTX_SELECT_ENV, "0") == "1"


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "TRTLLM_M3_CTX_SELECT_BUILD_DIR", os.path.join(_HERE, ".build_select_ctx")
    )
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="minimax_m3_select_blocks_ctx_ext",
        sources=[os.path.join(_HERE, "select_blocks_ctx.cu")],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        build_directory=build_dir,
        verbose=bool(os.environ.get("TRTLLM_M3_CTX_SELECT_VERBOSE")),
    )


def supports(max_score_kv: torch.Tensor, n_valid_blocks: torch.Tensor, topk: int) -> bool:
    """Whether select_blocks_ctx takes this call (and is worth it)."""
    if max_score_kv.dim() != 3 or max_score_kv.dtype != torch.float32 or not max_score_kv.is_cuda:
        return False
    heads, blocks, tokens = (int(x) for x in max_score_kv.shape)
    return (
        topk == 16
        and max_score_kv.stride(2) == 1
        and _MIN_BLOCKS <= blocks <= _MAX_BLOCKS
        and heads * tokens >= _MIN_ROWS
        and _is_sm10x(max_score_kv.device.index)
        # Context steps only: they are never graph-captured, and decode-sized calls keep the
        # packaged op (whose capture behaviour the graph runners were validated with).
        and not torch.cuda.is_current_stream_capturing()
    )


@functools.lru_cache(maxsize=None)
def _is_sm10x(device_index) -> bool:
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device_index)[0] == 10


def select_blocks_ctx(
    max_score_kv: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    head_major_output: bool,
    variant: int | None = None,
) -> torch.Tensor:
    """Drop-in for torch.ops.trtllm.minimax_m3_select_blocks (requires supports())."""
    return _ext().select_blocks_ctx(
        max_score_kv,
        n_valid_blocks,
        int(topk),
        int(init_blocks),
        int(local_blocks),
        bool(head_major_output),
        _VARIANT if variant is None else int(variant),
    )


@functools.lru_cache(maxsize=1)
def ctx_select_verify() -> bool:
    """Debug aid (TRTLLM_M3_CTX_SELECT_VERIFY=1): also run the packaged op and require equality.

    Synchronizes the stream on every call; for validation runs only, never in a CUDA graph.
    """
    return os.environ.get("TRTLLM_M3_CTX_SELECT_VERIFY", "0") == "1"


_VERIFY_STATS = {"calls": 0, "rows": 0}


def verify_against_packaged(
    selected, max_score_kv, n_valid_blocks, topk, init_blocks, local_blocks, head_major_output
) -> None:
    ref = torch.ops.trtllm.minimax_m3_select_blocks(
        max_score_kv, n_valid_blocks, topk, init_blocks, local_blocks, head_major_output
    )
    if not torch.equal(ref, selected):
        bad = (ref != selected).any(dim=-1).nonzero()
        raise RuntimeError(
            f"MiniMax-M3 context select differs from the packaged op in {bad.shape[0]} rows, "
            f"first {bad[:4].tolist()}; scores {tuple(max_score_kv.shape)}"
        )
    _VERIFY_STATS["calls"] += 1
    _VERIFY_STATS["rows"] += int(ref.shape[0]) * int(ref.shape[1])
    if _VERIFY_STATS["calls"] % 500 == 1:
        from tensorrt_llm.logger import logger

        logger.info(f"MiniMax-M3 context select verified bitwise: {_VERIFY_STATS}")


def prebuild() -> bool:
    """Build/load the extension now (warmup); returns False if it cannot be built here."""
    try:
        _ext()
        return True
    except Exception:  # noqa: BLE001 - a missing toolchain leaves the packaged op in charge
        return False


__all__ = [
    "CTX_SELECT_ENV",
    "ctx_select_enabled",
    "ctx_select_verify",
    "prebuild",
    "select_blocks_ctx",
    "supports",
    "verify_against_packaged",
]
