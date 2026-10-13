# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA (fmha_sm100) specific helpers for the MiniMax-M3 sparse backend."""

from __future__ import annotations

import functools
import importlib.util
import logging
import types
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._utils import is_sm_100f, maybe_pin_memory
from tensorrt_llm.logger import logger

from .msa_jit import configure_fmha_sm100_jit
from .paged_cache import write_kv_slots

# fmha_sm100 ships only head_dim 128 variants and the MiniMax-M3 checkpoint
# selects topk 16. Callers enforce these early so a misconfiguration fails
# with a clear message rather than a cryptic shape error inside the kernel.
MSA_REQUIRED_TOPK = 16
MSA_REQUIRED_HEAD_DIM = 128

# The fmha_sm100 sparse_mode each MSA call site runs; _fmha_sm100 in
# fmha_sm100/api.py derives it from the outputs and block table it is handed.
_FMHA_SPARSE_MODE_SPARSE = 0  # paged GQA over the indexer's top-k blocks
_FMHA_SPARSE_MODE_ONLY_SCORE = 2  # indexer proxy: per-block max score only
_FMHA_SPARSE_MODE_OFF = 3  # paged GQA over the full page table


def check_decode_span_shape(kernel: str, total_q: int, batch: int, query_len: int) -> None:
    """Reject a q that does not cover exactly the batch it was handed.

    The decode kernels derive the request id as token // query_len, so a longer
    q reads page table rows and lengths past the batch's last one. The caller
    names itself so the error does too, rather than surfacing as an assert
    several frames inside a kernel.
    """
    if total_q != batch * query_len:
        raise ValueError(
            f"{kernel}: total_q ({total_q}) must be batch ({batch}) * "
            f"decode_query_len ({query_len})."
        )


def is_msa_layer(attn) -> bool:
    """Whether this layer's attention is served by the MiniMax-M3 MSA kernels."""
    sparse_params = attn.sparse_params
    return (
        is_sm_100f()
        and sparse_params is not None
        and sparse_params.algorithm == "minimax_m3"
        and getattr(sparse_params, "implementation", None) == "msa"
    )


def _install_msa_cutlass_compatibility() -> None:
    """Provide legacy CuTe aliases still referenced by the packaged MSA sources."""
    try:
        import cutlass.cute as cute
    except ImportError:
        return

    # MSA has not yet migrated these aliases to their newer CUTLASS DSL
    # names. Keep this shim local to the MSA import path and remove it once the
    # packaged sources use the top-level thread classes and
    # cute.make_rmem_tensor directly.
    for thread_class_name in ("ThrCopy", "ThrMma"):
        if not hasattr(cute.core, thread_class_name):
            setattr(cute.core, thread_class_name, getattr(cute, thread_class_name))
    if not hasattr(cute, "make_fragment"):
        setattr(cute, "make_fragment", cute.make_rmem_tensor)


@functools.lru_cache(maxsize=1)
def msa_package_available() -> bool:
    """Prepare and report whether the packaged fmha_sm100 module can be imported.

    Cached: each call scans sys.path until fmha_sm100 is first imported, and
    FmhaManager construction checks once per attention layer. Whether the
    package is installed cannot change within a process.
    """
    if importlib.util.find_spec("fmha_sm100") is None:
        return False
    _install_msa_cutlass_compatibility()
    return True


class _TrtllmLogHandler(logging.Handler):
    """Re-emit log records through the TensorRT-LLM logger at their severity."""

    def emit(self, record: logging.LogRecord) -> None:
        message = f"{record.name}: {record.getMessage()}"
        if record.levelno >= logging.ERROR:
            logger.error(message)
        elif record.levelno >= logging.WARNING:
            logger.warning(message)
        else:
            logger.info(message)


@functools.lru_cache(maxsize=1)
def _forward_fmha_jit_log() -> None:
    """Send fmha_sm100's JIT log to the TensorRT-LLM logger.

    fmha_sm100 compiles a kernel variant with nvcc when a step first selects
    it, stalling that rank for a minute or more, and logs each build at INFO
    on a logger nothing configures, so the build would otherwise leave no
    trace in the server log.
    """
    jit_logger = logging.getLogger("fmha_sm100.jit")
    jit_logger.setLevel(logging.INFO)
    jit_logger.addHandler(_TrtllmLogHandler())
    # The handler is this logger's output now; propagating as well would print
    # each record twice wherever the root logger has a handler.
    jit_logger.propagate = False


def require_msa_module() -> types.ModuleType:
    """Import the packaged fmha_sm100 module or raise a clear error.

    The import is deferred to first kernel use so the MSA backend can be
    advertised in the config schema on systems where the kernels cannot load.
    A missing package is a hard error, never a silent fallback to another backend.
    The first import also limits the kernels fmha_sm100 compiles on first use
    to this device's architecture.
    """
    _install_msa_cutlass_compatibility()
    _forward_fmha_jit_log()
    try:
        import fmha_sm100
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-M3 MSA attention requires the fmha_sm100 kernels packaged "
            "with TensorRT-LLM. Reinstall TensorRT-LLM from a complete build."
        ) from exc
    configure_fmha_sm100_jit(fmha_sm100)
    return fmha_sm100


def fmha_context_variants(
    fmha_sm100: types.ModuleType,
    *,
    max_qo_len: int,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    attn_dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> List[Tuple]:
    """List the fmha_sm100 kernel variants a context step can dispatch to.

    A context step reaches fmha_sm100 at three sites: the indexer proxy over
    the MQA index-K cache, and the paged GQA of the sparse and of the dense
    layers. Each plans a paged cache with num_kv_splits=1, so the variant it
    runs follows from its dtype, its head counts and the longest query row of
    its plan alone, and walking every row length a step can hold covers them
    all. The selection mirrors _fmha_sm100_plan and _fmha_sm100 in
    fmha_sm100/api.py.

    Args:
        fmha_sm100: The imported fmha_sm100 package.
        max_qo_len: Longest query row a step can plan, its token budget.
        page_size: Tokens per KV cache page.
        num_q_heads: Per-rank attention query heads.
        num_kv_heads: Per-rank attention KV heads.
        num_index_heads: Per-rank indexer heads.
        attn_dtype: Query dtype of the paged GQA, that of the main KV cache.
        index_dtype: Query dtype of the indexer proxy, that of the index-K cache.

    Returns:
        Variant keys, each in fmha_sm100.jit.get_fmha_variant's argument order.
    """
    api = fmha_sm100.api
    sites = (
        # (query dtype, query heads, KV heads, sparse_mode)
        (index_dtype, num_index_heads, 1, _FMHA_SPARSE_MODE_ONLY_SCORE),
        (attn_dtype, num_q_heads, num_kv_heads, _FMHA_SPARSE_MODE_SPARSE),
        (attn_dtype, num_q_heads, num_kv_heads, _FMHA_SPARSE_MODE_OFF),
    )
    variants = set()
    # Rows past 128 tokens pack no heads and all run the 256-row tile, so a
    # longer row selects no new variant.
    for qo_len in range(1, min(max_qo_len, 129) + 1):
        for dtype, heads, kv_heads, sparse_mode in sites:
            is_sparse = sparse_mode == _FMHA_SPARSE_MODE_SPARSE
            if is_sparse and qo_len > api._prefill_qlen_threshold(True):
                # Longer sparse rows run the CuTe DSL sparse prefill instead.
                continue
            pack_factor = api._compute_pack_factor(qo_len, heads, kv_heads)
            # The sparse plan gives every query token a row of its own.
            packed_len = pack_factor if is_sparse else qo_len * pack_factor
            variants.add(
                (
                    fmha_sm100.jit._dlpack_dtype_code(dtype),
                    128 if packed_len <= 128 else 256,
                    packed_len <= 64,
                    sparse_mode,
                    page_size,
                    False,
                    pack_factor,
                )
            )
    return sorted(variants)


def prewarm_fmha_variants(fmha_sm100: types.ModuleType, variants: Sequence[Tuple]) -> None:
    """Build and load the given fmha_sm100 kernel variants now.

    fmha_sm100 otherwise compiles each of them with nvcc on first use, one at a
    time per process and on the order of a minute apiece. Here the builds run
    concurrently. Each takes the per-variant file lock and existence check
    get_fmha_variant uses, so processes sharing the kernel cache build every
    variant once between them. A variant that fails to build is left to
    compile on first use, as before.

    Args:
        fmha_sm100: The imported fmha_sm100 package.
        variants: Variant keys, as fmha_context_variants returns them.
    """
    jit = fmha_sm100.jit
    manager = jit._variant_manager
    # A build renders the variant's sources from jinja2 templates the manager
    # loads lazily, and unlocked: loading them here, before the builds fan out,
    # leaves the threads to only read them.
    manager._load_templates()

    def build(key: Tuple) -> None:
        name, params = jit._variant_key_from_runtime(*key)
        lock_fd = jit._acquire_file_lock(jit.CACHE_BASE / name / ".compile.lock")
        try:
            manager._compile_only(name, params)
        finally:
            jit._release_file_lock(lock_fd)

    with ThreadPoolExecutor(max_workers=max(1, len(variants))) as pool:
        builds = [(key, pool.submit(build, key)) for key in variants]
    built = 0
    for key, future in builds:
        try:
            future.result()
            jit.get_fmha_variant(*key)
        except (OSError, RuntimeError, ValueError) as e:
            # Reported rather than raised: the kernel still compiles on first
            # use, so a prewarm miss must not fail startup.
            logger.warning(f"fmha_sm100 variant {key} prewarm failed. {type(e).__name__}: {e}")
        else:
            built += 1
    logger.info(f"Prewarmed {built} of {len(variants)} fmha_sm100 kernel variants")


def msa_paged_kv(kv_cache_manager, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer paged K and V in fmha_sm100 HND layout, zero-copy.

    The cache is stored head-major (see `write_msa_main_kv`), so the "HND"
    buffer view is already the [num_slots, num_kv_heads, page_size, head_dim]
    layout fmha_sm100 expects. The kernel reads the page and head strides at
    runtime and needs only each page's [page_size, head_dim] block to be
    contiguous, which this view satisfies, so no copy is required.
    """
    buffers = kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
    return buffers[:, 0], buffers[:, 1]


def write_msa_main_kv(
    kv_cache_manager,
    layer_idx: int,
    out_cache_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_live_tokens: int,
) -> None:
    """Write new-token K and V into the paged main cache at out_cache_loc.

    fmha_sm100 reads the paged cache directly, so the new-token K and V must be
    resident before the sparse GQA runs. The write uses the head-major HND view
    so `msa_paged_kv` can return a zero-copy view.
    """
    buffers = kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
    k_view, v_view = buffers[:, 0], buffers[:, 1]
    num_kv_heads = int(k_view.shape[1])
    head_dim = int(k_view.shape[3])
    num_tokens = int(k.shape[0])
    write_kv_slots(
        k_view,
        out_cache_loc,
        k.reshape(num_tokens, num_kv_heads, head_dim),
        num_live_tokens,
        layout="HND",
    )
    write_kv_slots(
        v_view,
        out_cache_loc,
        v.reshape(num_tokens, num_kv_heads, head_dim),
        num_live_tokens,
        layout="HND",
    )


def write_msa_phase_kv(
    attn,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata,
    attention_input_type,
    *,
    token_offset: int,
) -> None:
    """Write one phase's new-token main K/V for one layer.

    The decode kernels and fmha_sm100 read the paged cache directly, so a
    phase's new-token K/V has to be resident before its kernel runs. Each FMHA
    library writes the rows it is about to attend, so a mixed batch's two
    phases cover the step between them and neither repeats the other's write.

    k and v are the phase's token slice, and token_offset its first token on
    the step's token axis, which is what msa_out_cache_loc is indexed by. A
    phase handed no K/V (k and v None) has nothing to write: that is how the
    MiniMax-M3 model layer, which stores the whole step's K/V itself through
    MiniMaxM3MsaSparseAttention.write_layer_caches ahead of its indexer,
    tells both libraries the cache is already resident.
    """
    if attention_input_type != AttentionInputType.mixed:
        raise NotImplementedError(
            "MiniMax-M3 MSA attention requires the mixed attention input type, "
            f"but this call passed {attention_input_type!r}. Its cache slots and "
            "top-k block table are indexed by whole-step token position."
        )
    if k is None or v is None:
        return
    num_tokens = int(k.shape[0])
    if num_tokens == 0:
        return
    write_msa_main_kv(
        metadata.kv_cache_manager,
        attn.layer_idx,
        metadata.msa_out_cache_loc[token_offset : token_offset + num_tokens],
        k,
        v,
        # k and v are this phase's own token slice, so every row owns a slot.
        num_tokens,
    )


def build_kv_page_indices(
    block_ids_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Build the flattened per-request page table fmha_sm100 consumes, on the host.

    Returns int32 global page ids concatenated per request, request b
    contributing the first ceil(kv_len / page_size) entries of its
    block_ids_cpu row. A request with kv_len <= 0 contributes nothing, matching
    the page count the plan derives from the same lengths. Page ids are global
    and non-contiguous in production, so they are not clamped to a per-request
    bound.

    block_ids_cpu is the [batch, max_blocks] host table
    build_paged_kv_slot_mapping obtains from the cache manager. Both use the
    manager's tokens_per_block as the page size, so
    req_to_token[b, p * page_size] // page_size equals block_ids_cpu[b, p] and
    the page table needs no device work. The result is pinned where that helps,
    so the caller stages it with one asynchronous copy.
    """
    pages = (kv_lens_cpu.to(torch.long) + (page_size - 1)) // page_size
    pages.clamp_(min=0)
    total_pages = int(pages.sum())
    if total_pages == 0:
        return torch.empty(0, dtype=torch.int32)
    batch = int(pages.shape[0])
    row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), pages)
    starts = torch.cumsum(pages, 0) - pages
    col = torch.arange(total_pages, dtype=torch.long) - starts[row]
    return maybe_pin_memory(block_ids_cpu[row, col].to(torch.int32))


def per_token_valid_blocks(
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    *,
    causal: bool,
    block_size: int,
) -> torch.Tensor:
    """Return the per-query number of valid KV blocks, on CPU.

    Expands per-request lengths and offsets to a per-token vector so block
    selection can honour each query token's own causal extent.
    """
    qo = qo_lens_cpu.to(torch.long)
    kv = kv_lens_cpu.to(torch.long)
    batch = int(qo.shape[0])
    total = int(qo.sum().item())
    if total == 0:
        return torch.zeros(0, dtype=torch.long)
    batch_row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), qo)
    starts = torch.zeros(batch, dtype=torch.long)
    if batch > 1:
        starts[1:] = torch.cumsum(qo, 0)[:-1]
    intra = torch.arange(total, dtype=torch.long) - starts[batch_row]
    kv_per = kv[batch_row]
    if causal:
        if qo_offset_cpu is not None:
            off = qo_offset_cpu.to(torch.long)[batch_row]
        else:
            off = (kv - qo)[batch_row]
        eff = torch.minimum(off + intra + 1, kv_per)
    else:
        eff = kv_per
    return (eff + block_size - 1) // block_size


def select_blocks_from_maxscore(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    head_major_output: bool = False,
) -> torch.Tensor:
    """Select per-query top-k blocks from per-KV-head block scores.

    Applies init and local forced blocks and per-query valid-block masking
    on the amax-reduced scores [num_kv_heads, n_blocks, total_q]. Returns
    [total_q, num_kv_heads, topk] int32 ascending block ids with -1 tail
    padding. When ``head_major_output`` is set, the logical result uses a
    head-major backing so ``result.permute(1, 0, 2)`` is contiguous without a
    copy.
    """
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    return torch.ops.trtllm.minimax_m3_select_blocks(
        max_score_kv,
        nvb,
        topk,
        init_blocks,
        local_blocks,
        head_major_output,
    )


__all__ = [
    "MSA_REQUIRED_HEAD_DIM",
    "MSA_REQUIRED_TOPK",
    "build_kv_page_indices",
    "check_decode_span_shape",
    "fmha_context_variants",
    "msa_package_available",
    "msa_paged_kv",
    "per_token_valid_blocks",
    "prewarm_fmha_variants",
    "require_msa_module",
    "select_blocks_from_maxscore",
    "write_msa_main_kv",
    "write_msa_phase_kv",
]
