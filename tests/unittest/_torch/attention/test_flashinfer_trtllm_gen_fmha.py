# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from fmha_test_utils import FakeAttention

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization.mode import QuantMode


def test_flashinfer_fp8_mode_remains_implementation_local() -> None:
    attn = FakeAttention()
    attn.quant_mode = int(QuantMode.from_description(use_fp8_kv_cache=True))
    fmha = FlashInferTrtllmGenFmha(attn)
    output = torch.empty(1, dtype=torch.bfloat16)

    assert fmha._use_fp8_context_fmha(output, AttentionInputType.context_only)
    assert fmha._use_fp8_context_fmha(output, AttentionInputType.mixed)
    assert not fmha._use_fp8_context_fmha(output, AttentionInputType.generation_only)


@pytest.mark.parametrize(
    "num_contexts,num_generations,dtype,expect_fallback",
    [
        (1, 0, torch.bfloat16, True),
        (4, 0, torch.bfloat16, True),
        (5, 0, torch.bfloat16, False),
        (1, 1, torch.bfloat16, False),
        (2, 9, torch.bfloat16, False),
        (2, 19, torch.bfloat16, False),
        (2, 38, torch.bfloat16, False),
        (4, 1, torch.bfloat16, False),
        (0, 1, torch.bfloat16, False),
        (1, 0, torch.float16, False),
    ],
)
def test_small_context_fallback_preserves_mixed_batch_generation(
    num_contexts: int,
    num_generations: int,
    dtype: torch.dtype,
    expect_fallback: bool,
) -> None:
    attn = FakeAttention()
    attn.head_dim = 256
    attn.sparse_params = None
    attn.position_embedding_type = 0
    fmha = FlashInferTrtllmGenFmha(attn)
    metadata = SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=None,
        tokens_per_block=32,
        is_cross=False,
        beam_width=1,
    )
    if num_contexts == 0:
        input_type = AttentionInputType.generation_only
        phases = (None, FmhaPhase.GENERATION)
    elif num_generations == 0:
        input_type = AttentionInputType.context_only
        phases = (None, FmhaPhase.CONTEXT)
    else:
        input_type = AttentionInputType.mixed
        phases = (None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION)
    num_tokens = num_contexts + num_generations
    q = torch.empty((num_tokens, 3 * attn.head_dim), dtype=dtype)
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_tokens, attn.head_dim), dtype=dtype),
        attention_input_type=input_type,
        is_fused_qkv=True,
    )

    for phase in phases:
        supported, reason = fmha._is_supported_with_reason(
            q, None, None, attn, metadata, forward_args, phase=phase
        )
        assert supported is not expect_fallback, reason
        if expect_fallback:
            assert "small-batch BF16 context attention" in reason


@pytest.mark.parametrize("kv_cache_dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize(
    "dtype,sm_version",
    [(torch.bfloat16, 100), (torch.float16, 103)],
    ids=["small_bf16_batch", "sm103_fp16"],
)
def test_flashinfer_quantized_kv_context_avoids_fp16_bf16_fallback(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: DataType,
    dtype: torch.dtype,
    sm_version: int,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen.get_sm_version",
        lambda: sm_version,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    monkeypatch.setattr(fmha, "_get_total_num_blocks", lambda _: 0)
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        position_embedding_type=0,
        head_dim=256,
        num_heads=32,
        num_kv_heads=2,
    )
    q_hidden_size = attn.num_heads * attn.head_dim
    qkv_hidden_size = q_hidden_size + 2 * attn.num_kv_heads * attn.head_dim
    q = torch.empty((1, qkv_hidden_size), dtype=dtype)
    metadata = SimpleNamespace(
        num_contexts=1,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=SimpleNamespace(dtype=kv_cache_dtype),
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=64,
        beam_width=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((1, q_hidden_size), dtype=dtype),
        attention_input_type=AttentionInputType.context_only,
        is_fused_qkv=True,
        update_kv_cache=True,
    )

    for phase in (None, FmhaPhase.CONTEXT):
        supported, reason = fmha._is_supported_with_reason(
            q,
            None,
            None,
            attn,
            metadata,
            forward_args,
            phase=phase,
        )

        assert supported, reason
        assert reason == ""


def test_decode_split_sm_count_scales_with_oversubscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen as fi_gen

    fmha = FlashInferTrtllmGenFmha(FakeAttention())
    with pytest.raises(RuntimeError, match="SM count"):
        fmha._decode_split_sm_count()
    fmha._multi_processor_count = 152
    assert fmha._decode_split_sm_count() == 152
    monkeypatch.setattr(fi_gen, "_GEN_DECODE_KV_SPLIT_OVERSUB", 4)
    assert fmha._decode_split_sm_count() == 4 * 152


@pytest.mark.parametrize("oversub", [1, 4])
def test_counter_block_covers_the_oversubscribed_split(
    monkeypatch: pytest.MonkeyPatch, oversub: int
) -> None:
    """The multi-CTA KV counters are sized for the SM count the launcher splits for, not the device's."""
    import tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen as fi_gen

    monkeypatch.setattr(fi_gen, "_GEN_DECODE_KV_SPLIT_OVERSUB", oversub)
    monkeypatch.setattr(fi_gen, "_get_workspace_size", lambda **_: 0)
    attn = FakeAttention()
    attn.num_heads, attn.num_kv_heads, attn.head_dim, attn.rope_dim = 4, 4, 128, 0
    attn.quant_mode = 0
    fmha = FlashInferTrtllmGenFmha(attn)
    fmha._multi_processor_count = 152
    q = torch.empty((20, attn.num_heads * attn.head_dim), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        max_num_sequences=20,
        max_num_requests=20,
        is_cuda_graph=False,
        num_ctx_tokens=0,
        max_context_length=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=AttentionInputType.generation_only,
        is_fused_qkv=False,
    )
    fmha.prepare_workspace(q, None, None, metadata, forward_args, torch.empty(0, dtype=torch.uint8))

    counters = fmha._get_multi_ctas_kv_counter_buffer()
    assert counters.numel() * counters.element_size() >= oversub * 152 * torch.int32.itemsize
    assert counters.numel() * counters.element_size() >= 4 * 20 * torch.int32.itemsize


def test_oversubscribed_decode_split_patches_only_for_the_call() -> None:
    fi_decode = pytest.importorskip("flashinfer.decode")
    import tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen as fi_gen

    original = fi_decode.get_device_sm_count
    with fi_gen._oversubscribed_decode_split(608):
        assert fi_decode.get_device_sm_count(torch.device("cuda", 0)) == 608
    assert fi_decode.get_device_sm_count is original
    with pytest.raises(ValueError), fi_gen._oversubscribed_decode_split(608):
        raise ValueError("restored on error too")
    assert fi_decode.get_device_sm_count is original


def _is_sm100f() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 10 and minor in (0, 3)


@pytest.mark.skipif(not _is_sm100f(), reason="trtllm-gen decode kernels are SM100/SM103 only")
@pytest.mark.parametrize("oversub", [2, 4])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("batch", [19, 20, 24])
def test_oversubscribed_decode_split_matches_default(oversub: int, q_len: int, batch: int) -> None:
    """An oversubscribed SM count only re-partitions the KV walk across CTAs.

    64 q heads over 4 KV heads, BF16 Q, FP8 KV, one long and many short sequences: at batch >= 20 the default
    launcher runs the persistent (unsplit) kernel on a 152-SM part and the oversubscribed one a multi-CTA
    split, so this compares the two kernels. Same output within the split-reduction rounding.
    """
    import flashinfer.decode as fi_decode

    import tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen as fi_gen

    page, head_dim, num_heads, num_kv_heads = 32, 128, 64, 4
    seq_lens = [page * 160 + 3] + [page * 9 + 7] * (batch - 1)
    gen = torch.Generator(device="cuda").manual_seed(3)
    pages = [(s + page - 1) // page for s in seq_lens]
    num_slots = sum(pages) + 5
    kv = (
        torch.randn((num_slots, 2, num_kv_heads, page, head_dim), device="cuda", generator=gen)
        .mul_(0.5)
        .to(torch.float8_e4m3fn)
    )
    perm = torch.randperm(num_slots, generator=gen, device="cuda").to(torch.int32)
    block_tables = torch.zeros((batch, max(pages)), device="cuda", dtype=torch.int32)
    offset = 0
    for row, n in enumerate(pages):
        block_tables[row, :n] = perm[offset : offset + n]
        offset += n
    q = torch.randn(
        (batch * q_len, num_heads, head_dim), device="cuda", generator=gen, dtype=torch.bfloat16
    )
    seq_lens_t = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    def run(split_sm_count: int) -> torch.Tensor:
        out = torch.zeros((batch * q_len, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
        counters = torch.zeros(
            fi_gen._get_multi_ctas_kv_counter_size(num_heads, batch, split_sm_count),
            device="cuda",
            dtype=torch.uint8,
        )
        workspace = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        with fi_gen._oversubscribed_decode_split(split_sm_count):
            fi_decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv,
                workspace_buffer=workspace,
                block_tables=block_tables,
                seq_lens=seq_lens_t,
                max_seq_len=max(seq_lens),
                bmm1_scale=head_dim**-0.5,
                bmm2_scale=1.0,
                out=out,
                kv_layout="HND",
                backend="trtllm-gen",
                q_len_per_req=q_len,
                multi_ctas_kv_counter_buffer=counters,
            )
        return out

    ref = run(sm_count)
    out = run(sm_count * oversub)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
