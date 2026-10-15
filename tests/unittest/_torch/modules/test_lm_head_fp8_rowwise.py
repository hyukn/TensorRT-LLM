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
"""LMHead FP8 rowwise projection (opt-in for Eagle3 draft heads that own their weight)."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_speculative import draft_lm_head_fp8_enabled
from tensorrt_llm._torch.modules.embedding import LMHead, fp8_rowwise_linear
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.mapping import Mapping

requires_fp8_mm = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="torch._scaled_mm rowwise needs SM90+",
)


def _reference(x, w_fp8, w_scale_t):
    """Dequantized math of fp8_rowwise_linear in fp32."""
    xf = x.float()
    xs = (xf.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    xq = (xf / xs).to(torch.float8_e4m3fn).float() * xs
    wq = w_fp8.float() * w_scale_t.t()
    return xq @ wq.t()


def _head(vocab, hidden):
    head = LMHead(
        vocab,
        hidden,
        dtype=torch.bfloat16,
        mapping=Mapping(),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    ).cuda()
    with torch.no_grad():
        head.weight.copy_((torch.randn(vocab, hidden, device="cuda") * 0.02).to(torch.bfloat16))
    return head


@requires_fp8_mm
@pytest.mark.parametrize("rows", [1, 4, 24])
def test_fp8_rowwise_head_matches_dequantized_reference(rows):
    torch.manual_seed(0)
    head = _head(4096, 1024)
    bf16_out = head(torch.zeros(1, 1024, device="cuda", dtype=torch.bfloat16))
    assert not head.uses_fp8_rowwise_projection
    head.enable_fp8_rowwise_projection(chunk_rows=1000)  # exercises a ragged last chunk
    assert head.uses_fp8_rowwise_projection
    assert head._fp8_weight.dtype == torch.float8_e4m3fn and head._fp8_weight_scale.shape == (
        1,
        4096,
    )
    # Each row's scale maps its amax onto the E4M3 max.
    w = head.weight.float()
    torch.testing.assert_close(head._fp8_weight_scale.t(), w.abs().amax(1, keepdim=True) / 448.0)

    x = torch.randn(rows, 1024, device="cuda").to(torch.bfloat16)
    got = head(x)
    ref = _reference(x, head._fp8_weight, head._fp8_weight_scale)
    torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)
    # The vocab-shard entry point takes the same FP8 path.
    torch.testing.assert_close(head.forward_vocab_shard(x).float(), got.float(), rtol=0, atol=0)
    # And it is close to (but not) the bf16 head: argmax mostly agrees.
    exact = x.float() @ w.t()
    agree = (got.float().argmax(1) == exact.argmax(1)).float().mean().item()
    assert agree >= 0.9
    assert bf16_out.shape == (1, 4096)


@requires_fp8_mm
def test_fp8_rowwise_linear_handles_zero_rows():
    w_fp8 = torch.zeros(64, 128, device="cuda", dtype=torch.float8_e4m3fn)
    w_scale_t = torch.ones(1, 64, device="cuda")
    x = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
    out = fp8_rowwise_linear(x, w_fp8, w_scale_t)
    assert out.dtype == torch.bfloat16 and torch.count_nonzero(out) == 0 and not out.isnan().any()


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_draft_lm_head_fp8_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TRTLLM_SPEC_DRAFT_LM_HEAD_FP8", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_SPEC_DRAFT_LM_HEAD_FP8", value)
    assert draft_lm_head_fp8_enabled() is expected
