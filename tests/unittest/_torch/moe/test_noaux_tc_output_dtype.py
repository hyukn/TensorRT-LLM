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
"""noaux_tc_op output_dtype: bfloat16 top-k weights from float32 scores, bitwise the float32 weights cast to bfloat16
(MiniMax-M3 TRTLLM_M3_ROUTING_BF16_SCALES drops that cast launch)."""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.routing import (
    MiniMaxM3MoeRoutingMethod,
    noaux_tc_supports_output_dtype,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or not noaux_tc_supports_output_dtype(),
        reason="needs a noaux_tc_op with output_dtype",
    ),
]

NUM_EXPERTS, TOP_K, SCALE = 128, 4, 2.5


def _inputs(num_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.float32) * 3
    bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32) * 0.1
    return logits, bias


@pytest.mark.parametrize("num_tokens", [1, 26, 104, 1024])
def test_bf16_weights_are_the_rounded_fp32_weights(num_tokens: int) -> None:
    logits, bias = _inputs(num_tokens)
    w32, i32 = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE)
    w16, i16 = torch.ops.trtllm.noaux_tc_op(
        logits, bias, 1, 1, TOP_K, SCALE, output_dtype=torch.bfloat16
    )
    assert w32.dtype == torch.float32 and w16.dtype == torch.bfloat16
    assert torch.equal(i16, i32)
    assert torch.equal(w16, w32.to(torch.bfloat16))
    # output_dtype equal to the scores dtype is the default path
    w32b, i32b = torch.ops.trtllm.noaux_tc_op(
        logits, bias, 1, 1, TOP_K, SCALE, output_dtype=torch.float32
    )
    assert torch.equal(w32b, w32) and torch.equal(i32b, i32)


def test_other_output_dtypes_are_rejected() -> None:
    logits, bias = _inputs(8)
    with pytest.raises(RuntimeError, match="output_dtype"):
        torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE, output_dtype=torch.float16)
    with pytest.raises(RuntimeError, match="output_dtype"):
        torch.ops.trtllm.noaux_tc_op(
            logits.to(torch.bfloat16), bias, 1, 1, TOP_K, SCALE, output_dtype=torch.float32
        )


@pytest.mark.parametrize("num_tokens", [26, 104])
def test_minimax_m3_routing_emits_bf16_weights(num_tokens: int) -> None:
    logits, bias = _inputs(num_tokens, seed=1)

    def routing(dtype):
        return MiniMaxM3MoeRoutingMethod(
            top_k=TOP_K,
            num_experts=NUM_EXPERTS,
            callable_e_score_correction_bias=lambda: bias,
            routed_scaling_factor=SCALE,
            output_dtype=dtype,
        )

    ids32, w32 = routing(torch.float32).apply(logits)
    ids16, w16 = routing(torch.bfloat16).apply(logits)
    assert w16.dtype == torch.bfloat16 and ids16.dtype == torch.int32
    assert torch.equal(ids16, ids32)
    assert torch.equal(w16, w32.to(torch.bfloat16))
