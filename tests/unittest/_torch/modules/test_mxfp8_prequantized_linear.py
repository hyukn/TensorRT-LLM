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
"""An MXFP8 Linear fed a pre-quantized activation runs the GEMM its quantize + GEMM path would."""

import pytest
import torch

from tensorrt_llm._torch.custom_ops.torch_custom_ops import IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    _mxfp8_cutlass_op_available,
    flashinfer_mxfp8_autotune,
    flashinfer_mxfp8_decode_graph_capture,
)
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not _mxfp8_cutlass_op_available() or not IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
    reason="needs the MXFP8 GEMM ops and FlashInfer CuTeDSL MXFP8 (SM100+)",
)


def _tuned_linear(out_f: int, in_f: int) -> Linear:
    torch.manual_seed(0)
    weight = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    weight_e4m3, weight_scale = quant_bf16_to_mxfp8(weight, 32)
    lin = Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
    ).cuda()
    lin.load_weights([{"weight": weight_e4m3, "weight_scale_inv": weight_scale}])
    method = lin.quant_method
    assert isinstance(method, MXFP8LinearMethod)
    assert method.enable_flashinfer_auto()
    method.tune_decode_graph_backends = True
    return lin


@pytest.mark.parametrize("num_tokens", [4, 24, 64])
@pytest.mark.parametrize("out_f, in_f", [(6144, 2048), (512, 768)])
def test_prequantized_matches_quantize_then_gemm(monkeypatch, num_tokens, out_f, in_f):
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    lin = _tuned_linear(out_f, in_f)
    x = torch.randn(num_tokens, in_f, dtype=torch.bfloat16, device="cuda")
    with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
        lin(x)  # warmup-only pass: tunes this bucket's quantize and GEMM tactics
    assert lin.quant_method.accepts_prequantized_input(lin) is False  # outside capture

    act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    pre = MxFp8QuantizedTensor(fp8_tensor=act, scaling_factor=act_sf, is_sf_swizzled=True)
    static_x = x.clone()
    graphs = []
    outs = []
    for inp in (static_x, pre):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            with flashinfer_mxfp8_decode_graph_capture():
                assert lin.quant_method.accepts_prequantized_input(lin)
                outs.append(lin(inp))
        graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


def test_prequantized_rejected_outside_decode_graph_path():
    lin = _tuned_linear(256, 512)
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    with pytest.raises(RuntimeError, match="pre-quantized"):
        lin(MxFp8QuantizedTensor(fp8_tensor=act, scaling_factor=act_sf, is_sf_swizzled=True))
    with flashinfer_mxfp8_decode_graph_capture():
        with pytest.raises(RuntimeError, match="pre-quantized"):
            lin(MxFp8QuantizedTensor(fp8_tensor=act, scaling_factor=act_sf, is_sf_swizzled=False))
