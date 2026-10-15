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
"""MiniMax-M3 MoE: routed-input NVFP4 quantize overlapped with the router GEMM.

``TRTLLM_M3_MOE_PREQUANT_OVERLAP=1`` moves the routed experts' input quantize
from behind the router GEMM (inside the fused MoE) onto the shared-expert
stream. The fused MoE must then see exactly the FP4 tensor and scales it would
have produced itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import TRTLLMGenNvfp4Base
from tensorrt_llm._torch.utils import EventType, Fp4QuantizedTensor

HIDDEN = 512
NUM_EXPERTS = 8


def _sm100_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


requires_sm100 = pytest.mark.skipif(not _sm100_or_newer(), reason="needs SM100+ for NVFP4")


class _NativeFp4OpBackend:
    def fp4_quantize(self, x, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout):
        return torch.ops.trtllm.fp4_quantize(
            x, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )


def _nvfp4_backend_stub(hidden: int, device: str = "cuda") -> SimpleNamespace:
    """The attributes TRTLLMGenNvfp4Base.quantize_input reads, with the real quantize op."""
    stub = SimpleNamespace(
        w3_w1_weight=torch.empty((1, 1, hidden // 2), dtype=torch.uint8, device=device),
        fc31_input_scale=torch.tensor([448.0 * 6.0 / 3.0], dtype=torch.float32, device=device),
        fc31_act_scale=None,
        scaling_vector_size=16,
        op_backend=_NativeFp4OpBackend(),
    )
    stub.quantize_input = lambda x, post_quant_comm=True: TRTLLMGenNvfp4Base.quantize_input(
        stub, x, post_quant_comm=post_quant_comm
    )
    return stub


def _quantize_input(backend, x):
    return TRTLLMGenNvfp4Base.quantize_input(backend, x, post_quant_comm=False)


@requires_sm100
@pytest.mark.parametrize("num_tokens", [1, 4, 24, 37])
def test_prequantized_input_round_trips_through_quantize_input(num_tokens: int) -> None:
    """quantize_input(Fp4QuantizedTensor(quantize_input(x))) == quantize_input(x), bitwise."""
    torch.manual_seed(0)
    backend = _nvfp4_backend_stub(HIDDEN)
    x = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda") * 3
    ref_q, ref_sf = _quantize_input(backend, x)
    pre_q, pre_sf = _quantize_input(backend, x)
    got_q, got_sf = _quantize_input(
        backend, Fp4QuantizedTensor(fp4_tensor=pre_q, scaling_factor=pre_sf, is_sf_swizzled=False)
    )
    assert got_q.shape == ref_q.shape and got_sf.shape == ref_sf.shape
    assert torch.equal(got_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(got_sf.view(torch.uint8), ref_sf.view(torch.uint8))


class _Gate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((NUM_EXPERTS, HIDDEN), dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states.float(), self.weight)


class _Experts(nn.Module):
    """Stands in for the fused MoE: quantizes (or unpacks) through the backend, then mixes."""

    def __init__(self, backend) -> None:
        super().__init__()
        self.backend = backend
        self.comm = None
        self.calls = []

    def calculate_num_chunks(self, all_rank_num_tokens) -> int:
        return 1

    def forward(self, x, router_logits, *, output_dtype=None, **kwargs) -> torch.Tensor:
        self.calls.append(type(x).__name__)
        x_q, x_sf = _quantize_input(self.backend, x)
        out_dtype = output_dtype if isinstance(x, Fp4QuantizedTensor) else x.dtype
        q = x_q.view(torch.uint8).float()
        sf = x_sf.view(torch.uint8).float()
        mixed = q[:, : HIDDEN // 2].repeat(1, 2) + sf.sum(-1, keepdim=True)
        return (mixed + router_logits.softmax(-1)[:, :1]).to(out_dtype)


class _Shared(nn.Module):
    def forward(self, hidden_states: torch.Tensor, lora_params=None) -> torch.Tensor:
        return hidden_states * 0.5


def _make_moe(backend) -> MiniMaxM3MoE:
    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.gate = _Gate()
    moe.experts = _Experts(backend)
    moe.shared_experts = _Shared()
    moe.routed_output_is_global = False
    moe.allreduce = None
    moe.aux_stream = torch.cuda.Stream()
    moe.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}
    moe._prequant_event = torch.cuda.Event()
    return moe


@requires_sm100
@pytest.mark.parametrize("num_tokens", [4, 24])
def test_prequant_overlap_matches_default_path_in_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int
) -> None:
    """Under a captured CUDA graph with two streams, the overlap path is bitwise identical."""
    torch.manual_seed(1)
    backend = _nvfp4_backend_stub(HIDDEN)
    monkeypatch.setattr(m3, "_moe_prequant_backend", lambda experts: experts.backend)
    moe = _make_moe(backend)
    meta = SimpleNamespace(all_rank_num_tokens=None)
    static_in = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")

    def run_graph(enabled: bool) -> torch.Tensor:
        monkeypatch.setattr(m3, "_MOE_PREQUANT_OVERLAP", enabled)
        moe.experts.calls.clear()
        with with_multi_stream(True):
            moe(static_in, meta)  # warm up outside capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = moe(static_in, meta)
        assert moe.experts.calls[-1] == ("Fp4QuantizedTensor" if enabled else "Tensor")
        outs = []
        for _ in range(2):
            static_in.copy_(torch.randn_like(static_in))
            graph.replay()
            torch.cuda.synchronize()
            outs.append((static_in.clone(), static_out.clone()))
        return outs

    torch.manual_seed(2)
    ref = run_graph(False)
    torch.manual_seed(2)
    got = run_graph(True)
    for (ref_in, ref_out), (got_in, got_out) in zip(ref, got):
        assert torch.equal(ref_in, got_in)
        assert torch.equal(ref_out, got_out)


@pytest.mark.cpu_only
def test_prequant_overlap_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only multi-stream (decode graph) forwards without LoRA take the overlap path."""
    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.shared_experts = _Shared()
    moe.aux_stream = object()
    moe.experts = SimpleNamespace(calculate_num_chunks=lambda tokens: 1)
    monkeypatch.setattr(m3, "_moe_prequant_backend", lambda experts: object())
    hidden = torch.zeros((4, HIDDEN))

    monkeypatch.setattr(m3, "_MOE_PREQUANT_OVERLAP", False)
    with with_multi_stream(True):
        assert not moe._use_prequant_overlap(hidden, None)

    monkeypatch.setattr(m3, "_MOE_PREQUANT_OVERLAP", True)
    assert not moe._use_prequant_overlap(hidden, None)  # no multi-stream: eager step
    with with_multi_stream(True):
        assert moe._use_prequant_overlap(hidden, None)
        assert not moe._use_prequant_overlap(hidden, {"lora": 1})
        assert not moe._use_prequant_overlap(hidden.unsqueeze(0), None)
        moe.experts = SimpleNamespace(calculate_num_chunks=lambda tokens: 2)
        assert not moe._use_prequant_overlap(hidden, None)
        moe.experts = SimpleNamespace(calculate_num_chunks=lambda tokens: 1)
        monkeypatch.setattr(m3, "_moe_prequant_backend", lambda experts: None)
        assert not moe._use_prequant_overlap(hidden, None)


@pytest.mark.cpu_only
def test_prequant_backend_requires_local_trtllm_gen_nvfp4() -> None:
    """Communication, DWDP, router-weight-on-input and non-NVFP4 backends are excluded."""

    class _Leaf(TRTLLMGenNvfp4Base):
        pass

    _Leaf.__abstractmethods__ = frozenset()  # isinstance only; never initialized or run
    nvfp4 = _Leaf.__new__(_Leaf)
    assert m3._moe_prequant_backend(SimpleNamespace(backend=nvfp4, comm=None)) is nvfp4
    assert m3._moe_prequant_backend(SimpleNamespace(backend=nvfp4, comm=object())) is None
    assert (
        m3._moe_prequant_backend(SimpleNamespace(backend=nvfp4, comm=None, enable_dwdp=True))
        is None
    )
    assert (
        m3._moe_prequant_backend(
            SimpleNamespace(backend=nvfp4, comm=None, apply_router_weight_on_input=True)
        )
        is None
    )
    assert m3._moe_prequant_backend(SimpleNamespace(backend=object(), comm=None)) is None
