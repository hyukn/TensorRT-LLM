# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInferMXFP8GemmRunner's swept CuTeDSL tactic (TRTLLM_MXFP8_CUTE_DSL_TUNED_CONFIG=1).

The decode CUDA graphs pick, per token bucket, between FlashInfer's CUTLASS MXFP8 GEMM and its CuTeDSL
GEMM with the heuristic config. The opt-in CUTE_DSL_TUNED tactic runs the CuTeDSL GEMM with the config
FlashInfer's own sweep picks for the shape. It is offered from M = 64 only (smaller sweeps picked configs
that differ in the last bf16 bit), and there it has to write bitwise what the existing tactics write at the
MiniMax-M3 TP4 decode shapes (MXFP8 attention projections, shared expert, dense-layer MLP).
"""

import pytest
import torch

from tensorrt_llm._torch.autotuner import OptimizationProfile
from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not tco.IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
    reason="needs SM100-family Blackwell with FlashInfer + CuTe DSL",
)

# (N, K) per TP4 rank: fused QKV + index projection, o_proj, shared expert gate_up / down,
# dense layers 0-2 QKV and MLP gate_up / down.
SHAPES = [
    (2560, 6144),
    (6144, 2048),
    (1536, 6144),
    (6144, 768),
    (2304, 6144),
    (6144, 6144),
    (6144, 3072),
]


def _quant(x):
    return torch.ops.trtllm.mxfp8_quantize(x, True)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [64, 128, 192, 256])
def test_tuned_cute_dsl_matches_other_tactics(n, k, m, monkeypatch):
    monkeypatch.setenv(tco._MXFP8_CUTE_DSL_TUNED_ENV, "1")
    torch.manual_seed(n + k + m)
    w, wsf = _quant(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05)
    a, asf = _quant(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    runner = tco.FlashInferMXFP8GemmRunner(torch.bfloat16)
    inputs = [a, asf, w, wsf]
    tactics = runner.get_valid_tactics(inputs, OptimizationProfile())
    assert tactics == [runner.CUTLASS, runner.CUTE_DSL, runner.CUTE_DSL_TUNED]
    ref = runner(inputs, tactic=runner.CUTE_DSL)
    for tactic in (runner.CUTE_DSL_TUNED, runner.CUTLASS):
        out = runner(inputs, tactic=tactic)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        assert torch.equal(out, ref), f"tactic {tactic}"


@pytest.mark.parametrize(
    "env,min_m,m,offered",
    [(None, None, 128, False), ("1", None, 32, False), ("1", None, 64, True), ("1", "1", 4, True)],
)
def test_tuned_tactic_gates(env, min_m, m, offered, monkeypatch):
    """Opt-in, and only from M = 64 (smaller sweeps are not bitwise) unless the floor is lowered."""
    if env is None:
        monkeypatch.delenv(tco._MXFP8_CUTE_DSL_TUNED_ENV, raising=False)
    else:
        monkeypatch.setenv(tco._MXFP8_CUTE_DSL_TUNED_ENV, env)
    if min_m is None:
        monkeypatch.delenv("TRTLLM_MXFP8_CUTE_DSL_TUNED_MIN_M", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_MXFP8_CUTE_DSL_TUNED_MIN_M", min_m)
    runner = tco.FlashInferMXFP8GemmRunner(torch.bfloat16)
    w, wsf = _quant(torch.randn(256, 512, device="cuda", dtype=torch.bfloat16))
    a, asf = _quant(torch.randn(m, 512, device="cuda", dtype=torch.bfloat16))
    expected = [runner.CUTLASS, runner.CUTE_DSL] + ([runner.CUTE_DSL_TUNED] if offered else [])
    assert runner.get_valid_tactics([a, asf, w, wsf], OptimizationProfile()) == expected
