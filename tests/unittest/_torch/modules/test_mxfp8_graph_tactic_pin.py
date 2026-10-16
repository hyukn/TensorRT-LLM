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
"""Pinning the MXFP8 decode-graph tactics (TLLM_AUTOTUNER_PIN_GRAPH_TACTICS)."""

import contextlib
import sys
from types import SimpleNamespace
from typing import List

import pytest
import torch

import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, TunableRunner
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    FlashInferMXFP8GemmRunner,
    _choose_mxfp8_tactic,
    _mxfp8_scale_infer_shape,
)
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_autotune
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PIN_GRAPH_TACTICS_ENV,
    graph_tactic_cache_paths,
)


def test_graph_tactic_cache_paths_off_by_default(monkeypatch):
    monkeypatch.delenv(PIN_GRAPH_TACTICS_ENV, raising=False)
    monkeypatch.setenv("TLLM_AUTOTUNER_CACHE_PATH", "/tmp/tactics.json")
    assert graph_tactic_cache_paths(0) == (None, None)


def test_graph_tactic_cache_paths_need_a_cache_path(monkeypatch):
    monkeypatch.setenv(PIN_GRAPH_TACTICS_ENV, "1")
    monkeypatch.delenv("TLLM_AUTOTUNER_CACHE_PATH", raising=False)
    assert graph_tactic_cache_paths(0) == (None, None)
    monkeypatch.setenv("TLLM_AUTOTUNER_CACHE_PATH", "")
    assert graph_tactic_cache_paths(0) == (None, None)


def test_graph_tactic_cache_paths_per_rank_flashinfer_file(monkeypatch):
    monkeypatch.setenv(PIN_GRAPH_TACTICS_ENV, "1")
    monkeypatch.setenv("TLLM_AUTOTUNER_CACHE_PATH", "/x/ll_tp4.json")
    assert graph_tactic_cache_paths(0) == ("/x/ll_tp4.json", "/x/ll_tp4.json.flashinfer.rank0.json")
    assert graph_tactic_cache_paths(3) == ("/x/ll_tp4.json", "/x/ll_tp4.json.flashinfer.rank3.json")


def _fake_flashinfer(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def autotune(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(autotune=autotune))
    return calls


def test_flashinfer_mxfp8_autotune_passes_cache(monkeypatch):
    calls = _fake_flashinfer(monkeypatch)
    with flashinfer_mxfp8_autotune(cache="/x/fi.json"):
        assert linear_module._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
    assert not linear_module._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
    assert calls == [((), {"cache": "/x/fi.json"})]


def test_flashinfer_mxfp8_autotune_default_has_no_cache(monkeypatch):
    calls = _fake_flashinfer(monkeypatch)
    with flashinfer_mxfp8_autotune():
        assert linear_module._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
    with flashinfer_mxfp8_autotune(cache=None):
        pass
    assert calls == [((), {}), ((), {})]
    assert not linear_module._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()


class _DecodeGemmRunner(TunableRunner):
    """Stand-in for FlashInferMXFP8GemmRunner: same tuning config, counted forwards."""

    tuning_config = FlashInferMXFP8GemmRunner.tuning_config
    calls: List[int] = []

    def unique_id(self):
        return (torch.bfloat16,)

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
    ) -> List[int]:
        return [-1, 0, 1]

    def forward(self, inputs: List[torch.Tensor], tactic: int = -1, **kwargs) -> torch.Tensor:
        type(self).calls.append(tactic)
        act, _, weight, _ = inputs
        out = act.float() @ weight.float().t()
        for _ in range(8 if tactic == 0 else 0):
            out = out + act.float() @ weight.float().t()
        return out


def _decode_gemm_inputs(m: int, n: int = 256, k: int = 128) -> List[torch.Tensor]:
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    act_scale = torch.zeros(_mxfp8_scale_infer_shape([[m, k]]), device="cuda", dtype=torch.uint8)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    weight_scale = torch.zeros(n * k // 32, device="cuda", dtype=torch.uint8)
    return [act, act_scale, weight, weight_scale]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AutoTuner profiling needs CUDA")
def test_decode_graph_pick_round_trips_through_saved_cache(tmp_path):
    """A pick saved after the graph warmup is reused by the next launch without profiling."""
    op = "test::graph_tactic_pin::gemm"
    path = str(tmp_path / "tactics.json")
    tuner = AutoTuner.get()
    tuner.profiling_cache.clear()
    runner = _DecodeGemmRunner()
    _DecodeGemmRunner.calls.clear()

    picks = {
        m: _choose_mxfp8_tactic(op, runner, _decode_gemm_inputs(m), tune=True) for m in (32, 16)
    }
    assert _DecodeGemmRunner.calls, "the first launch profiles the decode buckets"
    tuner.profiling_cache.save_cache(path, rank=0)

    # Next launch: fresh in-memory cache, then the autotuner pass loads the file.
    tuner.profiling_cache.clear()
    _DecodeGemmRunner.calls.clear()
    tuner.profiling_cache.load_cache(path, rank=0)
    for m, bucket in ((32, 32), (28, 32), (20, 32), (16, 16), (12, 16)):
        assert _choose_mxfp8_tactic(op, runner, _decode_gemm_inputs(m), tune=True) == picks[bucket]
    assert _DecodeGemmRunner.calls == [], "a pinned pick must not be profiled again"
    tuner.profiling_cache.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer profiling needs CUDA")
def test_flashinfer_cache_file_skips_profiling(tmp_path):
    """FlashInfer's own MXFP8 picks: a loaded config file is a hit, also in tuning mode."""
    fi_autotuner = pytest.importorskip("flashinfer.autotuner")
    path = str(tmp_path / "fi_tactics.json")

    class _FiRunner(fi_autotuner.TunableRunner):
        calls: List[int] = []

        def get_valid_tactics(self, inputs, profile):
            return [0, 1]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            type(self).calls.append(tactic)
            return inputs[0] * 2

    config = fi_autotuner.TuningConfig(
        dynamic_tensor_specs=(
            fi_autotuner.DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(8, 16),
                map_to_tuning_buckets=lambda x: x,
            ),
        )
    )
    tuner = fi_autotuner.AutoTuner.get()
    tuner.clear_cache()
    x = torch.randn(16, 64, device="cuda")
    with flashinfer_mxfp8_autotune(cache=path):
        _, tactic = tuner.choose_one("test_graph_tactic_pin", [_FiRunner()], config, [x])
    assert _FiRunner.calls, "the first launch profiles"

    tuner.clear_cache()
    _FiRunner.calls.clear()
    with flashinfer_mxfp8_autotune(cache=path):
        _, again = tuner.choose_one("test_graph_tactic_pin", [_FiRunner()], config, [x])
    assert again == tactic
    assert _FiRunner.calls == [], "a config from the file must not be profiled again"
    tuner.clear_cache()
