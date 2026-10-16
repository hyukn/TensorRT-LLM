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
"""TLLM_AUTOTUNER_CUDA_GRAPH_DELAY_US: keep host stalls out of the CUDA-graph profiling window."""

import statistics
import time
from typing import List

import pytest
import torch

import tensorrt_llm._torch.autotuner as autotuner_module
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)

ENV = "TLLM_AUTOTUNER_CUDA_GRAPH_DELAY_US"


def test_cuda_graph_delay_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert AutoTuner()._CUDA_GRAPH_DELAY_MICRO_SECS == 100


def test_cuda_graph_delay_from_env(monkeypatch):
    monkeypatch.setenv(ENV, "2000")
    assert AutoTuner()._CUDA_GRAPH_DELAY_MICRO_SECS == 2000
    assert AutoTuner._CUDA_GRAPH_DELAY_MICRO_SECS == 100, "the class default must stay untouched"


class _SmallGemmRunner(TunableRunner):
    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
    ) -> List[int]:
        return [0]

    def forward(self, inputs: List[torch.Tensor], tactic: int = 0, **kwargs) -> torch.Tensor:
        return inputs[0] @ inputs[1]


def _profile_us(tuner, inputs, trials=5):
    config = TuningConfig(use_cuda_graph=True)
    times = [
        tuner._profile_single_kernel(
            runner=_SmallGemmRunner(),
            inputs=inputs,
            tactic=0,
            tuning_config=config,
            use_cuda_graph=True,
        )
        * 1e3
        for _ in range(trials)
    ]
    return statistics.median(times)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="profiling needs CUDA")
def test_cuda_graph_profile_uses_env_delay(monkeypatch):
    monkeypatch.setenv(ENV, "1500")
    real = autotuner_module.delay_kernel
    seen = []

    def recording_delay(us, stream):
        seen.append(us)
        real(us, stream)

    monkeypatch.setattr(autotuner_module, "delay_kernel", recording_delay)
    inputs = [torch.randn(64, 256, device="cuda"), torch.randn(256, 256, device="cuda")]
    assert _profile_us(AutoTuner(), inputs, trials=1) > 0
    assert seen and set(seen) == {1500}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="profiling needs CUDA")
def test_host_stall_lands_in_window_unless_delay_covers_it(monkeypatch):
    """A 1 ms host stall between the start event and the replay inflates a microsecond kernel
    by ~90 us per repeat with the default 100 us delay; a 5 ms delay keeps it clean."""
    monkeypatch.setenv("TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE", "1")
    inputs = [torch.randn(64, 256, device="cuda"), torch.randn(256, 256, device="cuda")]
    real_replay = torch.cuda.CUDAGraph.replay

    monkeypatch.delenv(ENV, raising=False)
    clean = _profile_us(AutoTuner(), inputs)

    def stalled_replay(graph):
        time.sleep(0.001)  # the host thread stalls after queuing the start event
        return real_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", stalled_replay)
    stalled_default = _profile_us(AutoTuner(), inputs)
    monkeypatch.setenv(ENV, "5000")
    stalled_long_delay = _profile_us(AutoTuner(), inputs)

    assert stalled_default > clean + 30, (clean, stalled_default)
    assert stalled_long_delay < clean + 5, (clean, stalled_long_delay)
