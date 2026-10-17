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
"""TLLM_AUTOTUNER_TP_AGREE: TP ranks pick the candidate with the best worst-rank time."""

from types import SimpleNamespace
from typing import List

import pytest
import torch

import tensorrt_llm._torch.autotuner as autotuner_module
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    DistributedTuningStrategy,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)

OP = "trtllm::flashinfer_mxfp8_gemm_autotuned::gemm"


class _FakeTpDist:
    """tp_allgather returns this rank's object plus the given peers' (rank order)."""

    def __init__(self, peers, rank=0):
        self.peers = list(peers)
        self.rank = rank
        self.calls = 0

    def tp_allgather(self, obj, *, small_payload=False):
        self.calls += 1
        peers = self.peers.pop(0) if self.peers and isinstance(self.peers[0], list) else []
        out = list(peers)
        out.insert(self.rank, obj)
        return out


def _tuner(peers, monkeypatch, *, tp_size=4, attention_dp=False):
    monkeypatch.setenv(autotuner_module.TP_AGREE_ENV, "1")
    tuner = AutoTuner()
    tuner.mapping = SimpleNamespace(tp_size=tp_size, enable_attention_dp=attention_dp)
    tuner._dist = _FakeTpDist(peers)
    return tuner


class _Runner(TunableRunner):
    def __init__(self, tactics):
        self.tactics = tactics

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs):
        return list(self.tactics)

    def forward(self, inputs: List[torch.Tensor], tactic: int = -1, **kwargs):
        return inputs[0]


def _profile(tuner, monkeypatch, local_times, custom_op=OP, strategy=None):
    """Run _profile_runners with this rank's per-tactic times."""
    monkeypatch.setattr(
        tuner,
        "_profile_single_kernel",
        lambda runner, inputs, tactic, tuning_config, use_cuda_graph=False, **kw: local_times[
            tactic
        ],
    )
    config = TuningConfig(
        distributed_tuning_strategy=strategy or DistributedTuningStrategy.INDEPENDENT
    )
    inputs = [torch.zeros(4, 8)]
    profile = tuner._optimization_profiles(config, inputs)[0]
    tuner.stats.tuned_op_profiled_configs[custom_op] = 0
    tuner.stats.failed_profiling_count[custom_op] = set()
    runner_id, tactic, t, _ = tuner._profile_runners(
        custom_op, [_Runner(range(len(local_times)))], inputs, profile, config
    )
    return runner_id, tactic, t


@pytest.mark.parametrize("enabled", [True, False])
def test_profile_runners_picks_best_worst_rank(monkeypatch, enabled):
    # Tactic 1 is fastest here (rank 0) but slow on rank 2; tactic 0 is never worse than 5.
    local = [5.0, 1.0, 9.0]
    peers = [[[4.0, 2.0, 9.0], [5.0, 8.0, 9.0], [4.5, 2.5, 9.0]]]
    tuner = _tuner(peers, monkeypatch)
    if not enabled:
        monkeypatch.setenv(autotuner_module.TP_AGREE_ENV, "0")
    _, tactic, t = _profile(tuner, monkeypatch, local)
    if enabled:
        assert (tactic, t) == (0, 5.0)
        assert tuner._dist.calls == 1  # one all-gather per profiled shape
    else:
        assert (tactic, t) == (1, 1.0)
        assert tuner._dist.calls == 0


def test_profile_runners_agreement_scope(monkeypatch):
    peers = [[[1.0, 9.0], [1.0, 9.0], [1.0, 9.0]]]
    # Ops outside TLLM_AUTOTUNER_TP_AGREE_OPS keep the local pick.
    tuner = _tuner(list(peers), monkeypatch)
    assert _profile(tuner, monkeypatch, [3.0, 2.0], custom_op="trtllm::fp4_gemm::gemm")[1] == 1
    assert tuner._dist.calls == 0
    # So do non-INDEPENDENT strategies (they have their own cross-rank handling) ...
    tuner = _tuner(list(peers), monkeypatch)
    monkeypatch.setattr(tuner, "_maybe_sync_cache_data", lambda *a, **k: None)
    _profile(tuner, monkeypatch, [3.0, 2.0], strategy=DistributedTuningStrategy.BROADCAST)
    assert tuner._dist.calls == 0
    # ... attention DP and a single rank.
    for kw in ({"attention_dp": True}, {"tp_size": 1}):
        tuner = _tuner(list(peers), monkeypatch, **kw)
        assert _profile(tuner, monkeypatch, [3.0, 2.0])[1] == 1
        assert tuner._dist.calls == 0
    # The op list is configurable.
    monkeypatch.setenv(autotuner_module.TP_AGREE_OPS_ENV, "fp4_gemm")
    tuner = _tuner(list(peers), monkeypatch)
    assert _profile(tuner, monkeypatch, [3.0, 2.0], custom_op="trtllm::fp4_gemm::gemm")[1] == 0


def test_agree_over_tp_edge_cases(monkeypatch):
    inf = float("inf")
    # A candidate failing on any rank (inf) is out everywhere.
    tuner = _tuner([[[inf, 1.0], [2.0, 1.5], [2.0, 1.2]]], monkeypatch)
    assert tuner._agree_over_tp(OP, [(0, "a"), (0, "b")], [1.0, 1.0]) == (0, "b", 1.5)
    # None valid on every rank: no pick.
    tuner = _tuner([[[inf, 1.0], [2.0, 1.0], [2.0, inf]]], monkeypatch)
    assert tuner._agree_over_tp(OP, [(0, "a"), (0, "b")], [1.0, 1.0]) == (None, None, inf)
    # Mismatched candidate lists: keep the local pick (and warn).
    tuner = _tuner([[[3.0], [2.0, 1.0], [2.0, 1.0]]], monkeypatch)
    assert tuner._agree_over_tp(OP, [(0, "a"), (0, "b")], [1.0, 2.0]) == (0, "a", 1.0)


class _FakeFlashInferTuner:
    _CUDA_GRAPH_DELAY_MICRO_SECS = 100

    def __init__(self, times):
        self.times = times

    def _profile_single_kernel(self, runner, inputs, tactic, tuning_config, **kwargs):
        t = self.times[tactic]
        if isinstance(t, Exception):
            raise t
        return t


class CuteDSLMxfp8GemmRunner:
    pass


class OtherRunner:
    pass


def test_flashinfer_profile_wrapper(monkeypatch):
    tuner = _tuner([[2.0, 7.0, 3.0], [1.0, 1.0, 1.0]], monkeypatch)
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    wrapped = autotuner_module._tp_agreeing_profile(_FakeFlashInferTuner._profile_single_kernel)
    fi = _FakeFlashInferTuner({0: 4.0, 1: RuntimeError("bad config")})
    # Matching runner: the worst-rank time.
    assert wrapped(fi, CuteDSLMxfp8GemmRunner(), [], 0, None) == 7.0
    # A failing config still joins the all-gather, then re-raises.
    with pytest.raises(RuntimeError):
        wrapped(fi, CuteDSLMxfp8GemmRunner(), [], 1, None)
    assert tuner._dist.calls == 2
    # Other runners are untouched.
    assert wrapped(fi, OtherRunner(), [], 0, None) == 4.0
    assert tuner._dist.calls == 2


def test_install_flashinfer_overrides(monkeypatch):
    fi_autotuner = pytest.importorskip("flashinfer.autotuner.autotuner")
    cls = fi_autotuner.AutoTuner
    monkeypatch.setattr(cls, "_CUDA_GRAPH_DELAY_MICRO_SECS", cls._CUDA_GRAPH_DELAY_MICRO_SECS)
    monkeypatch.setattr(cls, "_profile_single_kernel", cls._profile_single_kernel)
    monkeypatch.delenv(autotuner_module.TP_AGREE_ENV, raising=False)
    monkeypatch.setenv(autotuner_module.FLASHINFER_GRAPH_DELAY_ENV, "2000")
    original = cls._profile_single_kernel
    autotuner_module.install_flashinfer_tuner_overrides()
    assert cls._CUDA_GRAPH_DELAY_MICRO_SECS == 2000
    assert cls._profile_single_kernel is original  # agreement off: profiler untouched
    monkeypatch.setenv(autotuner_module.TP_AGREE_ENV, "1")
    monkeypatch.setattr(cls, "_trtllm_tp_agree_installed", False, raising=False)
    autotuner_module.install_flashinfer_tuner_overrides()
    autotuner_module.install_flashinfer_tuner_overrides()  # idempotent
    assert cls._profile_single_kernel.__wrapped__ is original
