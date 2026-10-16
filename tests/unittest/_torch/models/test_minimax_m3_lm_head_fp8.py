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
"""MiniMax-M3: opt-in FP8 rowwise target lm_head (TRTLLM_M3_LM_HEAD_FP8) is built after weight load, TP only."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.models.modeling_minimaxm3 as m3

pytestmark = pytest.mark.cpu_only


class _Head:
    def __init__(self, adp=False, quant=False, dim=2):
        self.mapping = SimpleNamespace(enable_attention_dp=adp)
        self.has_any_quant = quant
        self.weight = torch.zeros((8,) * dim)
        self.calls = 0

    def enable_fp8_rowwise_projection(self):
        self.calls += 1


def _model(head):
    return SimpleNamespace(lm_head=head)


def test_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m3, "_LM_HEAD_FP8", False)
    head = _Head()
    assert not m3._maybe_enable_fp8_lm_head(_model(head))
    assert head.calls == 0


def test_built_on_every_load_for_tp_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m3, "_LM_HEAD_FP8", True)
    head = _Head()
    assert m3._maybe_enable_fp8_lm_head(_model(head))
    assert m3._maybe_enable_fp8_lm_head(_model(head))  # refit rebuilds the copy
    assert head.calls == 2


@pytest.mark.parametrize("kwargs", [{"adp": True}, {"quant": True}, {"dim": 3}])
def test_skipped_for_adp_quantized_or_odd_heads(monkeypatch: pytest.MonkeyPatch, kwargs) -> None:
    monkeypatch.setattr(m3, "_LM_HEAD_FP8", True)
    head = _Head(**kwargs)
    assert not m3._maybe_enable_fp8_lm_head(_model(head))
    assert head.calls == 0
    assert not m3._maybe_enable_fp8_lm_head(SimpleNamespace(lm_head=None))
