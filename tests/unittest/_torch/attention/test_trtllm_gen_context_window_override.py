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
"""TRTLLM context attention with ``context_window_left_override`` (Eagle3 context draft step, KV-write pass).

The override narrows only the trtllm-gen context kernel to a causal local window: the KV the call writes must be
exactly what the full-attention call writes (run_backend checks the cache contents), and the output must be the
sliding-window attention of the same inputs (VANILLA golden with ``sliding_window``), not the full attention.
"""

import dataclasses

import backend_case
import pytest
import torch
from backend_case import BackendCase, _tolerances, generate_inputs, run_backend
from utils.util import skip_pre_blackwell


def _case(window=None) -> BackendCase:
    return BackendCase(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        # The harness's context phase (no cached prefix); the first request is longer than the window.
        seq_lens=[1041, 73, 300],
        num_cached_tokens=[0, 0, 0],
        num_contexts=3,
        dtype="bfloat16",
        kv_dtype="fp8",
        page_size=32,
        sliding_window=window,
    )


@skip_pre_blackwell
@pytest.mark.parametrize("window", [256])
def test_context_window_override_keeps_kv_writes(
    monkeypatch: pytest.MonkeyPatch, window: int
) -> None:
    case = _case()
    inputs = generate_inputs(case, seed=3)
    kv_dtype = torch.float8_e4m3fn

    # Full attention; run_backend asserts the cache holds exactly the new K/V.
    full = run_backend(case, "TRTLLM", inputs, kv_dtype=kv_dtype, kv_layout="HND")

    created = []
    real_create = backend_case.create_attention

    def create_with_override(*args, **kwargs):
        attn = real_create(*args, **kwargs)
        attn.context_window_left_override = window - 1
        created.append(attn)
        return attn

    monkeypatch.setattr(backend_case, "create_attention", create_with_override)
    # Same inputs with the kernel window narrowed; the same cache-content assertion runs inside.
    local = run_backend(case, "TRTLLM", inputs, kv_dtype=kv_dtype, kv_layout="HND")
    monkeypatch.setattr(backend_case, "create_attention", real_create)
    assert created, "the TRTLLM backend was not built through create_attention"

    # The window took effect: queries past the first 256 positions lose their early keys.
    assert not torch.allclose(local.float(), full.float(), atol=1e-2)

    # And it is the causal sliding-window attention of the same inputs.
    golden = run_backend(
        dataclasses.replace(case, sliding_window=window),
        "VANILLA",
        inputs,
        kv_dtype=case.compute_dtype,
        kv_layout="NHD",
    )
    atol, rtol = _tolerances(case, kv_dtype)
    torch.testing.assert_close(local.float(), golden.float(), atol=atol, rtol=rtol)
