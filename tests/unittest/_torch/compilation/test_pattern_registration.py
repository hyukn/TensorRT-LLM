# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm.mapping import Mapping


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Undo the environment build_custom_passes writes to.

    The multi-GPU path sets DISABLE_LAMPORT_REDUCE_NORM_FUSION in os.environ, and a
    leaked value makes later tests in the same process compile through a different
    path. patch.dict snapshots os.environ and drops whatever the block added.
    """
    with patch.dict(os.environ, {}, clear=False):
        yield


def _build_multi_gpu_passes(enable_userbuffers: bool, mxfp8_out_fusion: str = "0"):
    mapping = Mapping(world_size=4, tp_size=4, rank=0)

    with (
        patch.dict(os.environ, {"TRTLLM_AR_MXFP8_OUT_FUSION": mxfp8_out_fusion}),
        patch("tensorrt_llm.mpi_world_size", return_value=4),
        patch(
            "tensorrt_llm.bindings.internal.userbuffers.ub_supported",
            return_value=enable_userbuffers,
        ),
    ):
        return Backend.build_custom_passes(enable_userbuffers, mapping)


def _num_patterns(passes, pass_name: str) -> int:
    (custom_pass,) = [p for p in passes if p.pass_name == pass_name]
    return sum(len(entries) for entries in custom_pass.patterns.values())


@pytest.mark.parametrize("enable_userbuffers", [False, True])
def test_multi_gpu_pattern_registration(enable_userbuffers: bool) -> None:
    _build_multi_gpu_passes(enable_userbuffers)


@pytest.mark.parametrize("enable_userbuffers", [False, True])
def test_ar_mxfp8_out_pattern_registration(enable_userbuffers: bool) -> None:
    """TRTLLM_AR_MXFP8_OUT_FUSION=1 adds one AR + MXFP8-quantize pattern per allreduce op, with or without UB."""
    default = _num_patterns(_build_multi_gpu_passes(enable_userbuffers), "ar_residual_norm_quant")
    fused = _num_patterns(
        _build_multi_gpu_passes(enable_userbuffers, mxfp8_out_fusion="1"), "ar_residual_norm_quant"
    )
    assert fused == default + 3


def test_single_gpu_pattern_registration() -> None:
    """Cover the world_size == 1 branch.

    It is the only caller of register_add_norm_quant, so without this the quantized
    add-norm registration is never exercised.
    """
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    with patch("tensorrt_llm.mpi_world_size", return_value=1):
        Backend.build_custom_passes(False, mapping)
