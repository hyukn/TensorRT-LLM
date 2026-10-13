# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the device-only build settings of fmha_sm100's just-in-time kernels.

They stand in for fmha_sm100.jit with a namespace that has the same hooks, so
no GPU or compiler is needed. One test checks the packaged module's own flag
layout when the package is installed.
"""

import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_jit
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
)

_DEFAULT_GENCODES = (
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
    "-gencode=arch=compute_100f,code=sm_100f",
)


def _default_nvcc_flags(cache_dir: Path, fmha: bool = True) -> str:
    """Mirror fmha_sm100.jit._get_nvcc_flags: the gencodes among other flags."""
    return " ".join(
        [
            "-O3",
            "-std=c++20",
            *_DEFAULT_GENCODES,
            "-Xcudafe --diag_suppress=2908",
            f"-I{cache_dir}",
            "-Xptxas",
            "-O1" if fmha else "-O3",
        ]
    )


def _fake_jit(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _get_nvcc_flags=_default_nvcc_flags,
        CACHE_BASE=tmp_path / "sources_sha256_0123",
        _variant_manager=SimpleNamespace(),
    )


def _gencode(sm_version: int) -> str:
    return f"-gencode=arch=compute_{sm_version}a,code=sm_{sm_version}a"


@pytest.mark.parametrize("fmha", [True, False])
@pytest.mark.parametrize("sm_version", [100, 103])
def test_build_keeps_only_the_device_image(tmp_path: Path, sm_version: int, fmha: bool) -> None:
    jit = _fake_jit(tmp_path)
    msa_jit.build_for_device(jit, sm_version)

    expected = [
        flag
        for flag in _default_nvcc_flags(tmp_path, fmha).split(" ")
        if flag == _gencode(sm_version) or flag not in _DEFAULT_GENCODES
    ]
    assert jit._get_nvcc_flags(tmp_path, fmha).split(" ") == expected
    assert jit.CACHE_BASE == tmp_path / f"sources_sha256_0123_sm_{sm_version}a"


def test_build_keeps_default_targets_without_device_image(tmp_path: Path) -> None:
    jit = _fake_jit(tmp_path)
    msa_jit.build_for_device(jit, 101)

    assert jit._get_nvcc_flags(tmp_path) == _default_nvcc_flags(tmp_path)


def test_variant_build_compiles_translation_units_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs["cwd"]))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(msa_jit.subprocess, "run", run)
    jit = _fake_jit(tmp_path)
    msa_jit.build_for_device(jit, 103)
    jit._variant_manager._run_ninja(tmp_path)

    assert calls == [(["ninja", "-j2"], str(tmp_path))]


def test_variant_build_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="nvcc error")

    monkeypatch.setattr(msa_jit.subprocess, "run", run)
    jit = _fake_jit(tmp_path)
    msa_jit.build_for_device(jit, 103)

    with pytest.raises(RuntimeError, match="nvcc error"):
        jit._variant_manager._run_ninja(tmp_path)


def test_configure_applies_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(msa_jit, "_configured", False)
    monkeypatch.setattr(msa_jit, "get_sm_version", lambda: 103)
    jit = _fake_jit(tmp_path)
    package = SimpleNamespace(jit=jit)

    msa_jit.configure_fmha_sm100_jit(package)
    msa_jit.configure_fmha_sm100_jit(package)

    assert jit.CACHE_BASE == tmp_path / "sources_sha256_0123_sm_103a"
    assert jit._get_nvcc_flags(tmp_path).count("-gencode=") == 1


def test_configure_leaves_other_architectures_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(msa_jit, "_configured", False)
    monkeypatch.setattr(msa_jit, "get_sm_version", lambda: 120)
    jit = _fake_jit(tmp_path)

    msa_jit.configure_fmha_sm100_jit(SimpleNamespace(jit=jit))

    assert jit.CACHE_BASE == tmp_path / "sources_sha256_0123"
    assert jit._get_nvcc_flags is _default_nvcc_flags
    assert not hasattr(jit._variant_manager, "_run_ninja")


@pytest.mark.skipif(not msa_package_available(), reason="fmha_sm100 is not installed")
@pytest.mark.parametrize("sm_version", [100, 103])
def test_packaged_build_narrows_to_device(tmp_path: Path, sm_version: int) -> None:
    from fmha_sm100 import jit as packaged_jit

    # Borrow the packaged flag builder without touching the packaged module.
    # An earlier require_msa_module() in this process may already have narrowed
    # it to the device the tests run on, so take the original underneath.
    jit = SimpleNamespace(
        _get_nvcc_flags=inspect.unwrap(packaged_jit._get_nvcc_flags),
        CACHE_BASE=tmp_path / "sources",
        _variant_manager=SimpleNamespace(),
    )
    msa_jit.build_for_device(jit, sm_version)

    for fmha in (True, False):
        flags = jit._get_nvcc_flags(tmp_path, fmha).split(" ")
        assert [flag for flag in flags if flag.startswith("-gencode=")] == [_gencode(sm_version)]
