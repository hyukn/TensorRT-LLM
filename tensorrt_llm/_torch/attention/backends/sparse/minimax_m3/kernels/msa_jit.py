# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build settings for the kernels fmha_sm100 compiles on first use.

fmha_sm100 compiles a kernel variant with nvcc the first time a call needs it,
on the calling thread. Its default build emits an image for every SM100-family
target and compiles the variant's two translation units one after the other,
although a process only ever loads the image of its own device. Narrowing the
build to that image and compiling the translation units concurrently shortens
the stall a first-use compile puts on the step that triggers it. The kernels
themselves are unchanged.
"""

from __future__ import annotations

import functools
import subprocess
import threading
import types
from pathlib import Path

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger

# A variant build compiles two independent translation units, the kernel
# instantiation and its FFI binding, then links them.
_VARIANT_BUILD_JOBS = 2

_configure_lock = threading.Lock()
_configured = False


def _run_variant_ninja(cache_dir: Path) -> None:
    """Run a variant's ninja build, compiling its translation units concurrently."""
    result = subprocess.run(
        ["ninja", f"-j{_VARIANT_BUILD_JOBS}"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FMHA variant compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def build_for_device(jit: types.ModuleType, sm_version: int) -> None:
    """Make fmha_sm100 build its kernels for one device's architecture only.

    Each build keeps just the arch-specific image of that device, the one the
    driver loads there, and variant builds compile their translation units
    concurrently. A device without an arch-specific image in fmha_sm100's
    default build keeps the default targets.

    Args:
        jit: The fmha_sm100.jit module.
        sm_version: SM version of the device, e.g. 103 for SM103.
    """
    target = f"sm_{sm_version}a"
    gencode = f"-gencode=arch=compute_{sm_version}a,code={target}"
    default_nvcc_flags = jit._get_nvcc_flags

    @functools.wraps(default_nvcc_flags)
    def nvcc_flags(cache_dir: Path, fmha: bool = True) -> str:
        flags = default_nvcc_flags(cache_dir, fmha).split(" ")
        if gencode not in flags:
            return " ".join(flags)
        return " ".join(f for f in flags if f == gencode or not f.startswith("-gencode="))

    jit._get_nvcc_flags = nvcc_flags
    # The cache key covers the kernel sources but not the build flags, so these
    # builds get a cache of their own: a default build in the shared cache may
    # be loaded by any SM100-family device. fmha_sm100 resolved its prebuilt
    # all-variants library at import, so that one still loads from the default
    # cache.
    jit.CACHE_BASE = jit.CACHE_BASE.with_name(f"{jit.CACHE_BASE.name}_{target}")
    jit._variant_manager._run_ninja = _run_variant_ninja


def configure_fmha_sm100_jit(fmha_sm100: types.ModuleType) -> None:
    """Build fmha_sm100's just-in-time kernels for the current device only.

    Takes effect once per process and must run before fmha_sm100 builds its
    first kernel; later calls return immediately.

    Args:
        fmha_sm100: The imported fmha_sm100 package.
    """
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        sm_version = get_sm_version()
        if is_sm_100f(sm_version):
            jit = fmha_sm100.jit
            build_for_device(jit, sm_version)
            logger.info(
                f"fmha_sm100 builds kernels for sm_{sm_version}a only, "
                f"{_VARIANT_BUILD_JOBS} jobs per variant, cached under {jit.CACHE_BASE}"
            )
        _configured = True


__all__ = [
    "build_for_device",
    "configure_fmha_sm100_jit",
]
