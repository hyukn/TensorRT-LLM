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
"""MiniMax-M3 attention QKV(+index) projection as an NVFP4 W4A4 GEMM (opt-in).

The checkpoint ships the projection in MXFP8. With ``TRTLLM_M3_QKV_NVFP4=1`` every decoder layer's
``self_attn.qkv_proj`` is re-quantized to NVFP4 after loading (``MXFP8LinearMethod.repack_to_nvfp4``)
and quantizes its activation per call with a static global scale from a calibration file:

  TRTLLM_M3_QKV_NVFP4_SCALES   JSON ``{"amax": {"<layer>": <max |input|>, ...}}`` (required)
  TRTLLM_M3_QKV_NVFP4_MARGIN   multiplier on the calibrated amax (default 2.0; larger inputs saturate)
  TRTLLM_M3_QKV_NVFP4_BACKENDS NVFP4 GEMM backends (default "cublaslt,cutedsl")
  TRTLLM_M3_QKV_NVFP4_KEEP_MXFP8=1 keeps the MXFP8 weight in memory (default: freed)

``TRTLLM_M3_QKV_NVFP4_CALIBRATE=<path>`` instead records each layer's running max |input| (MXFP8 GEMM
unchanged) and writes ``<path>.rank<r>`` (every 30 s from eager steps and at exit); merge the ranks
with ``merge_calibration``. Accuracy study: genperf round 12 (the QKV-only W4A4 subset is
quality-neutral, the other projections are not).
"""

import atexit
import glob
import json
import os
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ..modules.linear import MXFP8LinearMethod

ENABLE_ENV = "TRTLLM_M3_QKV_NVFP4"
SCALES_ENV = "TRTLLM_M3_QKV_NVFP4_SCALES"
MARGIN_ENV = "TRTLLM_M3_QKV_NVFP4_MARGIN"
BACKENDS_ENV = "TRTLLM_M3_QKV_NVFP4_BACKENDS"
KEEP_MXFP8_ENV = "TRTLLM_M3_QKV_NVFP4_KEEP_MXFP8"
CALIBRATE_ENV = "TRTLLM_M3_QKV_NVFP4_CALIBRATE"
DEFAULT_MARGIN = 2.0
DEFAULT_BACKENDS = ("cublaslt", "cutedsl")


def qkv_projections(layers: Iterable[nn.Module]) -> Dict[int, nn.Module]:
    """``{layer index: self_attn.qkv_proj}`` for the layers whose projection runs MXFP8 (and is not
    repacked to NVFP4 already: weight reloads after the repack are not supported)."""
    out = {}
    for idx, layer in enumerate(layers):
        proj = getattr(getattr(layer, "self_attn", None), "qkv_proj", None)
        if isinstance(getattr(proj, "quant_method", None), MXFP8LinearMethod) and not getattr(
            proj, "_nvfp4_repacked", False
        ):
            out[idx] = proj
    return out


class QkvAmaxRecorder:
    """Writes the calibration amax of every recorded projection to ``<path>.rank<rank>``."""

    def __init__(
        self, path: str, rank: int, projections: Dict[int, nn.Module], period_s: float = 30.0
    ):
        self.path = f"{path}.rank{rank}"
        self.projections = projections
        self.period_s = period_s
        self.last = time.monotonic()

    def flush(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last < self.period_s:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        amax = {str(i): float(p._nvfp4_calib_amax.item()) for i, p in self.projections.items()}
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"amax": amax}, f, indent=1)
        os.replace(tmp, self.path)
        self.last = time.monotonic()


def merge_calibration(paths: Sequence[str], out: Optional[str] = None) -> Dict[str, float]:
    """Per-layer max over calibration files (e.g. one per rank); written to ``out`` when given."""
    merged: Dict[str, float] = {}
    for path in paths:
        with open(path) as f:
            for layer, value in json.load(f)["amax"].items():
                merged[layer] = max(merged.get(layer, 0.0), float(value))
    if out is not None:
        with open(out, "w") as f:
            json.dump({"amax": merged, "sources": list(paths)}, f, indent=1)
    return merged


def load_scales(path: str) -> Dict[int, float]:
    with open(path) as f:
        return {int(layer): float(value) for layer, value in json.load(f)["amax"].items()}


def setup(layers: Iterable[nn.Module], rank: int) -> Tuple[str, int]:
    """Apply the env-selected QKV NVFP4 mode to ``layers``; returns (mode, number of projections)."""
    calibrate = os.environ.get(CALIBRATE_ENV)
    enabled = os.environ.get(ENABLE_ENV, "0") == "1"
    if not calibrate and not enabled:
        return "off", 0
    if calibrate and enabled:
        raise ValueError(f"{CALIBRATE_ENV} and {ENABLE_ENV}=1 are exclusive")
    projections = qkv_projections(layers)
    if calibrate:
        for proj in projections.values():
            proj.quant_method.enable_input_amax_calibration(proj)
        recorder = QkvAmaxRecorder(calibrate, rank, projections)
        if projections:
            first = projections[min(projections)]
            first._nvfp4_calib_flush = recorder.flush
            atexit.register(recorder.flush, True)
        logger.info(
            f"[M3 QKV NVFP4] calibrating {len(projections)} QKV projections -> {recorder.path}"
        )
        return "calibrate", len(projections)
    scales_path = os.environ.get(SCALES_ENV)
    if not scales_path:
        raise ValueError(f"{ENABLE_ENV}=1 needs the calibrated activation amax file {SCALES_ENV}")
    amax = load_scales(scales_path)
    missing = sorted(set(projections) - set(amax))
    if missing:
        raise ValueError(f"{scales_path} has no activation amax for layers {missing}")
    margin = float(os.environ.get(MARGIN_ENV, DEFAULT_MARGIN))
    backends = tuple(
        b.strip()
        for b in os.environ.get(BACKENDS_ENV, ",".join(DEFAULT_BACKENDS)).split(",")
        if b.strip()
    )
    free_mxfp8 = os.environ.get(KEEP_MXFP8_ENV, "0") != "1"
    for idx, proj in projections.items():
        proj.quant_method.repack_to_nvfp4(
            proj, amax[idx] * margin, allowed_backends=backends, free_mxfp8=free_mxfp8
        )
    if free_mxfp8 and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        f"[M3 QKV NVFP4] {len(projections)} QKV projections -> NVFP4 W4A4 (static activation scales from "
        f"{scales_path} x {margin}, backends {','.join(backends)}, MXFP8 weights "
        f"{'freed' if free_mxfp8 else 'kept'})"
    )
    return "nvfp4", len(projections)


def main() -> None:
    """``python -m tensorrt_llm._torch.models.minimax_m3_qkv_nvfp4 <calibration prefix> <out json>``."""
    import sys

    paths = sorted(glob.glob(f"{sys.argv[1]}.rank*"))
    merged = merge_calibration(paths, sys.argv[2])
    print(
        f"merged {len(paths)} files, {len(merged)} layers, amax {min(merged.values()):.3g}..{max(merged.values()):.3g}"
    )


if __name__ == "__main__":
    main()
