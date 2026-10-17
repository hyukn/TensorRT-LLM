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
"""NVFP4 fake quantization for accuracy studies (no NVFP4 GEMM involved).

``nvfp4_fake_quant(x)`` returns ``x`` rounded to what NVFP4 can represent: 16-element blocks along the last
dimension, one E4M3 block scale each under a per-tensor fp32 global scale (``448 * 6 / amax``), elements on
the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6} (round to nearest, ties to the even grid point, saturating at 6),
i.e. the format the NVFP4 MoE weights use. Computed in fp32 and returned in ``x``'s dtype.

``TRTLLM_FAKEQUANT_DENSE_NVFP4`` (MXFP8 dense linears, see ``MXFP8LinearMethod``):
  1: weights NVFP4 (fake-quantized once from the dequantized MXFP8 weight), bf16 activations and math (W4A16);
  2: also the activations fake-quantized to NVFP4 per call (dynamic per-tensor global scale) (W4A4);
  3: the operands of the W4A8 NVFP4 x FP8 GEMM (``fp4_fp8_gemm_trtllmgen``, ``W4A8NVFP4FP8LinearMethod``): E2M1
     weights with 32-element E4M3 block scales under a ``448 / amax`` global scale, activations FP8 E4M3 with a
     dynamic per-tensor scale (``amax / 448``);
  4: the operands of the MXFP4 x MXFP8 GEMMs (``w4a8_mxfp4_fp8_gemm``, the trtllm-gen MxE2m1 x MxE4m3 cubins): E2M1
     weights with 32-element power-of-two (UE8M0) block scales, activations MXFP8 as today;
  5: reference perturbation, no 4-bit anywhere: the MXFP8 weight as stored (exact in bf16) with bf16 activations
     (W8A16), i.e. today's GEMM minus its MXFP8 activation rounding.

``TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES`` (optional, comma list of ``ROLES``) restricts the fake quant to those
projections; the rest run today's MXFP8 GEMM. ``qkv`` is the Q / K / V rows of the attention input projection and
``index`` its fused index-Q / index-K rows, ``o`` the attention output projection, ``shared`` the MoE layers' shared
expert, ``dense`` the dense layers' MLP. Roles come from the module names (``tag_linear_roles``, MiniMax-M3 layout).
"""

import contextlib
import contextvars
import os

import torch

NVFP4_BLOCK = 16
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# E2M1 magnitudes; at a tie the grid point with the even mantissa wins (0, 1, 2, 4 even; 0.5, 1.5, 3, 6 odd).
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


ROLES = ("qkv", "index", "o", "shared", "dense")
_SUSPENDED = contextvars.ContextVar("trtllm_fakequant_dense_suspended", default=False)


def fake_quant_mode() -> int:
    if _SUSPENDED.get():
        return 0
    return int(os.environ.get("TRTLLM_FAKEQUANT_DENSE_NVFP4", "0") or 0)


@contextlib.contextmanager
def fake_quant_suspended():
    """Run today's path inside (``fake_quant_mode()`` is 0), e.g. for the projections the role filter leaves out."""
    token = _SUSPENDED.set(True)
    try:
        yield
    finally:
        _SUSPENDED.reset(token)


def selected_roles() -> frozenset:
    value = os.environ.get("TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES", "")
    roles = frozenset(r.strip() for r in value.split(",") if r.strip()) or frozenset(ROLES)
    unknown = roles - frozenset(ROLES)
    if unknown:
        raise ValueError(
            f"TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES: unknown roles {sorted(unknown)}, expected {ROLES}"
        )
    return roles


def tag_linear_roles(model: torch.nn.Module) -> int:
    """Tag the projections of a MiniMax-M3 model with their fake-quant role (``module._fq_role``) by name."""
    tagged = 0
    for name, module in model.named_modules():
        if name.endswith("self_attn.qkv_proj"):
            role = "qkv"
        elif name.endswith("self_attn.o_proj"):
            role = "o"
        elif ".shared_experts." in name:
            role = "shared"
        elif ".mlp." in name:
            role = "dense"
        else:
            continue
        module._fq_role = role
        tagged += 1
    return tagged


def fake_quant_rows(module: torch.nn.Module):
    """Output rows of ``module`` to fake-quantize under the role filter: None = all of them (no filter, or the
    module's roles are all selected), else a list of ``(start, end)`` row ranges (empty = run today's GEMM)."""
    roles = selected_roles()
    if roles == frozenset(ROLES):
        return None
    role = getattr(module, "_fq_role", None)
    if role is None:
        raise RuntimeError(
            "TRTLLM_FAKEQUANT_DENSE_NVFP4_ROLES needs projections tagged with tag_linear_roles (untagged linear "
            f"{tuple(module.weight.shape)})"
        )
    if role != "qkv":
        return None if role in roles else []
    total = module.weight.shape[0]
    sizes = getattr(module, "local_output_sizes", None)
    split = (
        sum(sizes[:3]) if sizes is not None and len(sizes) == 5 else total
    )  # [Q | K | V | index-Q | index-K]
    rows = []
    if "qkv" in roles:
        rows.append((0, split))
    if "index" in roles and split < total:
        rows.append((split, total))
    if rows and rows[0][0] == 0 and rows[-1][1] == total and sum(e - b for b, e in rows) == total:
        return None
    return rows


def _round_e2m1(a: torch.Tensor) -> torch.Tensor:
    """Round non-negative fp32 magnitudes to the E2M1 grid (nearest, ties to even, saturating).

    The grid spacing is 0.5 below 2, 1 in [2, 4) and 2 in [4, 6]; within each range the grid points are the
    multiples of the spacing, and an even multiple is a grid point with an even mantissa, so round-half-to-even of
    ``a / spacing`` is round-to-nearest-even on the grid. No tensor is created, so this is CUDA-graph capturable.
    """
    a = a.clamp(max=_E2M1_MAX)
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.round(a / step) * step


def nvfp4_fake_quant(
    x: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    block: int = NVFP4_BLOCK,
    global_target: float = _E4M3_MAX * _E2M1_MAX,
) -> torch.Tensor:
    """Round ``x`` to NVFP4 (``block`` elements along the last dim share an E4M3 scale, fp32 global scale
    ``global_target / amax``; the defaults are NVFP4's block 16 and 448 * 6)."""
    shape, dtype = x.shape, x.dtype
    k = shape[-1]
    assert k % block == 0, f"last dim {k} must be a multiple of {block}"
    xf = x.float().reshape(-1, k // block, block)
    amax = xf.abs().amax() if global_amax is None else global_amax.float()
    # Global scale maps the tensor amax to the largest block scale x the largest element.
    gs = global_target / amax.clamp(min=torch.finfo(torch.float32).tiny)
    block_amax = xf.abs().amax(dim=-1, keepdim=True)
    sf = (block_amax / _E2M1_MAX * gs).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn).float()
    # Effective per-block scale; an all-zero block (or one whose scale underflows) quantizes to zero.
    scale = sf / gs
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = _round_e2m1((xf / safe).abs()) * torch.sign(xf)
    out = torch.where(scale > 0, q * scale, torch.zeros_like(xf))
    return out.reshape(shape).to(dtype)


def mx_fake_quant(x: torch.Tensor, elem: str, block: int = 32) -> torch.Tensor:
    """Round ``x`` to an OCP MX format: ``block`` elements along the last dim share a power-of-two (UE8M0) scale, the
    smallest one that fits the block amax (2^ceil(log2(amax / elem_max)), as mxfp8_quantize / fp4_quantize with
    UE8M0 scales round it); elements E2M1 (``elem="e2m1"``, MXFP4) or E4M3 (``"e4m3"``, MXFP8)."""
    shape, dtype = x.shape, x.dtype
    k = shape[-1]
    assert k % block == 0, f"last dim {k} must be a multiple of {block}"
    elem_max = {"e2m1": _E2M1_MAX, "e4m3": _E4M3_MAX}[elem]
    xf = x.float().reshape(-1, k // block, block)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / elem_max)))
    if elem == "e2m1":
        q = _round_e2m1((xf / scale).abs()) * torch.sign(xf)
    else:
        q = (xf / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shape).to(dtype)


def fp8_per_tensor_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to FP8 E4M3 under a dynamic per-tensor scale ``amax / 448`` held in ``x``'s dtype, as
    ``quantize_e4m3_per_tensor`` returns it (saturating: a scale rounded down may map the amax just above 448)."""
    xf = x.float()
    scale = (
        (xf.abs().amax() / _E4M3_MAX).to(x.dtype).float().clamp(min=torch.finfo(torch.float32).tiny)
    )
    q = (xf / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).to(x.dtype)


def weight_fake_quant(w: torch.Tensor, mode: int) -> torch.Tensor:
    """The weight operand of fake-quant ``mode`` (see the module docstring)."""
    if mode == 3:
        return nvfp4_fake_quant(w, block=32, global_target=_E4M3_MAX)
    if mode == 4:
        return mx_fake_quant(w, "e2m1")
    if mode == 5:
        return w
    return nvfp4_fake_quant(w)


def activation_fake_quant(x: torch.Tensor, mode: int) -> torch.Tensor:
    """The activation operand of fake-quant ``mode`` (see the module docstring)."""
    if mode == 2:
        return nvfp4_fake_quant(x)
    if mode == 3:
        return fp8_per_tensor_fake_quant(x)
    if mode == 4:
        return mx_fake_quant(x, "e4m3")
    return x
