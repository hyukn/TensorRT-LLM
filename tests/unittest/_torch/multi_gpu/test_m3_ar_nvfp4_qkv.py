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
"""MiniMax-M3 TRTLLM_M3_AR_NVFP4_QKV: the layer-boundary MNNVL AllReduce + residual + RMSNorm emits the NVFP4
activation of an NVFP4-repacked QKV projection (RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 with the projection's static scale),
and the projection's GEMM on it matches the projection quantizing its bf16 input itself."""

import os
import pickle
import platform
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.modules.linear import Linear, MXFP8LinearMethod
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 6144
QKV_OUT = (
    1024  # a slice of the TP4 fused QKV+index projection (2944 columns) is enough for the handoff
)


def _repacked_qkv(input_amax: float) -> Linear:
    torch.manual_seed(2)
    weight = torch.randn(QKV_OUT, HIDDEN, dtype=torch.bfloat16) * 0.05
    w_e4m3, w_scale = quant_bf16_to_mxfp8(weight, 32)
    lin = Linear(
        in_features=HIDDEN,
        out_features=QKV_OUT,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
    ).cuda()
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": w_scale}])
    assert isinstance(lin.quant_method, MXFP8LinearMethod)
    lin.quant_method.repack_to_nvfp4(lin, input_amax)
    return lin


def _dequant(quant: torch.Tensor, sf: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        quant.cpu(), sf.cpu(), 1 / global_scale.cpu(), 16, 1, True
    )


@torch.inference_mode()
def ar_nvfp4_qkv_forward(tensor_parallel_size: int, num_tokens: int, use_input_add: bool) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        os.environ["TLLM_TEST_MNNVL"] = "1"
        torch.manual_seed(300 + rank)
        x = (torch.randn(num_tokens, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
        input_add = (
            (torch.randn(num_tokens, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
            if use_input_add
            else None
        )
        torch.manual_seed(11)
        residual = torch.randn(num_tokens, HIDDEN, device="cuda").to(torch.bfloat16)
        gamma = (torch.rand(HIDDEN, device="cuda") + 0.5).to(torch.bfloat16)
        MPI.COMM_WORLD.barrier()
        allreduce = AllReduce(
            mapping=Mapping(
                world_size=tensor_parallel_size, tp_size=tensor_parallel_size, rank=rank
            ),
            strategy=AllReduceStrategy.MNNVL,
            dtype=torch.bfloat16,
        )

        def run(op, scale=None):
            return allreduce(
                x.clone(),
                all_reduce_params=AllReduceParams(
                    fusion_op=op,
                    residual=residual.clone(),
                    norm_weight=gamma,
                    eps=1e-5,
                    scale=scale,
                    input_add=None if input_add is None else input_add.clone(),
                ),
            )

        ref_norm, ref_residual = run(AllReduceFusionOp.RESIDUAL_RMS_NORM)
        # A calibrated-style static bound: the observed amax with the default margin of 2.
        input_amax = 2.0 * float(ref_norm.abs().max())
        qkv = _repacked_qkv(input_amax)
        scale = qkv._nvfp4_input_scale
        norm, quant, sf, res = run(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4, scale)
        assert torch.equal(norm, ref_norm) and torch.equal(res, ref_residual)
        assert quant.shape == (num_tokens, HIDDEN // 2) and sf.dim() == 1

        # The fused quantization against the projection's own quantize of the same bf16 norm.
        sep_quant, sep_sf = torch.ops.trtllm.fp4_quantize(norm, scale, 16, False)
        fused_val, sep_val = _dequant(quant, sf, scale), _dequant(sep_quant, sep_sf, scale)
        mismatch = (fused_val != sep_val).float().mean().item()
        assert mismatch < 2e-3, f"{mismatch:.2e} of the NVFP4 values differ"

        # The repacked projection's GEMM on the handed-over activation vs on the bf16 norm.
        out_fused = qkv(
            Fp4QuantizedTensor(fp4_tensor=quant, scaling_factor=sf, is_sf_swizzled=True)
        )
        out_sep = qkv(norm)
        rel = ((out_fused.float() - out_sep.float()).norm() / out_sep.float().norm()).item()
        assert rel < 5e-3, f"relative difference {rel:.2e}"
        print(
            f"[ar-nvfp4-qkv] rank {rank} tokens {num_tokens} input_add {use_input_add}: "
            f"value mismatch {mismatch:.1e}, GEMM rel diff {rel:.1e}",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(
    platform.machine().lower() != "aarch64"
    or torch.cuda.device_count() < 4
    or not MnnvlMemory.supports_mnnvl(),
    reason="requires four GB200-class GPUs with fabric-backed MNNVL",
)
@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
@pytest.mark.parametrize("use_input_add", [False, True], ids=["plain", "input_add"])
@pytest.mark.parametrize("num_tokens", [8, 32, 124], ids=lambda n: f"tokens{n}")
def test_ar_nvfp4_qkv_handoff(mpi_pool_executor, num_tokens, use_input_add) -> None:
    """One-shot (8 tokens at TP4) and two-shot (32, 124) MNNVL kernels, with and without the MoE add fold."""
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        ar_nvfp4_qkv_forward,
        [tensor_parallel_size] * tensor_parallel_size,
        [num_tokens] * tensor_parallel_size,
        [use_input_add] * tensor_parallel_size,
    )
    assert all(results)
