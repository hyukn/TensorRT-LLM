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
"""RESIDUAL_RMS_NORM_OUT_QUANT_MXFP8 on the non-fused (NCCL) AllReduce path.

The op must equal RESIDUAL_RMS_NORM followed by ``torch.ops.trtllm.mxfp8_quantize(norm, True)`` bit for bit
(norm, residual, E4M3 values and the whole swizzled scale buffer, padding included), and under torch.compile
the ``ar_residual_norm_quant`` pass must fold that quantize into the AllReduce when
TRTLLM_AR_MXFP8_OUT_FUSION=1 (and leave the graph alone otherwise).
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.nn as nn
from mpi4py import MPI
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

_FUSION_ENV = "TRTLLM_AR_MXFP8_OUT_FUSION"


def _inputs(num_tokens: int, hidden: int, dtype: torch.dtype, rank: int):
    torch.manual_seed(100 + rank)
    x = (torch.randn(num_tokens, hidden, device="cuda") * 0.5).to(dtype)
    torch.manual_seed(7)
    residual = torch.randn(num_tokens, hidden, device="cuda").to(dtype)
    gamma = (torch.rand(hidden, device="cuda") + 0.5).to(dtype)
    return x, residual, gamma


def _assert_matches_quantize(norm, quant, scale, ref_norm):
    ref_quant, ref_scale = torch.ops.trtllm.mxfp8_quantize(ref_norm, True)
    assert quant.dtype == torch.float8_e4m3fn and quant.shape == ref_norm.shape
    assert torch.equal(quant.view(torch.uint8), ref_quant.reshape(quant.shape).view(torch.uint8))
    # The whole swizzled buffer: valid rows and the rows / columns padded to 128 x 4 (zero in both).
    assert scale.numel() == ref_scale.numel()
    assert torch.equal(scale.view(torch.uint8).reshape(-1), ref_scale.view(torch.uint8).reshape(-1))


def nccl_mxfp8_out_forward(
    tensor_parallel_size: int,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    strategy: AllReduceStrategy = AllReduceStrategy.NCCL,
) -> bool:
    """RESIDUAL_RMS_NORM_OUT_QUANT_MXFP8 (strategy) == NCCL RESIDUAL_RMS_NORM + mxfp8_quantize(norm, True), bitwise.

    With two ranks every strategy rounds the same two-operand sum, so NCCL_SYMMETRIC must match NCCL bit for bit.
    """
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        x, residual, gamma = _inputs(num_tokens, hidden, dtype, rank)
        MPI.COMM_WORLD.barrier()
        mapping = Mapping(world_size=tensor_parallel_size, tp_size=tensor_parallel_size, rank=rank)
        ref_allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=dtype)
        allreduce = AllReduce(mapping=mapping, strategy=strategy, dtype=dtype)

        def run(module, op):
            return module(
                x.clone(),
                all_reduce_params=AllReduceParams(
                    fusion_op=op, residual=residual.clone(), norm_weight=gamma, eps=1e-5
                ),
            )

        ref_norm, ref_residual = run(ref_allreduce, AllReduceFusionOp.RESIDUAL_RMS_NORM)
        norm, quant, scale, res = run(
            allreduce, AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_MXFP8
        )
        assert torch.equal(norm, ref_norm) and torch.equal(res, ref_residual)
        _assert_matches_quantize(norm, quant, scale, ref_norm)
    except Exception:
        traceback.print_exc()
        raise
    return True


_BITWISE_SHAPES = [
    (6144, torch.bfloat16),  # MiniMax-M3: the fused kernel
    (6144, torch.float16),
    (8192, torch.bfloat16),  # 1024 threads per row: the largest fused shape
    (2880, torch.bfloat16),  # 360 threads, not whole warps: norm + quantize kernels
    # More than one CTA's worth of 16-byte accesses: norm + quantize kernels.
    (16384, torch.bfloat16),
]
_BITWISE_TOKENS = [1, 20, 128, 130, 1000]
_SYMMETRIC_TOKENS = [1, 130, 1000, 16384]


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize(
    "hidden,dtype", _BITWISE_SHAPES, ids=lambda v: str(v).replace("torch.", "")
)
@pytest.mark.parametrize("num_tokens", _BITWISE_TOKENS, ids=lambda n: f"tokens{n}")
def test_nccl_mxfp8_out_fusion_is_bitwise(mpi_pool_executor, num_tokens, hidden, dtype) -> None:
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        nccl_mxfp8_out_forward,
        *zip(*[(tensor_parallel_size, num_tokens, hidden, dtype)] * tensor_parallel_size),
    )
    assert all(results)


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("num_tokens", _SYMMETRIC_TOKENS, ids=lambda n: f"tokens{n}")
def test_nccl_symmetric_mxfp8_out_matches_nccl(mpi_pool_executor, num_tokens) -> None:
    """The fused op on the NCCL_SYMMETRIC path (window buffers, symmetric kernels) at the M3 hidden size."""
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        nccl_mxfp8_out_forward,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    num_tokens,
                    6144,
                    torch.bfloat16,
                    AllReduceStrategy.NCCL_SYMMETRIC,
                )
            ]
            * tensor_parallel_size
        ),
    )
    assert all(results)


def compiled_mxfp8_out_forward(tensor_parallel_size: int, fusion_enabled: bool) -> bool:
    """torch.compile folds mxfp8_quantize(norm, True) after an NCCL RESIDUAL_RMS_NORM AllReduce (opt-in)."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    saved = os.environ.get(_FUSION_ENV)
    try:
        # A fresh Dynamo cache, so this Backend (built with the env below) compiles the graph.
        torch._dynamo.reset()
        os.environ[_FUSION_ENV] = "1" if fusion_enabled else "0"
        hidden, dtype = 6144, torch.bfloat16
        mapping = Mapping(world_size=tensor_parallel_size, tp_size=tensor_parallel_size, rank=rank)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.allreduce = AllReduce(
                    mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=dtype
                )

            def forward(self, x, residual, gamma):
                norm, residual = self.allreduce(
                    x,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=gamma,
                        eps=1e-5,
                    ),
                )
                # The MXFP8 linear's activation quantize, plus another reader of the norm (as the MoE router).
                quant, scale = torch.ops.trtllm.mxfp8_quantize(norm, True)
                router_in = norm.float().sum(-1)
                return quant, scale, router_in, residual

        model = Model()
        backend = Backend(enable_inductor=False, mapping=mapping)
        model_opt = torch.compile(model, backend=backend, fullgraph=True)
        with torch.inference_mode():
            for num_tokens in (130, 1000):
                x, residual, gamma = _inputs(num_tokens, hidden, dtype, rank)
                MPI.COMM_WORLD.barrier()
                ref = model(x.clone(), residual.clone(), gamma)
                out = model_opt(x.clone(), residual.clone(), gamma)
                assert torch.equal(out[0].view(torch.uint8), ref[0].view(torch.uint8))
                assert torch.equal(out[1].view(torch.uint8), ref[1].view(torch.uint8))
                assert torch.equal(out[2], ref[2])
                assert torch.equal(out[3], ref[3])
        assert backend.match_count_by_pass["ar_residual_norm_quant"] == (
            1 if fusion_enabled else 0
        ), dict(backend.match_count_by_pass)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if saved is None:
            os.environ.pop(_FUSION_ENV, None)
        else:
            os.environ[_FUSION_ENV] = saved
    return True


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("fusion_enabled", [True, False], ids=["fusion_on", "fusion_off"])
def test_compile_folds_mxfp8_quantize_into_nccl_allreduce(
    mpi_pool_executor, fusion_enabled
) -> None:
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        compiled_mxfp8_out_forward,
        *zip(*[(tensor_parallel_size, fusion_enabled)] * tensor_parallel_size),
    )
    assert all(results)


if __name__ == "__main__":
    # Direct launch, one process per rank under an MPI launcher (srun --mpi=pmix / mpirun), where spawning an MPI
    # pool is not available: python test_allreduce_mxfp8_out.py <nranks>
    n = int(sys.argv[1])
    for hidden, dtype in _BITWISE_SHAPES:
        for num_tokens in _BITWISE_TOKENS:
            assert nccl_mxfp8_out_forward(n, num_tokens, hidden, dtype)
    for num_tokens in _SYMMETRIC_TOKENS:
        assert nccl_mxfp8_out_forward(
            n, num_tokens, 6144, torch.bfloat16, AllReduceStrategy.NCCL_SYMMETRIC
        )
    for fusion_enabled in (True, False):
        assert compiled_mxfp8_out_forward(n, fusion_enabled)
    print(f"rank {tensorrt_llm.mpi_rank()} OK", flush=True)
