# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from ..distributed import allgather
from ..utils import create_lm_head_tp_mapping
from .linear import Linear, TensorParallelMode


def fp8_rowwise_linear(input: torch.Tensor, weight_fp8: torch.Tensor,
                       weight_scale_t: torch.Tensor) -> torch.Tensor:
    """``input @ dequant(weight_fp8).T`` with per-token E4M3 activations.

    ``weight_fp8`` is ``[N, K]`` E4M3 and ``weight_scale_t`` its ``[1, N]``
    fp32 row scales; the activation is quantized per row (amax / 448).
    """
    x = input.float()
    x_scale = (x.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)
    return torch._scaled_mm(x_fp8,
                            weight_fp8.t(),
                            scale_a=x_scale,
                            scale_b=weight_scale_t,
                            out_dtype=input.dtype)


class LMHead(Linear):
    """LM head layer.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        reduce_output: bool = True,
        use_custom_cublas_mm: bool = False,
        quant_config=None,
    ):
        local_in_features = embedding_dim
        local_out_features = num_embeddings
        mapping = mapping or Mapping()
        self.enable_lm_head_tp_in_adp = mapping.enable_attention_dp and \
            getattr(mapping, 'enable_lm_head_tp_in_adp', False)

        tp_size = mapping.tp_size

        # Attention DP doesn't work with embedding parallelization.
        if mapping.enable_attention_dp:
            tensor_parallel_mode = None

        if tensor_parallel_mode == TensorParallelMode.ROW:
            local_in_features = math.ceil(embedding_dim / tp_size)
            self.padding_size = tp_size * local_in_features - embedding_dim
        elif tensor_parallel_mode == TensorParallelMode.COLUMN:
            local_out_features = math.ceil(num_embeddings / tp_size)
            self.padding_size = tp_size * local_out_features - num_embeddings
        else:
            self.padding_size = 0

        # Linear re-shards only the dim selected by tensor_parallel_mode, so
        # only that dim may be passed as local*tp (to carry the ceil-div
        # padding); the other dim must be the true full size — Linear keeps it
        # as-is, and quant_method.create_weights sizes the (packed) quantized
        # weight and scales from it.
        in_features_full = (local_in_features * tp_size if tensor_parallel_mode
                            == TensorParallelMode.ROW else embedding_dim)
        out_features_full = (local_out_features *
                             tp_size if tensor_parallel_mode
                             == TensorParallelMode.COLUMN else num_embeddings)
        super().__init__(
            in_features_full,
            out_features_full,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
            quant_config=quant_config,
        )

        if tensor_parallel_mode == TensorParallelMode.ROW:
            if self.tp_rank == self.tp_size - 1:
                local_in_features -= self.padding_size
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if self.has_any_quant:
            # Keep the quantized weights created by Linear (e.g. NVFP4 packed
            # weight + scales). The plain-Parameter override below only exists
            # to (a) drop bias and (b) shrink the ROW-mode padded shard, both
            # of which assume a dense high-precision weight.
            if self.padding_size > 0 or tensor_parallel_mode == TensorParallelMode.ROW:
                raise NotImplementedError(
                    "Quantized LMHead does not support vocab/hidden padding or "
                    "ROW tensor-parallel mode")
            if self.enable_lm_head_tp_in_adp:
                logger.error(
                    "Quantized LMHead constructed with lm_head TP in ADP: "
                    "the spec-decoding head slices the raw weight, which is "
                    "incompatible with quantized (packed) weights. The "
                    "lm_head quant entry should have been dropped upstream "
                    f"(quant_algo={quant_config.quant_algo}).")
                raise NotImplementedError(
                    "Quantized LMHead does not support lm_head TP in ADP "
                    "(spec-decoding head slices the raw weight)")
            # lm_head has historically always been bf16; make the switch to a
            # quantized head visible so unexpected accuracy/perf behavior is
            # attributable without a profiler.
            logger.info(f"LMHead is quantized: quant_algo="
                        f"{quant_config.quant_algo}, weight shape "
                        f"{tuple(self.weight.shape)} dtype {self.weight.dtype}")
        else:
            weight_shape = (self.out_features, self.in_features)
            self.weight = Parameter(torch.empty(weight_shape, dtype=dtype))
            self.register_parameter("bias", None)

    @property
    def vocab_size_padded(self) -> int:
        if self.tp_mode == TensorParallelMode.COLUMN and self.gather_output:
            return self.out_features * self.tp_size
        else:
            return self.out_features

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        mapping_lm_head_tp: Optional[Mapping] = None,
        is_spec_decoding_head: bool = False,
    ) -> torch.Tensor:
        if is_spec_decoding_head and self.enable_lm_head_tp_in_adp:
            # For LM head TP in ADP, we need to slice the weight for the LM head
            tp_rank = mapping_lm_head_tp.tp_rank
            tp_size = mapping_lm_head_tp.tp_size
            slice_width = ceil_div(self.out_features, tp_size)
            slice_start = tp_rank * slice_width
            slice_end = min((tp_rank + 1) * slice_width, self.out_features)
            output = F.linear(input, self.weight[slice_start:slice_end, :],
                              None)
        else:
            output = super().forward(input, all_reduce_params=all_reduce_params)
        if (self.tp_mode == TensorParallelMode.COLUMN and self.gather_output
                and self.padding_size > 0):
            output = output[..., :-self.padding_size]

        return output

    def enable_fp8_rowwise_projection(self, chunk_rows: int = 4096) -> None:
        """Project with an FP8 copy of this head's weight from now on.

        E4M3 weight with one scale per vocab row (amax / 448) and a per-token
        dynamic E4M3 activation scale, through ``torch._scaled_mm``; the bf16
        weight stays allocated but unused by :meth:`apply_linear`. Meant for an
        Eagle3 draft head that owns its weight: draft tokens only steer which
        candidates the target verifies, so the generated tokens do not change,
        while the head's (memory-bound) weight read halves. The conversion
        runs once, here, in row chunks to bound the fp32 temporaries.
        """
        if self.has_any_quant or self.weight.dim() != 2:
            raise NotImplementedError(
                "FP8 rowwise LM head needs a dense high-precision weight")
        w = self.weight.detach()
        fp8 = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
        scale = torch.empty((w.shape[0], 1),
                            dtype=torch.float32,
                            device=w.device)
        for r0 in range(0, w.shape[0], chunk_rows):
            wf = w[r0:r0 + chunk_rows].float()
            sc = (wf.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
            fp8[r0:r0 + chunk_rows] = (wf / sc).to(torch.float8_e4m3fn)
            scale[r0:r0 + chunk_rows] = sc
        self._fp8_weight = fp8
        self._fp8_weight_scale = scale.t().contiguous()

    @property
    def uses_fp8_rowwise_projection(self) -> bool:
        return getattr(self, "_fp8_weight", None) is not None

    def apply_linear(self, input, bias, lora_params=None, layer_idx=None):
        if (self.uses_fp8_rowwise_projection and bias is None
                and not lora_params and input.dim() == 2):
            return fp8_rowwise_linear(input, self._fp8_weight,
                                      self._fp8_weight_scale)
        return super().apply_linear(input, bias, lora_params, layer_idx)

    def supports_vocab_shard_projection(self) -> bool:
        """Whether :meth:`forward_vocab_shard` can replace :meth:`forward`.

        True for a vocab-parallel head under plain tensor parallelism whose
        gathered output is exactly the rank-ordered concatenation of equal,
        unpadded shards, so rank ``r``'s shard covers vocab ids
        ``[r * out_features, (r + 1) * out_features)``.
        """
        return (self.tp_mode == TensorParallelMode.COLUMN and self.tp_size > 1
                and self.gather_output and self.padding_size == 0
                and self.gather_output_sizes is None
                and not self.mapping.enable_attention_dp and self.lora is None)

    def forward_vocab_shard(self, input: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of :meth:`forward`, without the all-gather.

        The same GEMM :meth:`forward` runs before gathering, so the result is
        bitwise the shard's columns of the gathered logits. A consumer that
        only needs each row's argmax combines the shards' maxima instead of
        gathering ``[rows, vocab]`` logits (see
        ``SpecWorkerBase.greedy_sample_draft_with_tp_gather``). Requires
        :meth:`supports_vocab_shard_projection`.
        """
        return self.apply_linear(input, self.bias)

    def forward_lm_head_tp_in_adp(
            self,
            input: torch.Tensor,
            num_rows_per_rank: int,
            lm_head_tp_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Mapping]:
        """Project the attention-DP group's rows onto this rank's vocab shard.

        Under attention DP every rank holds the whole weight, so a batch of a
        few rows re-reads all of it on every rank. Instead, every rank pads its
        rows to ``num_rows_per_rank``, the rows are all-gathered over the
        LM-head TP group, and each rank projects them onto its slice of the
        vocabulary only.

        Args:
            input: This rank's ``[num_rows, hidden_size]`` rows.
            num_rows_per_rank: The row count every rank pads to. It must be
                identical on all ranks and at least the largest ``num_rows``.
            lm_head_tp_size: The LM-head TP group size (a divisor of the
                attention-DP group size, identical on all ranks); None picks it
                from ``num_rows_per_rank`` (see ``create_lm_head_tp_mapping``).

        Returns:
            The ``[group_size * num_rows_per_rank, shard_size]`` logits, with
            the rows of LM-head TP rank ``r`` at ``[r * num_rows_per_rank,
            (r + 1) * num_rows_per_rank)``, and the LM-head TP mapping. The
            shard of rank ``r`` covers vocab ids ``[r * shard_size,
            (r + 1) * shard_size)``; ids past the vocabulary hold ``-inf``.
        """
        assert self.enable_lm_head_tp_in_adp, (
            "forward_lm_head_tp_in_adp requires enable_lm_head_tp_in_adp")
        num_rows = input.shape[0]
        if num_rows > num_rows_per_rank:
            raise ValueError(f"LM head input has {num_rows} rows, more than "
                             f"num_rows_per_rank={num_rows_per_rank}")
        mapping_lm_head_tp = create_lm_head_tp_mapping(self.mapping,
                                                       num_rows_per_rank,
                                                       lm_head_tp_size)
        if num_rows < num_rows_per_rank:
            input = F.pad(input, (0, 0, 0, num_rows_per_rank - num_rows))
        input = allgather(input, mapping_lm_head_tp, dim=0)
        output = self.forward(input,
                              mapping_lm_head_tp=mapping_lm_head_tp,
                              is_spec_decoding_head=True)
        shard_size = ceil_div(self.out_features, mapping_lm_head_tp.tp_size)
        if output.shape[-1] < shard_size:
            output = F.pad(output, (0, shard_size - output.shape[-1]),
                           value=float("-inf"))
        return output, mapping_lm_head_tp

    def forward_lm_head_tp_in_adp_full_vocab(
            self,
            input: torch.Tensor,
            num_rows_per_rank: int,
            lm_head_tp_size: Optional[int] = None) -> torch.Tensor:
        """Full-vocab logits of this rank's rows, computed vocab-parallel.

        Runs :meth:`forward_lm_head_tp_in_adp`, then gathers every LM-head TP
        rank's vocab shard of this rank's own rows, so the result has the
        ``[num_rows, vocab_size]`` layout of :meth:`forward`.
        """
        num_rows = input.shape[0]
        shards, mapping_lm_head_tp = self.forward_lm_head_tp_in_adp(
            input, num_rows_per_rank, lm_head_tp_size)
        group_size = mapping_lm_head_tp.tp_size
        # [shard owner, row owner, row, shard column]
        shards = allgather(shards, mapping_lm_head_tp,
                           dim=0).view(group_size, group_size,
                                       num_rows_per_rank, -1)
        logits = shards[:, mapping_lm_head_tp.tp_rank, :num_rows]
        logits = logits.transpose(0, 1).reshape(num_rows, -1)
        return logits[:, :self.out_features]

    def skip_forward(
            self,
            input: torch.Tensor,
            *,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        output_shape = input.shape[:-1] + (self.num_embeddings, )
        output = input.new_empty(output_shape)
        return output

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        original_weight = None
        if self.tp_mode == TensorParallelMode.COLUMN:
            if self.tp_rank == self.tp_size - 1 and self.padding_size > 0:
                original_weight = self.weight.data.zero_()
                self.weight.data = self.weight[:-self.padding_size, :]

        super().load_weights(weights,
                             allow_partial_loading=allow_partial_loading)

        if original_weight is not None:
            self.weight.data = original_weight


def get_masked_input_and_mask(
    input_: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_mask = (input_ >= vocab_start_index) & (input_ < vocab_end_index)
    valid_offset = vocab_start_index * vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask.unsqueeze(-1)


def pre_comm_embedding_ops(
    input_: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_mode: TensorParallelMode,
    vocab_start_index: int,
    vocab_end_index: int,
    gather_output: bool,
    padding_size: int,
):
    if tp_mode == TensorParallelMode.COLUMN:
        input_, input_mask = get_masked_input_and_mask(
            input_,
            vocab_start_index,
            vocab_end_index,
        )
    else:
        # flashinfer's rejection kernel (chain_speculative_sampling) pads non-accepted
        # tokens with -1. When the full vocab is local (non-TP or ROW TP), mask
        # out-of-range ids (e.g. -1) over [0, weight.shape[0]) to avoid an OOB
        # embedding lookup.
        input_, input_mask = get_masked_input_and_mask(
            input_,
            0,
            weight.shape[0],
        )

    # Get the embeddings.
    output = F.embedding(input_, weight)

    output.masked_fill_(input_mask, 0)

    if tp_mode == TensorParallelMode.ROW and gather_output:
        if tp_rank == tp_size - 1 and padding_size > 0:
            output = F.pad(output, (0, padding_size))

    return output


def embedding_skip_forward_impl(input: torch.Tensor, embedding_dim: int,
                                dtype: torch.dtype) -> torch.Tensor:
    output_shape = input.shape[:] + (embedding_dim, )
    output = input.new_empty(output_shape, dtype=dtype)
    return output


@torch.library.custom_op("trtllm::embedding_skip_forward", mutates_args=())
def embedding_skip_forward(input: torch.Tensor, embedding_dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    return embedding_skip_forward_impl(input, embedding_dim, dtype)


@embedding_skip_forward.register_fake
def _(input, embedding_dim, dtype):
    return embedding_skip_forward_impl(input, embedding_dim, dtype)


class Embedding(LMHead):
    """Embedding layer.

    Adapted from torch.nn.Embedding.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        reduce_output: bool = True,
        enable_torch_compile_for_embedding: Optional[bool] = False,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        self.enable_torch_compile_for_embedding = enable_torch_compile_for_embedding

        if self.tp_size > 1:
            slice_width = math.ceil(num_embeddings / self.tp_size)
            self.vocab_start_index = self.tp_rank * slice_width
            self.vocab_end_index = min((self.tp_rank + 1) * slice_width,
                                       num_embeddings)
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings

    def forward(self, input):
        if self.tp_size > 1:
            # Run the ops before all_reduce/all_gather.
            # We use torch.compile() to fuse the tiny pointwise ops before all_reduce/all_gather for Embedding module.
            embedding_ops_func = torch.compile(
                pre_comm_embedding_ops,
                options={"max-autotune": True},
                disable=not self.enable_torch_compile_for_embedding)
        else:
            # Skip torch.compile when TP size is 1 to avoid unnecessary host overhead
            embedding_ops_func = pre_comm_embedding_ops
        output = embedding_ops_func(input, self.weight, self.tp_size,
                                    self.tp_rank, self.tp_mode,
                                    self.vocab_start_index,
                                    self.vocab_end_index, self.gather_output,
                                    self.padding_size)

        # Run the all_reduce/all_gather.
        if self.tp_size > 1:
            if self.tp_mode == TensorParallelMode.COLUMN:
                # Reduce across all the model parallel GPUs.
                output = self.all_reduce(output)
            elif self.tp_mode == TensorParallelMode.ROW:
                if self.gather_output:
                    # Run allgather.
                    output = allgather(output, self.mapping)
                    # Remove the padding.
                    if self.padding_size > 0:
                        output = output[..., :-self.padding_size]

        return output

    def skip_forward(self, input):
        return embedding_skip_forward(input, self.embedding_dim, self.dtype)
