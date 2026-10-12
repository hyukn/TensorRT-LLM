# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vocab-sharded Eagle3 draft LM head under attention DP.

Under attention DP every rank keeps the full (replicated) draft LM head and projects its own
draft rows to the whole vocabulary, so each rank reads the full ``vocab x hidden`` weight once per
draft step (MiniMax-M3: 200064 x 6144 bf16 = 2.46 GB, ~0.35 ms on GB300, three times per step).
Draft tokens are greedy, so the full logits are never needed: the ranks of an LM-head-TP group
(``create_lm_head_tp_mapping``) all-gather their padded draft rows, each projects the group's rows
onto its 1/tp vocabulary shard, and ``greedy_sample_draft_with_tp_gather`` combines the shard
maxima into the global argmax -- the same scheme MTP-Eagle uses with
``enable_lm_head_tp_in_adp``, here applied to the Eagle3 one-model draft head.

Opt-in with ``TRTLLM_EAGLE3_DRAFT_LM_HEAD_TP_IN_ADP=1``. Every input to the decision is identical
on all ranks of the group, so they all take the same path and issue the same collective.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm.mapping import Mapping

ENV = "TRTLLM_EAGLE3_DRAFT_LM_HEAD_TP_IN_ADP"


def enabled(mapping: Optional[Mapping]) -> bool:
    """Whether the vocab-sharded draft head may be used for this mapping (env + ADP, TP > 1)."""
    return (
        os.environ.get(ENV, "0") == "1"
        and mapping is not None
        and bool(mapping.enable_attention_dp)
        and mapping.tp_size > 1
    )


def sharded_draft_logits(
    rows: torch.Tensor,
    weight: torch.Tensor,
    mapping: Mapping,
    max_rows: int,
) -> Optional[Tuple[torch.Tensor, Mapping]]:
    """Project the LM-head-TP group's draft rows onto this rank's vocabulary shard.

    Args:
        rows: ``[n, hidden]`` this rank's draft rows (``n <= max_rows``).
        weight: ``[vocab, hidden]`` replicated draft LM head weight.
        mapping: the model mapping (attention DP).
        max_rows: per-rank row capacity (``max_num_requests``), identical on every rank.

    Returns:
        ``(logits, mapping_lm_head_tp)`` with fp32 ``logits`` of shape
        ``[tp * max_rows, vocab // tp]`` (group rank ``g``'s rows at ``[g * max_rows, ...)``), or
        ``None`` when the group would have a single rank or the vocabulary does not split evenly
        (the caller then projects its own rows to the full vocabulary).
    """
    from ..distributed.ops import allgather
    from ..utils import create_lm_head_tp_mapping

    n = int(rows.shape[0])
    if n > max_rows:
        raise ValueError(f"Eagle3 draft LM-head TP: {n} rows exceed max_num_requests={max_rows}")
    lm_mapping = create_lm_head_tp_mapping(mapping, max_rows)
    tp = lm_mapping.tp_size
    vocab = int(weight.shape[0])
    if tp <= 1 or vocab % tp != 0:
        return None
    rows2d = rows.reshape(n, int(weight.shape[1]))
    if n < max_rows:
        rows2d = F.pad(rows2d, (0, 0, 0, max_rows - n))
    stacked = allgather(rows2d.contiguous(), lm_mapping, dim=0)
    width = vocab // tp
    shard = weight[lm_mapping.tp_rank * width : (lm_mapping.tp_rank + 1) * width]
    # LogitsProcessor returns fp32 logits; keep the same dtype for the sampler.
    return F.linear(stacked, shard).float(), lm_mapping
