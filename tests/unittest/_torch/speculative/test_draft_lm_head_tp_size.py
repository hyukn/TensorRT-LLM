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
"""LM-head TP group size of the Eagle3 draft-token projection (TRTLLM_SPEC_DRAFT_LM_HEAD_TP_SIZE)."""

import pytest

from tensorrt_llm._torch.speculative import eagle3
from tensorrt_llm._torch.utils import create_lm_head_tp_mapping
from tensorrt_llm.mapping import Mapping


def _adp_mapping(rank: int, world_size: int = 8, gpus_per_node: int = 2) -> Mapping:
    return Mapping(
        world_size=world_size,
        tp_size=world_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=True,
    )


def test_heuristic_is_capped_at_gpus_per_node(monkeypatch):
    # 26 draft rows per rank want 256 // 26 -> 16 ranks, but the heuristic caps the group at gpus_per_node.
    monkeypatch.delenv("LM_HEAD_TP_SIZE", raising=False)
    assert create_lm_head_tp_mapping(_adp_mapping(5), 26).tp_size == 2
    assert create_lm_head_tp_mapping(_adp_mapping(5, gpus_per_node=4), 26).tp_size == 4


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("rank", [0, 3, 5, 7])
def test_explicit_size_overrides_heuristic_and_env(monkeypatch, size, rank):
    monkeypatch.setenv("LM_HEAD_TP_SIZE", "2")
    m = create_lm_head_tp_mapping(_adp_mapping(rank), 26, size)
    assert m.tp_size == size
    assert m.pp_size == 8 // size
    assert m.tp_rank == rank % size
    assert m.rank == rank
    assert m.enable_attention_dp and m.enable_lm_head_tp_in_adp
    # Consecutive ranks form a group, so the group's rows are the row-stacked batch of ranks
    # [rank - tp_rank, rank - tp_rank + size).
    assert m.tp_group == list(range(rank - rank % size, rank - rank % size + size))


@pytest.mark.parametrize("size", [0, 3, 16])
def test_explicit_size_must_divide_the_group(size):
    with pytest.raises(AssertionError):
        create_lm_head_tp_mapping(_adp_mapping(0), 26, size)


def test_env_unset_keeps_the_default(monkeypatch):
    monkeypatch.setattr(eagle3, "_DRAFT_LM_HEAD_TP_SIZE", 0)
    assert eagle3.draft_lm_head_tp_size(_adp_mapping(0)) is None


def test_env_size_is_returned_when_it_divides_the_group(monkeypatch):
    monkeypatch.setattr(eagle3, "_DRAFT_LM_HEAD_TP_SIZE", 8)
    assert eagle3.draft_lm_head_tp_size(_adp_mapping(3)) == 8
    assert eagle3.draft_lm_head_tp_size(None) is None


def test_env_size_must_divide_the_group(monkeypatch):
    monkeypatch.setattr(eagle3, "_DRAFT_LM_HEAD_TP_SIZE", 3)
    with pytest.raises(ValueError, match="must divide"):
        eagle3.draft_lm_head_tp_size(_adp_mapping(0))
