# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The index TP split must stay off for a TP group that spans nodes (no IpcMemory peer access)."""

from types import SimpleNamespace

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import index_tp_split


def _mapping(rank, tp_size=8, gpus_per_node=4):
    return SimpleNamespace(
        rank=rank,
        tp_size=tp_size,
        pp_rank=0,
        local_rank=rank % gpus_per_node,
        node_rank=rank // gpus_per_node,
        tp_group=list(range(tp_size)),
        get_local_rank=lambda r: r % gpus_per_node,
        get_node_rank=lambda r: r // gpus_per_node,
    )


def test_multi_node_tp_group_disables_the_split():
    index_tp_split._PEER_ACCESS.clear()
    # TP8 over two 4-GPU nodes: every rank sees a peer on the other node.
    assert all(not index_tp_split.tp_split_peer_access(_mapping(r)) for r in range(8))
    # Cached per (rank, tp, pp).
    assert (3, 8, 0) in index_tp_split._PEER_ACCESS
    index_tp_split._PEER_ACCESS.clear()
