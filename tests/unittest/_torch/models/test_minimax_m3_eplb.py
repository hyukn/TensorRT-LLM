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
"""MiniMax-M3 is eligible for the MoE load balancer (EPLB) under expert parallelism."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    maybe_create_moe_load_balancer,
    moe_model_arch_list,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

M3_ARCHS = ("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")


@pytest.mark.parametrize("arch", M3_ARCHS)
def test_m3_archs_are_registered_and_eplb_eligible(arch: str) -> None:
    import tensorrt_llm._torch.models.modeling_minimaxm3  # noqa: F401  (registers the archs)

    assert arch in MODEL_CLASS_MAPPING
    assert arch in moe_model_arch_list


@pytest.mark.parametrize("arch", M3_ARCHS)
def test_m3_load_balancer_is_set_up_under_ep(arch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    def fake_init(self, ep_rank, ep_size, layer_updates_per_iter, *args, **kwargs):
        created.update(ep_rank=ep_rank, ep_size=ep_size, updates=layer_updates_per_iter)

    monkeypatch.setattr(MoeLoadBalancer, "__init__", fake_init)
    lb_config = MoeLoadBalancerConfig(num_slots=144, layer_updates_per_iter=0)
    mapping = SimpleNamespace(moe_ep_rank=3, moe_ep_size=8, moe_cluster_size=1)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[arch]),
        moe_load_balancer=lb_config,
    )
    maybe_create_moe_load_balancer(model_config, mapping)
    assert created == {"ep_rank": 3, "ep_size": 8, "updates": 0}
    assert lb_config.num_local_slots == 18
