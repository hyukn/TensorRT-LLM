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
"""Eagle3 first draft step on context batches: last query only (TRTLLM_EAGLE3_CTX_DRAFT_LAST_QUERY), gating."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.speculative.eagle3 as eagle3
from tensorrt_llm._torch.speculative.eagle3 import (
    Eagle3OneModelWorker,
    _draft_attention_backends,
    _local_draft_attention,
)

pytestmark = pytest.mark.cpu_only


class _Backend:
    def __init__(self, attention_chunk_size=None):
        self.attention_chunk_size = attention_chunk_size


class _AttnModule(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.attn = backend


class _Draft(nn.Module):
    def __init__(self, *backends):
        super().__init__()
        self.layers = nn.ModuleList([_AttnModule(b) for b in backends])


def test_local_window_is_set_and_restored() -> None:
    backends = [_Backend(), _Backend()]
    backends[1].context_window_left_override = 7
    found = _draft_attention_backends(_Draft(*backends))
    assert found == backends
    with _local_draft_attention(found, 256):
        assert [b.context_window_left_override for b in backends] == [255, 255]
    assert backends[0].context_window_left_override is None
    assert backends[1].context_window_left_override == 7
    with pytest.raises(RuntimeError):
        with _local_draft_attention(found, 64):
            raise RuntimeError("draft forward failed")
    assert backends[0].context_window_left_override is None
    assert backends[1].context_window_left_override == 7


def _worker(**overrides) -> Eagle3OneModelWorker:
    worker = Eagle3OneModelWorker.__new__(Eagle3OneModelWorker)
    worker.is_mtp_eagle = False
    worker.guided_decoder = None
    worker.sa_enhancer = None
    for key, value in overrides.items():
        setattr(worker, key, value)
    return worker


def _meta(new, kv):
    return SimpleNamespace(
        seq_lens=torch.tensor(new, dtype=torch.int32), kv_lens=torch.tensor(kv, dtype=torch.int32)
    )


# Three context requests: 16K new over ~1M / 300K cached, and 2K over 100K -> ~2.2e10 query-key pairs.
_BIG = _meta([16384, 16384, 2048], [1032192, 316384, 102048])
# A small eager step: 4K new over 60K cached -> 2.6e8 pairs, below the default threshold.
_SMALL = _meta([4096], [64096])


def test_ctx_draft_last_query_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _Draft(_Backend())

    def use(worker, meta=_BIG, num_contexts=3, num_gens=0, index_share=False, model=draft):
        return worker._use_ctx_draft_last_query(model, meta, num_contexts, num_gens, index_share)

    monkeypatch.setattr(eagle3, "_CTX_DRAFT_LAST_QUERY", False)
    assert not use(_worker())
    monkeypatch.setattr(eagle3, "_CTX_DRAFT_LAST_QUERY", True)
    assert use(_worker())
    # only where the saved attention outweighs the extra one-token pass
    assert not use(_worker(), meta=_SMALL, num_contexts=1)
    monkeypatch.setattr(eagle3, "_CTX_DRAFT_MIN_QK", 1e8)
    assert use(_worker(), meta=_SMALL, num_contexts=1)
    monkeypatch.setattr(eagle3, "_CTX_DRAFT_MIN_QK", 3e9)
    # only pure context batches
    assert not use(_worker(), num_contexts=0, num_gens=4)
    assert not use(_worker(), num_gens=1)
    # not for MTP-Eagle, index-share, guided decoding or SA drafts
    assert not use(_worker(is_mtp_eagle=True))
    assert not use(_worker(), index_share=True)
    assert not use(_worker(guided_decoder=object()))
    assert not use(_worker(sa_enhancer=object()))
    # a draft that already runs chunked attention, or has no attention backend, keeps the full step
    assert not use(_worker(), model=_Draft(_Backend(8192)))
    assert not use(_worker(), model=nn.Linear(2, 2))
    # never while tracing (compiled prefill steps keep the traced path)
    monkeypatch.setattr(eagle3, "is_torch_compiling", lambda: True)
    assert not use(_worker())


def test_window_override_reaches_the_context_kernel_call() -> None:
    """The trtllm-gen context path reads ``context_window_left_override`` for causal masks only."""
    import inspect

    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen

    src = inspect.getsource(flashinfer_trtllm_gen.FlashInferTrtllmGenFmha.run_context)
    assert 'getattr(attn, "context_window_left_override", None)' in src
    assert "window_left = window_left_override" in src
