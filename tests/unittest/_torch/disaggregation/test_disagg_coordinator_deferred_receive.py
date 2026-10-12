# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deferred publication of generation KV receives (TRTLLM_DISAGG_DEFER_GEN_RECEIVE).

With deferral on, ``receive_gen_init`` prepares resources but publishes no receive; the
executor loop calls ``publish_deferred_gen_receives`` after its forward, which host-waits each
request's admission event on the KV cache managers and only then starts the receives. Requests
that left DISAGG_GENERATION_INIT meanwhile are skipped. Synchronous transfer mode never defers.
"""

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import NoopDisaggCoordinator
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _async_transfer_mode(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)


class _FakeKvManager:
    """Records admission-event waits; shares one event log with the transceiver."""

    def __init__(self, log):
        self.defer_disagg_receive_ready = False
        self.log = log

    def wait_disagg_receive_ready(self, request_id):
        self.log.append(f"wait:{request_id}")


def _harness():
    h = CoordinatorHarness()
    log = h.transceiver.call_log
    mgr = _FakeKvManager(log)
    h.coordinator._kv_cache_manager = mgr
    return h, mgr, log


def _gen_init(h, rid):
    req = TransferRequest(
        rid, is_context_only_request=False, state=LlmRequestState.DISAGG_GENERATION_INIT
    )
    h.active.append(req)
    return req


def _receives(log):
    return [entry for entry in log if entry.startswith(("request_and_receive", "wait:"))]


def test_default_publishes_immediately():
    h, mgr, log = _harness()
    a, b = _gen_init(h, 1), _gen_init(h, 2)
    h.coordinator.receive_gen_init([a, b])
    assert mgr.defer_disagg_receive_ready is False
    assert _receives(log) == ["request_and_receive_async:1", "request_and_receive_async:2"]
    h.coordinator.publish_deferred_gen_receives()  # nothing deferred: no-op
    assert _receives(log) == ["request_and_receive_async:1", "request_and_receive_async:2"]


def test_deferred_waits_then_publishes_after_forward():
    h, mgr, log = _harness()
    h.coordinator.enable_deferred_gen_receive_publish(True)
    assert mgr.defer_disagg_receive_ready is True
    a, b = _gen_init(h, 1), _gen_init(h, 2)

    h.coordinator.receive_gen_init([a, b])
    # Resources prepared, nothing published, requests still waiting for their receive.
    assert h.effects.prepared == [[a, b]]
    assert _receives(log) == []
    assert a.state == b.state == LlmRequestState.DISAGG_GENERATION_INIT

    h.coordinator.publish_deferred_gen_receives()
    # Every admission event is awaited before the first receive is published.
    assert _receives(log) == [
        "wait:1",
        "wait:2",
        "request_and_receive_async:1",
        "request_and_receive_async:2",
    ]
    assert a.state != LlmRequestState.DISAGG_GENERATION_INIT
    # Published once only.
    h.coordinator.publish_deferred_gen_receives()
    assert len(_receives(log)) == 4


def test_deferred_skips_requests_that_left_init():
    h, mgr, log = _harness()
    h.coordinator.enable_deferred_gen_receive_publish(True)
    a, b = _gen_init(h, 1), _gen_init(h, 2)
    h.coordinator.receive_gen_init([a, b])
    a.state = LlmRequestState.GENERATION_COMPLETE  # terminated before the publish point
    h.coordinator.publish_deferred_gen_receives()
    assert _receives(log) == ["wait:2", "request_and_receive_async:2"]


def test_sync_transfer_mode_never_defers(monkeypatch):
    monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    h, mgr, log = _harness()
    h.coordinator.enable_deferred_gen_receive_publish(True)
    assert mgr.defer_disagg_receive_ready is False
    assert h.coordinator._defer_gen_receive_publish is False


def test_gen_only_benchmark_defers_only_the_admission_wait(monkeypatch):
    monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
    h, mgr, log = _harness()
    h.coordinator.enable_deferred_gen_receive_publish(True)
    assert mgr.defer_disagg_receive_ready is True
    a = _gen_init(h, 1)
    h.coordinator.receive_gen_init([a])
    # No receive exists in this benchmark: the request is complete right away, nothing deferred.
    assert a.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    h.coordinator.publish_deferred_gen_receives()
    assert _receives(log) == []


def test_disable_restores_immediate_path():
    h, mgr, log = _harness()
    h.coordinator.enable_deferred_gen_receive_publish(True)
    h.coordinator.enable_deferred_gen_receive_publish(False)
    assert mgr.defer_disagg_receive_ready is False
    a = _gen_init(h, 1)
    h.coordinator.receive_gen_init([a])
    assert _receives(log) == ["request_and_receive_async:1"]


def test_noop_coordinator_accepts_the_calls():
    noop = NoopDisaggCoordinator()
    noop.enable_deferred_gen_receive_publish(True)
    noop.publish_deferred_gen_receives()
