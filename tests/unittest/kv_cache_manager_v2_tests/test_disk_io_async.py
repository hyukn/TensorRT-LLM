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
"""Disk and host-to-host copies stay stream-ordered when their I/O runs on worker threads.

copy_disk_to_host, copy_host_to_disk, copy_disk_to_disk and copy_host_to_host hand their pread /
pwrite / memcpy work to a worker pool and make the stream wait for a completion flag
(TRTLLM_KV_DISK_IO_THREADS workers; 0 runs the I/O inside a CUDA host function as before). These
tests drive GPU -> host -> disk -> host -> GPU round trips on several streams at once, with no host
synchronization between the hops, and check every byte. Each hop's buffer is overwritten right
after it is consumed, so a hop that did not wait for the previous one reads the wrong bytes.

The pool is created once per process, so the thread-count variants run in subprocesses.

The KVCacheManagerV2 tests at the end drive multi-turn conversations through small GPU and host tiers
so that every conversation's prefix keeps cycling GPU -> host -> disk -> GPU, with
TRTLLM_KV_DISK_KEEP_COPY (committed pages promoted from disk keep their disk slot, and a later
demotion reuses it instead of rewriting the page) and TRTLLM_KV_HOST_FREE_WATERMARK (evict ahead to
disk). Every reused token is checked on the GPU by the fake engine, partial blocks are re-snapshotted
on every turn (a longer snapshot replaces a page that has a copy), and a tight disk quota forces
copies to be reclaimed and their slots reused. They run in subprocesses: the settings are read when
a manager is built, and a failed device-side check poisons the CUDA context.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

copy_helpers = pytest.importorskip(
    "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils",
    reason="the disk copy helpers are C++-backend only",
)

DiskAddress = copy_helpers.DiskAddress
DiskToDiskTask = copy_helpers.DiskToDiskTask
DiskToHostTask = copy_helpers.DiskToHostTask
HostToDiskTask = copy_helpers.HostToDiskTask
MemToMemTask = copy_helpers.MemToMemTask

POOL_THREAD_NAME = "kvcm-disk-io"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _check(result: int) -> None:
    assert result == 0, f"CUresult {result}"


def _pool_threads() -> int:
    count = 0
    for task in Path("/proc/self/task").iterdir():
        try:
            if (task / "comm").read_text().strip() == POOL_THREAD_NAME:
                count += 1
        except OSError:
            pass
    return count


def _round_trips(
    directory: str,
    num_streams: int = 6,
    pages_per_stream: int = 12,
    page_bytes: int = 1 << 20,
    rounds: int = 3,
) -> None:
    """Every stream moves its own pages through GPU -> host -> disk -> disk -> host -> host -> GPU.

    Stream s owns pages [s * P, (s + 1) * P) of two disk regions. One round on one stream:

      gpu_src --D2H--> host_a --H2Disk--> region 0 --Disk2Disk--> region 1 --Disk2H--> host_b
      host_b --H2H--> host_c --H2D--> gpu_dst

    after which host_a, host_b and host_c are overwritten on the same stream, before the next hop
    could have read them if it did not wait. The rounds reuse the same buffers and file pages with
    new data, all enqueued before a single synchronize; every round's GPU result is kept and
    compared at the end. A cross-stream pass then reads each stream's region-1 pages back on the
    next stream, ordered only by an event.
    """
    assert page_bytes % 16 == 0
    device = torch.device("cuda")
    pages = num_streams * pages_per_stream
    region_bytes = pages * page_bytes
    fd = os.open(os.path.join(directory, "kv_disk_tier.bin"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, 2 * region_bytes)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        generator = torch.Generator(device=device).manual_seed(1234)
        sources = [
            torch.randint(
                0, 256, (pages, page_bytes), dtype=torch.uint8, device=device, generator=generator
            )
            for _ in range(rounds)
        ]
        poison = torch.full((pages, page_bytes), 0xA5, dtype=torch.uint8, device=device)

        def pinned():
            return torch.empty((pages, page_bytes), dtype=torch.uint8, pin_memory=True)

        host_a, host_b, host_c = pinned(), pinned(), pinned()
        gpu_dst = torch.empty((pages, page_bytes), dtype=torch.uint8, device=device)
        results = [torch.empty_like(gpu_dst) for _ in range(rounds)]

        def rows(tensor, s):
            base = tensor.data_ptr()
            return [base + (s * pages_per_stream + i) * page_bytes for i in range(pages_per_stream)]

        def disk(region, s):
            return [
                DiskAddress(fd, region * region_bytes + (s * pages_per_stream + i) * page_bytes)
                for i in range(pages_per_stream)
            ]

        for r in range(rounds):
            for s, stream in enumerate(streams):
                handle = stream.cuda_stream
                with torch.cuda.stream(stream):
                    lo, hi = s * pages_per_stream, (s + 1) * pages_per_stream
                    _check(
                        copy_helpers.copy_device_to_host(
                            [
                                MemToMemTask(d, src)
                                for d, src in zip(rows(host_a, s), rows(sources[r], s))
                            ],
                            page_bytes,
                            handle,
                        )
                    )
                    _check(
                        copy_helpers.copy_host_to_disk(
                            [HostToDiskTask(d, src) for d, src in zip(disk(0, s), rows(host_a, s))],
                            page_bytes,
                            handle,
                        )
                    )
                    host_a[lo:hi].copy_(poison[lo:hi], non_blocking=True)
                    _check(
                        copy_helpers.copy_disk_to_disk(
                            [DiskToDiskTask(d, src) for d, src in zip(disk(1, s), disk(0, s))],
                            page_bytes,
                            handle,
                        )
                    )
                    _check(
                        copy_helpers.copy_disk_to_host(
                            [DiskToHostTask(d, src) for d, src in zip(rows(host_b, s), disk(1, s))],
                            page_bytes,
                            handle,
                        )
                    )
                    _check(
                        copy_helpers.copy_host_to_host(
                            [
                                MemToMemTask(d, src)
                                for d, src in zip(rows(host_c, s), rows(host_b, s))
                            ],
                            page_bytes,
                            handle,
                        )
                    )
                    host_b[lo:hi].copy_(poison[lo:hi], non_blocking=True)
                    _check(
                        copy_helpers.copy_host_to_device(
                            [
                                MemToMemTask(d, src)
                                for d, src in zip(rows(gpu_dst, s), rows(host_c, s))
                            ],
                            page_bytes,
                            handle,
                        )
                    )
                    host_c[lo:hi].copy_(poison[lo:hi], non_blocking=True)
                    results[r][lo:hi].copy_(gpu_dst[lo:hi], non_blocking=True)

        # Cross-stream: stream s + 1 reads stream s's region-1 pages (the last round's data) once an
        # event says stream s is done with them.
        cross = torch.empty_like(gpu_dst)
        cross_staging = pinned()
        events = [stream.record_event() for stream in streams]
        for s in range(num_streams):
            reader = streams[(s + 1) % num_streams]
            reader.wait_event(events[s])
            handle = reader.cuda_stream
            with torch.cuda.stream(reader):
                lo, hi = s * pages_per_stream, (s + 1) * pages_per_stream
                _check(
                    copy_helpers.copy_disk_to_host(
                        [
                            DiskToHostTask(d, src)
                            for d, src in zip(rows(cross_staging, s), disk(1, s))
                        ],
                        page_bytes,
                        handle,
                    )
                )
                cross[lo:hi].copy_(cross_staging[lo:hi], non_blocking=True)

        torch.cuda.synchronize()
        for r in range(rounds):
            mismatched = (results[r] != sources[r]).any(dim=1).nonzero().flatten().tolist()
            assert not mismatched, f"round {r}: pages {mismatched} differ after the round trip"
        assert torch.equal(cross, sources[-1]), "cross-stream disk read returned the wrong bytes"
    finally:
        os.close(fd)


def _run_in_subprocess(tmp_path: Path, threads: str | None, expect_pool_threads: int) -> None:
    env = dict(os.environ)
    env.pop("TRTLLM_KV_DISK_IO_THREADS", None)
    if threads is not None:
        env["TRTLLM_KV_DISK_IO_THREADS"] = threads
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path), str(expect_pool_threads)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _supports_stream_wait_value64() -> bool:
    # 64-bit stream memory operations: every GPU these copies target (Hopper, Blackwell).
    return torch.cuda.get_device_capability() >= (9, 0)


def test_round_trips_overlapping_streams(tmp_path: Path) -> None:
    """Default configuration, in this process."""
    _round_trips(str(tmp_path))


def test_empty_task_lists_are_no_ops() -> None:
    stream = torch.cuda.current_stream().cuda_stream
    assert copy_helpers.copy_disk_to_host([], 4096, stream) == 0
    assert copy_helpers.copy_host_to_disk([], 4096, stream) == 0
    assert copy_helpers.copy_disk_to_disk([], 4096, stream) == 0
    assert copy_helpers.copy_host_to_host([], 4096, stream) == 0
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "threads, expect_pool_threads",
    [
        pytest.param("0", 0, id="in-callback"),
        pytest.param("1", 1, id="one-worker"),
        pytest.param("3", 3, id="three-workers"),
        pytest.param(None, 8, id="default"),
    ],
)
def test_round_trips_per_thread_count(
    tmp_path: Path, threads: str | None, expect_pool_threads: int
) -> None:
    if expect_pool_threads > 0 and not _supports_stream_wait_value64():
        pytest.skip("the worker pool needs 64-bit stream memory operations")
    _run_in_subprocess(tmp_path, threads, expect_pool_threads)


def _process_io() -> dict[str, int]:
    fields = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":")
        fields[key.strip()] = int(value)
    return fields


def _conversations(
    directory: str,
    disk_quota: int,
    num_conversations: int = 8,
    num_turns: int = 5,
    new_tokens_per_turn: int = 100,
) -> dict[str, int]:
    """Multi-turn conversations, interleaved, each turn reusing its conversation's whole history.

    100 new tokens per turn with 32-token blocks leaves a partial block every turn, so each turn also
    replaces a partial snapshot page. 4 layers x (key + value) x 8 KiB = 64 KiB per block; GPU and
    host hold 32 blocks each while the conversations grow to 128, so prefixes come back from disk.
    """
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        CudaStream,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        TokenId,
    )

    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, test_dir)
    from cuda_test_utils import TemporaryCudaStream, init_cuda_once, round_up
    from fake_engine import FakeEngine, Role, Step

    init_cuda_once()
    tokens_per_block = 32
    config = KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=[
            GpuCacheTierConfig(quota=2 << 20),
            HostCacheTierConfig(quota=2 << 20),
            DiskCacheTierConfig(quota=disk_quota, path=directory),
        ],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(layer),
                buffers=[
                    BufferConfig(role=Role.KEY, size=8192),
                    BufferConfig(role=Role.VALUE, size=8192),
                ],
            )
            for layer in range(4)
        ],
    )
    engine = FakeEngine(config)
    manager = KVCacheManager(config)
    next_token = iter(range(1, 1 << 30))
    prompts: list[list] = [[] for _ in range(num_conversations)]
    reused_tokens = 0
    io_before = _process_io()
    with TemporaryCudaStream([]) as s:
        stream = CudaStream(s.handle)
        for _ in range(num_turns):
            for prompt in prompts:
                prompt.extend(TokenId(next(next_token)) for _ in range(new_tokens_per_turn))
                kv_cache = manager.create_kv_cache(None, prompt)
                num_reused = kv_cache.num_committed_tokens
                reused_tokens += num_reused
                assert kv_cache.resume(stream)
                assert kv_cache.resize(round_up(len(prompt), tokens_per_block))
                # Checks every reused token's values on the GPU, then writes the new ones.
                engine.execute([Step(kv_cache, prompt[num_reused:], prompt[:num_reused])], stream)
                kv_cache.commit(prompt[num_reused:])
                kv_cache.close()
    s.take_finish_event()
    import cuda.bindings.runtime as cudart

    (err,) = cudart.cudaDeviceSynchronize()
    assert int(err) == int(cudart.cudaError_t.cudaSuccess), f"device check failed: {err}"
    io_after = _process_io()
    manager.shutdown()
    return {
        "reused_tokens": reused_tokens,
        "bytes_written": io_after["wchar"] - io_before["wchar"],
        "bytes_read": io_after["rchar"] - io_before["rchar"],
    }


def _run_conversations(tmp_path: Path, env_overrides: dict[str, str], disk_quota: int) -> dict:
    env = dict(os.environ)
    for key in ("TRTLLM_KV_DISK_KEEP_COPY", "TRTLLM_KV_HOST_FREE_WATERMARK"):
        env.pop(key, None)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, __file__, "conversations", str(tmp_path), str(disk_quota)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_disk_copies_serve_reused_prefixes_and_save_writes(tmp_path: Path) -> None:
    """Kept disk copies give the same reuse, correct bytes, and far fewer disk writes.

    The same conversations run with and without TRTLLM_KV_DISK_KEEP_COPY; every reused token is
    checked, and demotions that reuse a copy write nothing.
    """
    baseline = _run_conversations(tmp_path, {}, disk_quota=64 << 20)
    kept = _run_conversations(tmp_path, {"TRTLLM_KV_DISK_KEEP_COPY": "1"}, disk_quota=64 << 20)
    assert baseline["reused_tokens"] > 0
    assert kept["reused_tokens"] == baseline["reused_tokens"]
    assert kept["bytes_written"] < 0.6 * baseline["bytes_written"], (baseline, kept)


@pytest.mark.parametrize(
    "env_overrides",
    [
        pytest.param({"TRTLLM_KV_DISK_KEEP_COPY": "1"}, id="keep-copy"),
        pytest.param(
            {"TRTLLM_KV_DISK_KEEP_COPY": "1", "TRTLLM_KV_HOST_FREE_WATERMARK": "0.25"},
            id="keep-copy-watermark",
        ),
        pytest.param({"TRTLLM_KV_HOST_FREE_WATERMARK": "0.25"}, id="watermark"),
    ],
)
def test_disk_copies_under_disk_pressure(tmp_path: Path, env_overrides: dict[str, str]) -> None:
    """No reused token is ever served from a stale copy when the disk tier is short.

    With a disk tier smaller than the conversations, copies are reclaimed, their slots rewritten by
    other pages, and pages dropped.
    """
    result = _run_conversations(tmp_path, env_overrides, disk_quota=6 << 20)
    assert result["reused_tokens"] > 0


if __name__ == "__main__":
    if sys.argv[1] == "conversations":
        print(json.dumps(_conversations(sys.argv[2], int(sys.argv[3]))))
        sys.exit(0)
    _round_trips(sys.argv[1])
    found = _pool_threads()
    expected = int(sys.argv[2])
    assert found == expected, f"{found} '{POOL_THREAD_NAME}' threads, expected {expected}"
