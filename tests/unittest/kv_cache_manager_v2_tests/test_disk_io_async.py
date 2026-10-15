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
"""

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


if __name__ == "__main__":
    _round_trips(sys.argv[1])
    found = _pool_threads()
    expected = int(sys.argv[2])
    assert found == expected, f"{found} '{POOL_THREAD_NAME}' threads, expected {expected}"
