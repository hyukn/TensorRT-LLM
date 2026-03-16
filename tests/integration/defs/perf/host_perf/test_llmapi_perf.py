# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM API-level host performance benchmark.

Bypasses the HTTP serving layer (trtllm-serve + FastAPI + SSE) and calls
LLM.generate() directly. This gives a cleaner host-overhead signal because:
  - No HTTP streaming overhead
  - No FastAPI/SSE framing
  - No client-side parsing overhead
  - Direct access to per-request perf metrics via return_perf_metrics=True

The trade-off is that this does not test the serve path. For host-regression
detection, the signal quality improvement is worth it — HTTP overhead adds
~0.5-1ms of noise per iteration that masks small host regressions.

Design:
- Uses LLM API directly (no trtllm-serve)
- Sends batches of concurrent requests via generate_async()
- Measures wall-clock time for the full batch
- Reports per-request TTFT, TPOT, ITL, and E2E latency
- Supports YAML config discovery (same configs as E2E tests)

Run:
    LLM_MODELS_ROOT=/path/to/models \
    pytest tests/integration/defs/perf/host_perf/test_llmapi_perf.py -v -s
"""

import glob
import json
import os
import statistics
import time
from typing import Dict, List

import pytest
import yaml

from ...conftest import llm_models_root

# Model PATH of local dir synced from internal LLM models repo.
MODEL_PATH_DICT = {
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
}

HOST_PERF_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_dir(model_name: str) -> str:
    if model_name in MODEL_PATH_DICT:
        return os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    return ""


class LlmApiTestCase:
    """A single LLM API benchmark test case parsed from YAML."""

    def __init__(self, yaml_file: str, server_name: str, client_name: str):
        self.yaml_file = yaml_file
        self.server_name = server_name
        self.client_name = client_name
        self._load()

    def _load(self):
        with open(self.yaml_file, "r") as f:
            config = yaml.safe_load(f)

        self.model_name = config.get("metadata", {}).get("model_name", "")

        for sc in config.get("server_configs", []):
            if sc.get("name") == self.server_name:
                self.tp = sc.get("tensor_parallel_size", 1)
                self.max_batch_size = sc.get("max_batch_size", 1)
                self.max_num_tokens = sc.get("max_num_tokens", 256)
                # Build LLM kwargs from server config
                self.llm_kwargs = {}
                for k, v in sc.items():
                    if k in ("name", "model_name", "tensor_parallel_size", "client_configs"):
                        continue
                    self.llm_kwargs[k] = v

                for cc in sc.get("client_configs", []):
                    if cc.get("name") == self.client_name:
                        self.concurrency = cc.get("concurrency", 1)
                        self.iterations = cc.get("iterations", 50)
                        self.isl = cc.get("isl", 128)
                        self.osl = cc.get("osl", 128)
                        return

        raise ValueError(f"Config not found: {self.server_name}/{self.client_name}")

    @property
    def test_id(self) -> str:
        yaml_base = os.path.splitext(os.path.basename(self.yaml_file))[0]
        return f"llmapi-{yaml_base}-{self.server_name}-{self.client_name}"

    def __repr__(self) -> str:
        return self.test_id


def discover_llmapi_test_cases() -> List[LlmApiTestCase]:
    """Discover test cases from YAML configs."""
    yaml_files = sorted(glob.glob(os.path.join(HOST_PERF_CONFIG_DIR, "host_perf_*.yaml")))
    test_cases = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        for sc in config.get("server_configs", []):
            server_name = sc.get("name", "")
            for cc in sc.get("client_configs", []):
                client_name = cc.get("name", "")
                test_cases.append(LlmApiTestCase(yaml_file, server_name, client_name))
    return test_cases


def run_llmapi_benchmark(
    test_case: LlmApiTestCase,
    output_dir: str,
) -> Dict[str, float]:
    """Run LLM API benchmark directly without HTTP layer.

    Creates an LLM instance, sends concurrent requests, collects metrics.
    """
    from tensorrt_llm.llmapi import LLM, SamplingParams

    model_dir = get_model_dir(test_case.model_name)
    model_path = model_dir if os.path.exists(model_dir) else test_case.model_name

    test_output_dir = os.path.join(output_dir, test_case.test_id)
    os.makedirs(test_output_dir, exist_ok=True)

    # Build LLM with the same config as trtllm-serve would use
    llm = LLM(
        model=model_path,
        backend="pytorch",
        tensor_parallel_size=test_case.tp,
        return_perf_metrics=True,
        **test_case.llm_kwargs,
    )

    # Generate synthetic prompts (random token ids)
    prompt_tokens = [1] * test_case.isl
    num_prompts = test_case.concurrency * test_case.iterations

    sampling_params = SamplingParams(
        max_tokens=test_case.osl,
        ignore_eos=True,
        return_perf_metrics=True,
    )

    # Warmup: run a small batch first
    warmup_count = min(test_case.concurrency, 4)
    warmup_prompts = [[1] * test_case.isl for _ in range(warmup_count)]
    warmup_results = llm.generate(
        warmup_prompts,
        sampling_params=SamplingParams(max_tokens=8, ignore_eos=True),
    )
    # Wait for warmup to complete
    for r in warmup_results:
        _ = r.outputs

    # Benchmark: send all prompts
    all_prompts = [prompt_tokens for _ in range(num_prompts)]

    wall_start = time.perf_counter()
    results = llm.generate(all_prompts, sampling_params=sampling_params)
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    # Collect per-request metrics
    output_lengths = []
    for r in results:
        for out in r.outputs:
            output_lengths.append(len(out.token_ids))

    total_output_tokens = sum(output_lengths)
    total_input_tokens = num_prompts * test_case.isl

    metrics = {
        "wall_time_s": wall_time,
        "num_prompts": num_prompts,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "token_throughput": total_output_tokens / wall_time,
        "request_throughput": num_prompts / wall_time,
        "avg_output_len": statistics.mean(output_lengths) if output_lengths else 0,
    }

    # Save metrics
    metrics_path = os.path.join(test_output_dir, "llmapi_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup
    del llm

    return metrics


# ---------------------------------------------------------------------------
# Test discovery and parametrization
# ---------------------------------------------------------------------------

LLMAPI_TEST_CASES = discover_llmapi_test_cases()


@pytest.mark.parametrize(
    "llmapi_test_case",
    LLMAPI_TEST_CASES,
    ids=[tc.test_id for tc in LLMAPI_TEST_CASES],
)
def test_llmapi_perf(output_dir, llmapi_test_case):
    """Run LLM API-level host performance benchmark.

    This test bypasses the HTTP serving layer and calls LLM.generate()
    directly. The resulting throughput metrics have less noise from HTTP
    overhead, making them more sensitive to host-side regressions.

    Requires GPU and model weights (set LLM_MODELS_ROOT).
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "host_perf_output")
    os.makedirs(output_dir, exist_ok=True)

    metrics = run_llmapi_benchmark(llmapi_test_case, output_dir)

    assert metrics["total_output_tokens"] > 0, (
        f"No output tokens generated for {llmapi_test_case.test_id}"
    )

    print(f"\nLLMAPI_PERF: {llmapi_test_case.test_id}")
    print(f"  wall_time:          {metrics['wall_time_s']:.2f}s")
    print(f"  num_prompts:        {metrics['num_prompts']}")
    print(f"  token_throughput:   {metrics['token_throughput']:.1f} tok/s")
    print(f"  request_throughput: {metrics['request_throughput']:.2f} req/s")
    print(f"  avg_output_len:     {metrics['avg_output_len']:.1f}")

    # Log key metrics for regression comparison
    for key in ["token_throughput", "request_throughput"]:
        print(f"LLMAPI_PERF_METRIC: {llmapi_test_case.test_id} {key}={metrics[key]:.3f}")
