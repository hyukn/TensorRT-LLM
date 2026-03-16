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
"""Host performance regression tests for TensorRT-LLM.

These tests use workload configurations where host (CPU) overhead is dominant
over GPU work, so that host performance regressions directly show up in
standard metrics (ITL, TPOT, throughput).

Design:
- Small models (DeepSeek-V3-Lite, Llama-3.1-8B) on 1 GPU
- Small batch sizes (1, 8) — GPU not saturated
- Short sequences (ISL=128, OSL=128-256) — high iteration rate
- Decode-heavy — many scheduling/sampling cycles per request

The tests follow the same serve+benchmark pattern as test_perf_sanity.py:
  1. Read YAML config
  2. Start trtllm-serve with server config
  3. Run benchmark_serving client
  4. Parse standard metrics from output
  5. Report structured results
"""

import copy
import glob
import json
import math
import os
import re
import statistics
import subprocess
from typing import Dict, List, Optional

import pytest
import yaml

from defs.trt_test_alternative import print_info
from tensorrt_llm._utils import get_free_port

from ...conftest import llm_models_root

# Model PATH of local dir synced from internal LLM models repo.
# Must match the names used in test_perf.py MODEL_PATH_DICT.
MODEL_PATH_DICT = {
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
}

# Regex patterns for parsing benchmark output metrics.
# These match the output format of tensorrt_llm.serve.scripts.benchmark_serving.
METRIC_PATTERNS = {
    "mean_itl": re.compile(r"Mean ITL \(ms\):\s+(-?[\d\.]+)"),
    "median_itl": re.compile(r"Median ITL \(ms\):\s+(-?[\d\.]+)"),
    "p99_itl": re.compile(r"P99 ITL \(ms\):\s+(-?[\d\.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT \(ms\):\s+(-?[\d\.]+)"),
    "median_tpot": re.compile(r"Median TPOT \(ms\):\s+(-?[\d\.]+)"),
    "p99_tpot": re.compile(r"P99 TPOT \(ms\):\s+(-?[\d\.]+)"),
    "mean_ttft": re.compile(r"Mean TTFT \(ms\):\s+(-?[\d\.]+)"),
    "median_ttft": re.compile(r"Median TTFT \(ms\):\s+(-?[\d\.]+)"),
    "p99_ttft": re.compile(r"P99 TTFT \(ms\):\s+(-?[\d\.]+)"),
    "mean_e2el": re.compile(r"Mean E2EL \(ms\):\s+(-?[\d\.]+)"),
    "seq_throughput": re.compile(r"Request throughput \(req/s\):\s+(-?[\d\.]+)"),
    "token_throughput": re.compile(r"Output token throughput \(tok/s\):\s+(-?[\d\.]+)"),
}

# Directory containing the YAML config files for host perf tests.
HOST_PERF_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Default server startup timeout in seconds.
DEFAULT_SERVER_TIMEOUT = 600


def get_model_dir(model_name: str) -> str:
    """Get model directory path from model name."""
    if model_name in MODEL_PATH_DICT:
        return os.path.join(llm_models_root(), MODEL_PATH_DICT[model_name])
    return ""


def wait_for_server_ready(
    url: str,
    timeout: int = DEFAULT_SERVER_TIMEOUT,
    check_files: Optional[List[str]] = None,
    server_proc: Optional[subprocess.Popen] = None,
):
    """Wait for a server health endpoint to become ready.

    Polls the health endpoint with retries until timeout.
    """
    from test_common.http_utils import wait_for_endpoint_ready

    wait_for_endpoint_ready(
        url, timeout=timeout, check_files=check_files or [], server_proc=server_proc
    )


def parse_metrics(output: str) -> Dict[str, float]:
    """Parse benchmark output and extract metrics."""
    metrics = {}
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(output)
        if match:
            metrics[metric_name] = float(match.group(1))
    return metrics


class HostPerfServerConfig:
    """Configuration for a trtllm-serve server instance."""

    def __init__(self, server_config_data: dict, model_name: str):
        self.name = server_config_data.get("name", "")
        self.model_name = server_config_data.get("model_name", model_name)
        self.tp = server_config_data.get("tensor_parallel_size", 1)
        self.max_batch_size = server_config_data.get("max_batch_size", 1)
        self.max_num_tokens = server_config_data.get("max_num_tokens", 256)
        self.extra_llm_api_config_data = {
            k: v
            for k, v in server_config_data.items()
            if k
            not in (
                "name",
                "model_name",
                "tensor_parallel_size",
                "concurrency",
                "gpus_per_node",
                "client_configs",
            )
        }

    def generate_extra_llm_api_config(self) -> str:
        """Generate extra-llm-api-config.yml content."""
        return yaml.dump(self.extra_llm_api_config_data, default_flow_style=False, sort_keys=False)

    def to_cmd(self, config_path: str) -> List[str]:
        """Generate trtllm-serve command."""
        model_dir = get_model_dir(self.model_name)
        model_path = model_dir if os.path.exists(model_dir) else self.model_name
        return [
            "trtllm-serve",
            model_path,
            "--backend",
            "pytorch",
            "--tp_size",
            str(self.tp),
            "--config",
            config_path,
        ]


class HostPerfClientConfig:
    """Configuration for a benchmark client."""

    def __init__(self, client_config_data: dict, model_name: str):
        self.name = client_config_data.get("name", "")
        self.model_name = model_name
        self.concurrency = client_config_data.get("concurrency", 1)
        self.iterations = client_config_data.get("iterations", 50)
        self.isl = client_config_data.get("isl", 128)
        self.osl = client_config_data.get("osl", 128)
        self.backend = client_config_data.get("backend", "openai")
        self.streaming = client_config_data.get("streaming", True)

    def to_cmd(self) -> List[str]:
        """Generate benchmark command."""
        model_dir = get_model_dir(self.model_name)
        model_path = model_dir if os.path.exists(model_dir) else self.model_name
        num_prompts = self.concurrency * self.iterations
        cmd = [
            "python",
            "-m",
            "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model",
            model_path,
            "--tokenizer",
            model_path,
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(self.concurrency),
            "--ignore-eos",
            "--no-test-input",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--dataset-name",
            "random",
            "--random-ids",
            "--tokenize-on-client",
            "--random-input-len",
            str(self.isl),
            "--random-output-len",
            str(self.osl),
            "--random-range-ratio",
            "0.0",
            "--trust-remote-code",
        ]
        if self.backend:
            cmd.extend(["--backend", self.backend])
        if not self.streaming:
            cmd.append("--non-streaming")
        return cmd


class HostPerfTestCase:
    """A single host perf test case: one server config + one client config."""

    def __init__(self, yaml_file: str, server_name: str, client_name: str):
        self.yaml_file = yaml_file
        self.server_name = server_name
        self.client_name = client_name

    @property
    def test_id(self) -> str:
        yaml_base = os.path.splitext(os.path.basename(self.yaml_file))[0]
        return f"{yaml_base}-{self.server_name}-{self.client_name}"

    def __repr__(self) -> str:
        return self.test_id


def discover_test_cases() -> List[HostPerfTestCase]:
    """Discover all host perf test cases from YAML config files."""
    yaml_files = sorted(glob.glob(os.path.join(HOST_PERF_CONFIG_DIR, "host_perf_*.yaml")))
    test_cases = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        for server_config_data in config.get("server_configs", []):
            server_name = server_config_data.get("name", "")
            for client_config_data in server_config_data.get("client_configs", []):
                client_name = client_config_data.get("name", "")
                test_cases.append(HostPerfTestCase(yaml_file, server_name, client_name))
    return test_cases


def load_configs(
    test_case: HostPerfTestCase,
) -> tuple[HostPerfServerConfig, HostPerfClientConfig]:
    """Load server and client configs for a test case."""
    with open(test_case.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("metadata", {}).get("model_name", "")

    for server_config_data in config.get("server_configs", []):
        if server_config_data.get("name") == test_case.server_name:
            server_config = HostPerfServerConfig(server_config_data, model_name)
            for client_config_data in server_config_data.get("client_configs", []):
                if client_config_data.get("name") == test_case.client_name:
                    client_config = HostPerfClientConfig(client_config_data, model_name)
                    return server_config, client_config

    raise ValueError(
        f"Could not find server={test_case.server_name}, "
        f"client={test_case.client_name} in {test_case.yaml_file}"
    )


def run_host_perf_test(
    test_case: HostPerfTestCase,
    output_dir: str,
) -> Dict[str, float]:
    """Run a single host perf test: start server, run benchmark, parse metrics.

    Returns dict of metric_name -> value.
    """
    server_config, client_config = load_configs(test_case)

    test_output_dir = os.path.join(output_dir, test_case.test_id)
    os.makedirs(test_output_dir, exist_ok=True)

    # Write extra-llm-api-config
    config_path = os.path.join(test_output_dir, "extra-llm-api-config.yml")
    with open(config_path, "w") as f:
        f.write(server_config.generate_extra_llm_api_config())

    # Start server
    server_port = get_free_port()
    server_hostname = "localhost"
    server_cmd = server_config.to_cmd(config_path) + [
        "--host",
        server_hostname,
        "--port",
        str(server_port),
    ]

    server_log_path = os.path.join(test_output_dir, "trtllm-serve.log")
    server_proc = None
    metrics = {}

    try:
        print_info(f"[host_perf] Starting server: {' '.join(server_cmd)}")
        with open(server_log_path, "w") as server_log:
            server_proc = subprocess.Popen(
                server_cmd,
                env=copy.deepcopy(os.environ),
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )

            wait_for_server_ready(
                f"http://{server_hostname}:{server_port}/health",
                timeout=DEFAULT_SERVER_TIMEOUT,
                check_files=[server_log_path],
                server_proc=server_proc,
            )

        # Run benchmark client
        client_cmd = client_config.to_cmd() + [
            "--host",
            server_hostname,
            "--port",
            str(server_port),
        ]

        print_info(f"[host_perf] Running benchmark: {' '.join(client_cmd)}")
        client_log_path = os.path.join(test_output_dir, "benchmark-client.log")

        output = subprocess.check_output(
            client_cmd,
            stderr=subprocess.STDOUT,
            env=copy.deepcopy(os.environ),
        ).decode()

        with open(client_log_path, "w") as f:
            f.write(output)

        # Parse metrics
        metrics = parse_metrics(output)
        print_info(f"[host_perf] Test {test_case.test_id} metrics: {metrics}")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    return metrics


# ---------------------------------------------------------------------------
# Test discovery and parametrization
# ---------------------------------------------------------------------------

HOST_PERF_TEST_CASES = discover_test_cases()


@pytest.mark.parametrize(
    "host_perf_test_case",
    HOST_PERF_TEST_CASES,
    ids=[tc.test_id for tc in HOST_PERF_TEST_CASES],
)
def test_host_perf(output_dir, host_perf_test_case):
    """Run a host performance regression test.

    The test starts a trtllm-serve instance with a host-overhead-dominant
    configuration, runs a benchmark client, and collects standard metrics.
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "host_perf_output")
    os.makedirs(output_dir, exist_ok=True)

    metrics = run_host_perf_test(host_perf_test_case, output_dir)

    # Verify we got meaningful metrics
    assert len(metrics) > 0, (
        f"No metrics collected for {host_perf_test_case.test_id}. "
        f"Check server and client logs in {output_dir}."
    )

    # Log key host-perf-sensitive metrics for regression comparison
    key_metrics = ["mean_itl", "mean_tpot", "p99_itl", "token_throughput"]
    for metric_name in key_metrics:
        if metric_name in metrics:
            print_info(
                f"HOST_PERF_METRIC: {host_perf_test_case.test_id} "
                f"{metric_name}={metrics[metric_name]}"
            )


# ---------------------------------------------------------------------------
# Multi-run statistical comparison
# ---------------------------------------------------------------------------

# Number of runs for statistical comparison. 3 is the minimum for Welch's t-test;
# 5 gives better confidence. Configurable via HOST_PERF_NUM_RUNS env var.
DEFAULT_NUM_RUNS = 3


def welch_t_test(sample_a: List[float], sample_b: List[float]) -> Dict[str, float]:
    """Perform Welch's t-test for two independent samples.

    Returns dict with t_statistic, degrees_of_freedom, and p_value.
    Uses the Welch–Satterthwaite approximation for degrees of freedom.
    No scipy dependency — computes the t-distribution p-value via
    the regularized incomplete beta function approximation.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return {"t_statistic": 0.0, "degrees_of_freedom": 0.0, "p_value": 1.0}

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)

    se_a = var_a / n_a
    se_b = var_b / n_b
    se_total = se_a + se_b

    if se_total == 0:
        return {"t_statistic": 0.0, "degrees_of_freedom": 0.0, "p_value": 1.0}

    t_stat = (mean_a - mean_b) / math.sqrt(se_total)

    # Welch-Satterthwaite degrees of freedom
    df = (se_total**2) / (se_a**2 / (n_a - 1) + se_b**2 / (n_b - 1))

    # Approximate two-tailed p-value using the t-distribution CDF
    # For small df, use a conservative approximation
    # For df >= 30, t ~ N(0,1) approximation is acceptable
    p_value = _approx_t_pvalue(abs(t_stat), df)

    return {
        "t_statistic": t_stat,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "pct_change": ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0,
    }


def _approx_t_pvalue(t: float, df: float) -> float:
    """Approximate two-tailed p-value for t-distribution.

    Uses the approximation: p ≈ 2 * (1 - Φ(t * sqrt(df/(df-2))))
    where Φ is the standard normal CDF. Accurate for df > 4.
    For very small df, returns a conservative (larger) p-value.
    """
    if df <= 2:
        return 1.0  # Conservative: not enough data

    # Adjust t-statistic to approximate normal
    adjusted_t = t * math.sqrt(df / (df - 2)) if df > 2 else t

    # Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)
    x = adjusted_t
    if x < 0:
        x = -x
    # Using the error function approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p_const = 0.3275911

    t_val = 1.0 / (1.0 + p_const * x)
    y = 1.0 - (((((a5 * t_val + a4) * t_val) + a3) * t_val + a2) * t_val + a1) * t_val * math.exp(
        -x * x / 2.0
    )

    # Two-tailed p-value
    return 2.0 * (1.0 - y)


def run_multi_run_benchmark(
    test_case: HostPerfTestCase,
    output_dir: str,
    num_runs: int = DEFAULT_NUM_RUNS,
) -> Dict[str, List[float]]:
    """Run a benchmark multiple times, reusing the same server.

    Returns dict of metric_name -> list of values (one per run).
    The server is started once and reused across all runs to isolate
    client-side measurement variance from server startup variance.
    """
    server_config, client_config = load_configs(test_case)

    test_output_dir = os.path.join(output_dir, test_case.test_id)
    os.makedirs(test_output_dir, exist_ok=True)

    # Write extra-llm-api-config
    config_path = os.path.join(test_output_dir, "extra-llm-api-config.yml")
    with open(config_path, "w") as f:
        f.write(server_config.generate_extra_llm_api_config())

    # Start server once
    server_port = get_free_port()
    server_hostname = "localhost"
    server_cmd = server_config.to_cmd(config_path) + [
        "--host",
        server_hostname,
        "--port",
        str(server_port),
    ]

    server_log_path = os.path.join(test_output_dir, "trtllm-serve.log")
    server_proc = None
    all_metrics: Dict[str, List[float]] = {}

    try:
        print_info(f"[host_perf_multi] Starting server for {num_runs} runs: {' '.join(server_cmd)}")
        with open(server_log_path, "w") as server_log:
            server_proc = subprocess.Popen(
                server_cmd,
                env=copy.deepcopy(os.environ),
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )

            wait_for_server_ready(
                f"http://{server_hostname}:{server_port}/health",
                timeout=DEFAULT_SERVER_TIMEOUT,
                check_files=[server_log_path],
                server_proc=server_proc,
            )

        # Run benchmark num_runs times
        for run_idx in range(num_runs):
            client_cmd = client_config.to_cmd() + [
                "--host",
                server_hostname,
                "--port",
                str(server_port),
            ]

            print_info(f"[host_perf_multi] Run {run_idx + 1}/{num_runs} for {test_case.test_id}")
            client_log_path = os.path.join(test_output_dir, f"benchmark-client-run{run_idx}.log")

            output = subprocess.check_output(
                client_cmd,
                stderr=subprocess.STDOUT,
                env=copy.deepcopy(os.environ),
            ).decode()

            with open(client_log_path, "w") as f:
                f.write(output)

            metrics = parse_metrics(output)
            for metric_name, value in metrics.items():
                all_metrics.setdefault(metric_name, []).append(value)

            print_info(f"[host_perf_multi] Run {run_idx + 1} metrics: {metrics}")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    return all_metrics


def report_multi_run_stats(test_id: str, all_metrics: Dict[str, List[float]], output_dir: str):
    """Report statistical summary of multi-run results."""
    summary = {}
    key_metrics = ["mean_itl", "mean_tpot", "p99_itl", "token_throughput"]

    print(f"\n{'=' * 60}")
    print(f"MULTI-RUN SUMMARY: {test_id} ({len(next(iter(all_metrics.values()), []))} runs)")
    print(f"{'=' * 60}")

    for metric_name in key_metrics:
        if metric_name not in all_metrics:
            continue

        values = all_metrics[metric_name]
        if len(values) < 2:
            continue

        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        cv = (stdev / mean * 100) if mean != 0 else 0  # coefficient of variation

        summary[metric_name] = {
            "mean": mean,
            "stdev": stdev,
            "cv_pct": cv,
            "values": values,
            "n": len(values),
        }

        print(f"  {metric_name}:")
        print(f"    values: {[f'{v:.2f}' for v in values]}")
        print(f"    mean:   {mean:.3f}")
        print(f"    stdev:  {stdev:.3f}")
        print(f"    CV:     {cv:.1f}%")

    print(f"{'=' * 60}\n")

    # Save summary to JSON for downstream comparison
    summary_path = os.path.join(output_dir, test_id, "multi_run_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


@pytest.mark.parametrize(
    "host_perf_test_case",
    HOST_PERF_TEST_CASES,
    ids=[tc.test_id for tc in HOST_PERF_TEST_CASES],
)
def test_host_perf_multi_run(output_dir, host_perf_test_case):
    """Run a host perf test multiple times and report statistical summary.

    By running the benchmark 3-5 times against the same server, we can
    compute mean and stdev for each metric. This enables:
    1. Confidence intervals for regression detection
    2. Welch's t-test comparison between baseline and candidate
    3. Detection of ~5% regressions (vs ~15% with single run)

    The number of runs is configurable via HOST_PERF_NUM_RUNS env var.
    Set to 0 to skip this test (it's slow: ~3x the single-run test time).
    """
    num_runs = int(os.environ.get("HOST_PERF_NUM_RUNS", DEFAULT_NUM_RUNS))
    if num_runs <= 0:
        pytest.skip("HOST_PERF_NUM_RUNS=0, skipping multi-run test")

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "host_perf_output")
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = run_multi_run_benchmark(host_perf_test_case, output_dir, num_runs)

    assert len(all_metrics) > 0, (
        f"No metrics collected for {host_perf_test_case.test_id}. "
        f"Check server and client logs in {output_dir}."
    )

    summary = report_multi_run_stats(host_perf_test_case.test_id, all_metrics, output_dir)

    # Verify coefficient of variation is reasonable (< 20%)
    # High CV indicates measurement instability
    for metric_name, stats in summary.items():
        if stats["cv_pct"] > 20:
            print_info(
                f"WARNING: High CV ({stats['cv_pct']:.1f}%) for "
                f"{host_perf_test_case.test_id}/{metric_name}. "
                f"Results may not be reliable for regression detection."
            )
