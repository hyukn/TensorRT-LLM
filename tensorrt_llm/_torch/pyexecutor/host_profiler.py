# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Host-side profiler for analyzing CPU overhead in the PyExecutor.

This module provides a HostProfiler class that wraps line_profiler to measure
line-by-line execution time of critical functions in the executor worker thread.

Usage:
    Set environment variable TLLM_LINE_PROFILER_PATH to enable:
        TLLM_LINE_PROFILER_PATH=./lp_results.txt pytest ...

    Or use programmatically:
        profiler = HostProfiler(output_path="./results.txt")
        profiler.start()
        # ... run code ...
        profiler.stop()
"""

import importlib
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from tensorrt_llm.logger import logger

# Environment variable to enable line_profiler output path.
LINE_PROFILER_PATH_ENV_VAR = "TLLM_LINE_PROFILER_PATH"

# Environment variable to specify additional functions to profile (comma-separated).
# Format: "module.Class.method,module.Class.method2,..."
LINE_PROFILER_FUNCTIONS_ENV_VAR = "TLLM_LINE_PROFILER_FUNCTIONS"


@dataclass
class ProfileTarget:
    """Represents a function to be profiled."""

    module_path: str
    class_name: str
    method_name: str

    @property
    def full_path(self) -> str:
        return f"{self.module_path}.{self.class_name}.{self.method_name}"

    def resolve(self) -> Optional[Callable]:
        """Resolve the target to an actual function object.

        Returns:
            The unwrapped function, or None if resolution fails.
        """
        try:
            module = importlib.import_module(self.module_path)
            cls = getattr(module, self.class_name)
            method = getattr(cls, self.method_name)
            return inspect.unwrap(method)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to resolve profile target {self.full_path}: {e}")
            return None


# Default functions to profile for host overhead analysis
# Hierarchical config: {module_path: {class_name: [method_names]}}
_PYEXEC = "tensorrt_llm._torch.pyexecutor"
_DEFAULT_PROFILE_CONFIG: Dict[str, Dict[str, List[str]]] = {
    f"{_PYEXEC}.py_executor": {
        "PyExecutor": [
            "_prepare_and_schedule_batch",
            "_schedule",
            "_forward_step",
            "_sample_async",
            "_update_requests",
            "_update_request_states",
            "_fetch_and_activate_new_requests",
            "_handle_responses",
            "_handle_canceled_requests",
            "_enqueue_responses",
        ],
    },
    f"{_PYEXEC}.sampler": {
        "TorchSampler": [
            "sample_async",
            "update_requests",
            "_process_requests",
            "_write_finish_reasons",
            "_prepare_beam_search",
            "_fast_greedy_sample_kernel",
            "_select_generated_logits",
            "_sample_batched_by_strategy",
            "_unbatch_sampling_results",
            "_handle_log_probs",
        ],
    },
    f"{_PYEXEC}.resource_manager": {
        "ResourceManager": ["prepare_resources", "update_resources", "free_resources"],
        "KVCacheManager": ["prepare_resources", "update_resources"],
    },
    f"{_PYEXEC}.scheduler": {
        "RequestScheduler": ["schedule_request"],
    },
    f"{_PYEXEC}.executor_request_queue": {
        "ExecutorRequestQueue": [
            "_fetch_new_requests_attention_tp",
            "_fetch_new_requests_attention_dp",
            "_fetch_and_process_requests",
            "_merge_requests",
            "fetch_new_requests",
        ],
    },
}


def _expand_profile_config(config: Dict[str, Dict[str, List[str]]]) -> List[ProfileTarget]:
    """Expand hierarchical config into a flat list of ProfileTarget objects."""
    targets = []
    for module_path, classes in config.items():
        for class_name, methods in classes.items():
            for method_name in methods:
                targets.append(ProfileTarget(module_path, class_name, method_name))
    return targets


DEFAULT_PROFILE_TARGETS: List[ProfileTarget] = _expand_profile_config(_DEFAULT_PROFILE_CONFIG)


class HostProfiler:
    """Host-side profiler for measuring CPU overhead in the executor.

    This class wraps line_profiler to provide line-by-line timing analysis
    of critical functions in the PyExecutor worker thread.

    Attributes:
        output_path: Path to save profiling results.
        targets: List of ProfileTarget objects specifying functions to profile.
        enabled: Whether profiling is currently active.

    Example:
        >>> profiler = HostProfiler(output_path="./results.txt")
        >>> profiler.add_target(
        ...     ProfileTarget(
        ...         module_path="my_module",
        ...         class_name="MyClass",
        ...         method_name="my_method",
        ...     )
        ... )
        >>> profiler.start()
        >>> # ... run code ...
        >>> profiler.stop()
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        targets: Optional[List[ProfileTarget]] = None,
        use_defaults: bool = True,
    ):
        """Initialize the host profiler.

        Args:
            output_path: Path to save results. If None, uses env var TLLM_LINE_PROFILER_PATH.
            targets: List of ProfileTarget objects. If None and use_defaults=True,
                     uses DEFAULT_PROFILE_TARGETS.
            use_defaults: Whether to include default profile targets.
        """
        self.output_path = output_path or os.environ.get(LINE_PROFILER_PATH_ENV_VAR)
        self.targets: List[ProfileTarget] = []
        self._line_profiler = None
        self._enabled = False

        # Add default targets if requested
        if use_defaults:
            self.targets.extend(DEFAULT_PROFILE_TARGETS)

        # Add custom targets
        if targets:
            self.targets.extend(targets)

        # Parse additional targets from environment variable
        self._parse_env_targets()

    def _parse_env_targets(self) -> None:
        """Parse additional profile targets from environment variable."""
        env_funcs = os.environ.get(LINE_PROFILER_FUNCTIONS_ENV_VAR, "")
        if not env_funcs:
            return

        for func_path in env_funcs.split(","):
            func_path = func_path.strip()
            if not func_path:
                continue

            parts = func_path.rsplit(".", 2)
            if len(parts) < 3:
                logger.warning(
                    f"Invalid function path '{func_path}'. Expected format: module.Class.method"
                )
                continue

            # Handle nested module paths
            method_name = parts[-1]
            class_name = parts[-2]
            module_path = ".".join(parts[:-2])

            self.targets.append(
                ProfileTarget(
                    module_path=module_path,
                    class_name=class_name,
                    method_name=method_name,
                    description=f"User-specified via {LINE_PROFILER_FUNCTIONS_ENV_VAR}",
                )
            )

    def add_target(self, target: ProfileTarget) -> "HostProfiler":
        """Add a profile target.

        Args:
            target: The ProfileTarget to add.

        Returns:
            Self for chaining.
        """
        self.targets.append(target)
        return self

    def add_function(
        self,
        module_path: str,
        class_name: str,
        method_name: str,
        description: str = "",
    ) -> "HostProfiler":
        """Add a function to profile by specifying its path.

        Args:
            module_path: The module path (e.g., "tensorrt_llm._torch.pyexecutor.sampler")
            class_name: The class name (e.g., "TorchSampler")
            method_name: The method name (e.g., "_process_requests")
            description: Optional description

        Returns:
            Self for chaining.
        """
        return self.add_target(
            ProfileTarget(
                module_path=module_path,
                class_name=class_name,
                method_name=method_name,
                description=description,
            )
        )

    @property
    def is_available(self) -> bool:
        """Check if line_profiler is available."""
        return True

    @property
    def should_profile(self) -> bool:
        """Check if profiling should be enabled (output path is set)."""
        return self.output_path is not None

    @property
    def enabled(self) -> bool:
        """Check if profiling is currently active."""
        return self._enabled

    def start(self) -> bool:
        """Start profiling.

        Returns:
            True if profiling started successfully, False otherwise.
        """
        if not self.should_profile:
            logger.debug("Line profiler not enabled (no output path specified)")
            return False

        if not self.is_available:
            logger.warning("line_profiler not installed. Install with: pip install line_profiler")
            return False

        if self._enabled:
            logger.warning("Line profiler already started")
            return True

        try:
            from line_profiler import LineProfiler

            self._line_profiler = LineProfiler()

            # Add all target functions
            resolved_count = 0
            for target in self.targets:
                func = target.resolve()
                if func is not None:
                    self._line_profiler.add_function(func)
                    resolved_count += 1
                    logger.debug(f"Added profile target: {target.full_path}")

            if resolved_count == 0:
                logger.warning("No profile targets could be resolved")
                self._line_profiler = None
                return False

            logger.info(
                f"Line profiler enabled with {resolved_count}/{len(self.targets)} targets. "
                f"Results will be saved to: {self.output_path}"
            )

            self._line_profiler.enable()
            self._enabled = True
            return True

        except Exception as e:
            logger.error(f"Failed to start line profiler: {e}")
            self._line_profiler = None
            return False

    def stop(self) -> bool:
        """Stop profiling and save results.

        Returns:
            True if results were saved successfully, False otherwise.
        """
        if not self._enabled or self._line_profiler is None:
            return False

        try:
            self._line_profiler.disable()
            self._enabled = False

            # Save results
            with open(self.output_path, "w") as f:
                self._line_profiler.print_stats(stream=f)

            logger.info(f"Line profiler results saved to: {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save line profiler results: {e}")
            return False

        finally:
            self._line_profiler = None

    @contextmanager
    def profile(self):
        """Context manager for profiling.

        Usage:
            with profiler.profile():
                # ... code to profile ...
        """
        started = self.start()
        try:
            yield self
        finally:
            if started:
                self.stop()

    def get_stats_string(self) -> Optional[str]:
        """Get profiling stats as a string (without saving to file).

        Returns:
            Stats string if profiling is active, None otherwise.
        """
        if self._line_profiler is None:
            return None

        import io

        stream = io.StringIO()
        self._line_profiler.print_stats(stream=stream)
        return stream.getvalue()

    def list_targets(self) -> List[str]:
        """List all configured profile targets.

        Returns:
            List of target paths.
        """
        return [t.full_path for t in self.targets]


# Global profiler instance for use in worker thread
_global_profiler: Optional[HostProfiler] = None


def get_global_profiler() -> Optional[HostProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def set_global_profiler(profiler: Optional[HostProfiler]) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


@contextmanager
def host_profiler_context(enable: bool = True, output_path: Optional[str] = None):
    """Context manager for host profiling in the worker thread.

    This is the main entry point for profiling in PyExecutor._event_loop_wrapper.

    Args:
        output_path: Path to save results. If None, uses env var.

    Usage:
        with host_profiler_context():
            # ... event loop code ...
    """
    if not enable:
        yield None
        return

    profiler = HostProfiler(output_path=output_path)
    set_global_profiler(profiler)

    started = profiler.start()
    try:
        yield profiler
    finally:
        if started:
            profiler.stop()
        set_global_profiler(None)
