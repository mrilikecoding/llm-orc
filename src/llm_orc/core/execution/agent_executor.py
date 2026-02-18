"""Agent execution coordination with parallel processing and resource management."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.adaptive_resource_manager import (
    SystemResourceMonitor,
)


class AgentExecutor:
    """Handles parallel execution of agents with resource management."""

    def __init__(
        self,
        performance_config: dict[str, Any],
        emit_performance_event: Callable[[str, dict[str, Any]], None],
        resolve_model_profile_to_config: Callable[..., Any],
        execute_agent_with_timeout: Callable[..., Any],
        get_agent_input: Callable[..., Any],
    ) -> None:
        """Initialize the agent executor.

        Args:
            performance_config: Performance configuration settings
            emit_performance_event: Function to emit performance events
            resolve_model_profile_to_config: Function to resolve model profiles
            execute_agent_with_timeout: Function to execute individual agents
            get_agent_input: Function to get input for specific agents
        """
        self._performance_config = performance_config
        self._emit_performance_event = emit_performance_event
        self._resolve_model_profile_to_config = resolve_model_profile_to_config
        self._execute_agent_with_timeout = execute_agent_with_timeout
        self._get_agent_input = get_agent_input

        # Initialize resource monitoring for performance feedback
        self._init_monitoring()

        # Track resource management statistics for this execution
        self.adaptive_stats: dict[str, Any] = {
            "adaptive_used": False,  # No longer using adaptive calculations
            "management_type": "user_configured",
            "concurrency_decisions": [],
            "resource_metrics": [],
        }

        # Track per-phase metrics for detailed performance analysis
        self._phase_metrics: list[dict[str, Any]] = []

    async def execute_agents_parallel(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Execute a list of agents in parallel with resource management.

        Args:
            agents: List of agent configurations to execute
            input_data: Either a string for uniform input, or a dict mapping
                       agent names to their specific enhanced input
            config: Ensemble configuration
            results_dict: Dictionary to store results
            agent_usage: Dictionary to store usage metrics
        """
        # Debug: Track that parallel execution started
        if not agents:
            return

        # Get concurrency limit from user configuration
        max_concurrent = await self._get_concurrency_limit(len(agents))

        # Start monitoring for performance feedback
        await self.monitor.start_execution_monitoring()

        # Record concurrency decision for metrics
        self.adaptive_stats["concurrency_decisions"].append(
            {
                "agent_count": len(agents),
                "configured_limit": max_concurrent,
                "management_type": "user_configured",
            }
        )

        self._emit_performance_event(
            "using_configured_concurrency",
            {
                "total_agents": len(agents),
                "concurrency_limit": max_concurrent,
                "management_type": "user_configured",
            },
        )

        # For small ensembles, run all in parallel
        # For large ensembles, use semaphore to limit concurrent execution
        if len(agents) <= max_concurrent:
            await self.execute_agents_unlimited(
                agents, input_data, config, results_dict, agent_usage
            )
        else:
            await self.execute_agents_with_semaphore(
                agents, input_data, config, results_dict, agent_usage, max_concurrent
            )

        # Collect execution metrics for performance feedback
        try:
            # Stop monitoring and get aggregated execution metrics
            execution_metrics = await self.monitor.stop_execution_monitoring()

            # Add detailed execution metrics to stats
            self.adaptive_stats["execution_metrics"] = execution_metrics

            # Add summary resource metrics entry for visualization
            self.adaptive_stats["resource_metrics"].append(
                {
                    "peak_cpu": execution_metrics["peak_cpu"],
                    "avg_cpu": execution_metrics["avg_cpu"],
                    "peak_memory": execution_metrics["peak_memory"],
                    "avg_memory": execution_metrics["avg_memory"],
                    "sample_count": execution_metrics["sample_count"],
                    "circuit_breaker_state": "N/A",
                    "measurement_point": "execution_summary",
                }
            )
        except Exception as e:
            # Final fallback - always provide SOME data for research
            print(f"⚠️  WARNING: All monitoring failed: {e}")
            self.adaptive_stats["execution_metrics"] = {
                "peak_cpu": 0.0,
                "avg_cpu": 0.0,
                "peak_memory": 0.0,
                "avg_memory": 0.0,
                "sample_count": 0,
            }
            self.adaptive_stats["resource_metrics"].append(
                {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "circuit_breaker_state": "ERROR",
                    "measurement_point": "fallback_error",
                }
            )

    def get_effective_concurrency_limit(self, agent_count: int) -> int:
        """Get effective concurrency limit based on configuration and agent count.

        Args:
            agent_count: Number of agents to execute

        Returns:
            Effective concurrency limit
        """
        # Check performance configuration first
        configured_limit = self._performance_config.get("concurrency", {}).get(
            "max_concurrent_agents", 0
        )

        # If explicitly configured and > 0, use it
        if isinstance(configured_limit, int) and configured_limit > 0:
            return configured_limit

        # Otherwise use smart defaults based on agent count and system resources
        if agent_count <= 3:
            return agent_count  # Small ensembles: run all in parallel
        elif agent_count <= 10:
            return 5  # Medium ensembles: limit to 5 concurrent
        elif agent_count <= 20:
            return 8  # Large ensembles: limit to 8 concurrent
        else:
            return 10  # Very large ensembles: cap at 10 concurrent

    async def execute_agents_unlimited(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Execute agents without concurrency limits (for small ensembles).

        Args:
            agents: List of agent configurations to execute
            input_data: Input data for agents
            config: Ensemble configuration
            results_dict: Dictionary to store results
            agent_usage: Dictionary to store usage metrics
        """
        try:

            async def execute_agent_task(
                agent_config: dict[str, Any],
            ) -> tuple[str, Any]:
                """Execute a single agent task."""
                agent_name = agent_config["name"]
                agent_start_time = time.time()

                # Emit agent started event
                self._emit_performance_event(
                    "agent_started",
                    {"agent_name": agent_name, "timestamp": agent_start_time},
                )

                try:
                    # Resolve config and execute - all happening in parallel per agent
                    enhanced_config = await self._resolve_model_profile_to_config(
                        agent_config
                    )
                    timeout = enhanced_config.get("timeout_seconds") or (
                        self._performance_config.get("execution", {}).get(
                            "default_timeout", 60
                        )
                    )
                    # Get the appropriate input for this agent
                    agent_input = self._get_agent_input(
                        input_data, agent_config["name"]
                    )
                    result = await self._execute_agent_with_timeout(
                        agent_config, agent_input, timeout
                    )

                    # Emit agent completed event with duration
                    agent_end_time = time.time()
                    duration_ms = int((agent_end_time - agent_start_time) * 1000)
                    self._emit_performance_event(
                        "agent_completed",
                        {
                            "agent_name": agent_name,
                            "timestamp": agent_end_time,
                            "duration_ms": duration_ms,
                        },
                    )

                    return agent_name, result
                except Exception as e:
                    # Emit agent completed event with error
                    agent_end_time = time.time()
                    duration_ms = int((agent_end_time - agent_start_time) * 1000)
                    self._emit_performance_event(
                        "agent_completed",
                        {
                            "agent_name": agent_name,
                            "timestamp": agent_end_time,
                            "duration_ms": duration_ms,
                            "error": str(e),
                        },
                    )

                    # Record error in results dict and return error indicator
                    results_dict[agent_name] = {
                        "error": str(e),
                        "status": "failed",
                    }
                    return agent_name, None

            # Create tasks using create_task to ensure they start immediately
            tasks = [
                asyncio.create_task(execute_agent_task(agent_config))
                for agent_config in agents
            ]

            # Wait for all tasks to complete
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            self.process_agent_results(agent_results, results_dict, agent_usage)

        except Exception as e:
            # Fallback: if gather fails, mark all agents as failed
            for agent_config in agents:
                agent_name = agent_config["name"]
                results_dict[agent_name] = {"error": str(e), "status": "failed"}

    async def execute_agents_with_semaphore(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        config: EnsembleConfig,
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
        max_concurrent: int,
    ) -> None:
        """Execute agents with semaphore-based concurrency control.

        Args:
            agents: List of agent configurations to execute
            input_data: Input data for agents
            config: Ensemble configuration
            results_dict: Dictionary to store results
            agent_usage: Dictionary to store usage metrics
            max_concurrent: Maximum number of concurrent agents
        """
        # Create semaphore to limit concurrent execution
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_agent_with_semaphore(
            agent_config: dict[str, Any],
        ) -> tuple[str, Any]:
            """Execute agent with semaphore control."""
            async with semaphore:
                agent_name = agent_config["name"]
                agent_start_time = time.time()

                # Emit agent started event
                self._emit_performance_event(
                    "agent_started",
                    {"agent_name": agent_name, "timestamp": agent_start_time},
                )

                try:
                    # Resolve config and execute
                    enhanced_config = await self._resolve_model_profile_to_config(
                        agent_config
                    )
                    timeout = enhanced_config.get("timeout_seconds") or (
                        self._performance_config.get("execution", {}).get(
                            "default_timeout", 60
                        )
                    )
                    # Get the appropriate input for this agent
                    agent_input = self._get_agent_input(
                        input_data, agent_config["name"]
                    )
                    result = await self._execute_agent_with_timeout(
                        agent_config, agent_input, timeout
                    )

                    # Emit agent completed event with duration
                    agent_end_time = time.time()
                    duration_ms = int((agent_end_time - agent_start_time) * 1000)
                    self._emit_performance_event(
                        "agent_completed",
                        {
                            "agent_name": agent_name,
                            "timestamp": agent_end_time,
                            "duration_ms": duration_ms,
                        },
                    )

                    return agent_name, result
                except Exception as e:
                    # Emit agent completed event with error
                    agent_end_time = time.time()
                    duration_ms = int((agent_end_time - agent_start_time) * 1000)
                    self._emit_performance_event(
                        "agent_completed",
                        {
                            "agent_name": agent_name,
                            "timestamp": agent_end_time,
                            "duration_ms": duration_ms,
                            "error": str(e),
                        },
                    )

                    # Record error in results dict and return error indicator
                    results_dict[agent_name] = {
                        "error": str(e),
                        "status": "failed",
                    }
                    return agent_name, None

        try:
            # Create tasks with semaphore control
            tasks = [
                asyncio.create_task(execute_agent_with_semaphore(agent_config))
                for agent_config in agents
            ]

            # Wait for all tasks to complete
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            self.process_agent_results(agent_results, results_dict, agent_usage)

        except Exception as e:
            # Fallback: if gather fails, mark all agents as failed
            for agent_config in agents:
                agent_name = agent_config["name"]
                results_dict[agent_name] = {"error": str(e), "status": "failed"}

    def process_agent_results(
        self,
        agent_results: list[Any],
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> None:
        """Process results from agent execution.

        Args:
            agent_results: List of results from agent execution
            results_dict: Dictionary to store processed results
            agent_usage: Dictionary to store usage metrics
        """
        for execution_result in agent_results:
            if isinstance(execution_result, Exception):
                # If we can't determine agent name from exception, skip this result
                # The agent task should handle its own error recording
                continue
            else:
                # execution_result is tuple[str, Any]
                agent_name, result = execution_result
                if result is None:
                    # Error was already recorded in execute_agent_task
                    continue

                # result is tuple[str, ModelInterface | None]
                response, model_instance = result
                results_dict[agent_name] = {
                    "response": response,
                    "status": "success",
                }
                # Collect usage metrics (only for LLM agents)
                if model_instance is not None:
                    usage = model_instance.get_last_usage()
                    if usage:
                        agent_usage[agent_name] = usage

    def _init_monitoring(self) -> None:
        """Initialize resource monitoring for performance feedback."""
        # Create system resource monitor for performance feedback
        self.monitor = SystemResourceMonitor(polling_interval=0.1)

        # Keep adaptive_manager for backward compatibility with existing code
        # But simplify it to just be a monitoring container
        self.adaptive_manager = type("SimpleMonitor", (), {"monitor": self.monitor})()

    async def _get_concurrency_limit(self, agent_count: int) -> int:
        """Get concurrency limit from user configuration."""
        # Use the configured max_concurrent directly - trust the user to set
        # appropriate limits
        limit = self.get_effective_concurrency_limit(agent_count)

        # Collect current resource metrics for user feedback (not for decision making)
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent

            self.adaptive_stats["resource_metrics"].append(
                {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "circuit_breaker_state": "N/A",
                    "measurement_point": "user_configured",
                }
            )

            guidance = self._get_concurrency_guidance(
                limit, agent_count, cpu_percent, memory_percent
            )
            print(
                f"🔧 Using configured concurrency: {limit} agents "
                f"(Current system: CPU {cpu_percent:.1f}%, "
                f"Memory {memory_percent:.1f}%)"
            )
            if guidance:
                print(f"💡 {guidance}")
        except Exception:
            print(f"🔧 Using configured concurrency: {limit} agents")
            print(
                "💡 Monitor performance and adjust max_concurrent in performance "
                "config as needed"
            )

        return limit

    def _get_concurrency_guidance(
        self, limit: int, agent_count: int, cpu_percent: float, memory_percent: float
    ) -> str | None:
        """Provide helpful guidance for concurrency settings based on current
        conditions."""
        # High resource usage - suggest reducing concurrency
        if cpu_percent > 80 or memory_percent > 85:
            return (
                f"High resource usage detected - consider reducing max_concurrent "
                f"from {limit} to {max(1, limit - 1)} for better stability"
            )

        # Low resource usage with small limit - suggest increasing
        if (
            cpu_percent < 20
            and memory_percent < 50
            and limit < agent_count
            and limit < 8
        ):
            return (
                f"System has capacity - consider increasing max_concurrent "
                f"from {limit} to {min(agent_count, limit + 1)} for faster execution"
            )

        # All agents running in parallel already
        if limit >= agent_count:
            return "All agents will run in parallel - optimal for this ensemble size"

        # No specific guidance needed
        return None

    def get_adaptive_stats(self) -> dict[str, Any]:
        """Get adaptive resource management statistics for this execution."""
        # Ensure we ALWAYS have execution_metrics for research purposes
        if "execution_metrics" not in self.adaptive_stats:
            try:
                import psutil

                current_cpu = psutil.cpu_percent(interval=None)
                current_memory = psutil.virtual_memory().percent
                self.adaptive_stats["execution_metrics"] = {
                    "peak_cpu": current_cpu,
                    "avg_cpu": current_cpu,
                    "peak_memory": current_memory,
                    "avg_memory": current_memory,
                    "sample_count": 1,
                    "raw_cpu_samples": [current_cpu],
                    "raw_memory_samples": [current_memory],
                }
                # Also ensure resource_metrics has at least one entry
                if not self.adaptive_stats["resource_metrics"]:
                    self.adaptive_stats["resource_metrics"].append(
                        {
                            "cpu_percent": current_cpu,
                            "memory_percent": current_memory,
                            "circuit_breaker_state": "N/A",
                            "measurement_point": "final_fallback",
                        }
                    )
            except Exception:
                # Absolute fallback
                self.adaptive_stats["execution_metrics"] = {
                    "peak_cpu": 0.0,
                    "avg_cpu": 0.0,
                    "peak_memory": 0.0,
                    "avg_memory": 0.0,
                    "sample_count": 0,
                }

        # Add phase metrics if available
        stats_copy = self.adaptive_stats.copy()
        if hasattr(self, "_phase_metrics") and self._phase_metrics:
            stats_copy["phase_metrics"] = self._phase_metrics

        return stats_copy

    def get_phase_metrics(self) -> list[dict[str, Any]]:
        """Get per-phase monitoring metrics."""
        if hasattr(self, "_phase_metrics") and isinstance(self._phase_metrics, list):
            return self._phase_metrics
        return []
