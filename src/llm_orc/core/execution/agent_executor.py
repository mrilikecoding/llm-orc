"""Agent execution coordination with parallel processing and resource management."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleConfig


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
        if not agents:
            return

        # Get concurrency limit from performance config or use sensible default
        max_concurrent = self.get_effective_concurrency_limit(len(agents))

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
