"""Agent orchestration for parallel execution."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from llm_orc.core.execution.input_enhancer import InputEnhancer


class AgentOrchestrator:
    """Orchestrates parallel execution of multiple agents."""

    def __init__(
        self,
        performance_config: dict[str, Any],
        event_emitter: Callable[[str, dict[str, Any]], None],
        config_resolver: Callable[[dict[str, Any]], dict[str, Any]],
        agent_executor: Callable[[dict[str, Any], str, int | None], tuple[str, Any]],
        input_enhancer: InputEnhancer,
        results_processor: Callable[[list[Any], dict[str, Any], dict[str, Any]], None],
    ) -> None:
        """Initialize orchestrator with required dependencies."""
        self._performance_config = performance_config
        self._emit_performance_event = event_emitter
        self._resolve_model_profile_to_config = config_resolver
        self._execute_agent_with_timeout = agent_executor
        self._input_enhancer = input_enhancer
        self._process_agent_results = results_processor

    async def execute_agents_parallel(
        self,
        agents: list[dict[str, Any]],
        input_data: str | dict[str, str],
        results_dict: dict[str, Any],
        agent_usage: dict[str, Any],
    ) -> bool:
        """Execute agents in parallel and return success status.

        Returns True if all agents succeeded, False if any failed.
        """
        if not agents:
            return True

        try:
            # Create tasks using create_task to ensure they start immediately
            tasks = [
                asyncio.create_task(self._execute_agent_task(agent_config, input_data))
                for agent_config in agents
            ]

            # Wait for all tasks to complete
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            self._process_agent_results(agent_results, results_dict, agent_usage)

            # Check if any agent failed
            return not any(
                results_dict.get(agent["name"], {}).get("status") == "failed"
                for agent in agents
            )

        except Exception as e:
            # Fallback: if gather fails, mark all agents as failed
            for agent_config in agents:
                agent_name = agent_config["name"]
                results_dict[agent_name] = {"error": str(e), "status": "failed"}
            return False

    async def _execute_agent_task(
        self, agent_config: dict[str, Any], input_data: str | dict[str, str]
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
            enhanced_config = self._resolve_model_profile_to_config(agent_config)
            timeout = enhanced_config.get("timeout_seconds") or (
                self._performance_config.get("execution", {}).get("default_timeout", 60)
            )

            # Get the appropriate input for this agent
            if isinstance(input_data, dict):
                # For dependency-based execution with per-agent input
                agent_input = input_data.get(agent_name, "")
            else:
                # For standard execution
                agent_input = self._input_enhancer.get_agent_input(
                    input_data, agent_config["name"]
                )

            result = self._execute_agent_with_timeout(
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

            # Return error indicator - will be processed by results processor
            return agent_name, e
