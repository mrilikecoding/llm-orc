"""Agent dispatch and parallel execution for ensemble phases."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from llm_orc.core.execution.agent_execution_coordinator import (
    AgentExecutionCoordinator,
)
from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.core.execution.progress_controller import ProgressController
from llm_orc.schemas.agent_config import (
    AgentConfig,
    LlmAgentConfig,
    ScriptAgentConfig,
)

# Type alias for the resolve profile callback
ResolveProfileFn = Callable[[AgentConfig], Awaitable[dict[str, Any]]]


class AgentDispatcher:
    """Dispatches and executes agents within ensemble phases.

    Handles parallel execution, monitoring, timeout management,
    and failure handling. Agent type routing remains on EnsembleExecutor
    to preserve test patchability.
    """

    def __init__(
        self,
        execution_coordinator: AgentExecutionCoordinator,
        dependency_resolver: DependencyResolver,
        progress_controller: ProgressController,
        emit_event_fn: Callable[[str, dict[str, Any]], None],
        resolve_profile_fn: ResolveProfileFn,
        performance_config: dict[str, Any],
    ) -> None:
        self._execution_coordinator = execution_coordinator
        self._dependency_resolver = dependency_resolver
        self._progress_controller = progress_controller
        self._emit_event = emit_event_fn
        self._resolve_profile = resolve_profile_fn
        self._performance_config = performance_config

    async def execute_agents_in_phase(
        self,
        phase_agents: list[AgentConfig],
        phase_input: str | dict[str, str],
    ) -> dict[str, Any]:
        """Execute agents in parallel within a phase."""
        tasks = [
            self._execute_single_agent_in_phase(agent_config, phase_input)
            for agent_config in phase_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        phase_results: dict[str, Any] = {}
        for result in results:
            if isinstance(result, BaseException):
                continue
            agent_name, agent_result = result
            phase_results[agent_name] = agent_result

        return phase_results

    def _determine_agent_type(self, agent_config: AgentConfig) -> str | None:
        """Determine agent type from configuration using isinstance."""
        if isinstance(agent_config, ScriptAgentConfig):
            return "script"
        if isinstance(agent_config, LlmAgentConfig):
            return "llm"
        return None

    async def _execute_single_agent_in_phase(
        self, agent_config: AgentConfig, phase_input: str | dict[str, str]
    ) -> tuple[str, dict[str, Any]]:
        """Execute a single agent in a phase and return (name, result)."""
        agent_name = agent_config.name

        if self._dependency_resolver.is_fan_out_instance_config(agent_config):
            base_input = (
                phase_input
                if isinstance(phase_input, str)
                else phase_input.get(agent_name, "")
            )
            phase_input = self._dependency_resolver.prepare_fan_out_instance_input(
                agent_config, base_input
            )

        try:
            return await self._execute_agent_with_monitoring(
                agent_config, agent_name, phase_input
            )
        except Exception as e:
            return await self._handle_agent_execution_failure(agent_config, e)

    async def _execute_agent_with_monitoring(
        self,
        agent_config: AgentConfig,
        agent_name: str,
        phase_input: str | dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        """Execute agent with full monitoring and progress tracking."""
        agent_start_time = time.time()

        self._emit_event(
            "agent_started",
            {"agent_name": agent_name, "timestamp": agent_start_time},
        )

        if self._progress_controller:
            await self._progress_controller.update_agent_progress(agent_name, "started")

        agent_input = self._dependency_resolver.get_agent_input(phase_input, agent_name)
        timeout = await self._get_agent_timeout(agent_config)

        (
            response,
            model_instance,
        ) = await self._execution_coordinator.execute_agent_with_timeout(
            agent_config, agent_input, timeout
        )

        await self._emit_agent_completion_events(agent_name, agent_start_time)

        return agent_name, {
            "response": response,
            "status": "success",
            "model_instance": model_instance,
        }

    async def _get_agent_timeout(self, agent_config: AgentConfig) -> int:
        """Get timeout for agent execution."""
        enhanced_config = await self._resolve_profile(agent_config)
        timeout = enhanced_config.get("timeout_seconds")
        if timeout is not None:
            return int(timeout)
        return int(
            self._performance_config.get("execution", {}).get("default_timeout", 60)
        )

    async def _emit_agent_completion_events(
        self, agent_name: str, start_time: float
    ) -> None:
        """Emit agent completion events and update progress."""
        agent_end_time = time.time()
        duration_ms = int((agent_end_time - start_time) * 1000)

        self._emit_event(
            "agent_completed",
            {
                "agent_name": agent_name,
                "timestamp": agent_end_time,
                "duration_ms": duration_ms,
            },
        )

        if self._progress_controller:
            await self._progress_controller.update_agent_progress(
                agent_name, "completed"
            )

    async def _handle_agent_execution_failure(
        self, agent_config: AgentConfig, error: Exception
    ) -> tuple[str, dict[str, Any]]:
        """Handle agent execution failure and return error result."""
        agent_name = agent_config.name
        agent_end_time = time.time()

        self._emit_event(
            "agent_completed",
            {
                "agent_name": agent_name,
                "timestamp": agent_end_time,
                "duration_ms": 0,
                "error": str(error),
            },
        )

        return agent_name, {
            "error": str(error),
            "status": "failed",
            "model_instance": None,
        }
