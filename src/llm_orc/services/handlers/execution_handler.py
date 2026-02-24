"""Execution handler for MCP server."""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
    from llm_orc.mcp.server import ProgressReporter


class ExecutionHandler:
    """Handles ensemble execution and streaming operations."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        ensemble_loader: EnsembleLoader,
        artifact_manager: ArtifactManager,
        get_executor_fn: Callable[[], EnsembleExecutor],
        find_ensemble_fn: Callable[[str], Any],
    ) -> None:
        """Initialize with dependencies.

        Args:
            config_manager: Configuration manager instance.
            ensemble_loader: Ensemble loader instance.
            artifact_manager: Artifact manager instance.
            get_executor_fn: Callback to get/create executor.
            find_ensemble_fn: Callback to find ensemble by name.
        """
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader
        self._artifact_manager = artifact_manager
        self._get_executor = get_executor_fn
        self._find_ensemble = find_ensemble_fn
        self._project_path: Path | None = None

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._config_manager = ctx.config_manager
        self._project_path = ctx.project_path

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute invoke tool.

        Args:
            arguments: Tool arguments including ensemble_name and input.

        Returns:
            Execution result.
        """
        ensemble_name = arguments.get("ensemble_name")
        input_data = arguments.get("input", "")

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        ensemble_dirs = self._config_manager.get_ensembles_dirs()
        config = None

        for ensemble_dir in ensemble_dirs:
            config = self._ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                break

        if not config:
            raise ValueError(f"Ensemble does not exist: {ensemble_name}")

        executor = self._get_executor()
        result = await executor.execute(config, input_data)

        raw_status: str | None = result.get("status")
        # Normalize internal status values to the API contract expected by clients.
        # "completed"             → "success"  (all agents succeeded)
        # "completed_with_errors" → "error"    (some agents failed)
        status_map = {"completed": "success", "completed_with_errors": "error"}
        status = status_map.get(raw_status, raw_status) if raw_status else raw_status

        return {
            "results": result.get("results", {}),
            "synthesis": result.get("synthesis"),
            "status": status,
        }

    async def execute_streaming(
        self,
        ensemble_name: str,
        input_data: str,
        reporter: ProgressReporter,
    ) -> dict[str, Any]:
        """Execute ensemble with streaming progress updates.

        Args:
            ensemble_name: Name of the ensemble to execute.
            input_data: Input data for the ensemble.
            reporter: Progress reporter for status updates.

        Returns:
            Execution result.
        """
        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        config = self._find_ensemble(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble does not exist: {ensemble_name}")

        executor = self._get_executor()
        total_agents = len(config.agents)
        state: dict[str, Any] = {
            "completed": 0,
            "result": {},
            "ensemble_name": ensemble_name,
            "input_data": input_data,
        }

        msg = f"Starting ensemble '{ensemble_name}' with {total_agents} agents"
        await reporter.info(msg)

        async for event in executor.execute_streaming(config, input_data):
            await self.handle_streaming_event(event, reporter, total_agents, state)

        result = state.get("result", {})
        if not isinstance(result, dict):
            result = {}
        return result

    async def handle_streaming_event(
        self,
        event: dict[str, Any],
        reporter: ProgressReporter,
        total_agents: int,
        state: dict[str, Any],
    ) -> None:
        """Handle a single streaming event from ensemble execution.

        Args:
            event: The streaming event.
            reporter: Progress reporter for status updates.
            total_agents: Total number of agents in ensemble.
            state: Mutable state dict with 'completed' count and 'result'.
        """
        event_type = event.get("type", "")
        event_data = event.get("data", {})

        if event_type == "execution_started":
            await reporter.report_progress(progress=0, total=total_agents)

        elif event_type == "agent_started":
            agent_name = event_data.get("agent_name", "unknown")
            await reporter.info(f"Agent '{agent_name}' started")

        elif event_type == "agent_completed":
            state["completed"] += 1
            agent_name = event_data.get("agent_name", "unknown")
            await reporter.report_progress(state["completed"], total_agents)
            await reporter.info(f"Agent '{agent_name}' completed")

        elif event_type == "execution_completed":
            results = event_data.get("results", {})
            synthesis = event_data.get("synthesis")
            status = event_data.get("status", "completed")
            state["result"] = {
                "results": results,
                "synthesis": synthesis,
                "status": status,
            }
            ensemble_name = state.get("ensemble_name", "unknown")
            input_data = state.get("input_data", "")
            self.save_execution_artifact(
                ensemble_name, input_data, results, synthesis, status
            )
            await reporter.report_progress(progress=total_agents, total=total_agents)

        elif event_type == "execution_failed":
            error_msg = event_data.get("error", "Unknown error")
            await reporter.error(f"Execution failed: {error_msg}")
            state["result"] = {
                "results": {},
                "synthesis": None,
                "status": "failed",
                "error": error_msg,
            }

        elif event_type == "agent_fallback_started":
            agent_name = event_data.get("agent_name", "unknown")
            msg = f"Agent '{agent_name}' falling back to alternate model"
            await reporter.warning(msg)

    def save_execution_artifact(
        self,
        ensemble_name: str,
        input_data: str,
        results: dict[str, Any],
        synthesis: Any,
        status: str,
    ) -> Path | None:
        """Save execution results as an artifact.

        Args:
            ensemble_name: Name of the executed ensemble.
            input_data: Input provided to the ensemble.
            results: Agent results dictionary.
            synthesis: Synthesis result (if any).
            status: Execution status.

        Returns:
            Path to the artifact directory or None if save failed.
        """
        artifact_data: dict[str, Any] = {
            "ensemble_name": ensemble_name,
            "input": input_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": status,
            "results": results,
            "synthesis": synthesis,
            "agents": [],
        }

        for agent_name, agent_result in results.items():
            if isinstance(agent_result, dict):
                artifact_data["agents"].append(
                    {
                        "name": agent_name,
                        "status": agent_result.get("status", "unknown"),
                        "result": agent_result.get("response", ""),
                    }
                )

        try:
            artifact_path = self._artifact_manager.save_execution_results(
                ensemble_name, artifact_data
            )
            return artifact_path
        except (OSError, TypeError, ValueError):
            return None

    async def invoke_streaming(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Invoke ensemble with streaming progress.

        Args:
            params: Invocation parameters.

        Yields:
            Progress events.
        """
        ensemble_name = params.get("ensemble_name")

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        ensemble_dirs = self._config_manager.get_ensembles_dirs()
        config = None

        for ensemble_dir in ensemble_dirs:
            config = self._ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                break

        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name")
            yield {
                "type": "agent_start",
                "agent": agent_name,
            }

            yield {
                "type": "agent_progress",
                "agent": agent_name,
                "progress": 50,
            }

            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "status": "success",
            }

        yield {
            "type": "execution_complete",
            "status": "success",
        }
