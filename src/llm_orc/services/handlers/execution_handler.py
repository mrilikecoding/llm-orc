"""Execution handler for MCP server."""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.results_processor import caller_status
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.services.handlers.preflight import DependencyReport, is_runnable
from llm_orc.services.handlers.run_preparation import NOT_EQUIPPED, RunRefusedError

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
    from llm_orc.mcp.server import ProgressReporter
    from llm_orc.services.handlers.provider_handler import Preflight

PreflightFn = Callable[..., Awaitable["Preflight"]]


@dataclass
class PreparedRun:
    """What the entry paths run: the root and the executor to run it on."""

    config: Any
    executor: EnsembleExecutor


class ExecutionHandler:
    """Handles ensemble execution and streaming operations."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        ensemble_loader: EnsembleLoader,
        artifact_manager: ArtifactManager,
        get_executor_fn: Callable[[], EnsembleExecutor],
        find_ensemble_fn: Callable[[str], Any],
        *,
        preflight_fn: PreflightFn,
    ) -> None:
        """Initialize with dependencies.

        Args:
            config_manager: Configuration manager instance.
            ensemble_loader: Ensemble loader instance.
            artifact_manager: Artifact manager instance.
            get_executor_fn: Callback to get/create executor.
            find_ensemble_fn: Callback to find ensemble by name.
            preflight_fn: The gate every run passes (``ProviderHandler.preflight``).
        """
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader
        self._artifact_manager = artifact_manager
        self._get_executor = get_executor_fn
        self._find_ensemble = find_ensemble_fn
        self._preflight = preflight_fn
        self._project_path: Path | None = None

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._config_manager = ctx.config_manager
        self._project_path = ctx.project_path

    def set_artifact_manager(self, manager: ArtifactManager) -> None:
        """Use a rebuilt artifact manager after a project switch."""
        self._artifact_manager = manager

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute invoke tool.

        Args:
            arguments: Tool arguments including ensemble_name and input.

        Returns:
            Execution result.
        """
        ensemble_name = arguments.get("ensemble_name")
        input_data = arguments.get("input", "")
        input_file = arguments.get("input_file")

        if not input_data and input_file:
            path = Path(input_file)
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            input_data = path.read_text()

        try:
            async with self._prepared(
                ensemble_name, self._lookup_in_tiers, "Ensemble does not exist"
            ) as run:
                result = await run.executor.execute(run.config, input_data)

                status, has_errors = caller_status(result.get("status"))

                return {
                    "results": result.get("results", {}),
                    "deliverable": result.get("deliverable"),
                    "status": status,
                    "has_errors": has_errors,
                    "raw_output": run.config.raw_output,
                }
        except RunRefusedError as refusal:
            return refusal.envelope()

    def _lookup_in_tiers(self, ensemble_name: str) -> Any:
        """The first tier directory that has the ensemble."""
        for ensemble_dir in self._config_manager.get_ensembles_dirs():
            config = self._ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                return config
        return None

    @asynccontextmanager
    async def _prepared(
        self,
        ensemble_name: str | None,
        lookup: Callable[[str], Any],
        missing: str,
    ) -> AsyncIterator[PreparedRun]:
        """The root and executor for one run, once the gate has passed.

        Raises ``RunRefusedError`` before any agent starts when the host
        cannot run the ensemble (Arc 4, ruling 1).
        """
        if not ensemble_name:
            raise ValueError("ensemble_name is required")
        config = lookup(ensemble_name)
        if not config:
            raise ValueError(f"{missing}: {ensemble_name}")
        outcome = await self._preflight(
            config,
            ensemble_name,
            config_manager=self._config_manager,
            project_dir=self._project_path,
        )
        if not is_runnable(outcome.reports):
            raise RunRefusedError(
                NOT_EQUIPPED, _unmet_message(outcome.reports), outcome.reports
            )
        yield PreparedRun(config, self._get_executor())

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
        try:
            async with self._prepared(
                ensemble_name, self._find_ensemble, "Ensemble does not exist"
            ) as run:
                return await self._stream(run, ensemble_name, input_data, reporter)
        except RunRefusedError as refusal:
            await reporter.error(f"Execution refused ({refusal.kind}): {refusal}")
            return refusal.envelope()

    async def _stream(
        self,
        run: PreparedRun,
        ensemble_name: str,
        input_data: str,
        reporter: ProgressReporter,
    ) -> dict[str, Any]:
        total_agents = len(run.config.agents)
        state: dict[str, Any] = {
            "completed": 0,
            "result": {},
            "ensemble_name": ensemble_name,
            "input_data": input_data,
        }

        msg = f"Starting ensemble '{ensemble_name}' with {total_agents} agents"
        await reporter.info(msg)

        async for event in run.executor.execute_streaming(run.config, input_data):
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
            deliverable = event_data.get("deliverable")
            raw_status = event_data.get("status", "completed")
            status, has_errors = caller_status(raw_status)
            state["result"] = {
                "results": results,
                "deliverable": deliverable,
                "status": status,
                "has_errors": has_errors,
            }
            ensemble_name = state.get("ensemble_name", "unknown")
            input_data = state.get("input_data", "")
            # The artifact keeps the raw internal status (completed /
            # completed_with_errors) — its own consumers read that value,
            # unrelated to the caller-facing success/error vocabulary above.
            self.save_execution_artifact(
                ensemble_name, input_data, results, deliverable, raw_status
            )
            await reporter.report_progress(progress=total_agents, total=total_agents)

        elif event_type == "execution_failed":
            error_msg = event_data.get("error", "Unknown error")
            await reporter.error(f"Execution failed: {error_msg}")
            status, has_errors = caller_status("failed")
            state["result"] = {
                "results": {},
                "deliverable": None,
                "status": status,
                "has_errors": has_errors,
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
        deliverable: str | None,
        status: str,
    ) -> Path | None:
        """Save execution results as an artifact.

        Args:
            ensemble_name: Name of the executed ensemble.
            input_data: Input provided to the ensemble.
            results: Agent results dictionary.
            deliverable: The executor-resolved deliverable (ADR-035 D1).
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
            "deliverable": deliverable,
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
        input_data = params.get("input", "")
        try:
            async with self._prepared(
                params.get("ensemble_name"),
                self._lookup_in_tiers,
                "Ensemble not found",
            ) as run:
                async for event in run.executor.execute_streaming(
                    run.config, input_data
                ):
                    yield event
        except RunRefusedError as refusal:
            yield {"type": "execution_failed", "data": {"error": refusal.error}}


def _unmet_message(reports: list[DependencyReport]) -> str:
    unmet = [f"{r.name} ({r.status.value})" for r in reports if not is_runnable([r])]
    return "this host cannot run the ensemble, unmet: " + ", ".join(unmet)
