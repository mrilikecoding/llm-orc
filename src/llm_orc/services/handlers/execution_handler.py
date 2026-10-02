"""Execution handler for MCP server."""

from __future__ import annotations

import datetime
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import (
    EnsembleLoader,
    child_ensemble_search_dirs,
)
from llm_orc.core.config.state import resolve_state_dir
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.results_processor import caller_status
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.services.handlers.preflight import DependencyReport, is_runnable
from llm_orc.services.handlers.run_preparation import (
    INVALID_REQUEST,
    NOT_EQUIPPED,
    RunRefusedError,
    only_pullable_unmet,
    pull_pullable,
    unmet_binding_rows,
    unnamed_bind_keys,
)
from llm_orc.services.handlers.run_request import (
    RunRequest,
    RunRequestError,
    apply_bindings,
    materialize,
)

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
    from llm_orc.mcp.server import ProgressReporter
    from llm_orc.services.handlers.provider_handler import Preflight

PreflightFn = Callable[..., Awaitable["Preflight"]]
LayerExecutorFn = Callable[[ConfigurationManager, bool], "EnsembleExecutor"]

#: The request keys a caller may send besides the input.
_REQUEST_KEYS = (
    "ensemble_name",
    "ensemble",
    "ensembles",
    "profiles",
    "scripts",
    "bind",
    "pull",
)
_PARSE_ERRORS = (
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    OSError,
    yaml.YAMLError,
)


@dataclass
class PreparedRun:
    """What the entry paths run: the root and the executor to run it on."""

    config: Any
    executor: EnsembleExecutor
    inline: bool = False
    bindings: dict[str, str] = field(default_factory=dict)
    pulled: list[str] = field(default_factory=list)


@dataclass
class _Layer:
    """The run layer's view, the inline root's file and the bindings."""

    view: ConfigurationManager | None = None
    root_path: Path | None = None
    applied: dict[str, str] = field(default_factory=dict)
    unmet: list[tuple[str, str]] = field(default_factory=list)


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
        layer_executor_fn: LayerExecutorFn,
    ) -> None:
        """Initialize with dependencies.

        Args:
            config_manager: Configuration manager instance.
            ensemble_loader: Ensemble loader instance.
            artifact_manager: Artifact manager instance.
            get_executor_fn: Callback to get/create executor.
            find_ensemble_fn: Callback to find ensemble by name.
            preflight_fn: The gate every run passes (``ProviderHandler.preflight``).
            layer_executor_fn: Builds the executor of a run with a layer from
                its config manager view and whether to save artifacts.
        """
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader
        self._artifact_manager = artifact_manager
        self._get_executor = get_executor_fn
        self._find_ensemble = find_ensemble_fn
        self._preflight = preflight_fn
        self._layer_executor = layer_executor_fn
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
        input_data = arguments.get("input", "")
        input_file = arguments.get("input_file")

        if not input_data and input_file:
            path = Path(input_file)
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            input_data = path.read_text()

        try:
            async with self._prepared(
                _request_data(arguments),
                self._lookup_in_tiers,
                "Ensemble does not exist",
            ) as run:
                result = await run.executor.execute(run.config, input_data)

                status, has_errors = caller_status(result.get("status"))

                return {
                    "results": result.get("results", {}),
                    "deliverable": result.get("deliverable"),
                    "status": status,
                    "has_errors": has_errors,
                    "raw_output": run.config.raw_output,
                    **_run_record(run),
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
        data: Mapping[str, Any],
        lookup: Callable[[str], Any],
        missing: str,
    ) -> AsyncIterator[PreparedRun]:
        """The root and executor for one run, once the gate has passed.

        Raises ``RunRefusedError`` before any agent starts when the
        request is malformed or the host cannot run the ensemble (Arc 4,
        rulings 1, 9 and 10). The run directory, when there is one, is
        removed on every way out: success, refusal, an exception and
        cancellation.
        """
        request = _parse(data)
        run_dir = self._new_run_dir() if request.needs_layer else None
        try:
            yield await self._prepare(request, run_dir, lookup, missing)
        finally:
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)

    def _new_run_dir(self) -> Path:
        runs = resolve_state_dir(self._config_manager.local_config_dir) / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=runs))

    async def _prepare(
        self,
        request: RunRequest,
        run_dir: Path | None,
        lookup: Callable[[str], Any],
        missing: str,
    ) -> PreparedRun:
        layer = self._open_layer(request, run_dir)
        manager = layer.view or self._config_manager
        inline = request.ensemble is not None
        config = self._load_root(request, layer.root_path, manager, lookup, missing)
        root_ref = str(
            request.ensemble["name"] if request.ensemble else request.ensemble_name
        )
        outcome = await self._gate(config, root_ref, manager)
        stray = unnamed_bind_keys(request.bind, outcome.closure.dependencies)
        if stray:
            raise RunRefusedError(
                INVALID_REQUEST,
                f"bind key {stray[0]!r} names no profile in the closure",
            )
        reports = [*outcome.reports, *unmet_binding_rows(layer.unmet)]
        pulled: list[str] = []
        if request.pull and only_pullable_unmet(reports):
            reports, pulled = await pull_pullable(reports, manager.get_model_profiles())
        if not is_runnable(reports):
            raise RunRefusedError(NOT_EQUIPPED, _unmet_message(reports), reports)
        if layer.view is None:
            return PreparedRun(config, self._get_executor(), inline, {}, pulled)
        executor = self._layer_executor(layer.view, not inline)
        return PreparedRun(config, executor, inline, layer.applied, pulled)

    def _open_layer(self, request: RunRequest, run_dir: Path | None) -> _Layer:
        """Materialize the request into ``run_dir`` and bind over it."""
        if run_dir is None:
            return _Layer()
        try:
            root_path = materialize(request, run_dir)
            view = self._config_manager.with_run_layer(run_dir)
            applied, unmet = apply_bindings(request.bind, view, run_dir)
        except RunRequestError as e:
            raise RunRefusedError(INVALID_REQUEST, str(e)) from e
        return _Layer(view, root_path, applied, unmet)

    def _load_root(
        self,
        request: RunRequest,
        root_path: Path | None,
        manager: ConfigurationManager,
        lookup: Callable[[str], Any],
        missing: str,
    ) -> Any:
        if root_path is None:
            name = str(request.ensemble_name)
            config = lookup(name)
            if not config:
                raise ValueError(f"{missing}: {name}")
            return config
        try:
            return self._ensemble_loader.load_from_file(
                str(root_path),
                search_dirs=child_ensemble_search_dirs(self._project_path, manager),
            )
        except _PARSE_ERRORS as e:
            raise RunRefusedError(
                INVALID_REQUEST, f"the inline ensemble does not load: {e}"
            ) from e

    async def _gate(
        self, config: Any, root_ref: str, manager: ConfigurationManager
    ) -> Preflight:
        try:
            return await self._preflight(
                config,
                root_ref,
                config_manager=manager,
                project_dir=self._project_path,
            )
        except _PARSE_ERRORS as e:
            raise RunRefusedError(
                INVALID_REQUEST, f"a child ensemble does not load: {e}"
            ) from e

    async def execute_streaming(
        self,
        ensemble_name: str | None,
        input_data: str,
        reporter: ProgressReporter,
        injection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute ensemble with streaming progress updates.

        Args:
            ensemble_name: Name of the ensemble to execute, or None when
                the root is inline (``injection["ensemble"]``).
            input_data: Input data for the ensemble.
            reporter: Progress reporter for status updates.
            injection: The run request's other keys: ``ensemble``,
                ``ensembles``, ``profiles``, ``scripts``, ``bind``, ``pull``.

        Returns:
            Execution result.
        """
        try:
            async with self._prepared(
                _request_data({**(injection or {}), "ensemble_name": ensemble_name}),
                self._find_ensemble,
                "Ensemble does not exist",
            ) as run:
                return await self._stream(run, ensemble_name, input_data, reporter)
        except RunRefusedError as refusal:
            await reporter.error(f"Execution refused ({refusal.kind}): {refusal}")
            return refusal.envelope()

    async def _stream(
        self,
        run: PreparedRun,
        ensemble_name: str | None,
        input_data: str,
        reporter: ProgressReporter,
    ) -> dict[str, Any]:
        total_agents = len(run.config.agents)
        name = ensemble_name or run.config.name
        state: dict[str, Any] = {
            "completed": 0,
            "result": {},
            "ensemble_name": name,
            "input_data": input_data,
            "save_artifact": not run.inline,
        }

        msg = f"Starting ensemble '{name}' with {total_agents} agents"
        await reporter.info(msg)

        async for event in run.executor.execute_streaming(run.config, input_data):
            await self.handle_streaming_event(event, reporter, total_agents, state)

        result = state.get("result", {})
        if not isinstance(result, dict):
            result = {}
        return {**result, **_run_record(run)} if result else result

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
            if state.get("save_artifact", True):
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
                _request_data(params), self._lookup_in_tiers, "Ensemble not found"
            ) as run:
                async for event in run.executor.execute_streaming(
                    run.config, input_data
                ):
                    yield event
        except RunRefusedError as refusal:
            yield {"type": "execution_failed", "data": {"error": refusal.error}}


def _run_record(run: PreparedRun) -> dict[str, Any]:
    """What the run was asked to do that the caller should see: the
    bindings applied and the models pulled, when there are any."""
    record: dict[str, Any] = {}
    if run.bindings:
        record["bindings"] = run.bindings
    if run.pulled:
        record["pulled"] = run.pulled
    return record


def _request_data(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The run request keys of ``arguments``; the rest (input, input_file)
    belong to the entry path, not the request."""
    return {
        k: arguments[k]
        for k in _REQUEST_KEYS
        if arguments.get(k) is not None and arguments[k] != ""
    }


def _parse(data: Mapping[str, Any]) -> RunRequest:
    try:
        return RunRequest.parse(data)
    except RunRequestError as e:
        raise RunRefusedError(INVALID_REQUEST, str(e)) from e


def _unmet_message(reports: list[DependencyReport]) -> str:
    unmet = [f"{r.name} ({r.status.value})" for r in reports if not is_runnable([r])]
    return "this host cannot run the ensemble, unmet: " + ", ".join(unmet)
