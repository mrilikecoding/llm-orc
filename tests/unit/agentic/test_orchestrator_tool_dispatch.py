"""Tests for the Orchestrator Tool Dispatch module.

Per `docs/agentic-serving/system-design.md` §Orchestrator Tool Dispatch
(L2) and §Integration Contracts (Orchestrator Runtime → Orchestrator
Tool Dispatch).

Covers scenarios:

* §Orchestrator tool surface is exactly the committed set (FC-5)
* §Invocation outside the tool set is rejected
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    EnsembleRuntimeExecutor,
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallError,
    ToolCallSuccess,
)
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig


class _RecordingExecutor:
    """Records calls and returns a canned result dict.

    Satisfies ``EnsembleRuntimeExecutor`` structurally. Not an
    ``EnsembleExecutor`` subclass — we are testing Tool Dispatch, not
    Ensemble Engine.
    """

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[tuple[str, str]] = []

    async def execute(self, config: EnsembleConfig, input_data: str) -> dict[str, Any]:
        self.calls.append((config.name, input_data))
        return self._result


class _StubExecutorProvider:
    """Raises if the Runtime accidentally calls through on a path
    that shouldn't reach Ensemble Engine in the test under scrutiny."""

    def __call__(self) -> EnsembleRuntimeExecutor:  # pragma: no cover
        raise AssertionError(
            "executor_provider should not be invoked for this test path"
        )


def _make_config_manager_with_ensembles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ensembles: dict[str, dict[str, Any]] | None = None,
) -> ConfigurationManager:
    """Construct a ConfigurationManager whose local ensembles dir has YAML files."""
    global_root = tmp_path / "xdg"
    global_root.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(global_root))
    monkeypatch.delenv("LLM_ORC_LIBRARY_PATH", raising=False)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    local_ensembles_dir = project_dir / ".llm-orc" / "ensembles"
    local_ensembles_dir.mkdir(parents=True)

    for filename, body in (ensembles or {}).items():
        (local_ensembles_dir / filename).write_text(yaml.safe_dump(body))

    return ConfigurationManager(project_dir=project_dir, provision=False)


class TestDispatchRejectsUnknownTool:
    """Scenario: Invocation outside the tool set is rejected.

    Tool Dispatch is the structural enforcement point for ADR-003 —
    the closed tool set. A name outside the five committed tools
    returns a typed tool error the Runtime can surface to the
    orchestrator LLM as an observation; the ReAct loop continues.
    """

    @pytest.mark.asyncio
    async def test_dispatch_rejects_unknown_tool_name(self) -> None:
        dispatch = OrchestratorToolDispatch(
            config_manager=None,  # type: ignore[arg-type]
            executor_provider=_StubExecutorProvider(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_abc",
                name="hallucinated_tool",
                arguments={},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.id == "call_abc"
        assert result.name == "hallucinated_tool"
        assert result.kind == "unknown_tool"


class TestClosedToolSet:
    """Scenario: Orchestrator tool surface is exactly the committed set (FC-5).

    The closed-set property (ADR-003) is enforced structurally: the
    five tool names in ``TOOL_NAMES`` correspond to exactly five
    async methods on the dispatch class. A sixth public async tool
    method would mean the closed set is no longer closed.
    """

    def test_tool_dispatch_exposes_exactly_five_tool_methods(self) -> None:
        tool_methods = {
            name
            for name, member in inspect.getmembers(OrchestratorToolDispatch)
            if inspect.iscoroutinefunction(member) and name in TOOL_NAMES
        }

        assert tool_methods == {
            "invoke_ensemble",
            "compose_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        }
        assert len(tool_methods) == 5

    def test_tool_names_set_matches_committed_five(self) -> None:
        """ADR-003's tool set is what the module advertises as ``TOOL_NAMES``."""
        assert TOOL_NAMES == frozenset(
            {
                "invoke_ensemble",
                "compose_ensemble",
                "list_ensembles",
                "query_knowledge",
                "record_outcome",
            }
        )


class TestNotYetWiredTools:
    """WP-C wires only invoke_ensemble and list_ensembles.

    The other three tools exist (to honor the closed-set property) but
    return a typed ``not_yet_wired`` tool error. The Runtime surfaces
    these to the orchestrator LLM as an observation — the LLM is free
    to plan around them.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "landing_wp"),
        [
            ("compose_ensemble", "WP-G"),
            ("query_knowledge", "WP-I"),
            ("record_outcome", "WP-I"),
        ],
    )
    async def test_not_yet_wired_tool_returns_typed_error(
        self, tool_name: str, landing_wp: str
    ) -> None:
        dispatch = OrchestratorToolDispatch(
            config_manager=None,  # type: ignore[arg-type]
            executor_provider=_StubExecutorProvider(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="call_1", name=tool_name, arguments={})
        )

        assert isinstance(result, ToolCallError)
        assert result.name == tool_name
        assert result.kind == "not_yet_wired"
        assert landing_wp in result.reason


class TestListEnsembles:
    """list_ensembles wires to the Ensemble Engine's library."""

    @pytest.mark.asyncio
    async def test_list_ensembles_returns_library_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager_with_ensembles(
            tmp_path,
            monkeypatch,
            ensembles={
                "analysis.yaml": {
                    "name": "analysis",
                    "description": "Analyzes code",
                    "agents": [
                        {
                            "name": "analyst",
                            "model_profile": "default",
                            "system_prompt": "You analyze code.",
                        }
                    ],
                },
            },
        )
        dispatch = OrchestratorToolDispatch(
            config_manager=cm, executor_provider=_StubExecutorProvider()
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="call_1", name="list_ensembles", arguments={})
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "list_ensembles"
        entries = result.content
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert entries[0]["name"] == "analysis"
        assert entries[0]["description"] == "Analyzes code"


class TestInvokeEnsemble:
    """invoke_ensemble resolves a library ensemble by name and executes it."""

    @pytest.mark.asyncio
    async def test_invoke_ensemble_executes_named_ensemble(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager_with_ensembles(
            tmp_path,
            monkeypatch,
            ensembles={
                "analysis.yaml": {
                    "name": "analysis",
                    "description": "Analyzes code",
                    "agents": [
                        {
                            "name": "analyst",
                            "model_profile": "default",
                            "system_prompt": "You analyze code.",
                        }
                    ],
                },
            },
        )
        recorded_result = {"agent_results": {"analyst": {"response": "ok"}}}
        executor = _RecordingExecutor(recorded_result)
        dispatch = OrchestratorToolDispatch(
            config_manager=cm, executor_provider=lambda: executor
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_7",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "refactor the parser"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.id == "call_7"
        assert result.name == "invoke_ensemble"
        assert result.content == recorded_result
        assert executor.calls == [("analysis", "refactor the parser")]

    @pytest.mark.asyncio
    async def test_invoke_ensemble_returns_error_when_name_not_in_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hallucinated ensemble name becomes an observation, not a crash."""
        cm = _make_config_manager_with_ensembles(tmp_path, monkeypatch)

        dispatch = OrchestratorToolDispatch(
            config_manager=cm, executor_provider=_StubExecutorProvider()
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_8",
                name="invoke_ensemble",
                arguments={"name": "does-not-exist", "input": "anything"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.name == "invoke_ensemble"
        assert result.kind == "invocation_failed"
        assert "does-not-exist" in result.reason

    @pytest.mark.asyncio
    async def test_invoke_ensemble_rejects_missing_name_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Input validation — missing or empty ``name`` is a typed error."""
        cm = _make_config_manager_with_ensembles(tmp_path, monkeypatch)
        dispatch = OrchestratorToolDispatch(
            config_manager=cm, executor_provider=_StubExecutorProvider()
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="call_9", name="invoke_ensemble", arguments={})
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invalid_arguments"
