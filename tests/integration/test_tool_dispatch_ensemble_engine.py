"""Boundary integration test: Orchestrator Tool Dispatch → Ensemble Engine.

Per ``docs/agentic-serving/system-design.md`` §Test Architecture:

    Orchestrator Tool Dispatch → Ensemble Engine
    test_invoke_ensemble_executes_real_ensemble — End-to-end ensemble
    execution with real ``EnsembleExecutor``.

The Group 2 unit tests use a scripted ``EnsembleOperations`` double.
This test exercises the production call chain:
``OrchestratorToolDispatch → OrchestraService.invoke → ExecutionHandler →
EnsembleLoader.find_ensemble → EnsembleExecutor.execute → MockModel``.

Uses ``mock`` model-name routing in ``ModelFactory`` (names starting
with ``mock`` construct ``MockModel``) so the test stays fully local
and deterministic — no network, no API keys, no provider quirks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.services.orchestra_service import OrchestraService


def _make_dispatch(service: OrchestraService) -> OrchestratorToolDispatch:
    """Construct Tool Dispatch with a real Harness pointed at the test library.

    The ``analysis`` fixture ensemble is flagged ``raw_output: true`` so the
    Harness takes the pass-through branch without requiring a summarizer
    ensemble in the library. The Tool Dispatch → Harness → Ensemble Engine
    summarization path is covered separately by the Group 5 boundary test.
    """
    harness = ResultSummarizerHarness(
        invoker=service, summarizer_name="agentic-result-summarizer"
    )
    return OrchestratorToolDispatch(operations=service, harness=harness)


def _write_local_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a local ``.llm-orc`` library with one mock-model ensemble.

    Returns the project directory the ConfigurationManager reads from.
    The global XDG root is redirected into ``tmp_path`` so this test
    doesn't see the developer's real configuration.
    """
    global_root = tmp_path / "xdg"
    global_root.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(global_root))
    monkeypatch.delenv("LLM_ORC_LIBRARY_PATH", raising=False)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    local = project_dir / ".llm-orc"
    local.mkdir()

    # Model profile that routes to MockModel (any name starting with "mock").
    (local / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model_profiles": {
                    "local-mock": {"model": "mock", "provider": "mock"},
                }
            }
        )
    )

    ensembles_dir = local / "ensembles"
    ensembles_dir.mkdir()
    (ensembles_dir / "analysis.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "analysis",
                "description": "Analyze input with a mock model.",
                "raw_output": True,
                "agents": [
                    {
                        "name": "analyst",
                        "model_profile": "local-mock",
                        "system_prompt": ("You are an analyst. Summarize the input."),
                    }
                ],
            }
        )
    )
    return project_dir


class TestInvokeEnsembleReachesEnsembleEngine:
    """End-to-end ensemble execution from the orchestrator tool surface."""

    @pytest.mark.asyncio
    async def test_invoke_ensemble_executes_real_ensemble(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_e2e",
                name="invoke_ensemble",
                arguments={
                    "name": "analysis",
                    "input": "traffic spiked at 03:14 UTC",
                },
            )
        )

        # Success result flows through OrchestraService.invoke's
        # normalized shape: {results, synthesis, status}.
        assert isinstance(result, ToolCallSuccess)
        assert result.id == "call_e2e"
        assert result.name == "invoke_ensemble"
        content = result.content
        assert isinstance(content, dict)
        assert content.get("status") == "success"
        # MockModel echoes the input substring into its response.
        analyst_result = content["results"]["analyst"]
        response_text = (
            analyst_result["response"]
            if isinstance(analyst_result, dict)
            else str(analyst_result)
        )
        assert "traffic spiked" in response_text

    @pytest.mark.asyncio
    async def test_invoke_ensemble_not_found_surfaces_as_tool_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real handler raises ``ValueError``; Tool Dispatch converts."""
        from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallError

        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_missing",
                name="invoke_ensemble",
                arguments={"name": "does-not-exist", "input": ""},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert "does-not-exist" in result.reason


class TestListEnsemblesReachesLibrary:
    """Boundary: list_ensembles → OrchestraService.read_ensembles → library."""

    @pytest.mark.asyncio
    async def test_list_ensembles_enumerates_local_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(id="call_list", name="list_ensembles", arguments={})
        )

        assert isinstance(result, ToolCallSuccess)
        entries = result.content
        assert isinstance(entries, list)
        names = [e["name"] for e in entries]
        assert "analysis" in names
        # ResourceHandler decorates entries with source / path /
        # agent_count in addition to name and description.
        analysis = next(e for e in entries if e["name"] == "analysis")
        assert analysis["description"] == "Analyze input with a mock model."
        assert analysis["source"] == "local"
        assert analysis["agent_count"] == 1
