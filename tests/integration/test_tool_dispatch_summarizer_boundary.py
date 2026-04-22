"""Boundary integration test: Tool Dispatch → RSH → Ensemble Engine (summarize path).

Per ``docs/agentic-serving/system-design.md`` §Test Architecture (as
amended by Design Amendment #3):

    Orchestrator Tool Dispatch → Result Summarizer Harness
    test_runtime_never_sees_unsummarized_result — ``invoke_ensemble``
    returning a large dict reaches the Runtime as a summary via the
    Harness; escape-hatch flag bypasses. Interposition lives on Tool
    Dispatch per Amendment #3.

The existing ``test_tool_dispatch_ensemble_engine.py`` exercises the
raw_output=True pass-through branch (the ``analysis`` fixture there is
flagged to skip the Harness). This test covers the complementary
summarize branch — the default path AS-7 makes structural:

- Real ``OrchestraService`` (project services layer)
- Real ``OrchestratorToolDispatch``
- Real ``ResultSummarizerHarness`` pointed at a real summarizer
  ensemble configured in the test library
- Real ``EnsembleExecutor`` execution of both ensembles, driven by
  ``MockModel`` for determinism

A regression in any layer (Harness extraction logic, Tool Dispatch
interposition, OrchestraService delegation) that breaks the summarize
path would fail this test at the boundary it governs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.services.orchestra_service import OrchestraService

_SUMMARIZER_NAME = "test-summarizer"


def _write_library_with_summarizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Lay out a local ``.llm-orc`` library with an analysis ensemble and a
    single-agent summarizer ensemble, both routed to MockModel.

    The ``analysis`` ensemble does NOT set ``raw_output``, so Tool
    Dispatch must interpose the Harness, which invokes the summarizer
    ensemble. Both ensembles are single-agent so the Harness's
    ``results[agent]["response"]`` fallback applies (the dependency-
    based execution model leaves ``synthesis`` unpopulated for single-
    agent ensembles — see ``core/execution/results_processor``).
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
                "description": "Analyze input with a mock model (summarize path).",
                "agents": [
                    {
                        "name": "analyst",
                        "model_profile": "local-mock",
                        "system_prompt": "You are an analyst.",
                    }
                ],
            }
        )
    )
    (ensembles_dir / f"{_SUMMARIZER_NAME}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": _SUMMARIZER_NAME,
                "description": "Single-agent summarizer on MockModel.",
                "agents": [
                    {
                        "name": "summarizer",
                        "model_profile": "local-mock",
                        "system_prompt": "Condense the JSON payload.",
                    }
                ],
            }
        )
    )
    return project_dir


def _make_dispatch(service: OrchestraService) -> OrchestratorToolDispatch:
    """Wire Tool Dispatch with a real Harness pointed at ``test-summarizer``."""
    harness = ResultSummarizerHarness(invoker=service, summarizer_name=_SUMMARIZER_NAME)
    policy = AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL)
    return OrchestratorToolDispatch(
        operations=service, harness=harness, autonomy_policy=policy
    )


class TestInvokeEnsembleRoutesThroughSummarizer:
    """``invoke_ensemble`` on a non-raw-output ensemble returns a summary.

    Given a library with an ``analysis`` ensemble (raw_output unset)
    and a single-agent summarizer ensemble on MockModel, ``invoke_ensemble``
    must dispatch analysis, pass its full result to the Harness, and the
    Harness must invoke the summarizer and return the summarizer's
    response — not the raw analysis dict.
    """

    @pytest.mark.asyncio
    async def test_invoke_ensemble_returns_summary_not_raw_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_library_with_summarizer(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_summarize",
                name="invoke_ensemble",
                arguments={
                    "name": "analysis",
                    "input": "traffic spiked at 03:14 UTC",
                },
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.id == "call_summarize"
        assert result.name == "invoke_ensemble"
        content = result.content
        # The Harness's success path wraps the summary string in a
        # {"summary": ...} dict — the orchestrator observes the summary,
        # never the raw ensemble result.
        assert isinstance(content, dict)
        assert set(content.keys()) == {"summary"}
        summary = content["summary"]
        assert isinstance(summary, str)
        assert summary
        # MockModel echoes a deterministic prefix plus a snippet of
        # whatever message reached it. The Harness sends the JSON-
        # encoded raw analysis result as the summarizer's input, so the
        # summary carries MockModel's signature phrase.
        assert "Analysis of the data shows interesting patterns" in summary
        # And the raw analyst response MUST NOT leak into the summary
        # field — if it did, the interposition isn't running.
        assert "raw_output" not in content
        assert "results" not in content

    @pytest.mark.asyncio
    async def test_summarizer_missing_surfaces_as_typed_tool_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Harness failure → ``ToolCallError(kind="summarization_failed")``.

        Configures the Harness to look for a summarizer ensemble that
        does not exist. ``OrchestraService.invoke`` raises
        ``ValueError``; the Harness catches and returns
        ``SummarizationFailure``; Tool Dispatch converts to a typed
        tool error — never a raw-dict leak.
        """
        from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallError

        project_dir = _write_library_with_summarizer(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        # Point the Harness at a non-existent summarizer.
        harness = ResultSummarizerHarness(
            invoker=service, summarizer_name="does-not-exist"
        )
        policy = AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL)
        dispatch = OrchestratorToolDispatch(
            operations=service, harness=harness, autonomy_policy=policy
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_missing_summarizer",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "anything"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "summarization_failed"
        assert result.name == "invoke_ensemble"
        # Error observation carries a reason, never the raw dict.
        assert "does-not-exist" in result.reason
