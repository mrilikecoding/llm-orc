"""Boundary integration: Tool Dispatch → Composition Validator (WP-G).

Per ``docs/agentic-serving/system-design.md`` §Test Architecture:

    Orchestrator Tool Dispatch → Composition Validator
    test_compose_ensemble_rejects_cycle — Cyclic reference graph
    rejected at composition time.

    Composition Validator ↔ Ensemble Engine (shared)
    test_shared_validator_same_result_both_paths —
    ``validate_ensemble_reference_graph`` returns identical outcome
    when called from load path and composition path on the same input
    (scenarios regression).

The tests below use real project infrastructure: ``ConfigurationManager``
pointed at a ``tmp_path``, a real :class:`ConfigManagerPrimitiveRegistry`,
a real :class:`CompositionValidator`, and a real
:class:`ConfigManagerEnsembleWriter`. The dispatch path is exercised
end-to-end so a regression in any layer (registry existence check,
shared cycle validator, local-tier write) fails at the boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.composition_validator import (
    CompositionValidator,
    ConfigManagerEnsembleWriter,
    ConfigManagerPrimitiveRegistry,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallError,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import (
    EnsembleLoader,
    validate_ensemble_reference_graph,
)
from llm_orc.services.orchestra_service import OrchestraService


def _write_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing_ensembles: dict[str, dict[str, Any]] | None = None,
    profiles: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Lay out a local ``.llm-orc`` project for the boundary test.

    Returns the project directory ``ConfigurationManager`` reads from.
    The global XDG root is redirected into ``tmp_path`` so the test
    doesn't touch the developer's real config.
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

    resolved_profiles = profiles or {
        "default": {"model": "mock", "provider": "mock"},
    }
    (local / "config.yaml").write_text(
        yaml.safe_dump({"model_profiles": resolved_profiles})
    )

    ensembles_dir = local / "ensembles"
    ensembles_dir.mkdir()
    for name, body in (existing_ensembles or {}).items():
        (ensembles_dir / f"{name}.yaml").write_text(yaml.safe_dump(body))

    return project_dir


def _make_dispatch(service: OrchestraService) -> OrchestratorToolDispatch:
    """Wire a real Tool Dispatch with composition dependencies."""
    harness = ResultSummarizerHarness(
        invoker=service, summarizer_name="unused-no-invoke-in-this-test"
    )
    policy = AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL)
    registry = ConfigManagerPrimitiveRegistry(service.config_manager)
    validator = CompositionValidator(primitives=registry)
    writer = ConfigManagerEnsembleWriter(service.config_manager)
    return OrchestratorToolDispatch(
        operations=service,
        harness=harness,
        autonomy_policy=policy,
        composition_validator=validator,
        local_ensemble_writer=writer,
    )


class TestComposeEnsembleRejectsCycle:
    """``Orchestrator Tool Dispatch → Composition Validator`` boundary row."""

    @pytest.mark.asyncio
    async def test_compose_ensemble_rejects_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing a→b→c dangling; proposed c→a closes a→b→c→a cycle.

        Real validator walks the reference graph from the proposed
        name, catches the cycle, Tool Dispatch surfaces the typed
        tool error. No partial state on rejection — ``c.yaml`` never
        appears in the local ensembles directory.
        """
        project_dir = _write_library(
            tmp_path,
            monkeypatch,
            existing_ensembles={
                "a": {
                    "name": "a",
                    "description": "a",
                    "agents": [{"name": "ref_b", "ensemble": "b"}],
                },
                "b": {
                    "name": "b",
                    "description": "b",
                    "agents": [{"name": "ref_c", "ensemble": "c"}],
                },
            },
        )
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_c",
                name="compose_ensemble",
                arguments={
                    "name": "c",
                    "description": "closes cycle",
                    "agents": [{"name": "ref_a", "ensemble": "a"}],
                },
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert "cycle" in result.reason.lower()
        assert "c" in result.reason
        assert "a" in result.reason
        assert "b" in result.reason

        # AS-2: no partial state persists.
        target = project_dir / ".llm-orc" / "ensembles" / "c.yaml"
        assert not target.exists()


class TestComposeEnsembleAcceptsAndWrites:
    """Accept-path boundary: proposal validates, writer persists to local tier."""

    @pytest.mark.asyncio
    async def test_compose_ensemble_writes_validated_config_to_local_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_new",
                name="compose_ensemble",
                arguments={
                    "name": "new-combo",
                    "description": "assembled from existing primitives",
                    "agents": [
                        {"name": "thinker", "model_profile": "default"},
                    ],
                },
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "compose_ensemble"
        content = result.content
        assert isinstance(content, dict)
        assert content["name"] == "new-combo"
        assert content["path"].endswith(".llm-orc/ensembles/new-combo.yaml")

        written = (
            project_dir / ".llm-orc" / "ensembles" / "new-combo.yaml"
        ).read_text()
        parsed = yaml.safe_load(written)
        assert parsed["name"] == "new-combo"
        assert parsed["description"] == "assembled from existing primitives"
        agents = parsed["agents"]
        assert len(agents) == 1
        assert agents[0]["name"] == "thinker"
        assert agents[0]["model_profile"] == "default"


class TestComposeEnsembleRejectsCollision:
    """AS-2: composition never overwrites an existing ensemble on disk."""

    @pytest.mark.asyncio
    async def test_compose_ensemble_rejects_name_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_library(
            tmp_path,
            monkeypatch,
            existing_ensembles={
                "combo": {
                    "name": "combo",
                    "description": "preexisting",
                    "agents": [{"name": "a", "model_profile": "default"}],
                }
            },
        )
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_dup",
                name="compose_ensemble",
                arguments={
                    "name": "combo",
                    "description": "duplicate",
                    "agents": [
                        {"name": "b", "model_profile": "default"},
                    ],
                },
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert "already exists" in result.reason

        # The original file is untouched — the collision failure is
        # atomic with respect to the on-disk state.
        remaining = (project_dir / ".llm-orc" / "ensembles" / "combo.yaml").read_text()
        assert "preexisting" in remaining


class TestSharedValidatorSameBothPaths:
    """FC-6 regression: load-path and composition-time validator share one routine.

    Same proposed spec → same cycle outcome by calling
    ``validate_ensemble_reference_graph`` directly (load path would
    invoke the same function). The composition path's rejection,
    produced through Tool Dispatch, must name the same cycle as the
    shared routine raises in isolation.
    """

    @pytest.mark.asyncio
    async def test_shared_validator_same_result_both_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_library(
            tmp_path,
            monkeypatch,
            existing_ensembles={
                "a": {
                    "name": "a",
                    "description": "a",
                    "agents": [{"name": "ref_b", "ensemble": "b"}],
                },
                "b": {
                    "name": "b",
                    "description": "b",
                    "agents": [{"name": "ref_c", "ensemble": "c"}],
                },
            },
        )
        ensembles_dir = project_dir / ".llm-orc" / "ensembles"

        # Path 1: the shared validator called directly (as the load path does).
        from llm_orc.schemas.agent_config import parse_agent_config

        proposed_agents = [parse_agent_config({"name": "ref_a", "ensemble": "a"})]
        direct_error: str | None = None
        try:
            validate_ensemble_reference_graph(
                "c", proposed_agents, [str(ensembles_dir)]
            )
        except ValueError as exc:
            direct_error = str(exc)

        assert direct_error is not None
        assert "cycle" in direct_error.lower()

        # Path 2: the composition path through Tool Dispatch.
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        dispatch = _make_dispatch(service)
        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_c",
                name="compose_ensemble",
                arguments={
                    "name": "c",
                    "description": "cycle closer",
                    "agents": [{"name": "ref_a", "ensemble": "a"}],
                },
            )
        )

        assert isinstance(result, ToolCallError)
        # Both paths surface the same cycle path string — the composition-
        # time rejection reason is the direct validator's error.
        assert direct_error in result.reason

        # Structural property: the load path and the composition path
        # both reach :func:`validate_ensemble_reference_graph` through
        # the single canonical module. Grep the validator module for
        # the import — it is the function-by-name in the composition
        # path, identical to the load path. (Canonical module relocated
        # to core/validation at Cycle-8 WP-B8.)
        import llm_orc.core.config.ensemble_config as ec
        import llm_orc.core.validation.composition_validator as cv_module

        cv_source = Path(cv_module.__file__).read_text()
        assert "validate_ensemble_reference_graph" in cv_source
        assert "from llm_orc.core.config.ensemble_config import" in cv_source, (
            "composition validator must import the shared routine by module"
        )
        assert EnsembleLoader.load_from_file.__module__ == ec.__name__


class TestComposeEnsembleNeverAuthorsPrimitives:
    """Scenario: Composition never authors scripts or profiles (structural)."""

    def test_tool_names_do_not_include_script_or_profile_authoring(self) -> None:
        """AS-6 closure: the committed tool set has no script/profile writer.

        This is a structural assertion — the closed set is the
        enforcement point. A future tool that wrote scripts or
        profiles would require adding a name to ``TOOL_NAMES``, which
        would fail FC-5 (exactly five entries).
        """
        from llm_orc.agentic.orchestrator_tool_dispatch import TOOL_NAMES

        assert TOOL_NAMES == frozenset(
            {
                "invoke_ensemble",
                "compose_ensemble",
                "list_ensembles",
                "query_knowledge",
                "record_outcome",
            }
        )
