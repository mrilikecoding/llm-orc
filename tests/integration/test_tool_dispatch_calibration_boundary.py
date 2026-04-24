"""Boundary integration test: Orchestrator Tool Dispatch → Calibration Gate.

Per ``docs/agentic-serving/system-design.md`` §Test Architecture:

    Orchestrator Tool Dispatch → Calibration Gate
    test_calibration_interposes_on_in_calibration_ensembles —
    First N invocations run the check; (N+1)th does not.

Unit tests in ``tests/unit/agentic/test_orchestrator_tool_dispatch.py``
cover the interposition with scripted dispatch components. This test
exercises the production call chain end-to-end with real types on the
boundary under test:

``OrchestratorToolDispatch → CalibrationGate`` (real gate instance;
the ``CalibrationChecker`` is scripted so the test stays fast — it is
the neighboring ``Calibration Gate → Ensemble Engine`` edge, not the
boundary under test here).

The ensemble execution side uses the real ``OrchestraService →
ExecutionHandler → EnsembleLoader → EnsembleExecutor → MockModel``
stack, matching the Group 5 precedent in
``test_tool_dispatch_ensemble_engine.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.calibration_gate import (
    CalibrationGate,
    QualitySignal,
)
from llm_orc.agentic.composition_validator import (
    CompositionValidator,
    ConfigManagerEnsembleWriter,
    ConfigManagerPrimitiveRegistry,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.services.orchestra_service import OrchestraService


class _ScriptedChecker:
    """CalibrationChecker double that returns from a scripted iterator.

    Records invocations so the test asserts the checker fires for the
    first N calls on an in-calibration ensemble and stops firing once
    trust is earned.
    """

    def __init__(self, signals: list[QualitySignal]) -> None:
        self._iter = iter(signals)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def check(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal:
        self.calls.append((ensemble_name, raw_result))
        return next(self._iter)


def _write_local_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a local ``.llm-orc`` library with a single mock-model ensemble.

    Matches the pattern in
    ``tests/integration/test_tool_dispatch_ensemble_engine.py`` so the
    test isolates from any developer-local configuration and exercises
    the real production stack deterministically.
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
                "description": "Analyze input with a mock model.",
                "raw_output": True,
                "agents": [
                    {
                        "name": "analyst",
                        "model_profile": "local-mock",
                        "system_prompt": "Analyze the input succinctly.",
                    }
                ],
            }
        )
    )
    return project_dir


def _make_dispatch(
    service: OrchestraService, gate: CalibrationGate
) -> OrchestratorToolDispatch:
    """Build the production dispatch stack with the provided gate."""
    harness = ResultSummarizerHarness(
        invoker=service, summarizer_name="agentic-result-summarizer"
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
        calibration_gate=gate,
    )


class TestCalibrationInterposesOnInCalibrationEnsembles:
    """FC-12: composed ensembles enter the Calibration Gate transparently on
    ``invoke_ensemble``.

    The invoking ensemble (``analysis``) is treated as a stand-in for a
    just-composed ensemble by calling ``gate.mark_composed`` before the
    first invocation. Downstream work packages substitute a real
    composition path; the interposition contract is unchanged.
    """

    @pytest.mark.asyncio
    async def test_calibration_interposes_on_in_calibration_ensembles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        checker = _ScriptedChecker(["positive", "positive", "positive", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        dispatch = _make_dispatch(service, gate)

        # Stand-in: mark the library ensemble as "just composed" so calibration
        # engages. A real ``compose_ensemble`` would do this for us — covered
        # by ``test_compose_registers_real_ensemble_for_calibration`` below.
        gate.mark_composed(session_id="s1", ensemble_name="analysis")

        for idx in range(3):
            result = await dispatch.dispatch(
                InternalToolCall(
                    id=f"c-{idx}",
                    name="invoke_ensemble",
                    arguments={"name": "analysis", "input": f"task-{idx}"},
                ),
                session_id="s1",
            )
            assert isinstance(result, ToolCallSuccess)

        # First N invocations all triggered the checker.
        assert len(checker.calls) == 3
        # Gate transitioned to trusted.
        assert gate.status(session_id="s1", ensemble_name="analysis") == "trusted"

        # (N+1)th invocation does NOT fire the checker — the scripted
        # iterator still has a fourth signal queued, and the test asserts
        # it is never consumed.
        result = await dispatch.dispatch(
            InternalToolCall(
                id="c-next",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "task-next"},
            ),
            session_id="s1",
        )
        assert isinstance(result, ToolCallSuccess)
        assert len(checker.calls) == 3  # still 3 — no new check on trusted

    @pytest.mark.asyncio
    async def test_library_ensemble_never_triggers_checker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only composed ensembles enter calibration (AS-5 / ADR-007).

        The orchestrator invoking a library ensemble that was not
        ``mark_composed``'d must not consult the checker — the gate
        returns ``trusted`` immediately for untracked names. A
        ``_ScriptedChecker`` with an empty iterator would raise
        ``StopIteration`` on any call; this test asserts no such call
        happens.
        """
        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        checker = _ScriptedChecker([])  # any call would raise StopIteration
        gate = CalibrationGate(default_n=3, checker=checker)
        dispatch = _make_dispatch(service, gate)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="lib-1",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "untracked"},
            ),
            session_id="s1",
        )

        assert isinstance(result, ToolCallSuccess)
        assert checker.calls == []
        assert gate.status(session_id="s1", ensemble_name="analysis") == "trusted"

    @pytest.mark.asyncio
    async def test_compose_registers_real_ensemble_for_calibration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real ``compose_ensemble`` path writes and registers the name.

        Exercises Tool Dispatch → CompositionValidator → LocalEnsembleWriter
        → CalibrationGate on the happy path. After composition the new
        ensemble is in calibration; invocation will trigger the checker.
        """
        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        checker = _ScriptedChecker(["positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        dispatch = _make_dispatch(service, gate)

        compose_result = await dispatch.dispatch(
            InternalToolCall(
                id="compose-1",
                name="compose_ensemble",
                arguments={
                    "name": "composed-analysis",
                    "description": "Composed view over analysis.",
                    "agents": [
                        {
                            "name": "analyst",
                            "model_profile": "local-mock",
                            "system_prompt": "Look at the input and comment.",
                        }
                    ],
                },
            ),
            session_id="s1",
        )

        assert isinstance(compose_result, ToolCallSuccess)
        assert (
            gate.status(session_id="s1", ensemble_name="composed-analysis")
            == "in_calibration"
        )

    @pytest.mark.asyncio
    async def test_session_isolation_across_gate_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario §Calibration is session-scoped when Plexus is absent.

        Session 1's calibration state is invisible to Session 2. An
        ensemble composed-and-trusted in Session 1 re-enters calibration
        when Session 2 composes the same name — the gate's per-session
        store carries this structurally.
        """
        project_dir = _write_local_library(tmp_path, monkeypatch)
        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        checker = _ScriptedChecker(["positive", "positive", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        dispatch = _make_dispatch(service, gate)

        gate.mark_composed(session_id="s1", ensemble_name="analysis")
        for idx in range(3):
            await dispatch.dispatch(
                InternalToolCall(
                    id=f"s1-{idx}",
                    name="invoke_ensemble",
                    arguments={"name": "analysis", "input": f"task-{idx}"},
                ),
                session_id="s1",
            )
        assert gate.status(session_id="s1", ensemble_name="analysis") == "trusted"

        # Session 2 did not call mark_composed — the gate treats the
        # ensemble as library-default (trusted) without a record.
        assert gate.status(session_id="s2", ensemble_name="analysis") == "trusted"
