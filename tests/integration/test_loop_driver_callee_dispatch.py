"""Integration: Loop Driver callee delegation through the real Tool Dispatch.

The loop-driver unit tests exercise the callee path against a
``_FakeToolDispatch``. This test replaces that stub with the real
:class:`OrchestratorToolDispatch` driving a real (mock-model, $0) capability
ensemble, verifying the boundary the unit tests could not: that the
loop-driver's ``invoke_ensemble`` argument shape (``{name, input}``) is
accepted by the real dispatch's argument validation, the ensemble executes,
and the deliverable flows back into a client ``write`` tool call.

Deliverable *content fidelity* for substrate-routed capability ensembles is
the Artifact Bridge's job (WP-LB-D); the Client-Tool-Action Terminal marshals
the inline ``primary`` directly here. This test asserts the structural
integration of the callee boundary, not large-deliverable fidelity.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Loop Driver
(callee delegation; FC-44) and the build skill's Step 5 (Integration
Verification — replace ``MockX`` with the real ``X`` at one boundary).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.agentic.artifact_bridge import ArtifactBridge
from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.client_tool_action_terminal import ClientToolActionTerminal
from llm_orc.agentic.composition_validator import (
    CompositionValidator,
    ConfigManagerEnsembleWriter,
    ConfigManagerPrimitiveRegistry,
)
from llm_orc.agentic.loop_driver import LoopDriver
from llm_orc.agentic.orchestrator_chunk import ClientToolCall, OrchestratorChunk
from llm_orc.agentic.orchestrator_tool_dispatch import OrchestratorToolDispatch
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.models.base import ToolCall, ToolCallingResponse
from llm_orc.services.orchestra_service import OrchestraService


class _FixedSeatFiller:
    """Seat-filler double emitting one scripted invoke_ensemble decision."""

    def __init__(self, response: ToolCallingResponse) -> None:
        self._response = response

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        return self._response


class _FakeJudgmentSeat:
    """Judgment-seat double — the contexts here never reach a judgment."""

    async def generate_response(self, message: str, role_prompt: str) -> str:
        return "VERDICT: REMAINING\n"


def _write_capability_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a local ``.llm-orc`` with one mock-model capability ensemble."""
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
            {"model_profiles": {"local-mock": {"model": "mock", "provider": "mock"}}}
        )
    )
    ensembles_dir = local / "ensembles"
    ensembles_dir.mkdir()
    (ensembles_dir / "code-gen-mock.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "code-gen-mock",
                "description": "Generate code with a mock model.",
                "raw_output": True,
                "agents": [
                    {
                        "name": "coder",
                        "model_profile": "local-mock",
                        "system_prompt": "You are a code generator.",
                    }
                ],
            }
        )
    )
    return project_dir


def _make_dispatch(service: OrchestraService) -> OrchestratorToolDispatch:
    return OrchestratorToolDispatch(
        operations=service,
        harness=ResultSummarizerHarness(
            invoker=service, summarizer_name="agentic-result-summarizer"
        ),
        autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
        composition_validator=CompositionValidator(
            primitives=ConfigManagerPrimitiveRegistry(service.config_manager)
        ),
        local_ensemble_writer=ConfigManagerEnsembleWriter(service.config_manager),
    )


async def _collect(
    chunks: AsyncIterator[OrchestratorChunk],
) -> list[OrchestratorChunk]:
    return [chunk async for chunk in chunks]


@pytest.mark.asyncio
async def test_callee_generation_dispatches_real_ensemble_into_a_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _write_capability_library(tmp_path, monkeypatch)
    config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
    service = OrchestraService(config_manager=config_manager)
    dispatch = _make_dispatch(service)

    seat_filler = _FixedSeatFiller(
        ToolCallingResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="invoke_ensemble",
                    arguments_json=json.dumps(
                        {
                            "name": "code-gen-mock",
                            "input": "write a fibonacci function",
                            "filePath": "fib.py",
                        }
                    ),
                )
            ],
            finish_reason="tool_calls",
        )
    )
    driver = LoopDriver(
        seat_filler=seat_filler,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=dispatch,
        action_record=SessionActionRecord(),
        judgment_seat=_FakeJudgmentSeat(),
        budget=BudgetController(turn_limit=1_000, token_limit=1_000_000),
    )
    terminal = ClientToolActionTerminal(
        loop_driver=driver,
        bridge=ArtifactBridge(SessionArtifactStore(agentic_sessions_root=tmp_path)),
    )
    context = SessionContext(
        messages=[ChatMessage(role="user", content="write a fibonacci function")],
        tools=[{"type": "function", "function": {"name": "write"}}],
        state=SessionState(
            identity=SessionIdentity(value="callee-int", method="user_field")
        ),
    )

    chunks = await _collect(terminal.run(context))

    tool_calls = [c for c in chunks if isinstance(c, ClientToolCall)]
    assert len(tool_calls) == 1
    invocation = tool_calls[0].tool_calls[0]
    assert invocation.name == "write"
    args = json.loads(invocation.arguments)
    assert args["filePath"] == "fib.py"
    # The real ensemble executed (MockModel echoes the input into its
    # response), so the marshalled deliverable carries it — proving the
    # invoke_ensemble call passed the real dispatch's validation and routed
    # through the real Ensemble Engine, not a stub.
    assert "write a fibonacci function" in args["content"]
