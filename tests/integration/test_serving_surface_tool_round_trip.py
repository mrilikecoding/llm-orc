"""Integration: the real ASGI ``/v1/chat/completions`` tool round-trip.

The practitioner's "don't fly blind" directive (cycle-status §Feed-Forward
WP-LB-D #3; memory ``validate-against-real-client-not-harness``): WP-LB-C must
drive the *actual* Serving Layer surface, not a loop-driver fixture — the WP-A
scar was that "the architecture being kept was never run through the client it
exists to serve." So this test stands up the real ASGI app and exercises the
**real** Client-Tool-Action Terminal composition (real surface-mode
discriminator → real Loop Driver → real Artifact Bridge → real SSE / body
shaping); only the seat-filler (the model) and the Tool Dispatch are doubled,
so the round-trip is deterministic and $0.

It covers the parity round-trip the loop-back exists to deliver:

* a tool-carrying request yields a real ``finish_reason: "tool_calls"`` response
  carrying **bridge-marshalled** (substrate-routed, full-fidelity) content —
  streaming and non-streaming (FC-47 + FC-49);
* a ``role: "tool"`` follow-up routes the client's tool result back into the
  loop driver, which decides the next turn (FC-50).

The remaining north-star validation — a real OpenCode session against
``llm-orc serve`` — is the $0 manual smoke test in
``docs/agentic-serving/references/opencode-smoke-test.md``; it cannot run in CI
($0 but needs a real OpenCode install + local Ollama).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.loop_driver import compose_form_directive
from llm_orc.agentic.orchestrator_config import (
    BudgetDefaults,
    CalibrationDefaults,
    OrchestratorConfig,
    OverrideBounds,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
from llm_orc.agentic.session_registry import SessionRegistry
from llm_orc.agentic.session_start import SessionStartCache
from llm_orc.models.base import ToolCall, ToolCallingResponse
from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app

_DELIVERABLE = "def quicksort(xs):\n    return sorted(xs)\n"
_WRITE_TOOLS = [{"type": "function", "function": {"name": "write"}}]


def _minimal_config() -> OrchestratorConfig:
    """A canned config; observability defaults serve the lifecycle helpers."""
    return OrchestratorConfig(
        model_profile="test-profile",
        budget=BudgetDefaults(turn_limit=10, token_limit=10_000),
        autonomy_level="operator-as-tool-user",
        plexus_enabled=False,
        override_bounds=OverrideBounds(
            allow_budget_override=True,
            max_turn_limit=100,
            max_token_limit=100_000,
        ),
        allowed_profiles=("test-profile",),
        summarizer_ensemble="agentic-result-summarizer",
        orchestrator_system_prompt="",
        calibration=CalibrationDefaults(
            default_n=3, checker_ensemble="agentic-calibration-checker"
        ),
    )


class _FakeConfigResolver:
    """Returns a canned config.

    ``resolve_validated`` feeds the serving body; ``resolve`` feeds
    ``get_loop_driver`` (the AS-3 budget read, FC-69) — both return the
    same canned config here.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

    def resolve_validated(self) -> OrchestratorConfig:
        return self._config

    def resolve(self) -> OrchestratorConfig:
        return self._config


class _ScriptedSeatFiller:
    """Seat-filler double returning one scripted response per turn.

    Records the messages it was handed each turn so the loop-participation
    test can assert the client's ``role: "tool"`` result reached the driver.
    """

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self._responses = list(responses)
        self.surfaced: list[list[dict[str, Any]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        self.surfaced.append(messages)
        return self._responses.pop(0)

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """The judgment seat shares the seat-filler's model (FC-68 default).

        ADR-037: a trailing tool-result tail opens with the termination
        judgment. REMAINING keeps this test on the call-2 path, so the
        scripted action responses (and the FC-50 assertion that the tool
        result reached the driver) ride unchanged.
        """
        return "VERDICT: REMAINING\nThe requested follow-up is still open."


class _StoreWritingDispatch:
    """Tool-dispatch double that routes its deliverable through the store.

    Mimics a production ``output_substrate: artifact`` capability ensemble
    (ADR-025): writes the deliverable to the Session Artifact Store and returns
    an envelope whose ``primary`` is a summary and whose ``artifacts[0]`` points
    at the stored content — so the Terminal's Artifact Bridge must read the
    store to marshal full-fidelity content.
    """

    def __init__(self, store: SessionArtifactStore, deliverable: str) -> None:
        self._store = store
        self._deliverable = deliverable
        self.calls: list[InternalToolCall] = []

    async def dispatch(
        self,
        call: InternalToolCall,
        *,
        session_id: str = "",
        model_profile_override: str | None = None,
    ) -> ToolCallResult:
        self.calls.append(call)
        ref = self._store.write_deliverable(
            session_id=session_id or "session",
            dispatch_id="dispatch-1",
            deliverable_name="deliverable",
            content=self._deliverable,
            content_type="application/python",
        )
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content="(substrate-routed)",
            envelope=DispatchEnvelope(
                status="success",
                primary="deliverable.py: a quicksort — summary, not the content",
                artifacts=[dataclasses.asdict(ref)],
            ),
        )


def _wire_real_terminal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seat_filler: _ScriptedSeatFiller,
    dispatch: _StoreWritingDispatch,
    store: SessionArtifactStore,
) -> TestClient:
    """Stand up the app with the REAL terminal/loop-driver/bridge composition.

    Only the seat-filler (model), the Tool Dispatch, the artifact store, and the
    config/registry are doubled; ``get_client_tool_action_terminal`` and
    ``get_loop_driver`` are left real so the production wiring is exercised.
    """
    monkeypatch.setattr(v1_chat_completions, "get_session_registry", SessionRegistry)
    monkeypatch.setattr(
        v1_chat_completions, "get_session_start_cache", SessionStartCache
    )
    monkeypatch.setattr(
        v1_chat_completions,
        "get_orchestrator_config_resolver",
        lambda: _FakeConfigResolver(_minimal_config()),
    )

    async def _resolve_seat_filler() -> _ScriptedSeatFiller:
        return seat_filler

    monkeypatch.setattr(
        v1_chat_completions, "_resolve_seat_filler", _resolve_seat_filler
    )
    monkeypatch.setattr(
        v1_chat_completions, "get_orchestrator_tool_dispatch", lambda: dispatch
    )
    monkeypatch.setattr(
        v1_chat_completions, "get_session_artifact_store", lambda: store
    )
    return TestClient(create_app())


def _invoke_ensemble_turn() -> ToolCallingResponse:
    return ToolCallingResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments_json=json.dumps(
                    {
                        "name": "code-generator",
                        "input": "write a quicksort",
                        "filePath": "quicksort.py",
                    }
                ),
            )
        ],
        finish_reason="tool_calls",
    )


def _parse_sse_frames(body: bytes) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for block in body.split(b"\n\n"):
        stripped = block.strip()
        if not stripped or not stripped.startswith(b"data: "):
            continue
        payload = stripped[len(b"data: ") :]
        if payload == b"[DONE]":
            continue
        frames.append(json.loads(payload))
    return frames


def test_non_streaming_tool_request_emits_bridge_marshalled_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    seat_filler = _ScriptedSeatFiller([_invoke_ensemble_turn()])
    dispatch = _StoreWritingDispatch(store, _DELIVERABLE)
    client = _wire_real_terminal(
        monkeypatch, seat_filler=seat_filler, dispatch=dispatch, store=store
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "write a quicksort to a file"}],
            "tools": _WRITE_TOOLS,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "write"
    args = json.loads(tool_call["function"]["arguments"])
    assert args["filePath"] == "quicksort.py"
    # Full bridge-marshalled content, not the envelope's summary line.
    assert args["content"] == _DELIVERABLE
    assert dispatch.calls, "the loop driver delegated generation to the ensemble"
    # FC-53 at the real serving composition: the callee dispatch input
    # carries the write-keyed bare-output form directive (ADR-035) after
    # the seat-filler's generation task.
    dispatched_input = dispatch.calls[0].arguments["input"]
    assert dispatched_input.startswith("write a quicksort")
    assert compose_form_directive("write") in dispatched_input


def test_no_server_side_write_to_the_client_filesystem_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FC-48 — the deliverable reaches the client only via a tool call.

    The server-side write lands in the artifact store (substrate), not at the
    client's target path: the Terminal has no client-filesystem-write code path
    (Spike π Phase A's rejected shape is absent — satisfied by absence, guarded
    here). The client executes the ``write`` itself.
    """
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    seat_filler = _ScriptedSeatFiller([_invoke_ensemble_turn()])
    dispatch = _StoreWritingDispatch(store, _DELIVERABLE)
    client = _wire_real_terminal(
        monkeypatch, seat_filler=seat_filler, dispatch=dispatch, store=store
    )

    client.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "write a quicksort to a file"}],
            "tools": _WRITE_TOOLS,
        },
    )

    # The client's target path was never written server-side; the deliverable
    # lives only in the artifact store under the agentic-sessions root.
    assert not (tmp_path / "quicksort.py").exists()
    assert not Path("quicksort.py").exists()
    stored = list(tmp_path.rglob("deliverable.py"))
    assert len(stored) == 1
    assert stored[0].read_text() == _DELIVERABLE


def test_streaming_tool_request_emits_bridge_marshalled_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    seat_filler = _ScriptedSeatFiller([_invoke_ensemble_turn()])
    dispatch = _StoreWritingDispatch(store, _DELIVERABLE)
    client = _wire_real_terminal(
        monkeypatch, seat_filler=seat_filler, dispatch=dispatch, store=store
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "write a quicksort to a file"}],
            "tools": _WRITE_TOOLS,
            "stream": True,
        },
    )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    tool_call_frames = [
        frame for frame in frames if frame["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_call_frames) == 1
    frame = tool_call_frames[0]
    assert frame["choices"][0]["finish_reason"] == "tool_calls"
    delta_call = frame["choices"][0]["delta"]["tool_calls"][0]
    assert delta_call["function"]["name"] == "write"
    args = json.loads(delta_call["function"]["arguments"])
    assert args["content"] == _DELIVERABLE


def test_tool_result_follow_up_continues_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FC-50 — a ``role: "tool"`` follow-up reaches the driver, which finishes.

    The judgment seat returns REMAINING (``_ScriptedSeatFiller``), so the
    trailing tail falls through to the action call where the tool result is
    surfaced (FC-50). The seat then stalls; the F-σ.1 REMAINING-retry re-asks
    once and, on a second stall, the driver finishes — so two stall responses
    are scripted (the AS-3 cap remains the ultimate backstop beyond that).
    """
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    seat_filler = _ScriptedSeatFiller(
        [
            ToolCallingResponse(content="Done — quicksort written.", tool_calls=[]),
            ToolCallingResponse(content="Done — quicksort written.", tool_calls=[]),
        ]
    )
    dispatch = _StoreWritingDispatch(store, _DELIVERABLE)
    client = _wire_real_terminal(
        monkeypatch, seat_filler=seat_filler, dispatch=dispatch, store=store
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [
                {"role": "user", "content": "write a quicksort to a file"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": json.dumps(
                                    {
                                        "filePath": "quicksort.py",
                                        "content": _DELIVERABLE,
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": "ok: wrote quicksort.py (42 bytes)",
                },
            ],
            "tools": _WRITE_TOOLS,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Done — quicksort written."
    assert not dispatch.calls, "a finish turn delegates no generation"
    # The client's tool result reached the driver's per-turn decision (FC-50).
    surfaced = seat_filler.surfaced[0]
    assert any(
        message.get("content") == "ok: wrote quicksort.py (42 bytes)"
        for message in surfaced
    )
