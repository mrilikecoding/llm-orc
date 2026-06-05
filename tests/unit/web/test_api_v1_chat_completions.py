"""Tests for the Serving Layer ``/v1/chat/completions`` endpoint.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3) and
ADR-027. Cycle 7 WP-A swapped the handler's caller from
``OrchestratorRuntime`` to the framework-driven :class:`DispatchPipeline`.
The endpoint:

- Parses the OpenAI-compatible request (messages, model, tools, user,
  stream).
- Resolves a ``SessionIdentity`` via Session Registry.
- Calls ``resolve_session_start_context`` exactly once per session
  start (FC-9); Phase 1 returns ``[]``.
- Drives the Dispatch Pipeline (plan → dispatch → synthesize) and shapes
  its ``OrchestratorChunk`` stream into a ``chat.completion`` body or SSE.

Shape/streaming/session tests drive a :class:`_StubPipeline`; the
``TestDispatchPipelineHandlerSwap`` acceptance tests drive a real
:class:`DispatchPipeline` wired to stub ports through the HTTP boundary.
"""

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_orc.agentic.artifact_bridge import ArtifactBridge
from llm_orc.agentic.client_tool_action_terminal import ClientToolActionTerminal
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.dispatch_pipeline import DispatchPipeline, DispatchPlan
from llm_orc.agentic.loop_driver import LoopDriver, TurnDecision
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.agentic.orchestrator_config import (
    BudgetDefaults,
    CalibrationDefaults,
    OrchestratorConfig,
    OverrideBounds,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
from llm_orc.agentic.session_registry import SessionIdentity, SessionRegistry
from llm_orc.agentic.session_start import (
    PromptFragment,
    SessionContext,
    SessionStartCache,
)
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCallingResponse
from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app


class _StubPipeline:
    """A scripted Dispatch Pipeline double for shape/streaming/session tests.

    Yields a fixed ``OrchestratorChunk`` sequence; the default is a single
    ``Completion(stop)`` so the response body is empty content +
    ``finish_reason: stop`` — the skeleton behavior the preserved
    shape/streaming/session tests assert.
    """

    def __init__(self, chunks: list[OrchestratorChunk] | None = None) -> None:
        self._chunks: list[OrchestratorChunk] = (
            list(chunks) if chunks is not None else [Completion(finish_reason="stop")]
        )
        self.contexts: list[SessionContext] = []

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        self.contexts.append(context)
        for chunk in self._chunks:
            yield chunk


class _FakeSeatFiller:
    """Seat-filler double returning a fixed tool-calling response.

    Stands in for the resolved Model Profile so loop-driver wiring tests do
    not stand up real model resolution.
    """

    def __init__(self, response: ToolCallingResponse) -> None:
        self._response = response

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        return self._response


class _NoToolDispatch:
    """Tool-dispatch double that must not be called (finish-with-text paths)."""

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        raise AssertionError("the finish-with-text path should not dispatch")


class _CapturingSink:
    """Event sink recording every event the substrate fans out."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def consume(self, event: object) -> None:
        self.events.append(event)


class _FakeJudgmentSeat:
    """Judgment-seat double — the contexts here never reach a judgment."""

    async def generate_response(self, message: str, role_prompt: str) -> str:
        return "VERDICT: REMAINING\n"


def _real_terminal(
    seat_filler: _FakeSeatFiller, substrate: DispatchEventSubstrate
) -> ClientToolActionTerminal:
    """The tool-driven caller: the real Terminal over the real Loop Driver.

    The endpoint drives the Terminal (the ``_ChatCompletionsCaller``); the
    Terminal composes the Loop Driver, which still emits the ``TurnDecision``
    diagnostics the FC-42 tests assert.
    """
    store = SessionArtifactStore(agentic_sessions_root=Path("unused-by-finish-path"))
    return ClientToolActionTerminal(
        loop_driver=LoopDriver(
            seat_filler=seat_filler,
            enforcer=SingleStepEnforcer(),
            tool_dispatch=_NoToolDispatch(),
            action_record=SessionActionRecord(),
            judgment_seat=_FakeJudgmentSeat(),
            event_substrate=substrate,
        ),
        bridge=ArtifactBridge(store),
    )


class _StubPlanner:
    """Stage-1 RoutingPlanner port double returning a fixed plan."""

    def __init__(self, plan: DispatchPlan | None) -> None:
        self._plan = plan

    async def plan(self, *, request: str) -> DispatchPlan | None:
        return self._plan


class _StubSynthesizer:
    """Stage-3 ResponseSynthesizer port double yielding fixed text.

    Records each call so acceptance tests can assert the pipeline fed the
    synthesizer the dispatch output (capability path) or an empty
    DISPATCH RESULTS section (direct path).
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    async def synthesize(
        self,
        *,
        original_request: str,
        dispatched: list[str],
        planned_but_not_run: list[str],
        dispatch_results: list[tuple[str, str]],
    ) -> AsyncIterator[str]:
        self.calls.append(
            {
                "original_request": original_request,
                "dispatched": dispatched,
                "dispatch_results": dispatch_results,
            }
        )
        yield self._text


class _StubDispatcher:
    """Stage-2 ToolDispatcher port double returning a fixed success result."""

    def __init__(self, content: str = "DISPATCH OUTPUT") -> None:
        self._content = content
        self.calls: list[InternalToolCall] = []

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        self.calls.append(call)
        return ToolCallSuccess(id=call.id, name=call.name, content=self._content)


def _default_orchestrator_config() -> OrchestratorConfig:
    """An OrchestratorConfig used when tests don't care about its contents.

    The pipeline factory and lifecycle helpers read it via the
    :class:`_FakeConfigResolver`; the field values are arbitrary defaults.
    """
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
    """Returns a canned ``OrchestratorConfig`` without touching the filesystem.

    Satisfies the duck-typed surface of ``OrchestratorConfigResolver``
    the handler reads — only ``resolve_validated`` is used.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

    def resolve_validated(self) -> OrchestratorConfig:
        return self._config


def _parse_sse_frames(body: bytes) -> list[dict[str, Any]]:
    """Split an SSE body into parsed frames.

    Each frame is either the JSON payload of a ``data:`` line or the
    sentinel ``{"__done__": True}`` marking the literal ``[DONE]``
    terminator. Blank lines between frames are preserved as separators
    by SSE but don't produce frames here.
    """
    frames: list[dict[str, Any]] = []
    for block in body.split(b"\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        assert stripped.startswith(b"data: "), f"unexpected SSE line: {block!r}"
        payload = stripped[len(b"data: ") :]
        if payload == b"[DONE]":
            frames.append({"__done__": True})
            continue
        frames.append(json.loads(payload))
    return frames


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry: SessionRegistry | None = None,
    session_start_spy: (Callable[[SessionContext], list[PromptFragment]] | None) = None,
    pipeline: "_StubPipeline | DispatchPipeline | None" = None,
    terminal: "_StubPipeline | ClientToolActionTerminal | None" = None,
    config: OrchestratorConfig | None = None,
) -> tuple[TestClient, SessionRegistry, SessionStartCache]:
    """Wire a TestClient with isolated Registry, cache, and Dispatch Pipeline.

    Overrides the module-level factories in ``v1_chat_completions`` so each
    test runs the real Serving Layer wiring (session resolution, SSE
    framing, heartbeat + context-sink lifecycle) but with a stub Dispatch
    Pipeline (ADR-027) in place of the ensemble-backed pipeline. The
    default :class:`_StubPipeline` yields a single ``Completion(stop)`` —
    empty assistant content + ``finish_reason: stop`` — preserving the
    skeleton behavior the shape/streaming/session tests assert.

    Acceptance tests pass a real :class:`DispatchPipeline` wired to stub
    ports via ``pipeline=`` to exercise plan → dispatch → synthesize
    through the HTTP boundary.
    """
    shared_registry = registry or SessionRegistry()
    monkeypatch.setattr(
        v1_chat_completions, "get_session_registry", lambda: shared_registry
    )
    cache = (
        SessionStartCache(resolver=session_start_spy)
        if session_start_spy is not None
        else SessionStartCache()
    )
    monkeypatch.setattr(v1_chat_completions, "get_session_start_cache", lambda: cache)

    stub_pipeline: _StubPipeline | DispatchPipeline = pipeline or _StubPipeline()

    async def _pipeline_factory() -> _StubPipeline | DispatchPipeline:
        return stub_pipeline

    monkeypatch.setattr(v1_chat_completions, "get_dispatch_pipeline", _pipeline_factory)

    stub_terminal: _StubPipeline | ClientToolActionTerminal = (
        terminal or _StubPipeline()
    )

    async def _terminal_factory() -> _StubPipeline | ClientToolActionTerminal:
        return stub_terminal

    monkeypatch.setattr(
        v1_chat_completions, "get_client_tool_action_terminal", _terminal_factory
    )

    resolver = _FakeConfigResolver(config or _default_orchestrator_config())
    monkeypatch.setattr(
        v1_chat_completions,
        "get_orchestrator_config_resolver",
        lambda: resolver,
    )

    return TestClient(create_app()), shared_registry, cache


class TestChatCompletionsResponseShape:
    """``POST /v1/chat/completions`` returns an OpenAI-shaped body.

    Group 4 skeleton: empty assistant content + ``finish_reason: "stop"``
    + zero ``usage``. WP-C replaces this with real Runtime output.
    """

    def test_returns_openai_chat_completion_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "refactor the parser"}],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert isinstance(body["id"], str)
        assert body["id"]
        assert isinstance(body["created"], int)
        assert body["model"] == "primary"

    def test_returns_single_choice_with_stop_finish_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        body = response.json()
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == ""

    def test_returns_zero_usage_in_skeleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skeleton does no model calls, so usage is zero across the board.

        WP-C wires real token accounting via Budget Controller →
        Session Registry.
        """
        client, _, _ = _build_client(monkeypatch)

        body = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
            },
        ).json()

        assert body["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestChatCompletionsRequestParsing:
    """The endpoint accepts optional ``tools`` and ``user`` fields."""

    def test_accepts_tools_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client-declared tools parse; under ADR-033 they engage the loop-driver.

        WP-LB-A routes a ``tools[]`` request to the loop-driver surface
        (here the default :class:`_StubPipeline` loop-driver double), which
        finishes with a stop completion — the request still returns 200.
        Surface-mode discrimination itself is asserted in
        :class:`TestSurfaceModeDiscrimination`.
        """
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bash", "description": "run shell"},
                    }
                ],
            },
        )

        assert response.status_code == 200

    def test_accepts_user_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``user`` field supplies Session identity correlation."""
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )

        assert response.status_code == 200


class TestSurfaceModeDiscrimination:
    """``POST /v1/chat/completions`` routes by client-tools presence (ADR-033 D1).

    A request carrying client ``tools[]`` engages the layer-A loop-driver;
    a request with no client tools continues through ADR-027's single-turn
    Dispatch Pipeline. The discriminator is the presence of client tools.
    Each surface is injected as a distinguishable :class:`_StubPipeline`
    caller so the route is observable in the response content at the HTTP
    boundary.
    """

    @staticmethod
    def _scripted(label: str) -> "_StubPipeline":
        return _StubPipeline(
            [ContentDelta(content=label), Completion(finish_reason="stop")]
        )

    def test_tool_request_engages_loop_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline_caller = self._scripted("PIPELINE")
        terminal_caller = self._scripted("LOOPDRIVER")
        client, _, _ = _build_client(
            monkeypatch, pipeline=pipeline_caller, terminal=terminal_caller
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "write the config file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "write", "description": "write a file"},
                    }
                ],
            },
        )

        body = response.json()
        assert body["choices"][0]["message"]["content"] == "LOOPDRIVER"
        assert terminal_caller.contexts, "loop-driver should be driven"
        assert not pipeline_caller.contexts, "pipeline should not be driven"

    def test_non_tool_request_uses_single_turn_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline_caller = self._scripted("PIPELINE")
        terminal_caller = self._scripted("LOOPDRIVER")
        client, _, _ = _build_client(
            monkeypatch, pipeline=pipeline_caller, terminal=terminal_caller
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "what is a monad?"}],
            },
        )

        body = response.json()
        assert body["choices"][0]["message"]["content"] == "PIPELINE"
        assert pipeline_caller.contexts, "pipeline should be driven"
        assert not terminal_caller.contexts, "loop-driver should not be driven"

    def test_tool_request_engages_loop_driver_on_streaming_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discrimination happens before the stream/non-stream branch (wiring)."""
        pipeline_caller = self._scripted("PIPELINE")
        terminal_caller = self._scripted("LOOPDRIVER")
        client, _, _ = _build_client(
            monkeypatch, pipeline=pipeline_caller, terminal=terminal_caller
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "edit the parser"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "edit", "description": "edit a file"},
                    }
                ],
            },
        )

        frames = _parse_sse_frames(response.content)
        deltas = [
            frame["choices"][0]["delta"].get("content", "")
            for frame in frames
            if not frame.get("__done__")
        ]
        assert "".join(deltas) == "LOOPDRIVER"
        assert terminal_caller.contexts
        assert not pipeline_caller.contexts

    async def test_terminal_factory_composes_the_real_loop_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WP-LB-C: the production caller factory builds the real Terminal.

        :func:`get_loop_driver` builds the real Loop Driver (seat-filler
        resolved via the overridable :func:`_resolve_seat_filler` seam, a
        double here so the test does not stand up model resolution);
        :func:`get_client_tool_action_terminal` composes it into the Terminal
        the discriminator engages. The Terminal's ``run`` drives the real
        driver — a no-action seat-filler finishes with a stop completion.
        """
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="ok", tool_calls=[], finish_reason="stop")
        )

        async def _resolve() -> _FakeSeatFiller:
            return seat_filler

        monkeypatch.setattr(v1_chat_completions, "_resolve_seat_filler", _resolve)
        monkeypatch.setattr(
            v1_chat_completions,
            "get_orchestrator_tool_dispatch",
            lambda: _NoToolDispatch(),
        )

        assert isinstance(await v1_chat_completions.get_loop_driver(), LoopDriver)
        caller = await v1_chat_completions.get_client_tool_action_terminal()

        assert isinstance(caller, ClientToolActionTerminal)
        registry = SessionRegistry()
        state = registry.get_or_create_state(
            SessionIdentity(value="seam-test", method="user_field")
        )
        context = SessionContext(messages=[], tools=[], state=state)
        chunks = [chunk async for chunk in caller.run(context)]
        assert any(
            isinstance(chunk, Completion) and chunk.finish_reason == "stop"
            for chunk in chunks
        )

    def test_tool_request_drives_real_loop_driver_emitting_turn_decision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FC-42 (event-based): a ``tools[]`` request drives the real driver.

        WP-LB-A verified routing via distinguishable callers; the FC-42 event
        assertion deferred to WP-LB-B verifies the discrimination engages the
        *real* Loop Driver, which emits a ``TurnDecision`` diagnostic — not
        faked against a stub.
        """
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="ack", tool_calls=[], finish_reason="stop")
        )
        client, _, _ = _build_client(
            monkeypatch, terminal=_real_terminal(seat_filler, substrate)
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "write"}}],
            },
        )

        assert response.status_code == 200
        turn_decisions = [e for e in sink.events if isinstance(e, TurnDecision)]
        assert len(turn_decisions) == 1
        assert turn_decisions[0].action == "finish"

    def test_non_tool_request_emits_no_turn_decision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The non-tool surface routes to the pipeline, not the loop-driver."""
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="ack", tool_calls=[], finish_reason="stop")
        )
        client, _, _ = _build_client(
            monkeypatch, terminal=_real_terminal(seat_filler, substrate)
        )

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "what is a monad?"}],
            },
        )

        assert not [e for e in sink.events if isinstance(e, TurnDecision)]


class TestNonStreamingToolCallEmission:
    """FC-47 (non-streaming) — a tool-call outcome shapes ``message.tool_calls``.

    The streaming path emits tool_calls via the SSE formatter (the formatter's
    ``ClientToolCall`` case was retained by ``0a7a822``); the non-streaming body
    shapes the same ``ClientToolCall`` into ``message.tool_calls`` + ``content:
    null`` + ``finish_reason: "tool_calls"`` (the OpenAI non-streaming tool-call
    shape — the collector + body pieces ``0a7a822`` removed, re-introduced on
    the tool-driven terminal path per ADR-034).
    """

    def test_tool_call_chunk_becomes_message_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        invocation = ToolCallInvocation(
            id="w1",
            name="write",
            arguments=json.dumps({"filePath": "f.py", "content": "x = 1\n"}),
        )
        terminal = _StubPipeline([ClientToolCall(tool_calls=(invocation,))])
        client, _, _ = _build_client(monkeypatch, terminal=terminal)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "write f.py"}],
                "tools": [{"type": "function", "function": {"name": "write"}}],
            },
        )

        body = response.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tool_calls = choice["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "write"
        assert json.loads(tool_calls[0]["function"]["arguments"])["filePath"] == "f.py"


class TestStreamingPath:
    """``stream: true`` returns an SSE response (WP-B Group 5).

    The skeleton response emits the OpenAI-expected opener (role delta)
    and a single ``finish_reason: stop`` completion, terminated by
    ``data: [DONE]``. WP-C replaces the stub stream handoff with the
    real Runtime, which will interleave content deltas and internal
    tool-call observations.
    """

    def test_stream_true_returns_event_stream_content_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        content_type = response.headers["content-type"]
        assert content_type.startswith("text/event-stream")

    def test_stream_body_frames_role_delta_stop_and_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skeleton stream yields opener → stop completion → DONE."""
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        frames = _parse_sse_frames(response.content)
        assert len(frames) == 3
        first, second, terminator = frames

        assert first["choices"][0]["delta"] == {"role": "assistant"}
        assert first["choices"][0]["finish_reason"] is None

        assert second["choices"][0]["delta"] == {}
        assert second["choices"][0]["finish_reason"] == "stop"

        assert terminator == {"__done__": True}

    def test_stream_chunks_carry_request_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every chunk's ``model`` field mirrors the request's ``model``."""
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "my-profile",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        frames = _parse_sse_frames(response.content)
        data_chunks = [f for f in frames if f != {"__done__": True}]
        assert data_chunks
        assert all(chunk["model"] == "my-profile" for chunk in data_chunks)

    def test_stream_chunks_share_one_stream_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All data chunks in a single stream share a stable ``id``.

        OpenAI clients correlate chunks by id while reconstructing the
        completion; the id must not vary mid-stream.
        """
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        frames = _parse_sse_frames(response.content)
        ids = {f["id"] for f in frames if f != {"__done__": True}}
        assert len(ids) == 1


class TestStreamingSessionStart:
    """FC-9 holds under streaming: ``resolve_session_start_context`` runs
    once per session regardless of whether the request is streaming or
    not. The cache mediates this; the streaming branch must not bypass
    ``_resolve_context``.
    """

    def test_streaming_request_fires_session_start_exactly_once_per_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal call_count
            call_count += 1
            return []

        client, _, _ = _build_client(monkeypatch, session_start_spy=spy)

        for _ in range(2):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "primary",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "user": "client-abc",
                },
            )
            # Consume body so the StreamingResponse actually runs.
            response.content  # noqa: B018

        assert call_count == 1

    def test_streaming_and_non_streaming_share_session_start_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One streaming and one non-streaming request on the same identity
        must resolve the session-start resolver exactly once between them.
        """
        call_count = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal call_count
            call_count += 1
            return []

        client, _, _ = _build_client(monkeypatch, session_start_spy=spy)

        streaming = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "user": "client-abc",
            },
        )
        streaming.content  # noqa: B018
        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )

        assert call_count == 1


class TestNonStreamingPath:
    """The Group 4 non-streaming body is preserved under Group 5."""

    def test_non_streaming_default_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requests that omit ``stream`` default to non-streaming and succeed."""
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200

    def test_stream_false_explicit_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``stream: false`` is the Group 4 happy path."""
        client, _, _ = _build_client(monkeypatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200


class TestSessionStartIntegration:
    """Verify the Serving Layer → ``resolve_session_start_context`` contract.

    Covers the boundary integration test called out in the system
    design: ``test_session_start_context_is_empty_in_phase_1`` — function
    called once per session start, returns ``[]``, never touches Plexus
    (FC-9, test architecture table).
    """

    def test_session_start_fires_on_first_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[SessionContext] = []

        def spy(context: SessionContext) -> list[PromptFragment]:
            calls.append(context)
            return []

        client, _, _ = _build_client(monkeypatch, session_start_spy=spy)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "start"}],
                "user": "client-abc",
            },
        )

        assert len(calls) == 1
        assert calls[0].messages[0].content == "start"
        assert calls[0].tools == []

    def test_session_start_returns_empty_list_phase_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase 1 returns ``[]``; a spy that returns ``[]`` simulates Phase 1."""
        calls: list[list[PromptFragment]] = []

        def spy(context: SessionContext) -> list[PromptFragment]:
            fragments: list[PromptFragment] = []
            calls.append(fragments)
            return fragments

        client, _, _ = _build_client(monkeypatch, session_start_spy=spy)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "start"}],
            },
        )

        assert calls == [[]]

    def test_session_start_fires_exactly_once_per_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FC-9: the function is called once per session, not per request.

        Two requests that resolve to the same ``SessionIdentity`` (same
        ``user`` field) must trigger exactly one session-start
        resolution. The second request continues the established
        session; Phase 2 re-injecting on every request would spam the
        Plexus Adapter.
        """
        call_count = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal call_count
            call_count += 1
            return []

        client, _, _ = _build_client(monkeypatch, session_start_spy=spy)

        for _ in range(2):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "primary",
                    "messages": [{"role": "user", "content": "turn"}],
                    "user": "client-abc",
                },
            )

        assert call_count == 1

    def test_session_start_fires_per_distinct_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different ``SessionIdentity`` values each trigger their own start.

        A new ``user`` field resolves to a distinct identity; the
        previously-cached session does not suppress the new session's
        start-time resolution.
        """
        call_count = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal call_count
            call_count += 1
            return []

        client, _, _ = _build_client(monkeypatch, session_start_spy=spy)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "turn"}],
                "user": "client-abc",
            },
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "turn"}],
                "user": "client-xyz",
            },
        )

        assert call_count == 2

    def test_session_start_context_is_empty_in_phase_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Named integration test from system-design.md Test Architecture table.

        Verifies the Serving Layer → ``resolve_session_start_context``
        contract with the **real** Phase 1 resolver (default cache; no
        spy). A second request to the same session confirms the cache
        suppresses re-resolution — both requests succeed without the
        real resolver needing to do anything beyond returning ``[]``.
        The "never touches Plexus Adapter" property is structural: the
        resolver has no Plexus imports in Phase 1.
        """
        client, _, _ = _build_client(monkeypatch)

        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )
        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )

        assert first.status_code == 200
        assert second.status_code == 200

    def test_cache_retains_fragments_across_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache retains fragments so Runtime (WP-C) can read them.

        Group 4 stores the resolver's result in the cache on first
        request. The test asserts the retained value is exactly what
        the resolver returned — WP-C will read it from the same cache
        each iteration.
        """

        def spy(context: SessionContext) -> list[PromptFragment]:
            return [PromptFragment(content="sys-prime", source="spy")]

        client, _, cache = _build_client(monkeypatch, session_start_spy=spy)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )

        from llm_orc.agentic.session_registry import SessionIdentity, SessionState

        identity = SessionIdentity(value="client-abc", method="user_field")
        follow_up = SessionContext(
            messages=[],
            tools=[],
            state=SessionState(identity=identity),
        )

        assert cache.resolve(follow_up) == [
            PromptFragment(content="sys-prime", source="spy")
        ]


class TestServingResolvesSessionIdentity:
    """Boundary integration: Serving Layer → Session Registry.

    Per ``docs/agentic-serving/system-design.md`` Test Architecture
    table, edge ``Serving Layer → Session Registry``: an HTTP request
    with or without session continuity correlates to the correct
    ``SessionState``. Each HTTP request goes through ``_resolve_context``
    which calls ``SessionRegistry.resolve_identity`` and
    ``SessionRegistry.get_or_create_state``; the integration contract
    is that identity derivation and state lookup are coherent across
    requests — the same derivation inputs yield the same ``SessionState``
    instance, so Budget Controller, Autonomy Policy, and (later) the
    Orchestrator Runtime all read a single coherent view.
    """

    def test_same_user_field_resolves_to_same_session_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two requests with the same ``user`` field share one ``SessionState``.

        The Session Registry hands out the same instance so that turn
        count and token spend accumulated by the first request are
        observed by the second (cycle-status §FF 43: shared-reference
        lifecycle pattern).
        """
        client, registry, _ = _build_client(monkeypatch)

        for _ in range(2):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "primary",
                    "messages": [{"role": "user", "content": "hi"}],
                    "user": "client-abc",
                },
            )

        identity = SessionIdentity(value="client-abc", method="user_field")
        state = registry.get_or_create_state(identity)
        assert state.identity == identity
        assert list(registry._states.keys()).count(identity) == 1

    def test_state_mutation_between_requests_carries_across(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation through the retained state is visible on the next request.

        The follow-up request must resolve to the same ``SessionState`` so
        an accumulated turn count and token spend persist across requests.
        """
        client, registry, _ = _build_client(monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )

        identity = SessionIdentity(value="client-abc", method="user_field")
        state_between = registry.get_or_create_state(identity)
        state_between.record_iteration(tokens=128)
        state_between.record_iteration(tokens=64)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "client-abc",
            },
        )

        state_after = registry.get_or_create_state(identity)
        assert state_after is state_between
        # The same SessionState is retained across requests, so the manual
        # between-request mutations persist. Under ADR-027 the framework-
        # driven pipeline records no ReAct iterations (no orchestrator-LLM
        # loop on this surface), so only the two manual increments show.
        assert state_after.turn_count == 2
        assert state_after.token_spend == 192

    def test_distinct_user_fields_resolve_to_distinct_states(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two requests with different ``user`` values get distinct Sessions."""
        client, registry, _ = _build_client(monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "one"}],
                "user": "client-abc",
            },
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "two"}],
                "user": "client-xyz",
            },
        )

        id_abc = SessionIdentity(value="client-abc", method="user_field")
        id_xyz = SessionIdentity(value="client-xyz", method="user_field")
        state_abc = registry.get_or_create_state(id_abc)
        state_xyz = registry.get_or_create_state(id_xyz)
        assert state_abc is not state_xyz

    def test_message_prefix_derivation_when_user_field_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requests without ``user`` but with user messages derive by prefix hash.

        Two requests whose first user message is identical resolve to
        the same ``SessionState``; requests whose first user message
        differs resolve to distinct ones. This is the fallback identity
        path per Session Registry's integration contract.
        """
        client, registry, _ = _build_client(monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [
                    {"role": "user", "content": "refactor the parser"},
                    {"role": "assistant", "content": "ok"},
                ],
            },
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [
                    {"role": "user", "content": "refactor the parser"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "now the lexer"},
                ],
            },
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "unrelated task"}],
            },
        )

        prefix_identities = [
            i for i in registry._states if i.method == "message_prefix"
        ]
        assert len(prefix_identities) == 2

    def test_cold_start_request_gets_fresh_state_each_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``user`` field and no user-role message → cold-start fresh identity.

        Each cold-start request is its own Session so consumers don't
        accidentally alias unrelated clients through a shared cold
        bucket.
        """
        client, registry, _ = _build_client(monkeypatch)

        for _ in range(2):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "primary",
                    "messages": [{"role": "system", "content": "be helpful"}],
                },
            )

        cold_identities = [i for i in registry._states if i.method == "cold_start"]
        assert len(cold_identities) == 2
        assert cold_identities[0] != cold_identities[1]


class TestDispatchPipelineHandlerSwap:
    """ADR-027 / WP-A: the handler drives the framework-driven pipeline.

    These exercise a real :class:`DispatchPipeline` wired to stub ports
    through the HTTP boundary — the handler-level integration the
    pipeline's own unit tests (``test_dispatch_pipeline.py``) do not
    cover. Scenarios 1, 2, and 4 of ``scenarios.md`` §Framework-Driven
    Dispatch Pipeline.
    """

    def test_capability_match_flows_through_plan_dispatch_synthesize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario 1: plan(dispatch) → dispatch → synthesize → response."""
        synth = _StubSynthesizer("SYNTH OUTPUT")
        dispatcher = _StubDispatcher(content="raw dispatch result")
        pipeline = DispatchPipeline(
            planner=_StubPlanner(
                DispatchPlan(
                    action="dispatch",
                    ensemble="code-generator",
                    input="write a sort",
                    rationale="capability match",
                )
            ),
            synthesizer=synth,
            tool_dispatch=dispatcher,
            capability_names=frozenset({"code-generator"}),
        )
        client, _, _ = _build_client(monkeypatch, pipeline=pipeline)

        body = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "write a sort"}],
            },
        ).json()

        assert body["choices"][0]["message"]["content"] == "SYNTH OUTPUT"
        assert body["choices"][0]["finish_reason"] == "stop"
        # Dispatch fired once against the matched capability via invoke_ensemble.
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0].name == "invoke_ensemble"
        assert dispatcher.calls[0].arguments["ensemble_name"] == "code-generator"
        # The synthesizer received the dispatch output as structured input.
        assert synth.calls[0]["dispatched"] == ["code-generator"]
        assert synth.calls[0]["dispatch_results"] == [
            ("code-generator", "raw dispatch result")
        ]

    def test_no_capability_match_flows_through_direct_synthesize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario 2: plan(direct) → skip dispatch → synthesize empty results."""
        synth = _StubSynthesizer("DIRECT ANSWER")
        dispatcher = _StubDispatcher()
        pipeline = DispatchPipeline(
            planner=_StubPlanner(
                DispatchPlan(
                    action="direct",
                    ensemble=None,
                    input=None,
                    rationale="no capability match",
                )
            ),
            synthesizer=synth,
            tool_dispatch=dispatcher,
            capability_names=frozenset({"code-generator"}),
        )
        client, _, _ = _build_client(monkeypatch, pipeline=pipeline)

        body = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "what's the weather?"}],
            },
        ).json()

        assert body["choices"][0]["message"]["content"] == "DIRECT ANSWER"
        # No dispatch fired; the synthesizer saw an empty DISPATCH RESULTS.
        assert dispatcher.calls == []
        assert synth.calls[0]["dispatched"] == []
        assert synth.calls[0]["dispatch_results"] == []

    def test_orchestrator_runtime_is_not_constructed_on_this_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario 4: the runtime-construction path is gone from the handler.

        The handler no longer constructs ``OrchestratorRuntime`` — the
        loader/runtime/compaction factories were removed. The class
        remains in the codebase (ADR-027 disposition (a)) but has no
        caller here. A request resolves entirely via the pipeline.
        """
        assert not hasattr(v1_chat_completions, "get_orchestrator_llm_loader")
        assert not hasattr(v1_chat_completions, "_build_runtime")
        assert not hasattr(v1_chat_completions, "get_conversation_compaction")

        client, _, _ = _build_client(
            monkeypatch,
            pipeline=_StubPipeline(
                [ContentDelta(content="ok"), Completion(finish_reason="stop")]
            ),
        )
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
            },
        ).json()
        assert body["choices"][0]["message"]["content"] == "ok"

    def test_client_tool_reusing_reserved_internal_name_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reserved TOOL_NAMES stay enforced as a defensive input guard.

        The pipeline does not route by ``tools`` membership under ADR-027,
        but the Cycle 1 reserved-name commitment is preserved: a client
        tool reusing an internal name is rejected with HTTP 400.
        """
        client, _, _ = _build_client(monkeypatch)
        reserved = sorted(TOOL_NAMES)[0]

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"type": "function", "function": {"name": reserved}},
                ],
            },
        )

        assert response.status_code == 400
        # The app's custom HTTPException handler wraps ``detail`` under an
        # ``"error"`` key (see ``web/server.py``); the inner payload is the
        # guard's structured detail dict.
        payload = response.json()["error"]
        assert payload["error"] == "reserved_tool_name"
        assert reserved in payload["reserved_names"]


class _StubConfigManager:
    def __init__(self, profiles: dict[str, dict[str, Any]]) -> None:
        self._profiles = profiles

    def get_model_profiles(self) -> dict[str, dict[str, Any]]:
        return self._profiles


class _StubService:
    def __init__(self, profiles: dict[str, dict[str, Any]]) -> None:
        self.config_manager = _StubConfigManager(profiles)


class _ResolveOnlyResolver:
    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

    def resolve(self) -> OrchestratorConfig:
        return self._config


class _ToolCallingModel:
    supports_tool_calling = True


class _CapturingModelFactory:
    """A ModelFactory double recording the kwargs ``load_model`` was handed."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def load_model(
        self, model_name: str, provider: str | None = None, **kwargs: Any
    ) -> _ToolCallingModel:
        _CapturingModelFactory.last_kwargs = {
            "model_name": model_name,
            "provider": provider,
            **kwargs,
        }
        return _ToolCallingModel()


class TestResolveSeatFiller:
    """The seat-filler resolution threads the profile's connection config.

    A local OpenAI-compatible (Ollama) seat-filler profile carries
    ``base_url: http://localhost:11434/v1``; if the resolution drops it, the
    adapter defaults to ``api.openai.com`` and a $0 local seat-filler 401s.
    The harness override of ``_resolve_seat_filler`` hides this, so the
    threading is asserted here directly.
    """

    async def test_resolve_seat_filler_threads_profile_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profiles = {
            "test-profile": {
                "model": "qwen3:14b",
                "provider": "openai-compatible/ollama",
                "base_url": "http://localhost:11434/v1",
            }
        }
        _CapturingModelFactory.last_kwargs = {}
        monkeypatch.setattr(
            v1_chat_completions, "get_orchestra_service", lambda: _StubService(profiles)
        )
        monkeypatch.setattr(
            v1_chat_completions,
            "get_orchestrator_config_resolver",
            lambda: _ResolveOnlyResolver(_default_orchestrator_config()),
        )
        monkeypatch.setattr(
            v1_chat_completions, "CredentialStorage", lambda config_manager: object()
        )
        monkeypatch.setattr(v1_chat_completions, "ModelFactory", _CapturingModelFactory)

        await v1_chat_completions._resolve_seat_filler()

        assert (
            _CapturingModelFactory.last_kwargs.get("base_url")
            == "http://localhost:11434/v1"
        )
