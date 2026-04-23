"""Tests for the Serving Layer ``/v1/chat/completions`` endpoint.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3) and
§Integration Contracts (Serving Layer → Session Registry; Serving Layer
→ ``resolve_session_start_context``; Serving Layer → Orchestrator
Runtime). Group 4 is the **non-streaming skeleton** — roadmap WP-B
Group 4 (see ``docs/agentic-serving/roadmap.md``). The endpoint:

- Parses the OpenAI-compatible request (messages, model, tools, user,
  stream).
- Resolves a ``SessionIdentity`` via Session Registry.
- Calls ``resolve_session_start_context`` exactly once per session
  start (FC-9); Phase 1 returns ``[]``.
- Hands off to a stubbed Orchestrator Runtime. WP-C replaces the stub
  with the real ReAct loop.
- Returns a minimal OpenAI-shaped ``chat.completion`` body.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from llm_orc.agentic.autonomy_policy import (
    BASELINE_LEVEL,
    PURE_TOOL_USER_VISIBLE_LEVEL,
    AutonomyPolicy,
)
from llm_orc.agentic.orchestrator_config import (
    BudgetDefaults,
    OrchestratorConfig,
    OverrideBounds,
)
from llm_orc.agentic.orchestrator_runtime import OrchestratorLLM, ToolDispatcher
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallResult,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.agentic.session_registry import SessionIdentity, SessionRegistry
from llm_orc.agentic.session_start import (
    PromptFragment,
    SessionContext,
    SessionStartCache,
)
from llm_orc.models.base import ToolCall, ToolCallingResponse, ToolCallUsage
from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app


class _RejectingValidator:
    """Composition validator stub that rejects every request.

    Tests at the Serving Layer that dispatch ``compose_ensemble``
    incidentally (e.g., autonomy-and-promotion acceptance) do not care
    about the composition outcome — the orchestrator LLM observes a
    ``ToolCallError`` and the gate narration, tests assert on the
    narration. Tests that care about accept-path behavior pass a real
    validator wired through ``v1_chat_completions.get_orchestrator_tool_dispatch``.
    """

    def validate(self, request: Any) -> Any:
        from llm_orc.agentic.composition_validator import CompositionRejected

        return CompositionRejected(
            kind="missing_primitive",
            reason="rejecting validator for non-compose serving-layer tests",
        )


class _UnusedWriter:
    """Local ensemble writer stub paired with :class:`_RejectingValidator`."""

    def write(self, config: Any) -> str:  # pragma: no cover
        raise AssertionError(
            f"local_ensemble_writer unused in this test: {config.name!r}"
        )


def _dispatch_kwargs_without_composition() -> dict[str, Any]:
    """Returns composition dependency kwargs for tests not exercising compose.

    The validator rejects every request so compose dispatches produce
    a typed ``ToolCallError``; the writer fails loudly if reached.
    """
    return {
        "composition_validator": _RejectingValidator(),
        "local_ensemble_writer": _UnusedWriter(),
    }


class _ScriptedLLM:
    """Plays back a prepared sequence of tool-calling responses.

    Satisfies the ``OrchestratorLLM`` Protocol structurally. Records
    each call so tests can assert on message state per iteration.
    """

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        self.calls.append((list(messages), list(tools)))
        if not self._responses:
            raise AssertionError("_ScriptedLLM ran out of canned responses")
        return self._responses.pop(0)


class _AlwaysStopLLM:
    """Default LLM for tests that do not exercise tool calling.

    Returns ``finish_reason: stop`` with empty content for every
    request — preserves the behavior the pre-WP-C stub produced so
    tests that predate the Runtime wiring continue to pass.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        self.calls.append((list(messages), list(tools)))
        return _stop_response()


class _StubToolDispatch:
    """Returns canned tool results keyed by tool-call id."""

    def __init__(self, results: dict[str, ToolCallResult] | None = None) -> None:
        self._results = dict(results or {})
        self.calls: list[InternalToolCall] = []

    async def dispatch(self, call: InternalToolCall) -> ToolCallResult:
        self.calls.append(call)
        if call.id not in self._results:
            raise AssertionError(
                f"_StubToolDispatch received unexpected call id={call.id!r}"
            )
        return self._results[call.id]


def _stop_response(content: str = "", total_tokens: int = 0) -> ToolCallingResponse:
    """Build a minimal stop-reason response for default-path tests."""
    return ToolCallingResponse(
        content=content,
        tool_calls=[],
        usage=ToolCallUsage(
            prompt_tokens=0,
            completion_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
        finish_reason="stop",
    )


def _default_orchestrator_config() -> OrchestratorConfig:
    """A Runtime-compatible OrchestratorConfig used when tests don't care.

    The model_profile name is arbitrary — tests inject a scripted LLM
    loader that ignores the profile argument.
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
    )


class _FakeConfigResolver:
    """Returns a canned ``OrchestratorConfig`` without touching the filesystem.

    Satisfies the duck-typed surface of ``OrchestratorConfigResolver``
    that ``_build_runtime`` calls — only ``resolve_validated`` is used.
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
    llm: _ScriptedLLM | None = None,
    tool_dispatch: ToolDispatcher | None = None,
    config: OrchestratorConfig | None = None,
) -> tuple[TestClient, SessionRegistry, SessionStartCache]:
    """Wire a TestClient with isolated Registry, cache, LLM, and dispatch.

    Overrides the module-level factories in ``v1_chat_completions`` so
    each test runs the real Serving Layer → Runtime → Tool Dispatch
    wiring but with scripted doubles in place of the real orchestrator
    LLM and ensemble facade. Defaults keep the prior skeleton behavior
    (empty-content stop) so tests that predate the wiring continue to
    pass unchanged.
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

    scripted_llm: _ScriptedLLM | _AlwaysStopLLM = llm or _AlwaysStopLLM()

    async def loader(_model_profile: str) -> OrchestratorLLM:
        return scripted_llm

    monkeypatch.setattr(
        v1_chat_completions, "get_orchestrator_llm_loader", lambda: loader
    )

    dispatch = tool_dispatch or _StubToolDispatch()
    monkeypatch.setattr(
        v1_chat_completions, "get_orchestrator_tool_dispatch", lambda: dispatch
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
        """Client-declared tools are received but not yet routed (WP-F)."""
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

        Simulates what WP-C's Orchestrator Runtime will do — record a
        ReAct iteration mid-session. The follow-up request must resolve
        to the same ``SessionState`` so Budget Controller sees the
        accumulated turn count and token spend.
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
        # The retained state includes both the manual between-requests
        # increments and the two iterations the Runtime itself recorded
        # (one per request under the default scripted stop LLM).
        assert state_after.turn_count == 4
        # The scripted stop LLM reports zero tokens so the Runtime's
        # contribution to token_spend is zero — only the manual
        # mutations between requests are reflected here.
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


class TestOrchestratorEndToEnd:
    """§Tool user completes a task against the stateless orchestrator.

    Real Serving Layer → real Runtime → real Tool Dispatch facade (wrapping
    a ``_StubToolDispatch`` at the operation boundary) → scripted LLM.
    Proves the happy-path ReAct round trip surfaces correctly in the
    OpenAI-compatible response body.
    """

    def test_non_streaming_tool_round_trip_returns_final_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = _ScriptedLLM(
            responses=[
                # Iteration 1: LLM decides to call list_ensembles.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="list_ensembles",
                            arguments_json="{}",
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=5, total_tokens=25
                    ),
                    finish_reason="tool_calls",
                ),
                # Iteration 2: LLM sees the tool result and wraps up.
                ToolCallingResponse(
                    content="I found 2 ensembles.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=45, completion_tokens=8, total_tokens=53
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        # The Tool Dispatch stub impersonates OrchestratorToolDispatch —
        # its ``dispatch`` method matches the ToolDispatcher Protocol.
        from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallSuccess

        tool_dispatch = _StubToolDispatch(
            results={
                "call_1": ToolCallSuccess(
                    id="call_1",
                    name="list_ensembles",
                    content=[{"name": "a"}, {"name": "b"}],
                )
            }
        )
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=tool_dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "list available ensembles"}],
            },
        )

        assert response.status_code == 200
        body = response.json()
        # The assistant's final content is what the LLM produced after
        # seeing the tool result.
        assert body["choices"][0]["message"]["content"] == "I found 2 ensembles."
        assert body["choices"][0]["finish_reason"] == "stop"
        # Usage reflects the two iterations' token totals (25 + 53).
        assert body["usage"]["total_tokens"] == 78
        # Tool Dispatch saw the expected call.
        assert len(tool_dispatch.calls) == 1
        assert tool_dispatch.calls[0].name == "list_ensembles"
        # LLM was called twice; the second call saw the tool result
        # message.
        assert len(llm.calls) == 2
        second_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"

    def test_streaming_tool_round_trip_yields_expected_sse_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="Looking that up...",
                    tool_calls=[
                        ToolCall(
                            id="call_s",
                            name="list_ensembles",
                            arguments_json="{}",
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="Done.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=30, completion_tokens=2, total_tokens=32
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallSuccess

        tool_dispatch = _StubToolDispatch(
            results={
                "call_s": ToolCallSuccess(
                    id="call_s", name="list_ensembles", content=[]
                )
            }
        )
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=tool_dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "list"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        frames = _parse_sse_frames(response.content)
        # Content chunks arrive inline with the loop — both iterations'
        # content surface in the stream.
        content_chunks = [
            f
            for f in frames
            if f != {"__done__": True} and f["choices"][0]["delta"].get("content")
        ]
        assert [c["choices"][0]["delta"]["content"] for c in content_chunks] == [
            "Looking that up...",
            "Done.",
        ]
        # Terminated by a stop-reason completion then DONE.
        completions = [
            f
            for f in frames
            if f != {"__done__": True}
            and f["choices"][0].get("finish_reason") == "stop"
        ]
        assert len(completions) == 1
        assert frames[-1] == {"__done__": True}


class _StubEnsembleOperations:
    """Stub ``EnsembleOperations`` + ``SummarizerInvoker``.

    Both Protocols share the ``async def invoke(arguments) -> dict`` shape.
    A single stub satisfies both so the real Tool Dispatch and Harness wire
    up against the same object — tests inspect ``calls`` to assert whether
    the summarizer ran or was skipped.

    ``library_entries`` optionally surfaces ensemble metadata through
    :meth:`read_ensembles` — the shape that ``ResourceHandler.read_ensembles``
    produces in production (name, source, relative_path, agent_count,
    description). Scenario (c) uses it to verify the orchestrator can infer
    a file-content dependency from the ensemble's description.
    """

    def __init__(
        self,
        results_by_ensemble: dict[str, dict[str, Any]],
        *,
        library_entries: list[dict[str, Any]] | None = None,
    ) -> None:
        self._results = dict(results_by_ensemble)
        self._library_entries = list(library_entries or [])
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        name = arguments.get("ensemble_name")
        if not isinstance(name, str) or name not in self._results:
            raise ValueError(f"ensemble '{name}' not in stub library")
        return self._results[name]

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return list(self._library_entries)


class TestRawOutputEscapeHatchAcceptance:
    """Acceptance test for ``scenarios.md`` §Raw-output escape hatch.

    Wires a real :class:`OrchestratorToolDispatch` + real
    :class:`ResultSummarizerHarness` over a stub ``EnsembleOperations``,
    then exercises the scenario through ``/v1/chat/completions``:

        Given an ensemble configured with the raw-output escape-hatch
        flag (per ADR-004)
        When the Orchestrator Agent calls ``invoke_ensemble`` on that
        ensemble
        Then the raw result is passed directly into the orchestrator's
        context without invoking the summarizer, and the behavior is
        opt-in — not a default.

    The test sits at the Serving Layer boundary (real request → real
    Runtime → real Tool Dispatch → real Harness → stubbed ensemble
    operations). It proves the escape hatch is observable end-to-end,
    not just inside the Harness's unit tests. Complement to the FC-8
    static check: FC-8 proves the summarize path can't be bypassed
    structurally; this test proves the documented opt-in path is wired
    through the real Serving Layer.
    """

    def test_raw_output_flag_passes_raw_dict_to_orchestrator_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_result = {
            "results": {"analyst": {"response": "traffic spiked at 03:14 UTC"}},
            "synthesis": None,
            "status": "success",
            "raw_output": True,
        }
        operations = _StubEnsembleOperations({"analysis": raw_result})
        harness = ResultSummarizerHarness(
            invoker=operations, summarizer_name="test-summarizer"
        )
        dispatch = OrchestratorToolDispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
            **_dispatch_kwargs_without_composition(),
        )

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_raw",
                            name="invoke_ensemble",
                            arguments_json=(
                                '{"name": "analysis", "input": "summarize traffic"}'
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="Spike acknowledged.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=40, completion_tokens=3, total_tokens=43
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "watch the log"}],
            },
        )

        assert response.status_code == 200
        # The summarizer ensemble must NOT appear in operations.calls —
        # raw_output is an opt-in bypass per ADR-004.
        invoked_ensembles = [c["ensemble_name"] for c in operations.calls]
        assert invoked_ensembles == ["analysis"], (
            "Summarizer ensemble was invoked despite raw_output=True — "
            "ADR-004's escape hatch is not being honored by the wiring."
        )

        # The second LLM call must see the raw dict as a role: tool
        # observation — not a {"summary": ...} dict.
        assert len(llm.calls) == 2
        second_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        observed = json.loads(tool_msgs[0]["content"])
        assert observed == raw_result, (
            f"Orchestrator observation is not the raw ensemble dict. "
            f"Expected raw pass-through; got {observed!r}."
        )

    def test_raw_output_false_default_routes_through_summarizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The opt-in flag's negation: without raw_output, summarize runs.

        Closes the "is this behavior actually opt-in?" question — the
        scenario says "the behavior is opt-in — not a default". Same
        wiring, same raw dict shape minus the flag, different path.
        """
        raw_result = {
            "results": {"analyst": {"response": "traffic spiked at 03:14 UTC"}},
            "synthesis": None,
            "status": "success",
        }
        summary_result = {
            "results": {"summarizer": {"response": "Traffic spike at 03:14."}},
            "synthesis": None,
            "status": "success",
        }
        operations = _StubEnsembleOperations(
            {"analysis": raw_result, "test-summarizer": summary_result}
        )
        harness = ResultSummarizerHarness(
            invoker=operations, summarizer_name="test-summarizer"
        )
        dispatch = OrchestratorToolDispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
            **_dispatch_kwargs_without_composition(),
        )

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_sum",
                            name="invoke_ensemble",
                            arguments_json=(
                                '{"name": "analysis", "input": "summarize traffic"}'
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="ok",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=1, total_tokens=21
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "watch the log"}],
            },
        )

        assert response.status_code == 200
        invoked_ensembles = [c["ensemble_name"] for c in operations.calls]
        assert invoked_ensembles == ["analysis", "test-summarizer"], (
            "Summarizer did not run for a raw_output=False ensemble — "
            "AS-7's default must be summarize-always."
        )

        second_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        observed = json.loads(tool_msgs[0]["content"])
        assert observed == {"summary": "Traffic spike at 03:14."}, (
            f"Orchestrator observation is not a summary payload. "
            f"Expected summary; got {observed!r}."
        )


class TestAutonomyAndPromotionAcceptance:
    """Acceptance tests for ``scenarios.md`` §Autonomy and Promotion.

    Wires the real :class:`OrchestratorToolDispatch`, real
    :class:`AutonomyPolicy`, and real :class:`ResultSummarizerHarness`
    over a stub ``EnsembleOperations`` and drives
    ``/v1/chat/completions`` end-to-end. Four scenarios:

    * Default Autonomy Level permits invocation, permits composition,
      gates promotion — structural (``TOOL_NAMES`` closed) + behavioral
      (gate allows the five committed tools).
    * Tool user without operator role observes composition events when
      configured — tightened level narrates on ``delta.content``.
    * Pure tool-user session at default Autonomy Level experiences
      silent composition — baseline level emits zero narration.
    * Script authorship is never permitted at any Autonomy Level —
      AS-6 closure via the unknown-tool filter; parametrized over
      multiple levels (including a synthetic future level).
    """

    def _assistant_content(self, response_json: dict[str, Any]) -> str:
        choices = response_json["choices"]
        assert isinstance(choices, list)
        assert choices
        content = choices[0]["message"]["content"]
        assert isinstance(content, str)
        return content

    def _build_dispatch(
        self, operations: _StubEnsembleOperations, level: str
    ) -> OrchestratorToolDispatch:
        harness = ResultSummarizerHarness(
            invoker=operations, summarizer_name="test-summarizer"
        )
        policy = AutonomyPolicy(level_provider=lambda: level)
        return OrchestratorToolDispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=policy,
            **_dispatch_kwargs_without_composition(),
        )

    def test_default_level_permits_invoke_and_compose_no_promotion_in_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario: Default Autonomy Level permits invocation, permits
        composition, gates promotion.

        Structural: promotion is not an orchestrator tool — ``TOOL_NAMES``
        has no entry for it, so self-promotion is mechanically impossible.
        Behavioral: at baseline, the gate allows ``invoke_ensemble`` and
        ``compose_ensemble`` (the latter returns ``not_yet_wired`` pending
        WP-G; the point is the gate does not block the attempt).
        """
        assert "promote_ensemble" not in TOOL_NAMES
        assert "promote" not in TOOL_NAMES
        assert TOOL_NAMES == frozenset(
            {
                "invoke_ensemble",
                "compose_ensemble",
                "list_ensembles",
                "query_knowledge",
                "record_outcome",
            }
        )

        analysis_result = {
            "results": {"a": {"response": "done"}},
            "synthesis": None,
            "status": "success",
            "raw_output": True,  # bypass summarizer to keep the test focused
        }
        operations = _StubEnsembleOperations({"analysis": analysis_result})
        dispatch = self._build_dispatch(operations, BASELINE_LEVEL)

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_inv",
                            name="invoke_ensemble",
                            arguments_json=('{"name": "analysis", "input": "x"}'),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=1, total_tokens=11
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_com",
                            name="compose_ensemble",
                            arguments_json='{"name": "new"}',
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=1, total_tokens=21
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="both calls completed",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=30, completion_tokens=3, total_tokens=33
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "do both"}],
            },
        )

        assert response.status_code == 200
        assert operations.calls == [{"ensemble_name": "analysis", "input": "x"}]

        # Second LLM turn saw invoke_ensemble's raw-output pass-through.
        # Third LLM turn saw compose_ensemble's invocation_failed observation —
        # WP-G wires the validator; the test-scope ``_RejectingValidator``
        # rejects any request, which maps to ``invocation_failed``.
        third_messages = llm.calls[2][0]
        compose_tool_msgs = [
            m
            for m in third_messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_com"
        ]
        assert len(compose_tool_msgs) == 1
        observed = json.loads(compose_tool_msgs[0]["content"])
        assert observed.get("error") == "invocation_failed"

        # Baseline emits no composition narration.
        content = self._assistant_content(response.json())
        assert "[composition:" not in content

    def test_tool_user_observes_composition_events_at_tightened_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario: Tool user without operator role observes composition
        events when configured.

        At ``pure-tool-user-visible``, ``compose_ensemble`` produces a
        :class:`VisibilityEvent` that renders on ``delta.content`` — the
        tool user sees ``[composition: {...}]`` inline in their client's
        assistant message text.
        """
        operations = _StubEnsembleOperations({})
        dispatch = self._build_dispatch(operations, PURE_TOOL_USER_VISIBLE_LEVEL)

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="Composing a specialist.",
                    tool_calls=[
                        ToolCall(
                            id="call_com",
                            name="compose_ensemble",
                            arguments_json=(
                                '{"name": "pricing-analyst", "profiles": ["default"]}'
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=1, total_tokens=11
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="That composer is not wired yet.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=2, total_tokens=22
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "find a specialist"}],
            },
        )

        assert response.status_code == 200
        content = self._assistant_content(response.json())
        # Narration sits between the two assistant content segments.
        assert "[composition:" in content
        # The rendered payload includes the composed ensemble's name so
        # the tool user sees *what* was composed — not only *that*
        # composition happened.
        assert "pricing-analyst" in content
        assert content.index("Composing a specialist.") < content.index("[composition:")
        assert content.index("[composition:") < content.index(
            "That composer is not wired yet."
        )

    def test_default_level_silent_composition_no_narration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario: Pure tool-user session at default Autonomy Level
        experiences silent composition.

        Same flow as the tightened-level scenario, but at
        ``operator-as-tool-user``. No narration appears — the tool user
        sees only the LLM's own content and the error observation flow.
        """
        operations = _StubEnsembleOperations({})
        dispatch = self._build_dispatch(operations, BASELINE_LEVEL)

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_com",
                            name="compose_ensemble",
                            arguments_json='{"name": "silent-specialist"}',
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=1, total_tokens=11
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="Nothing to show you.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=2, total_tokens=22
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "find a specialist"}],
            },
        )

        assert response.status_code == 200
        content = self._assistant_content(response.json())
        assert content == "Nothing to show you."
        assert "[composition:" not in content
        assert "silent-specialist" not in content

    @pytest.mark.parametrize(
        "level",
        [
            BASELINE_LEVEL,
            PURE_TOOL_USER_VISIBLE_LEVEL,
            "some-future-loosest-level",
        ],
    )
    def test_script_authorship_never_permitted_at_any_level(
        self, monkeypatch: pytest.MonkeyPatch, level: str
    ) -> None:
        """Scenario: Script authorship is never permitted at any Autonomy
        Level.

        AS-6 closure lives in ``TOOL_NAMES``: any tool name outside the
        five committed operations returns ``unknown_tool`` before the
        gate is consulted. Parametrized over baseline, the current
        tightened level, and a synthetic future level to demonstrate
        that no configuration opens a path to primitive authorship.
        """
        operations = _StubEnsembleOperations({})
        dispatch = self._build_dispatch(operations, level)

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_bad",
                            name="author_script",
                            arguments_json=('{"name": "evil.py", "content": "..."}'),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=1, total_tokens=11
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="I cannot author scripts.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=2, total_tokens=22
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "write a script"}],
            },
        )

        assert response.status_code == 200

        # Ensemble operations were never touched — no script authorship
        # reached any library surface.
        assert operations.calls == []

        # The second LLM call saw an unknown_tool observation — not an
        # allowed authorship, not a denied_by_autonomy. AS-6 closure is
        # structural; the gate doesn't need to weigh in.
        second_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        observed = json.loads(tool_msgs[0]["content"])
        assert observed.get("error") == "unknown_tool"


class TestEnsembleCompositionWithValidationAcceptance:
    """Acceptance for ``scenarios.md`` §Ensemble Composition with Validation.

    Exercises the full HTTP path — ``/v1/chat/completions`` → scripted
    Orchestrator LLM → real :class:`OrchestratorToolDispatch` with real
    :class:`CompositionValidator` and real
    :class:`ConfigManagerEnsembleWriter` — so a regression anywhere on
    the Serving Layer → Runtime → Tool Dispatch → Validator → Writer
    path fails here. The unit validator tests and the boundary
    integration in ``test_tool_dispatch_composition.py`` cover the six
    rejection branches in isolation; this class asserts the end-to-end
    observability from the orchestrator's perspective: what the LLM
    sees on the turn after the compose tool call returns.
    """

    def _build_real_dispatch(
        self, project_dir: Path
    ) -> tuple[OrchestratorToolDispatch, Any]:
        from llm_orc.agentic.composition_validator import (
            CompositionValidator,
            ConfigManagerEnsembleWriter,
            ConfigManagerPrimitiveRegistry,
        )
        from llm_orc.core.config.config_manager import ConfigurationManager
        from llm_orc.services.orchestra_service import OrchestraService

        config_manager = ConfigurationManager(project_dir=project_dir, provision=False)
        service = OrchestraService(config_manager=config_manager)
        harness = ResultSummarizerHarness(
            invoker=service, summarizer_name="unused-no-invoke-in-compose-tests"
        )
        policy = AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL)
        registry = ConfigManagerPrimitiveRegistry(config_manager)
        validator = CompositionValidator(primitives=registry)
        writer = ConfigManagerEnsembleWriter(config_manager)
        dispatch = OrchestratorToolDispatch(
            operations=service,
            harness=harness,
            autonomy_policy=policy,
            composition_validator=validator,
            local_ensemble_writer=writer,
        )
        return dispatch, service

    def _prepare_library(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        existing_ensembles: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        import yaml as _yaml

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
            _yaml.safe_dump(
                {"model_profiles": {"default": {"model": "mock", "provider": "mock"}}}
            )
        )
        ensembles_dir = local / "ensembles"
        ensembles_dir.mkdir()
        for name, body in (existing_ensembles or {}).items():
            (ensembles_dir / f"{name}.yaml").write_text(_yaml.safe_dump(body))
        return project_dir

    def test_compose_happy_path_writes_new_ensemble_and_reports_to_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario: Composition with only profiles and scripts succeeds."""
        project_dir = self._prepare_library(tmp_path, monkeypatch)
        dispatch, _service = self._build_real_dispatch(project_dir)

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_com",
                            name="compose_ensemble",
                            arguments_json=(
                                '{"name": "combo", "description": "combine",'
                                ' "agents": [{"name": "think",'
                                ' "model_profile": "default"}]}'
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=1, total_tokens=6
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="Composed combo.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "compose a combo"}],
            },
        )

        assert response.status_code == 200

        # File persisted to local tier.
        target = project_dir / ".llm-orc" / "ensembles" / "combo.yaml"
        assert target.exists()
        written = yaml.safe_load(target.read_text())
        assert written["name"] == "combo"
        assert written["agents"][0]["model_profile"] == "default"

        # Second LLM call saw the ToolCallSuccess content.
        second_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        observed = json.loads(tool_msgs[0]["content"])
        assert observed["name"] == "combo"
        assert observed["path"].endswith(".llm-orc/ensembles/combo.yaml")

    def test_compose_rejects_cycle_and_leaves_local_tier_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario: Composition that would introduce a reference-graph cycle fails.

        Full HTTP path. Dispatch's ``invocation_failed`` flows back as
        a ``role: tool`` observation the LLM's second iteration sees.
        """
        project_dir = self._prepare_library(
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
        dispatch, _service = self._build_real_dispatch(project_dir)

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_c",
                            name="compose_ensemble",
                            arguments_json=(
                                '{"name": "c", "description": "closes cycle",'
                                ' "agents": [{"name": "ref_a", "ensemble": "a"}]}'
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=1, total_tokens=6
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="Composition rejected — cycle detected.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "attempt cycle"}],
            },
        )

        assert response.status_code == 200

        # AS-2: no partial state on validation failure.
        target = project_dir / ".llm-orc" / "ensembles" / "c.yaml"
        assert not target.exists()

        # Second LLM call saw the invocation_failed observation naming the cycle.
        second_messages = llm.calls[1][0]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        observed = json.loads(tool_msgs[0]["content"])
        assert observed.get("error") == "invocation_failed"
        assert "cycle" in observed.get("reason", "").lower()


class TestClientToolSurfaceCommitment:
    """Acceptance tests for ``scenarios.md`` §Client Tool Surface Commitment.

    Option C per system-design §Client Tool Surface Commitment: the
    orchestrator's internal tool surface stays at the five ADR-003 tools;
    client-declared tools (the ``tools[]`` array on ``/v1/chat/completions``)
    flow through as a **response surface** — when a turn needs a client-side
    action, the Runtime closes with ``finish_reason: tool_calls`` and the
    next request resumes the same Session with the client's ``role: tool``
    messages as observations.

    Scenario (a) — turn-boundary delegation.
    Scenario (b) — Session continuity across a client-tool round trip.
    Scenario (c) — pre-invoke delegation when an ensemble's first agent
        needs a client-filesystem file.
    Scenario (d) — retry pattern for an un-predicted mid-execution need.
    Scenario (negative) — composed ensemble without the structured
        ``needs_client_tool`` signal silently degrades to a quality failure.

    The tests wire a real :class:`OrchestratorRuntime` over a
    :class:`_StubToolDispatch` constructed with no results — any internal
    dispatch call trips ``AssertionError``, which proves Option C's
    "no internal tool dispatched for a client-declared tool" property
    structurally, not only by counting.
    """

    def _file_read_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read a file from the client's filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read.",
                        }
                    },
                    "required": ["path"],
                },
            },
        }

    def _bash_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the client environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        }

    def test_orchestrator_delegates_client_tool_at_turn_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§Orchestrator delegates a client-declared tool at the turn boundary.

        Given a Session whose initial request carries ``tools: [file_read,
        bash]``; when the Runtime's ReAct iteration produces a ``file_read``
        tool call; then the turn closes with ``finish_reason: tool_calls``
        carrying the client-tool delegation on ``message.tool_calls`` and no
        Orchestrator Tool from the closed five is dispatched during that
        turn.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_fr_1",
                            name="file_read",
                            arguments_json='{"path": "src/auth.py"}',
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=15, completion_tokens=8, total_tokens=23
                    ),
                    finish_reason="tool_calls",
                ),
            ]
        )
        # Empty-results stub: any dispatch call raises AssertionError.
        tool_dispatch = _StubToolDispatch()
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=tool_dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [
                    {"role": "user", "content": "read src/auth.py"},
                ],
                "tools": [
                    self._file_read_tool_schema(),
                    self._bash_tool_schema(),
                ],
            },
        )

        assert response.status_code == 200
        body = response.json()
        choice = body["choices"][0]
        # Option C's wire signal: finish_reason is tool_calls, not stop.
        assert choice["finish_reason"] == "tool_calls", (
            f"Expected finish_reason='tool_calls' for client-tool delegation; "
            f"got {choice['finish_reason']!r}."
        )
        # No internal tool was dispatched — the stub's `calls` list is empty
        # because the name 'file_read' is not in TOOL_NAMES and the Runtime
        # must route it as a ClientToolCall instead.
        assert tool_dispatch.calls == [], (
            "A client-declared tool was routed through Tool Dispatch. Option "
            "C requires client-tool names (outside TOOL_NAMES) to emit a "
            "ClientToolCall chunk without any internal dispatch."
        )
        message = choice["message"]
        # OpenAI shape: delegated tool calls live on message.tool_calls; the
        # assistant content is empty / absent for a pure tool_calls turn.
        assert message.get("content") in (None, "")
        tool_calls = message["tool_calls"]
        assert isinstance(tool_calls, list)
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "file_read"
        assert tool_calls[0]["function"]["arguments"] == '{"path": "src/auth.py"}'
        # LLM saw the client tools alongside the five internal tools on
        # iteration 1 — the Runtime must advertise the union of both surfaces.
        assert len(llm.calls) == 1
        _, tools_seen = llm.calls[0]
        advertised_names = {t["function"]["name"] for t in tools_seen}
        assert "file_read" in advertised_names
        assert "bash" in advertised_names
        assert TOOL_NAMES.issubset(advertised_names), (
            f"Internal tool surface missing from LLM prompt; saw "
            f"{advertised_names - TOOL_NAMES} only."
        )

    def test_session_continuity_across_client_tool_round_trip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§Session turn count and token spend accumulate across a client-tool
        round trip.

        Given a SessionState at ``turn_count=5`` and ``token_spend=12000``;
        when the client sends the next request carrying the accumulated
        history plus a ``role: tool`` message with the file-read result;
        then the same ``SessionState`` is resolved, the Runtime resumes its
        ReAct loop with the tool result as an observation, ``turn_count``
        continues accumulating from 5 (not reset to 0), and ``token_spend``
        continues accumulating on the same Budget.
        """
        registry = SessionRegistry()
        seeded_identity = SessionIdentity(value="agent-client-42", method="user_field")
        seeded_state = registry.get_or_create_state(seeded_identity)
        seeded_state.turn_count = 5
        seeded_state.token_spend = 12000

        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="I read src/auth.py and found the authenticate helper.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=50, completion_tokens=12, total_tokens=62
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        # Empty-results stub proves no internal dispatch fires on the
        # resumption turn — the orchestrator reasons from the tool result
        # alone.
        tool_dispatch = _StubToolDispatch()
        # Scenario framing: Budget is sized for an extended agentic coding
        # session, so a mid-session token_spend of 12000 is well within
        # the configured ceiling.
        config = OrchestratorConfig(
            model_profile="test-profile",
            budget=BudgetDefaults(turn_limit=500, token_limit=10_000_000),
            autonomy_level="operator-as-tool-user",
            plexus_enabled=False,
            override_bounds=OverrideBounds(
                allow_budget_override=True,
                max_turn_limit=1_000,
                max_token_limit=20_000_000,
            ),
            allowed_profiles=("test-profile",),
            summarizer_ensemble="agentic-result-summarizer",
            orchestrator_system_prompt="",
        )
        client, shared_registry, _ = _build_client(
            monkeypatch,
            registry=registry,
            llm=llm,
            tool_dispatch=tool_dispatch,
            config=config,
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "user": "agent-client-42",
                "messages": [
                    {"role": "user", "content": "read src/auth.py"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_fr_1",
                                "type": "function",
                                "function": {
                                    "name": "file_read",
                                    "arguments": '{"path": "src/auth.py"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_fr_1",
                        "content": "def authenticate(user):\n    return True\n",
                    },
                ],
                "tools": [self._file_read_tool_schema()],
            },
        )

        assert response.status_code == 200
        body = response.json()
        choice = body["choices"][0]
        # Turn closes normally — the orchestrator has the tool result and
        # completes its reasoning.
        assert choice["finish_reason"] == "stop"
        assert (
            choice["message"]["content"]
            == "I read src/auth.py and found the authenticate helper."
        )

        # Session continuity: same SessionState resolved; accounting accumulates.
        state_after = shared_registry.get_or_create_state(seeded_identity)
        assert state_after is seeded_state, (
            "The second request resolved a different SessionState — Session "
            "continuity across a client-tool round trip is broken."
        )
        assert state_after.turn_count == 6, (
            f"turn_count did not carry the prior session's accumulation. "
            f"Expected 5 + 1 = 6; got {state_after.turn_count}."
        )
        assert state_after.token_spend == 12062, (
            f"token_spend did not carry the prior session's accumulation. "
            f"Expected 12000 + 62 = 12062; got {state_after.token_spend}."
        )

        # The orchestrator LLM saw the tool-round-trip observation on its
        # first (and only) iteration of this turn.
        assert len(llm.calls) == 1
        messages_seen, _ = llm.calls[0]
        assistant_with_tool_calls = [
            m
            for m in messages_seen
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_with_tool_calls) == 1, (
            "The LLM did not see its own prior turn's tool_calls on replay. "
            "OpenAI tool-round-trip coherence requires the assistant message "
            "with tool_calls to ride alongside the tool result."
        )
        assert assistant_with_tool_calls[0]["tool_calls"][0]["id"] == "call_fr_1"
        tool_messages = [m for m in messages_seen if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_fr_1"
        assert "authenticate" in tool_messages[0]["content"]

    def test_mixed_batch_rejected_and_retried_without_silent_loss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mixed-batch discipline — reject, feed tool errors, let the LLM retry.

        When the orchestrator LLM emits a single batch containing both
        internal (TOOL_NAMES) and client-declared tool calls, Option C's
        one-kind-per-turn property is violated (internal dispatches
        synchronously in-process; client delegations close the turn). The
        Runtime rejects the batch by feeding a ``mixed_batch`` error
        observation per call, then continues the loop so the LLM can retry
        with a pure batch — no silent data loss, no mis-routing.

        Three LLM iterations verify the full round trip:
          1. LLM emits mixed ``[list_ensembles, file_read]``. Runtime
             rejects both; neither Tool Dispatch nor ClientToolCall fires.
          2. LLM observes the errors, retries with internal only; dispatch
             runs, observation flows back.
          3. LLM emits client-only ``file_read``; Runtime closes the turn.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_mixed_a",
                            name="list_ensembles",
                            arguments_json="{}",
                        ),
                        ToolCall(
                            id="call_mixed_b",
                            name="file_read",
                            arguments_json='{"path": "src/auth.py"}',
                        ),
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=10, total_tokens=30
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_retry_internal",
                            name="list_ensembles",
                            arguments_json="{}",
                        ),
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=50, completion_tokens=5, total_tokens=55
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_retry_client",
                            name="file_read",
                            arguments_json='{"path": "src/auth.py"}',
                        ),
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=70, completion_tokens=8, total_tokens=78
                    ),
                    finish_reason="tool_calls",
                ),
            ]
        )
        from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallSuccess

        tool_dispatch = _StubToolDispatch(
            results={
                "call_retry_internal": ToolCallSuccess(
                    id="call_retry_internal",
                    name="list_ensembles",
                    content=[{"name": "auth-analyzer"}],
                )
            }
        )
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=tool_dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "look up and then read"}],
                "tools": [self._file_read_tool_schema()],
            },
        )

        assert response.status_code == 200
        body = response.json()
        # Final turn emits the file_read ClientToolCall from iteration 3.
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        final_tool_calls = body["choices"][0]["message"]["tool_calls"]
        assert len(final_tool_calls) == 1
        assert final_tool_calls[0]["function"]["name"] == "file_read"
        # Tool Dispatch saw only the retry's internal call — not the
        # mixed-batch call_mixed_a (which was rejected).
        assert [call.name for call in tool_dispatch.calls] == ["list_ensembles"]
        assert tool_dispatch.calls[0].id == "call_retry_internal"
        # LLM was called 3 times. Iteration 2 must have seen a mixed_batch
        # error observation for BOTH rejected calls — no silent loss.
        assert len(llm.calls) == 3
        iter2_messages = llm.calls[1][0]
        rejection_messages = [m for m in iter2_messages if m.get("role") == "tool"]
        assert len(rejection_messages) == 2, (
            "Both mixed-batch calls must surface as role:tool error "
            "observations on the LLM's retry — otherwise the LLM has no "
            "signal to correct its batch discipline."
        )
        observed_ids = {m["tool_call_id"] for m in rejection_messages}
        assert observed_ids == {"call_mixed_a", "call_mixed_b"}
        for message in rejection_messages:
            payload = json.loads(message["content"])
            assert payload["error"] == "mixed_batch", (
                "Rejection payload must name the discipline violated so "
                "the LLM can reason about how to retry."
            )

    def test_client_tool_shadowing_internal_name_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Name-collision guard — TOOL_NAMES are reserved.

        The five internal tool names (ADR-003) are llm-orc's action
        surface. A client-declared tool whose name shadows any of them
        would silently misroute — the Runtime would treat the internal
        call as a client delegation. Reject at the Serving Layer with
        HTTP 400 so operators learn about the conflict immediately rather
        than through a mysterious stream of ``finish_reason: tool_calls``
        when they expected ``list_ensembles`` to run.
        """
        llm = _ScriptedLLM(responses=[])  # never called
        tool_dispatch = _StubToolDispatch()  # never called
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=tool_dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "invoke_ensemble",
                            "description": (
                                "Client attempting to shadow an internal tool."
                            ),
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

        assert response.status_code == 400
        body = response.json()
        # The app's custom HTTPException handler wraps ``detail`` under an
        # ``"error"`` key (see ``web/server.py``). The inner payload is
        # the dict this test actually cares about.
        payload = body["error"] if isinstance(body.get("error"), dict) else body
        assert payload["error"] == "reserved_tool_name"
        assert "invoke_ensemble" in payload["message"]
        # Collision rejected before Runtime — LLM never invoked, dispatch
        # never consulted.
        assert llm.calls == []

    def test_streaming_client_tool_delegation_yields_tool_calls_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario (a) under streaming — SSE framing carries the delegation.

        Proves the full wire path: real Runtime → OpenAiSseFormatter → SSE
        bytes. The last content-bearing chunk is a ``chat.completion.chunk``
        with ``delta.tool_calls`` and ``finish_reason: tool_calls``, then
        ``[DONE]`` — matching what an OpenAI-compat client (OpenCode, Roo
        Code, Cline) parses to surface a tool call to the user.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_fr_s",
                            name="file_read",
                            arguments_json='{"path": "src/auth.py"}',
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=15, completion_tokens=8, total_tokens=23
                    ),
                    finish_reason="tool_calls",
                ),
            ]
        )
        tool_dispatch = _StubToolDispatch()
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=tool_dispatch)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "stream": True,
                "messages": [{"role": "user", "content": "read src/auth.py"}],
                "tools": [self._file_read_tool_schema()],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _parse_sse_frames(response.content)
        # Stream opener + ClientToolCall chunk + [DONE] terminator.
        assert any(frame.get("__done__") for frame in frames), (
            "Stream did not terminate with [DONE] — formatter contract broken."
        )
        tool_call_frames = [
            frame
            for frame in frames
            if not frame.get("__done__")
            and frame.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
        ]
        assert len(tool_call_frames) == 1, (
            f"Expected exactly one chunk carrying delta.tool_calls for the "
            f"client-tool delegation; got {len(tool_call_frames)}."
        )
        framed = tool_call_frames[0]
        choice = framed["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tool_calls = choice["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "file_read"
        assert tool_calls[0]["function"]["arguments"] == '{"path": "src/auth.py"}'
        # No internal dispatch fired.
        assert tool_dispatch.calls == []

    def test_pre_invoke_delegation_reads_file_before_invoking_ensemble(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§Scenario (c): pre-invoke delegation for a file-content ensemble.

        An ensemble whose first agent consumes file content (not a file
        path) is invoked by the orchestrator via the retry-free happy
        path: read the file at the prior turn boundary, fold the content
        into ``input_data``, then invoke the ensemble atomically. The
        Ensemble Engine never suspends; Layer 3 (ADR-001/002) is
        unchanged.

        Exercises two HTTP requests with session continuity in between:

          Request 1 — ``tools: [file_read]``, user asks for analysis:
            * Iteration 1: LLM calls ``list_ensembles``; Tool Dispatch
              returns ``auth-analyzer`` with its description (the WP-F
              build-time decision — ``description`` is on the schema).
            * Iteration 2: LLM reasons from the description that
              ``auth-analyzer`` consumes file content, emits ``file_read``.
              Turn closes with ``finish_reason: tool_calls``.

          Request 2 — client returns the file content as ``role: tool``:
            * Iteration 1: LLM emits ``invoke_ensemble`` with the file
              content folded into ``input`` alongside the task description.
            * Iteration 2: LLM observes the ensemble's summary and
              emits a stop response.

        The test runs the real :class:`OrchestratorToolDispatch` + real
        :class:`ResultSummarizerHarness` over a stub
        :class:`_StubEnsembleOperations` so the invariant that
        ``list_ensembles`` surfaces ``description`` is verified through
        the production dispatch path — not only inside the stub.
        """
        ensemble_run_result = {
            "results": {
                "auth-analyzer-agent": {
                    "response": (
                        "authenticate uses a hardcoded True — replace with "
                        "real credential validation."
                    )
                }
            },
            "synthesis": None,
            "status": "success",
        }
        summary_result = {
            "results": {
                "summarizer": {
                    "response": (
                        "auth-analyzer flagged a hardcoded True in authenticate."
                    )
                }
            },
            "synthesis": None,
            "status": "success",
        }
        operations = _StubEnsembleOperations(
            {
                "auth-analyzer": ensemble_run_result,
                "test-summarizer": summary_result,
            },
            library_entries=[
                {
                    "name": "auth-analyzer",
                    "source": "local",
                    "relative_path": "auth-analyzer.yaml",
                    "agent_count": 1,
                    "description": (
                        "Analyzes Python source file contents for "
                        "authentication weaknesses. Pass the file's "
                        "source code as input."
                    ),
                },
            ],
        )
        harness = ResultSummarizerHarness(
            invoker=operations, summarizer_name="test-summarizer"
        )
        dispatch = OrchestratorToolDispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
            **_dispatch_kwargs_without_composition(),
        )

        file_content = "def authenticate(user):\n    return True\n"
        llm = _ScriptedLLM(
            responses=[
                # Request 1, iteration 1: list the library.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_list",
                            name="list_ensembles",
                            arguments_json="{}",
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=15, completion_tokens=4, total_tokens=19
                    ),
                    finish_reason="tool_calls",
                ),
                # Request 1, iteration 2: ensemble description says
                # "pass the file's source code" — delegate file_read.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_fr_precall",
                            name="file_read",
                            arguments_json='{"path": "src/auth.py"}',
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=60, completion_tokens=8, total_tokens=68
                    ),
                    finish_reason="tool_calls",
                ),
                # Request 2, iteration 1: fold the file content into the
                # ensemble's input_data and invoke atomically.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_invoke",
                            name="invoke_ensemble",
                            arguments_json=json.dumps(
                                {
                                    "name": "auth-analyzer",
                                    "input": (
                                        "Analyze authentication in "
                                        "src/auth.py. Source:\n" + file_content
                                    ),
                                }
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=120, completion_tokens=15, total_tokens=135
                    ),
                    finish_reason="tool_calls",
                ),
                # Request 2, iteration 2: summarize and stop.
                ToolCallingResponse(
                    content=("auth-analyzer found a hardcoded True in authenticate."),
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=160, completion_tokens=12, total_tokens=172
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        # ---- Request 1: orchestrator learns the ensemble exists and
        #      delegates file_read at the turn boundary.
        first_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "user": "pre-invoke-client",
                "messages": [
                    {
                        "role": "user",
                        "content": "Check src/auth.py for auth issues.",
                    }
                ],
                "tools": [self._file_read_tool_schema()],
            },
        )

        assert first_response.status_code == 200
        first_body = first_response.json()
        first_choice = first_body["choices"][0]
        assert first_choice["finish_reason"] == "tool_calls"
        delegated = first_choice["message"]["tool_calls"]
        assert len(delegated) == 1
        assert delegated[0]["function"]["name"] == "file_read"
        assert delegated[0]["function"]["arguments"] == '{"path": "src/auth.py"}'
        delegated_tool_call_id = delegated[0]["id"]

        # Iteration 1 fired list_ensembles through the real dispatch.
        invoked_ensemble_names = [c["ensemble_name"] for c in operations.calls]
        assert invoked_ensemble_names == [], (
            "Expected zero ensemble invocations on request 1 — only "
            "list_ensembles (metadata read) should have run."
        )
        # The LLM's iteration-2 context must include the list_ensembles
        # result with the auth-analyzer description visible. Otherwise
        # the scenario's "inference from description" premise has no
        # evidence for the test's claims.
        iter2_messages = llm.calls[1][0]
        iter2_tool_msgs = [m for m in iter2_messages if m.get("role") == "tool"]
        assert len(iter2_tool_msgs) == 1
        observed_library = json.loads(iter2_tool_msgs[0]["content"])
        # ``list_ensembles`` returns the library metadata list directly —
        # no summarization interposes on this tool (Harness runs only on
        # ``invoke_ensemble``). The description field is the load-bearing
        # evidence the orchestrator uses to infer a file-content
        # dependency; verify it reaches the LLM context untouched.
        assert isinstance(observed_library, list)
        assert len(observed_library) == 1
        auth_entry = observed_library[0]
        assert auth_entry["name"] == "auth-analyzer"
        assert "authentication weaknesses" in auth_entry["description"], (
            "Ensemble description missing or altered in the list_ensembles "
            "tool result — the orchestrator needs it to decide on pre-invoke "
            f"delegation. Got entry: {auth_entry!r}"
        )

        # ---- Request 2: client returns file_read result; orchestrator
        #      invokes auth-analyzer with content folded into input_data.
        second_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "user": "pre-invoke-client",
                "messages": [
                    {
                        "role": "user",
                        "content": "Check src/auth.py for auth issues.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": delegated_tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "file_read",
                                    "arguments": '{"path": "src/auth.py"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": delegated_tool_call_id,
                        "content": file_content,
                    },
                ],
                "tools": [self._file_read_tool_schema()],
            },
        )

        assert second_response.status_code == 200
        second_body = second_response.json()
        second_choice = second_body["choices"][0]
        assert second_choice["finish_reason"] == "stop"
        assert (
            second_choice["message"]["content"]
            == "auth-analyzer found a hardcoded True in authenticate."
        )

        # invoke_ensemble ran with the file content folded into input_data
        # — the load-bearing assertion for scenario (c).
        ensemble_invocations = [
            c for c in operations.calls if c.get("ensemble_name") == "auth-analyzer"
        ]
        assert len(ensemble_invocations) == 1, (
            "Expected a single auth-analyzer invocation on request 2."
        )
        # OrchestratorToolDispatch.invoke_ensemble translates the LLM's
        # ``arguments.input`` into ``{"ensemble_name": ..., "input": ...}``
        # before calling the EnsembleOperations facade — so the stub sees
        # it under the ``input`` key.
        folded_input = ensemble_invocations[0].get("input", "")
        assert "def authenticate" in folded_input, (
            "The file content was not folded into invoke_ensemble's "
            "input — pre-invoke delegation's whole point is that the "
            f"ensemble sees the content, not a path. Got: {folded_input!r}"
        )
        assert "src/auth.py" in folded_input

    def _bash_client_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the client environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        }

    def test_retry_pattern_resolves_mid_execution_client_tool_need(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§Scenario (d): retry pattern for un-predicted mid-execution needs.

        A composed ensemble whose second-phase agent depends on a client-
        tool result the orchestrator did NOT predict at invoke-time emits
        a structured ``{"needs_client_tool": {"tool": ..., "args": ...}}``
        signal. The Result Summarizer Harness preserves that signal
        through to the orchestrator's tool-call result (build-time
        convention: see updated summarizer ensemble's prompt). The
        orchestrator observes the signal, closes the next turn with a
        ``bash`` client-tool delegation, and re-invokes the ensemble on
        the subsequent request with the grep output folded into
        ``input_data`` alongside the original task — the retry pattern.
        The DAG engine never suspends (ADR-001/002: Layer 3 unchanged).

        Two HTTP requests, four LLM iterations:

          Request 1 — ``tools: [bash]``, user asks for TODO scan:
            * Iter 1: invoke_ensemble(repo-scanner, <task>).
              The stub summarizer returns a ``needs_client_tool`` JSON
              string (the preservation is verified through the real
              Harness + real Tool Dispatch path).
            * Iter 2: LLM reads the signal, emits bash ClientToolCall.
              Turn closes.

          Request 2 — client returns grep output as role:tool:
            * Iter 1: LLM re-invokes repo-scanner with the grep output
              folded into input alongside the original task. Stub
              summarizer returns a normal prose summary this time.
            * Iter 2: LLM emits stop with the final answer.
        """
        # First invoke-ensemble: the composed ensemble emits the
        # ``needs_client_tool`` retry signal (phase-2 agent cannot
        # proceed). The stubbed summarizer preserves the JSON structure
        # verbatim per the updated convention.
        needs_signal_json = json.dumps(
            {
                "needs_client_tool": {
                    "tool": "bash",
                    "args": {"command": 'grep -r "TODO" /client/repo'},
                }
            }
        )
        first_ensemble_raw = {
            "results": {
                "scanner-phase-1": {"response": "list ready"},
                "scanner-phase-2": {"response": needs_signal_json},
            },
            "synthesis": None,
            "status": "success",
        }
        first_summarizer_result = {
            "results": {"summarizer": {"response": needs_signal_json}},
            "synthesis": None,
            "status": "success",
        }
        # Second invoke-ensemble (the retry): ensemble runs to completion
        # with the grep output folded into input_data. Summarizer produces
        # a normal prose summary this time.
        second_ensemble_raw = {
            "results": {
                "scanner-phase-1": {"response": "list ready"},
                "scanner-phase-2": {
                    "response": "Found 3 TODOs: auth.py:12, db.py:47, ui.py:3."
                },
            },
            "synthesis": None,
            "status": "success",
        }
        second_summarizer_result = {
            "results": {
                "summarizer": {
                    "response": "Scanned repo — 3 TODOs in auth.py, db.py, ui.py.",
                }
            },
            "synthesis": None,
            "status": "success",
        }

        # One stub handles both invocations of repo-scanner by returning
        # different results per call — the list is popped in order so the
        # first invoke returns the needs_signal result, the second returns
        # the normal one.
        class _SequencedOperations:
            def __init__(self) -> None:
                self._repo_results = [first_ensemble_raw, second_ensemble_raw]
                self._summarizer_results = [
                    first_summarizer_result,
                    second_summarizer_result,
                ]
                self.calls: list[dict[str, Any]] = []

            async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
                self.calls.append(arguments)
                name = arguments.get("ensemble_name")
                if name == "repo-scanner":
                    return self._repo_results.pop(0)
                if name == "test-summarizer":
                    return self._summarizer_results.pop(0)
                raise ValueError(f"unexpected ensemble '{name}'")

            async def read_ensembles(self) -> list[dict[str, Any]]:
                return []

        operations = _SequencedOperations()
        harness = ResultSummarizerHarness(
            invoker=operations, summarizer_name="test-summarizer"
        )
        dispatch = OrchestratorToolDispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
            **_dispatch_kwargs_without_composition(),
        )

        grep_output = (
            "/client/repo/auth.py:12: # TODO: validate credentials\n"
            "/client/repo/db.py:47: # TODO: connection pool\n"
            "/client/repo/ui.py:3: # TODO: accessibility\n"
        )
        llm = _ScriptedLLM(
            responses=[
                # Request 1, iter 1: initial invoke_ensemble.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_invoke_first",
                            name="invoke_ensemble",
                            arguments_json=json.dumps(
                                {
                                    "name": "repo-scanner",
                                    "input": "Scan /client/repo for TODOs.",
                                }
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=10, total_tokens=30
                    ),
                    finish_reason="tool_calls",
                ),
                # Request 1, iter 2: LLM sees needs_client_tool signal,
                # emits bash ClientToolCall (turn closes).
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_bash",
                            name="bash",
                            arguments_json=(
                                '{"command": "grep -r \\"TODO\\" /client/repo"}'
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=60, completion_tokens=8, total_tokens=68
                    ),
                    finish_reason="tool_calls",
                ),
                # Request 2, iter 1: re-invoke with grep output folded in.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_invoke_retry",
                            name="invoke_ensemble",
                            arguments_json=json.dumps(
                                {
                                    "name": "repo-scanner",
                                    "input": (
                                        "Scan /client/repo for TODOs.\n"
                                        "Grep output:\n" + grep_output
                                    ),
                                }
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=120, completion_tokens=15, total_tokens=135
                    ),
                    finish_reason="tool_calls",
                ),
                # Request 2, iter 2: final answer.
                ToolCallingResponse(
                    content="Found 3 TODOs across auth.py, db.py, ui.py.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=160, completion_tokens=12, total_tokens=172
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        client, _, _ = _build_client(monkeypatch, llm=llm, tool_dispatch=dispatch)

        # ---- Request 1: orchestrator runs the ensemble, observes the
        #      retry signal in the summary, delegates bash.
        first_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "user": "retry-pattern-client",
                "messages": [
                    {
                        "role": "user",
                        "content": "Scan /client/repo for TODOs.",
                    }
                ],
                "tools": [self._bash_client_tool_schema()],
            },
        )

        assert first_response.status_code == 200
        first_body = first_response.json()
        first_choice = first_body["choices"][0]
        assert first_choice["finish_reason"] == "tool_calls"
        bash_delegation = first_choice["message"]["tool_calls"]
        assert len(bash_delegation) == 1
        assert bash_delegation[0]["function"]["name"] == "bash"
        bash_call_id = bash_delegation[0]["id"]

        # Verify the retry signal reached the LLM's iter 2 context.
        iter2_messages = llm.calls[1][0]
        iter2_tool_msgs = [m for m in iter2_messages if m.get("role") == "tool"]
        assert len(iter2_tool_msgs) == 1
        summary_payload = json.loads(iter2_tool_msgs[0]["content"])
        # Harness wraps summary in {"summary": <str>}; the summary string
        # IS the needs_client_tool JSON preserved verbatim.
        summary_str = summary_payload.get("summary", "")
        assert "needs_client_tool" in summary_str, (
            "Summarizer Harness did not preserve the needs_client_tool "
            f"signal through to the orchestrator. Got: {summary_payload!r}"
        )
        parsed_signal = json.loads(summary_str)
        assert parsed_signal["needs_client_tool"]["tool"] == "bash"

        # ---- Request 2: client returns grep output; orchestrator
        #      re-invokes the ensemble with the result folded in.
        second_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "user": "retry-pattern-client",
                "messages": [
                    {
                        "role": "user",
                        "content": "Scan /client/repo for TODOs.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": bash_call_id,
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": (
                                        '{"command": "grep -r \\"TODO\\" /client/repo"}'
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": bash_call_id,
                        "content": grep_output,
                    },
                ],
                "tools": [self._bash_client_tool_schema()],
            },
        )

        assert second_response.status_code == 200
        second_body = second_response.json()
        second_choice = second_body["choices"][0]
        assert second_choice["finish_reason"] == "stop"
        assert "Found 3 TODOs" in second_choice["message"]["content"]

        # Load-bearing assertion: the re-invocation carried the grep
        # output folded into its input. This is the retry pattern's
        # defining property.
        repo_scanner_invocations = [
            c for c in operations.calls if c.get("ensemble_name") == "repo-scanner"
        ]
        assert len(repo_scanner_invocations) == 2, (
            "Expected exactly two repo-scanner invocations — the initial "
            "invoke and the retry with grep output folded in."
        )
        retry_input = repo_scanner_invocations[1].get("input", "")
        assert "Grep output:" in retry_input
        assert "/client/repo/auth.py:12" in retry_input, (
            "Retry invocation did not carry the grep output in input — "
            "the retry pattern's defining property is the client-tool "
            f"result folded into input_data. Got: {retry_input!r}"
        )

    def test_composed_ensemble_without_retry_signal_silently_degrades(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§Scenario (negative): silent quality failure when convention fails.

        A composed ensemble whose second-phase agent depends on a
        client-tool result but does NOT emit the ``needs_client_tool``
        convention produces a normal-shaped prose summary. The
        orchestrator has no signal to motivate a retry, accepts the
        (hallucinated) result, and returns a final completion. The
        Session's *structural* behavior is correct — no crash, Budget
        enforces, turn_count and token_spend advance normally, no
        spurious ClientToolCall — even though the answer is wrong.

        Acceptance is structural correctness, not result correctness.
        Catching the quality failure belongs to WP-H's Calibration
        Gate (ADR-007) — documented here as carried-forward WP-H scope.
        """
        # Prose-only summary — no needs_client_tool signal. This models
        # what happens when the ensemble was authored or composed
        # without convention compliance.
        prose_ensemble_raw = {
            "results": {
                "scanner-phase-1": {"response": "list ready"},
                "scanner-phase-2": {
                    "response": (
                        "Repo contains 3 Python files: auth.py, db.py, ui.py. "
                        "No obvious TODOs found in skim review."
                    )
                },
            },
            "synthesis": None,
            "status": "success",
        }
        prose_summary_result = {
            "results": {
                "summarizer": {
                    "response": (
                        "Scanned repo — no TODOs found in auth.py, db.py, ui.py."
                    )
                }
            },
            "synthesis": None,
            "status": "success",
        }
        operations = _StubEnsembleOperations(
            {
                "repo-scanner": prose_ensemble_raw,
                "test-summarizer": prose_summary_result,
            }
        )
        harness = ResultSummarizerHarness(
            invoker=operations, summarizer_name="test-summarizer"
        )
        dispatch = OrchestratorToolDispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
            **_dispatch_kwargs_without_composition(),
        )

        llm = _ScriptedLLM(
            responses=[
                # Iter 1: invoke the ensemble.
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_invoke_prose",
                            name="invoke_ensemble",
                            arguments_json=json.dumps(
                                {
                                    "name": "repo-scanner",
                                    "input": "Scan /client/repo for TODOs.",
                                }
                            ),
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=10, total_tokens=30
                    ),
                    finish_reason="tool_calls",
                ),
                # Iter 2: no retry signal visible — orchestrator accepts
                # the summary as final and emits stop with the
                # (hallucinated) content.
                ToolCallingResponse(
                    content="No TODOs found in the repo.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=60, completion_tokens=8, total_tokens=68
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        registry = SessionRegistry()
        client, shared_registry, _ = _build_client(
            monkeypatch, registry=registry, llm=llm, tool_dispatch=dispatch
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "user": "negative-path-client",
                "messages": [
                    {
                        "role": "user",
                        "content": "Scan /client/repo for TODOs.",
                    }
                ],
                "tools": [self._bash_client_tool_schema()],
            },
        )

        assert response.status_code == 200
        body = response.json()
        choice = body["choices"][0]
        # Structural assertion A: turn closed with stop, not tool_calls.
        assert choice["finish_reason"] == "stop"
        # Structural assertion B: no tool_calls emitted on the response
        # surface — no retry triggered because no signal was visible.
        message = choice["message"]
        assert message.get("tool_calls") in (None, []), (
            f"Negative scenario emitted a ClientToolCall despite no "
            f"signal in the summary. Message: {message!r}"
        )
        # Structural assertion C: bash was never dispatched on the
        # client side. The only ensemble invocations were the single
        # repo-scanner call and its summarizer.
        invoked_ensembles = [c["ensemble_name"] for c in operations.calls]
        # (bash isn't an ensemble; this assertion guards against a
        # future typo misrouting bash through the ensemble facade.)
        assert "bash" not in invoked_ensembles
        assert invoked_ensembles.count("repo-scanner") == 1, (
            "Expected one repo-scanner invocation — no retry path under "
            "the negative scenario."
        )

        # Structural assertion D: Session Budget state advances normally
        # — two iterations' worth of tokens accumulated, turn_count == 2.
        seeded_identity = SessionIdentity(
            value="negative-path-client", method="user_field"
        )
        state = shared_registry.get_or_create_state(seeded_identity)
        assert state.turn_count == 2, (
            f"Expected turn_count=2 (invoke + stop); got {state.turn_count}."
        )
        # tokens: 30 + 68 from the two scripted responses.
        assert state.token_spend == 98

        # Commentary: the orchestrator's content ("No TODOs found") is
        # factually wrong but the Session executed correctly. Catching
        # this quality failure is WP-H's Calibration Gate scope.
