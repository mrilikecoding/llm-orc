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
from typing import Any

import pytest
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
    """

    def __init__(self, results_by_ensemble: dict[str, dict[str, Any]]) -> None:
        self._results = dict(results_by_ensemble)
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        name = arguments.get("ensemble_name")
        if not isinstance(name, str) or name not in self._results:
            raise ValueError(f"ensemble '{name}' not in stub library")
        return self._results[name]

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return []


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
            operations=operations, harness=harness, autonomy_policy=policy
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
        # Third LLM turn saw compose_ensemble's not_yet_wired observation.
        third_messages = llm.calls[2][0]
        compose_tool_msgs = [
            m
            for m in third_messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_com"
        ]
        assert len(compose_tool_msgs) == 1
        observed = json.loads(compose_tool_msgs[0]["content"])
        assert observed.get("error") == "not_yet_wired"

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
