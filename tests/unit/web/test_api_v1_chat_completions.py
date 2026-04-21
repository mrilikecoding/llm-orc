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

from llm_orc.agentic.session_registry import SessionIdentity, SessionRegistry
from llm_orc.agentic.session_start import (
    PromptFragment,
    SessionContext,
    SessionStartCache,
)
from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app


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
) -> tuple[TestClient, SessionRegistry, SessionStartCache]:
    """Wire a TestClient with an isolated Session Registry and session-start cache.

    Follows the pattern in ``test_api_v1_models.py`` — override the
    module-level factories so each test gets its own state. Returns the
    client, registry, and cache so tests can inspect accumulated state.
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
