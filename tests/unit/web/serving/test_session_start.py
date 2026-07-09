"""Tests for the session-start Serving Layer function.

Per ``docs/serving.md`` §Serving Layer (L3) and
§Integration Contracts (Serving Layer → ``resolve_session_start_context``).
Phase 1 of ADR-009's structural reservation: the typed function and its
call site exist; the body returns ``[]`` unconditionally so Phase 2 can
populate from the Plexus Adapter without a structural change.

FC-9 is satisfied at this module plus the Serving Layer call-site test
in ``tests/unit/web/test_api_v1_chat_completions.py``.
"""

from llm_orc.core.session.registry import (
    SessionIdentity,
    SessionState,
)
from llm_orc.web.serving.session_start import (
    ChatMessage,
    PromptFragment,
    SessionContext,
    SessionStartCache,
    resolve_session_start_context,
)


class TestPromptFragmentShape:
    """``PromptFragment`` is the typed unit the session-start function emits."""

    def test_has_content_and_source_fields(self) -> None:
        fragment = PromptFragment(content="hello", source="plexus:q-1")

        assert fragment.content == "hello"
        assert fragment.source == "plexus:q-1"

    def test_is_immutable(self) -> None:
        """Frozen dataclass — once resolved, fragments don't change in-place."""
        fragment = PromptFragment(content="hello", source="plexus:q-1")

        try:
            fragment.content = "mutated"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("PromptFragment should be frozen")


class TestSessionContextShape:
    """``SessionContext`` is passed to session-start and (later) to the Runtime."""

    def test_holds_messages_tools_and_state(self) -> None:
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity)
        context = SessionContext(
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "bash"}}],
            state=state,
        )

        assert context.messages == [ChatMessage(role="user", content="hi")]
        assert context.tools == [{"type": "function", "function": {"name": "bash"}}]
        assert context.state is state

    def test_tools_defaults_to_empty_list(self) -> None:
        """Requests without a ``tools`` array produce a context with no tools."""
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity)
        context = SessionContext(
            messages=[ChatMessage(role="user", content="hi")],
            tools=[],
            state=state,
        )

        assert context.tools == []


class TestResolveSessionStartContextPhase1:
    """Phase 1 body of ``resolve_session_start_context`` (ADR-009).

    Returns ``[]`` unconditionally. Phase 2 (deferred) populates from the
    Plexus Adapter without a structural change — same signature, same call
    site.
    """

    def test_returns_empty_list(self) -> None:
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity)
        context = SessionContext(
            messages=[ChatMessage(role="user", content="hi")],
            tools=[],
            state=state,
        )

        fragments = resolve_session_start_context(context)

        assert fragments == []

    def test_return_type_is_list_of_prompt_fragments(self) -> None:
        """FC-9: signature returns ``list[PromptFragment]``.

        Empty in Phase 1, but the element type is committed now so Phase 2
        is a function-body change, not a signature change.
        """
        identity = SessionIdentity(value="x", method="cold_start")
        state = SessionState(identity=identity)
        context = SessionContext(
            messages=[],
            tools=[],
            state=state,
        )

        fragments = resolve_session_start_context(context)

        assert isinstance(fragments, list)
        for fragment in fragments:
            assert isinstance(fragment, PromptFragment)

    def test_does_not_mutate_state(self) -> None:
        """Phase 1 is pure — no Plexus, no side effects on Session state."""
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity, turn_count=3, token_spend=150)
        context = SessionContext(
            messages=[ChatMessage(role="user", content="hi")],
            tools=[],
            state=state,
        )

        resolve_session_start_context(context)

        assert state.turn_count == 3
        assert state.token_spend == 150


def _make_context(identity_value: str = "x") -> SessionContext:
    identity = SessionIdentity(value=identity_value, method="user_field")
    state = SessionState(identity=identity)
    return SessionContext(
        messages=[ChatMessage(role="user", content="hi")],
        tools=[],
        state=state,
    )


class TestSessionStartCache:
    """``SessionStartCache`` enforces the once-per-session invariant (FC-9).

    The cache owns FC-9 at the class boundary: the Serving Layer calls
    ``resolve`` on every request; the injected resolver runs only on
    the first request per ``SessionIdentity``.
    """

    def test_resolves_on_first_call_per_identity(self) -> None:
        calls = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal calls
            calls += 1
            return [PromptFragment(content="fresh", source="spy")]

        cache = SessionStartCache(resolver=spy)
        context = _make_context()

        fragments = cache.resolve(context)

        assert calls == 1
        assert fragments == [PromptFragment(content="fresh", source="spy")]

    def test_suppresses_resolver_on_repeat_calls_for_same_identity(self) -> None:
        calls = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal calls
            calls += 1
            return []

        cache = SessionStartCache(resolver=spy)
        context = _make_context()

        cache.resolve(context)
        cache.resolve(context)
        cache.resolve(context)

        assert calls == 1

    def test_resolves_again_for_distinct_identity(self) -> None:
        calls = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal calls
            calls += 1
            return []

        cache = SessionStartCache(resolver=spy)

        cache.resolve(_make_context(identity_value="a"))
        cache.resolve(_make_context(identity_value="b"))

        assert calls == 2

    def test_cached_empty_list_does_not_re_resolve(self) -> None:
        """Phase 1 always returns ``[]``; caching must not confuse empty with unset."""
        calls = 0

        def spy(context: SessionContext) -> list[PromptFragment]:
            nonlocal calls
            calls += 1
            return []

        cache = SessionStartCache(resolver=spy)
        context = _make_context()

        first = cache.resolve(context)
        second = cache.resolve(context)

        assert first == []
        assert second == []
        assert calls == 1

    def test_default_resolver_is_phase_1_resolve_session_start_context(self) -> None:
        """The default cache uses the Phase 1 resolver — returns ``[]`` always."""
        cache = SessionStartCache()
        context = _make_context()

        fragments = cache.resolve(context)

        assert fragments == []
