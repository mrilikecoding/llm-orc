"""Tests for the Session Registry module.

Per `docs/serving.md` §Session Registry (L3) and
§Integration Contracts (Serving Layer → Session Registry; consumers →
Session Registry).
"""

from llm_orc.core.session.registry import (
    SessionIdentity,
    SessionRegistry,
    SessionState,
)
from llm_orc.web.serving.session_start import ChatMessage


class TestSessionIdentityDerivation:
    """Identity is derived from request features.

    Phase 1 supports two methods: client-supplied `user` field and
    message-prefix hash. Cold-start falls back to a fresh identity.
    """

    def test_uses_user_field_when_provided(self) -> None:
        registry = SessionRegistry()
        messages = [ChatMessage(role="user", content="hello")]

        identity = registry.resolve_identity(messages=messages, user_field="client-abc")

        assert identity.value == "client-abc"
        assert identity.method == "user_field"

    def test_hashes_first_user_message_when_user_field_absent(self) -> None:
        registry = SessionRegistry()
        messages = [ChatMessage(role="user", content="refactor the parser")]

        identity = registry.resolve_identity(messages=messages, user_field=None)

        assert identity.method == "message_prefix"
        assert identity.value  # nonempty

    def test_identity_is_deterministic_for_same_prefix(self) -> None:
        """Two requests with the same first user message yield the same identity."""
        registry = SessionRegistry()
        messages_a = [
            ChatMessage(role="user", content="refactor the parser"),
            ChatMessage(role="assistant", content="ok"),
        ]
        messages_b = [
            ChatMessage(role="user", content="refactor the parser"),
            ChatMessage(role="assistant", content="ok"),
            ChatMessage(role="user", content="now the lexer"),
        ]

        id_a = registry.resolve_identity(messages=messages_a, user_field=None)
        id_b = registry.resolve_identity(messages=messages_b, user_field=None)

        assert id_a == id_b

    def test_identity_differs_for_different_first_user_message(self) -> None:
        registry = SessionRegistry()
        id_a = registry.resolve_identity(
            messages=[ChatMessage(role="user", content="task one")],
            user_field=None,
        )
        id_b = registry.resolve_identity(
            messages=[ChatMessage(role="user", content="task two")],
            user_field=None,
        )

        assert id_a != id_b

    def test_cold_start_when_no_user_message(self) -> None:
        """No user message and no user field: fresh identity per request."""
        registry = SessionRegistry()
        messages = [ChatMessage(role="system", content="you are helpful")]

        id_a = registry.resolve_identity(messages=messages, user_field=None)
        id_b = registry.resolve_identity(messages=messages, user_field=None)

        assert id_a.method == "cold_start"
        assert id_b.method == "cold_start"
        assert id_a != id_b

    def test_user_field_takes_precedence_over_message_prefix(self) -> None:
        registry = SessionRegistry()
        messages = [ChatMessage(role="user", content="a task")]

        identity = registry.resolve_identity(
            messages=messages, user_field="explicit-id"
        )

        assert identity.method == "user_field"
        assert identity.value == "explicit-id"


class TestSessionStateAccounting:
    """SessionState tracks cumulative turn count and token spend."""

    def test_state_initializes_with_zero_counts(self) -> None:
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity)

        assert state.turn_count == 0
        assert state.token_spend == 0

    def test_record_iteration_increments_turn_count(self) -> None:
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity)

        state.record_iteration(tokens=42)

        assert state.turn_count == 1
        assert state.token_spend == 42

    def test_record_iteration_accumulates_across_calls(self) -> None:
        identity = SessionIdentity(value="x", method="user_field")
        state = SessionState(identity=identity)

        state.record_iteration(tokens=10)
        state.record_iteration(tokens=15)
        state.record_iteration(tokens=7)

        assert state.turn_count == 3
        assert state.token_spend == 32


class TestSessionRegistryLookup:
    """Registry returns the same state object for the same identity."""

    def test_get_or_create_returns_same_state_for_same_identity(self) -> None:
        registry = SessionRegistry()
        identity = SessionIdentity(value="x", method="user_field")

        first = registry.get_or_create_state(identity)
        second = registry.get_or_create_state(identity)

        assert first is second

    def test_get_or_create_returns_distinct_state_for_different_identity(
        self,
    ) -> None:
        registry = SessionRegistry()
        id_a = SessionIdentity(value="a", method="user_field")
        id_b = SessionIdentity(value="b", method="user_field")

        state_a = registry.get_or_create_state(id_a)
        state_b = registry.get_or_create_state(id_b)

        assert state_a is not state_b
        assert state_a.identity == id_a
        assert state_b.identity == id_b

    def test_caller_mutation_visible_through_subsequent_lookup(self) -> None:
        """The registry aliases its retained state with the returned reference.

        Lifecycle-composition check: a caller that mutates the returned state
        via `record_iteration` must see the mutation on the next lookup by
        the same identity, because Budget Controller and Autonomy Policy
        will both read session accounting through separate `get_or_create_state`
        calls and need a coherent view.
        """
        registry = SessionRegistry()
        identity = SessionIdentity(value="x", method="user_field")

        first = registry.get_or_create_state(identity)
        first.record_iteration(tokens=17)
        first.record_iteration(tokens=23)

        second = registry.get_or_create_state(identity)

        assert second.turn_count == 2
        assert second.token_spend == 40

    def test_missing_session_falls_back_to_cold_start_defaults(self) -> None:
        """Consumers that query state for an unresolved identity get
        cold-session defaults, not an exception.

        Per system-design §Integration Contracts (Autonomy/Calibration/Budget
        → Session Registry): 'Missing Session (identity unresolved) is treated
        as cold-session defaults for every consumer.'
        """
        registry = SessionRegistry()
        identity = SessionIdentity(value="unknown", method="user_field")

        state = registry.get_or_create_state(identity)

        assert state.turn_count == 0
        assert state.token_spend == 0


class TestSessionCloseCallbacks:
    """Cycle 6 WP-E (ADR-025): Session Registry fires close-callbacks.

    Per system-design.agents.md §Session Registry → Session Artifact
    Store: ``on_session_close(identity)`` triggers retention cleanup of
    ``session``-retention artifacts. The dependency direction is L3 →
    L1; Session Registry owns the lifecycle hook surface and consumers
    (Session Artifact Store; future cycles' lifecycle-coupled L1 / L2
    modules) register a callback to be fired at close.
    """

    def test_register_close_callback_then_close_fires_with_identity(self) -> None:
        registry = SessionRegistry()
        identity = SessionIdentity(value="sess-1", method="user_field")
        registry.get_or_create_state(identity)

        fired: list[str] = []
        registry.register_close_callback(lambda i: fired.append(i.value))
        registry.close_session(identity)

        assert fired == ["sess-1"]

    def test_multiple_callbacks_fire_in_registration_order(self) -> None:
        """All registered callbacks fire on close, in registration order —
        Session Artifact Store cleanup and future lifecycle hooks
        compose without ordering surprise."""
        registry = SessionRegistry()
        identity = SessionIdentity(value="sess-2", method="user_field")
        registry.get_or_create_state(identity)

        log: list[str] = []
        registry.register_close_callback(lambda i: log.append(f"first:{i.value}"))
        registry.register_close_callback(lambda i: log.append(f"second:{i.value}"))
        registry.close_session(identity)

        assert log == ["first:sess-2", "second:sess-2"]

    def test_close_session_removes_state(self) -> None:
        """After close, ``get_or_create_state`` on the same identity
        produces a fresh state — turn_count and token_spend reset.
        The contract is "close ends the session's lifetime"; subsequent
        requests under the same identity start a new accumulation."""
        registry = SessionRegistry()
        identity = SessionIdentity(value="sess-3", method="user_field")
        state = registry.get_or_create_state(identity)
        state.record_iteration(tokens=99)
        assert state.turn_count == 1

        registry.close_session(identity)

        # New state allocated; prior accounting cleared.
        fresh = registry.get_or_create_state(identity)
        assert fresh.turn_count == 0
        assert fresh.token_spend == 0

    def test_close_session_for_unregistered_identity_is_noop(self) -> None:
        """Closing an identity that was never resolved is a no-op —
        callbacks fire with the identity (the contract is "close means
        end-of-session lifecycle"; the registry does not gate on prior
        state existence). The call does not raise so the serve layer
        can always fire close at request lifecycle boundaries."""
        registry = SessionRegistry()
        identity = SessionIdentity(value="never-seen", method="user_field")

        fired: list[str] = []
        registry.register_close_callback(lambda i: fired.append(i.value))
        registry.close_session(identity)  # no raise

        # Callback still fires per the close-lifecycle contract — the
        # registry does not gate the hook on prior state existence.
        assert fired == ["never-seen"]

    def test_callback_exceptions_do_not_block_subsequent_callbacks(self) -> None:
        """A failing callback (e.g., Session Artifact Store filesystem
        error) does not prevent later callbacks from firing — the
        Registry's close contract is best-effort fan-out, mirroring the
        operator-resilience pattern used elsewhere (heartbeat scheduler
        cleanup, dispatch-log write)."""
        registry = SessionRegistry()
        identity = SessionIdentity(value="sess-4", method="user_field")
        registry.get_or_create_state(identity)

        fired: list[str] = []

        def boom(i: SessionIdentity) -> None:
            raise RuntimeError("filesystem unavailable")

        registry.register_close_callback(boom)
        registry.register_close_callback(lambda i: fired.append(i.value))
        registry.close_session(identity)

        # Despite the first callback raising, the second still fires.
        assert fired == ["sess-4"]
