"""Tests for the Session Action Record (Cycle 7 loop-back #5 WP-LB-K, ADR-037).

The Session Action Record is the framework-owned digest's home (L1) — the
evidence base the termination judgment reads. Records derive from the
framework's own emission records joined with the client's per-call tool
results, never reconstructed from client-serialized messages alone (FC-64;
Spike θ round 1 measured that reconstruction's failure). Scenarios from
``docs/agentic-serving/scenarios.md`` §"Session-Termination Mechanism —
Two-Call Trailing Composition (ADR-037)" — the digest-provenance group.
"""

from __future__ import annotations

from llm_orc.agentic.session_action_record import (
    ActionRecord,
    SessionActionRecord,
)


class TestSessionActionRecordAccumulates:
    """Scenario: the framework-owned digest derives from the framework's
    own records — record at emission, join the client result next request.
    """

    def test_recorded_action_is_retrievable_with_result_pending(self) -> None:
        store = SessionActionRecord()

        store.record_action(
            "session-1", action_kind="write", target_path="string_utils.py"
        )

        records = store.records("session-1")
        assert records == (
            ActionRecord(
                action_kind="write",
                target_path="string_utils.py",
                result=None,
            ),
        )

    def test_join_result_attaches_client_result_to_the_pending_record(self) -> None:
        store = SessionActionRecord()
        store.record_action(
            "session-1", action_kind="write", target_path="string_utils.py"
        )

        store.join_result("session-1", "Wrote file successfully")

        records = store.records("session-1")
        assert records == (
            ActionRecord(
                action_kind="write",
                target_path="string_utils.py",
                result="Wrote file successfully",
            ),
        )

    def test_join_result_targets_the_earliest_pending_record(self) -> None:
        """Single-step enforcement keeps at most one record pending, but the
        join must stay well-defined if that invariant ever loosens: results
        attach in emission order, never to an already-joined record.
        """
        store = SessionActionRecord()
        store.record_action(
            "session-1", action_kind="write", target_path="string_utils.py"
        )
        store.join_result("session-1", "ok-1")
        store.record_action(
            "session-1", action_kind="write", target_path="test_string_utils.py"
        )

        store.join_result("session-1", "ok-2")

        assert [record.result for record in store.records("session-1")] == [
            "ok-1",
            "ok-2",
        ]

    def test_join_result_without_pending_record_is_a_no_op(self) -> None:
        """A client tool result the framework never emitted an action for
        cannot occur on this surface (the driver is the only emitter); if
        one arrives anyway it must not fabricate a record (FC-64).
        """
        store = SessionActionRecord()

        store.join_result("session-1", "unsolicited result")

        assert store.records("session-1") == ()

    def test_sessions_are_isolated(self) -> None:
        store = SessionActionRecord()
        store.record_action("session-1", action_kind="write", target_path="a.py")
        store.record_action("session-2", action_kind="read", target_path="b.py")

        store.join_result("session-2", "contents of b")

        assert store.records("session-1") == (
            ActionRecord(action_kind="write", target_path="a.py", result=None),
        )
        assert store.records("session-2") == (
            ActionRecord(
                action_kind="read", target_path="b.py", result="contents of b"
            ),
        )


class TestSessionActionRecordLifecycle:
    """Lifecycle rides session scope (the Session Artifact Store retention
    pattern): close clears the session's records.
    """

    def test_cleanup_session_clears_records(self) -> None:
        store = SessionActionRecord()
        store.record_action("session-1", action_kind="write", target_path="a.py")

        store.cleanup_session("session-1")

        assert store.records("session-1") == ()

    def test_cleanup_of_unknown_session_is_not_an_error(self) -> None:
        store = SessionActionRecord()

        store.cleanup_session("never-seen")

        assert store.records("never-seen") == ()
