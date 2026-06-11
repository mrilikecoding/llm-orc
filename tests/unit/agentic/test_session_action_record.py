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


class TestSessionActionRecordContent:
    """ADR-039 (loop-back #7, V-03/V-04): the record carries the resolved
    deliverable content forward, the extensible meta-record seam's second
    increment. Content is captured at the Terminal (where the Artifact Bridge
    resolves the deliverable), so the Loop Driver builds the content anchor
    from records it already holds — no Session Artifact Store edge. Scenarios
    from ``docs/agentic-serving/scenarios.md`` §"Content Anchor (ADR-039,
    Finding H)".
    """

    def test_record_content_attaches_to_the_latest_record(self) -> None:
        store = SessionActionRecord()
        store.record_action(
            "session-1", action_kind="write", target_path="converters.py"
        )

        store.record_content("session-1", "def c_to_f(c: float) -> float: ...")

        records = store.records("session-1")
        assert records[-1].content == "def c_to_f(c: float) -> float: ..."

    def test_record_content_without_a_record_is_a_no_op(self) -> None:
        """Content only ever follows an emitted action; an orphan capture with
        no prior record must not fabricate one (the FC-64 no-fabrication rule).
        """
        store = SessionActionRecord()

        store.record_content("session-1", "orphan content")

        assert store.records("session-1") == ()

    def test_content_defaults_to_none_until_captured(self) -> None:
        """A recorded action carries no content until the Terminal resolves and
        captures the deliverable; carries and failed dispatches stay ``None``.
        """
        store = SessionActionRecord()
        store.record_action("session-1", action_kind="read", target_path="a.py")

        assert store.records("session-1")[-1].content is None


class TestSessionActionRecordRequested:
    """J-3 persist-once (Spike σ): the requested-deliverable set is captured
    once per session (turn 1 carries the guaranteed-full task) and held stable,
    so a later turn whose task text the client has compacted cannot clear or
    overwrite it. Scenarios from ``docs/agentic-serving/scenarios.md``
    §"Deterministic completeness (Finding I / Spike σ)".
    """

    def test_requested_defaults_to_empty(self) -> None:
        store = SessionActionRecord()

        assert store.requested("session-1") == frozenset()

    def test_set_requested_persists_and_is_retrievable(self) -> None:
        store = SessionActionRecord()

        store.set_requested_if_absent("session-1", frozenset({"a.py", "b.py"}))

        assert store.requested("session-1") == frozenset({"a.py", "b.py"})

    def test_empty_set_is_a_no_op(self) -> None:
        """A turn that names no files must not persist an empty set — that would
        pin the session to the no-files judge fallback for the rest of its life.
        """
        store = SessionActionRecord()

        store.set_requested_if_absent("session-1", frozenset())

        assert store.requested("session-1") == frozenset()

    def test_first_non_empty_set_wins(self) -> None:
        """A later (compacted) turn re-deriving a partial or empty set cannot
        overwrite the set captured from the first turn that named files.
        """
        store = SessionActionRecord()
        store.set_requested_if_absent("session-1", frozenset({"a.py", "b.py"}))

        store.set_requested_if_absent("session-1", frozenset({"c.py"}))

        assert store.requested("session-1") == frozenset({"a.py", "b.py"})

    def test_requested_sets_are_session_isolated(self) -> None:
        store = SessionActionRecord()
        store.set_requested_if_absent("session-1", frozenset({"a.py"}))
        store.set_requested_if_absent("session-2", frozenset({"b.py"}))

        assert store.requested("session-1") == frozenset({"a.py"})
        assert store.requested("session-2") == frozenset({"b.py"})


class TestSessionActionRecordLifecycle:
    """Lifecycle rides session scope (the Session Artifact Store retention
    pattern): close clears the session's records.
    """

    def test_cleanup_session_clears_records(self) -> None:
        store = SessionActionRecord()
        store.record_action("session-1", action_kind="write", target_path="a.py")

        store.cleanup_session("session-1")

        assert store.records("session-1") == ()

    def test_cleanup_session_clears_the_requested_set(self) -> None:
        store = SessionActionRecord()
        store.set_requested_if_absent("session-1", frozenset({"a.py"}))

        store.cleanup_session("session-1")

        assert store.requested("session-1") == frozenset()

    def test_cleanup_of_unknown_session_is_not_an_error(self) -> None:
        store = SessionActionRecord()

        store.cleanup_session("never-seen")

        assert store.records("never-seen") == ()
