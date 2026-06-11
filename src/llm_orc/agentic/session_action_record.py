"""Session Action Record (Cycle 7 loop-back #5 WP-LB-K, ADR-037) — L1.

The framework-owned digest's home: per-session action records (action kind,
target file path, client result) accumulated from the framework's own
emissions and joined with the client's per-call tool results — never
reconstructed from client-serialized messages alone (FC-64; Spike θ round 1
measured that reconstruction's failure: the client serialization drops what
was written).

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Session Action
Record. The record schema is the **extensible meta-record seam**: the
write-log fields here (action kind, path, result) are the first increment,
not the final form — the false-stop share (FC-67) is the extend-on-evidence
enrichment trigger. A leaf store importing nothing from the agentic package,
keeping the Loop Driver → Session Action Record edge cycle-free.

Lifecycle rides session scope (the Session Artifact Store retention
pattern): the serving layer registers :meth:`cleanup_session` as a
session-close callback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "ActionRecord",
    "SessionActionRecord",
]


@dataclass(frozen=True)
class ActionRecord:
    """One framework-emitted client-tool action and its observed result.

    ``result`` is ``None`` between the framework's emission (the driver
    decided the action this turn) and the client's per-call ``role: tool``
    result arriving on the next request — the join that makes the record
    judgment-grade evidence (FC-64).

    ``content`` is the Terminal-resolved deliverable content (ADR-039, the
    content anchor's source). ``None`` until the Terminal captures it (a carry
    or a failed dispatch never resolves content), it is the extensible
    meta-record seam's second increment — the produced file's bytes the Loop
    Driver anchors a dependent callee against, sourced from the record it
    already holds rather than re-read from the artifact store.
    """

    action_kind: str
    target_path: str
    result: str | None
    content: str | None = None


class SessionActionRecord:
    """Per-process store of per-session action records."""

    def __init__(self) -> None:
        self._records: dict[str, list[ActionRecord]] = {}
        self._requested: dict[str, frozenset[str]] = {}

    def record_action(
        self, session_id: str, *, action_kind: str, target_path: str
    ) -> None:
        """Record a framework-emitted action at the driver's decision time."""
        self._records.setdefault(session_id, []).append(
            ActionRecord(action_kind=action_kind, target_path=target_path, result=None)
        )

    def join_result(self, session_id: str, tool_result: str) -> None:
        """Join the client's per-call tool result to the pending record.

        Results attach in emission order to the earliest record still
        awaiting one (single-step enforcement keeps at most one pending in
        practice). A result with no pending record is dropped — the store
        never fabricates a record from client-supplied content (FC-64).
        """
        session_records = self._records.get(session_id, [])
        for index, record in enumerate(session_records):
            if record.result is None:
                session_records[index] = replace(record, result=tool_result)
                return

    def record_content(self, session_id: str, content: str) -> None:
        """Capture the Terminal-resolved deliverable content on the latest record.

        The content anchor's source (ADR-039): the Terminal calls this after
        the Artifact Bridge resolves a delegated write's deliverable, so the
        produced file's bytes ride forward in the record the Loop Driver
        already holds. Single-step enforcement keeps one record per turn, so
        the latest record is the write just emitted. With no record the capture
        is dropped — content never fabricates a record (FC-64).
        """
        session_records = self._records.get(session_id, [])
        if session_records:
            session_records[-1] = replace(session_records[-1], content=content)

    def records(self, session_id: str) -> tuple[ActionRecord, ...]:
        """The session's accumulated records, in emission order."""
        return tuple(self._records.get(session_id, ()))

    def set_requested_if_absent(
        self, session_id: str, requested: frozenset[str]
    ) -> None:
        """Persist the requested-deliverable set once per session (J-3, Spike σ).

        The first non-empty set wins; empty sets and repeat calls are no-ops.
        The Loop Driver captures the set from turn 1's task (where the client is
        guaranteed to send the full ask), so a later turn whose task text the
        client has compacted cannot clear or overwrite it — the completeness
        gate reads a stable set rather than re-deriving it from a possibly
        truncated conversation each turn. An empty set is never persisted: that
        would pin the session to the no-files judge fallback for the rest of its
        life.
        """
        if requested and session_id not in self._requested:
            self._requested[session_id] = requested

    def requested(self, session_id: str) -> frozenset[str]:
        """The session's persisted requested-deliverable set (empty if unset).

        Empty means no turn has named files yet, which routes completeness to
        the ADR-037 stochastic-judge fallback (the general-task path).
        """
        return self._requested.get(session_id, frozenset())

    def cleanup_session(self, session_id: str) -> None:
        """Clear a closed session's records (session-close callback target)."""
        self._records.pop(session_id, None)
        self._requested.pop(session_id, None)
