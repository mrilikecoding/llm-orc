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
    """

    action_kind: str
    target_path: str
    result: str | None


class SessionActionRecord:
    """Per-process store of per-session action records."""

    def __init__(self) -> None:
        self._records: dict[str, list[ActionRecord]] = {}

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

    def records(self, session_id: str) -> tuple[ActionRecord, ...]:
        """The session's accumulated records, in emission order."""
        return tuple(self._records.get(session_id, ()))

    def cleanup_session(self, session_id: str) -> None:
        """Clear a closed session's records (session-close callback target)."""
        self._records.pop(session_id, None)
