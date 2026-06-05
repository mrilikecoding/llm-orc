"""Integration: Session Registry close fires Session Action Record cleanup.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Session
Action Record (Cycle 7 loop-back #5, ADR-037): lifecycle rides session
scope — the Session Artifact Store retention pattern. The Registry owns
the lifecycle hook; the record store owns the cleanup; the integration
is the registration handshake the serve layer performs.

Real Registry firing close against the real record store (no stubs).
"""

from __future__ import annotations

from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_registry import SessionIdentity, SessionRegistry


def test_session_close_drives_action_record_cleanup() -> None:
    store = SessionActionRecord()
    registry = SessionRegistry()

    # The serve-layer integration: register cleanup as a close callback.
    registry.register_close_callback(
        lambda identity: store.cleanup_session(identity.value)
    )

    identity = SessionIdentity(value="sess-act-1", method="user_field")
    registry.get_or_create_state(identity)
    store.record_action(identity.value, action_kind="write", target_path="a.py")
    store.join_result(identity.value, "Wrote file successfully")

    registry.close_session(identity)

    assert store.records(identity.value) == ()
