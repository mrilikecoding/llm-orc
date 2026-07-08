"""Integration: Session Registry close fires Session Artifact Store cleanup.

Per ``docs/agentic-serving/system-design.agents.md`` §Session Registry
→ Session Artifact Store integration contract (Cycle 6 WP-E, ADR-025):
the L3 → L1 close hook drives ``cleanup_session(session_id)`` on the
artifact store. The Registry owns the lifecycle hook; the Store owns
the retention cleanup; the integration is the registration handshake
the serve layer performs at session-start.

The build skill's Step 5 framing applied early: the contract is
verified against the real artifact store with real filesystem
operations (no stubs), the real Registry firing close, and the real
ArtifactReference shape consumers find at envelope.artifacts[0].
"""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.session.artifact_store import SessionArtifactStore
from llm_orc.core.session.registry import SessionIdentity, SessionRegistry


def test_session_close_drives_artifact_store_cleanup(tmp_path: Path) -> None:
    """Registering ``SessionArtifactStore.cleanup_session`` as the
    Registry's close callback wires the L3 → L1 lifecycle edge. After
    ``close_session`` fires, ``retention: session`` artifacts under the
    closed session's directory are gone; durable artifacts persist.
    """
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    registry = SessionRegistry()

    # The serve-layer integration: register cleanup as a close callback.
    # The lambda adapts the SessionIdentity → session_id (str) shape.
    registry.register_close_callback(
        lambda identity: store.cleanup_session(identity.value)
    )

    identity = SessionIdentity(value="sess-int-1", method="user_field")
    registry.get_or_create_state(identity)

    # Write two artifacts under the session: one session, one durable.
    store.write_deliverable(
        session_id=identity.value,
        dispatch_id="dispatch-001",
        deliverable_name="ephemeral_x",
        content="session-scoped output",
        content_type="text/plain",
        retention="session",
    )
    durable_ref = store.write_deliverable(
        session_id=identity.value,
        dispatch_id="dispatch-002",
        deliverable_name="durable_y",
        content="durable output",
        content_type="text/plain",
        retention="durable",
    )

    # Close the session via the Registry — the registered cleanup
    # callback fires; the artifact store removes retention=session
    # artifacts under sess-int-1/.
    registry.close_session(identity)

    # Session-retention dispatch dir gone.
    assert not (tmp_path / "sess-int-1" / "dispatch-001").exists()
    # Durable artifact survives.
    durable_disk_path = tmp_path / Path(durable_ref.path).relative_to(
        "agentic-sessions"
    )
    assert durable_disk_path.exists()
    assert durable_disk_path.read_text() == "durable output"


def test_close_session_with_no_artifacts_does_not_raise(tmp_path: Path) -> None:
    """A session that never wrote any artifact closes cleanly — the
    cleanup callback fires, the artifact store's no-such-dir branch
    runs the no-op, and the Registry's close lifecycle completes
    without raising. The serve layer can fire close at request
    lifecycle boundaries regardless of whether the session produced
    substrate-routed deliverables."""
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    registry = SessionRegistry()
    registry.register_close_callback(lambda i: store.cleanup_session(i.value))

    identity = SessionIdentity(value="sess-int-empty", method="user_field")
    registry.get_or_create_state(identity)

    registry.close_session(identity)
    assert not (tmp_path / "sess-int-empty").exists()


def test_close_session_state_clear_composes_with_artifact_cleanup(
    tmp_path: Path,
) -> None:
    """After close, the Registry has reset the per-session state AND the
    artifact store has cleaned up. Both invariants hold post-close —
    composing the L3 lifecycle (state clear) with the L1 lifecycle
    (artifact cleanup) under one close call is the contract the serve
    layer relies on."""
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    registry = SessionRegistry()
    registry.register_close_callback(lambda i: store.cleanup_session(i.value))

    identity = SessionIdentity(value="sess-int-2", method="user_field")
    state = registry.get_or_create_state(identity)
    state.record_iteration(tokens=50)
    store.write_deliverable(
        session_id=identity.value,
        dispatch_id="dispatch-001",
        deliverable_name="x",
        content="session-scoped",
        content_type="text/plain",
        retention="session",
    )

    registry.close_session(identity)

    # L3 — fresh state on subsequent get_or_create_state.
    fresh = registry.get_or_create_state(identity)
    assert fresh.turn_count == 0
    # L1 — session dir gone (only session-retention content).
    assert not (tmp_path / "sess-int-2").exists()
