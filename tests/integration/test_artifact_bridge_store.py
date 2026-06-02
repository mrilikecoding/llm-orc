"""Integration: Artifact Bridge ↔ Session Artifact Store (Cycle 7 loop-back).

The bridge unit tests already exercise a real ``SessionArtifactStore``;
this file is the explicit Step 5 integration gate for the
Artifact Bridge → Session Artifact Store boundary (system-design.agents.md
§Test Architecture: ``test_artifact_bridge_reads_deliverable_from_store``),
plus the **FC-49 fidelity-at-scale gate** the spike evidence could not
cover — Spike ρ used trivially-small content (``hello.py`` / ``calc.py``);
the large/complex case is BUILD-added (ARCHITECT advisory #4; scenarios.md
Cycle 7 loop-back Acceptance Criteria Table row 3).

Real types on both sides of the boundary: a deliverable written through
``write_deliverable`` round-trips back through ``read_deliverable`` into
the bridge's marshalled content, byte-identical. The retention-window
case is a lifecycle-sequence verification (write → dispose via
``cleanup_session`` → re-read the stale reference).

Per the build skill's Step 5 (replace ``MockX`` with the real ``X`` at one
boundary) and ADR-034 §Decision 3 (the artifact-bridge fidelity FC-49).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llm_orc.agentic.artifact_bridge import ArtifactBridge
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.session_artifact_store import (
    ArtifactNotFoundError,
    ArtifactReference,
    SessionArtifactStore,
)


def _substrate_envelope(ref: ArtifactReference, *, summary: str) -> DispatchEnvelope:
    """Build a substrate-routed envelope (``primary`` a summary, the typed
    reference serialized at ``artifacts[0]``) as Orchestrator Tool Dispatch
    composes it for an ``output_substrate: artifact`` ensemble."""
    return DispatchEnvelope(
        status="success",
        primary=summary,
        artifacts=[dataclasses.asdict(ref)],
    )


def test_artifact_bridge_reads_deliverable_from_store(tmp_path: Path) -> None:
    """A deliverable written to the store round-trips through the bridge
    into byte-identical marshalled content — NOT the envelope summary."""
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    bridge = ArtifactBridge(store)
    content = "# config\nname = 'agentic-serving'\nversion = '7'\n"
    ref = store.write_deliverable(
        session_id="2026-06-02T12:00:00Z-cc33",
        dispatch_id="dispatch-001",
        deliverable_name="config",
        content=content,
        content_type="text/x-python",
    )
    envelope = _substrate_envelope(ref, summary="config.py: 3 lines")

    assert bridge.marshal(envelope) == content


def test_artifact_bridge_marshals_large_complex_deliverable_with_fidelity(
    tmp_path: Path,
) -> None:
    """FC-49 fidelity at scale: a large deliverable (> 256 KB) carrying
    multi-byte UTF-8 round-trips byte-identical — the large/complex case
    the trivially-small spike content did not cover."""
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    bridge = ArtifactBridge(store)
    content = "".join(
        f"def transform_{i}(value: int) -> int:\n"
        f"    # rotate by {i} — café/ω/🜍 unicode fidelity check\n"
        f"    return (value << {i % 13}) ^ {i}\n\n"
        for i in range(6000)
    )
    assert len(content.encode("utf-8")) > 256 * 1024  # genuinely large on disk
    ref = store.write_deliverable(
        session_id="2026-06-02T12:00:00Z-dd44",
        dispatch_id="dispatch-002",
        deliverable_name="transforms",
        content=content,
        content_type="application/python",
    )
    envelope = _substrate_envelope(ref, summary="transforms.py: 6000 functions")

    marshalled = bridge.marshal(envelope)

    assert marshalled == content
    assert isinstance(marshalled, str)  # text deliverable marshals as str


def test_artifact_bridge_raises_after_retention_cleanup(tmp_path: Path) -> None:
    """Lifecycle-sequence verification: a session-retention deliverable is
    disposed by ``cleanup_session``; the bridge re-reading the now-stale
    reference raises ``ArtifactNotFoundError`` (the read must occur within
    the retention window — Test Architecture, Artifact Bridge → Session
    Artifact Store)."""
    store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    bridge = ArtifactBridge(store)
    ref = store.write_deliverable(
        session_id="sess-lifecycle",
        dispatch_id="dispatch-001",
        deliverable_name="ephemeral_doc",
        content="to be cleaned up at session close\n",
        content_type="text/markdown",
        retention="session",
    )
    envelope = _substrate_envelope(ref, summary="ephemeral_doc.md")

    # Within the retention window: resolves with fidelity.
    assert bridge.marshal(envelope) == "to be cleaned up at session close\n"

    # Session close disposes the session-retention artifact.
    store.cleanup_session("sess-lifecycle")
    with pytest.raises(ArtifactNotFoundError):
        bridge.marshal(envelope)
