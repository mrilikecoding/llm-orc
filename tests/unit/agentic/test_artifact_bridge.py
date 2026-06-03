"""Artifact Bridge unit tests (Cycle 7 loop-back, ADR-034 Decision 3 / WP-LB-D).

The bridge owns the artifact-vs-inline branch (system-design.agents.md
§Module: Artifact Bridge): substrate-routed deliverables
(``output_substrate: artifact`` — ``envelope.artifacts`` populated,
``primary`` a summary) are read from the Session Artifact Store with
fidelity; inline deliverables (``envelope.artifacts`` absent) are read
from ``primary`` directly and the store read is a no-op.

A real :class:`SessionArtifactStore` (filesystem-backed, ``tmp_path``) is
used rather than a stub — the read path is the behavior under test, and
the store is cheap to exercise for real.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_orc.agentic.artifact_bridge import ArtifactBridge
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.session_artifact_store import SessionArtifactStore


class TestArtifactBridgeMarshal:
    """``ArtifactBridge.marshal`` resolves a dispatch deliverable into the
    content for a client tool-call ``content`` argument."""

    def test_marshal_returns_inline_primary_when_no_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Inline-substrate deliverable skips the bridge step — the
        marshalled content equals the inline ``primary`` (scenarios.md
        §ADR-034 "Inline-substrate deliverable skips the bridge step")."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store)
        envelope = DispatchEnvelope(
            status="success",
            primary="print('hello from an inline ensemble')\n",
        )
        assert bridge.marshal(envelope) == "print('hello from an inline ensemble')\n"

    def test_marshal_reads_substrate_routed_deliverable_with_fidelity(
        self, tmp_path: Path
    ) -> None:
        """Substrate-routed deliverable: the bridge reads the full content
        from the store, NOT ``primary``'s summary (scenarios.md §ADR-034
        "Artifact-bridge reads the substrate-routed deliverable and
        marshals it into tool-call content"; the artifact-bridge fidelity
        FC-49)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        content = (
            "class Calculator:\n"
            "    def add(self, a: int, b: int) -> int:\n"
            "        return a + b\n"
        )
        ref = store.write_deliverable(
            session_id="2026-06-02T11:00:00Z-bb22",
            dispatch_id="dispatch-007",
            deliverable_name="calculator",
            content=content,
            content_type="application/python",
        )
        envelope = DispatchEnvelope(
            status="success",
            primary="calculator.py: 3 lines — a Calculator class",  # summary
            artifacts=[dataclasses.asdict(ref)],
        )
        bridge = ArtifactBridge(store)

        marshalled = bridge.marshal(envelope)

        assert marshalled == content
        assert marshalled != envelope.primary  # fidelity: the content, not the summary


class TestFormGateSeam:
    """FC-57 — the named FormGate seam on the marshal surface (ADR-035 §4).

    ``marshal`` receives the turn's destination tool and evaluates the
    content at the FormGate before returning it. The default gate is
    pass-through (identity): the form contract's primary mechanism is
    the boundary directive (FC-53/54), not bridge-side shaping. The
    detect-and-refuse escalation installs at this seam on PLAY evidence
    with zero Terminal edits — the seam existing is the fitness
    property; form-gating composes with, never weakens, the fidelity
    contract (no paraphrase, no summary, no multi-fence extraction).
    """

    def test_form_gate_seam_default_passthrough(self, tmp_path: Path) -> None:
        """The default gate is identity — content is marshalled unchanged,
        destination tool in hand (scenarios.md §ADR-035 defense-in-depth:
        the seam exists; the contract lives upstream)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store)
        content = "def f():\n    return 42\n"
        envelope = DispatchEnvelope(status="success", primary=content)

        assert bridge.marshal(envelope, destination_tool="write") == content

    def test_destination_tool_defaults_for_legacy_callers(self, tmp_path: Path) -> None:
        """Callers that predate the seam still marshal (no signature break)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store)
        envelope = DispatchEnvelope(status="success", primary="x = 1\n")

        assert bridge.marshal(envelope) == "x = 1\n"
