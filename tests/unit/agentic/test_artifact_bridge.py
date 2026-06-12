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

import pytest

from llm_orc.agentic.artifact_bridge import (
    ArtifactBridge,
    FormRefusedError,
    parse_check_form_gate,
)
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

    def test_refusing_gate_raises_through_the_seam(self, tmp_path: Path) -> None:
        """The refusal channel: a gate refuses by raising FormRefusedError.

        The detect-and-refuse escalation (ADR-035 §4) installs as a gate
        that raises; the channel pre-exists so installation touches only
        the seam (FC-57's zero-Terminal-edits criterion).
        """

        def refusing_gate(
            content: str | bytes, tool: str | None, path: str | None
        ) -> str | bytes:
            raise FormRefusedError(f"clearly non-bare deliverable for {tool!r}")

        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store, form_gate=refusing_gate)
        envelope = DispatchEnvelope(status="success", primary="```python\n...\n```")

        with pytest.raises(FormRefusedError, match="write"):
            bridge.marshal(envelope, destination_tool="write")


class TestParseCheckFormGate:
    """ADR-041 §Decision 1 — the deterministic destination-validity gate.

    The committed detect-and-refuse gate (promoted from ADR-035 §4's
    reserved escalation by Spike π): the marshalled content must parse as
    what its destination *path* claims. A ``.py`` must ``ast.parse``; a
    ``.json`` must ``json.loads``; other destinations (prose/``.md``) pass
    un-inspected — the parse/validity determinism boundary (ADR-041
    §Decision 6). The gate inspects bytes deterministically and never
    extracts or normalizes (Spike χ F-χ.1 / corpus Fork-1): it only
    recognizes a structurally-wrong deliverable and refuses it. Installed
    at the production construction site via the ``form_gate=`` seam (FC-57,
    zero-Terminal-edits).
    """

    def test_invalid_python_refuses(self, tmp_path: Path) -> None:
        """The σ form-bleed seam: a ``.py`` with trailing prose does not
        parse, so the gate refuses (ADR-041; Spike π corpus C1)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store, form_gate=parse_check_form_gate)
        bled = "def f():\n    return 1\nThis function returns one.\n"
        envelope = DispatchEnvelope(status="success", primary=bled)

        with pytest.raises(FormRefusedError, match="cli.py"):
            bridge.marshal(
                envelope, destination_tool="write", destination_path="cli.py"
            )

    def test_wrong_language_refuses(self, tmp_path: Path) -> None:
        """The η intent-divergence seam: JavaScript in a ``.py`` slot does
        not parse as Python, so the gate refuses (ADR-041; Spike π corpus C4)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store, form_gate=parse_check_form_gate)
        js = "const add = (a, b) => a + b;\n"
        envelope = DispatchEnvelope(status="success", primary=js)

        with pytest.raises(FormRefusedError, match="util.py"):
            bridge.marshal(
                envelope, destination_tool="write", destination_path="util.py"
            )

    def test_invalid_json_refuses(self, tmp_path: Path) -> None:
        """A ``.json`` destination must ``json.loads`` (ADR-041 §Decision 1)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store, form_gate=parse_check_form_gate)
        envelope = DispatchEnvelope(status="success", primary="{not valid json,}")

        with pytest.raises(FormRefusedError, match="config.json"):
            bridge.marshal(
                envelope, destination_tool="write", destination_path="config.json"
            )

    def test_valid_python_passes_through(self, tmp_path: Path) -> None:
        """Valid Python marshals unchanged — the gate is a no-op on good
        content (fidelity preserved, FC-49)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store, form_gate=parse_check_form_gate)
        content = "def add(a: int, b: int) -> int:\n    return a + b\n"
        envelope = DispatchEnvelope(status="success", primary=content)

        marshalled = bridge.marshal(
            envelope, destination_tool="write", destination_path="adder.py"
        )

        assert marshalled == content

    def test_prose_destination_passes_uninspected(self, tmp_path: Path) -> None:
        """A ``.md`` destination is not structurally checkable, so it passes
        un-inspected — the principled determinism edge (ADR-041 §Decision 6)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bridge = ArtifactBridge(store, form_gate=parse_check_form_gate)
        prose = "# Title\n\nThis is documentation, not code.\n"
        envelope = DispatchEnvelope(status="success", primary=prose)

        marshalled = bridge.marshal(
            envelope, destination_tool="write", destination_path="README.md"
        )

        assert marshalled == prose
