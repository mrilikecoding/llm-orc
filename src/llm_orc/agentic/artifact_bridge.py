"""Artifact Bridge — marshals a dispatch deliverable into tool-call content.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Artifact
Bridge (L2, *new in Cycle 7 loop-back per ADR-034 §Decision 3* — the
F-ρ.1 marshalling step). Owns the read-and-marshal step that moves a
capability ensemble's deliverable into a client tool-call ``content``
argument:

* **Substrate-routed** deliverable (``output_substrate: artifact`` per
  ADR-025 — ``envelope.artifacts`` carries an ``ArtifactReference`` and
  ``envelope.primary`` is a summary): read the full content from the
  Session Artifact Store via ``read_deliverable`` and place it in the
  tool-call ``content`` argument.
* **Inline-response** deliverable (``output_substrate: inline`` —
  ``envelope.artifacts`` absent): read ``envelope.primary`` directly; the
  store read is a no-op.

The bridge is deterministic framework code, not an LLM generation — the
bytes-to-tool-call step does not reintroduce the orchestrator-LLM
confabulation failure mode (PLAY note 22). The fidelity contract: the
marshalled content equals the stored deliverable, not a summary or
paraphrase (FC-49; AS-7 result-summarization is upstream and does not
apply to the bridge).

Consumer: the Client-Tool-Action Terminal (WP-LB-C) calls the bridge to
resolve the turn's deliverable content before emitting a ``write`` tool
call. WP-LB-D builds the bridge testable in isolation against a fixture
artifact (the terminal does not exist yet).
"""

from __future__ import annotations

from typing import Any

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.session_artifact_store import (
    ArtifactReference,
    SessionArtifactStore,
)

__all__ = ["ArtifactBridge"]


class ArtifactBridge:
    """Marshals a dispatch deliverable into client tool-call content.

    Constructed with the Session Artifact Store the substrate-routed
    deliverables were written to; the store's ``read_deliverable``
    accessor (the read-side API added for WP-LB-D) is the only
    dependency.
    """

    def __init__(self, store: SessionArtifactStore) -> None:
        self._store = store

    def marshal(self, envelope: DispatchEnvelope) -> str | bytes:
        """Return the deliverable content for a client tool-call argument.

        Reads from the Session Artifact Store for substrate-routed
        deliverables (``envelope.artifacts`` populated) and from
        ``envelope.primary`` for inline ones (the artifact-vs-inline
        branch the bridge owns). Raises
        :class:`~llm_orc.agentic.session_artifact_store.ArtifactNotFoundError`
        (from ``read_deliverable``) when a substrate reference does not
        resolve — the Terminal degrades that to a dispatch-failure
        completion (system-design.agents.md §Client-Tool-Action Terminal
        error handling).
        """
        reference = _artifact_reference(envelope)
        if reference is None:
            return envelope.primary
        return self._store.read_deliverable(reference)


def _artifact_reference(envelope: DispatchEnvelope) -> ArtifactReference | None:
    """Reconstruct the typed reference from ``envelope.artifacts[0]``.

    The envelope carries artifacts in serialized (dict) form so the
    chat-completion response shape stays ``dataclasses.asdict``-clean
    (``dispatch_envelope.py``). ``None``/empty ``artifacts`` means an
    inline deliverable (read ``envelope.primary`` directly).
    """
    artifacts = envelope.artifacts
    if not artifacts:
        return None
    return _reference_from_dict(artifacts[0])


def _reference_from_dict(data: dict[str, Any]) -> ArtifactReference:
    """Build an :class:`ArtifactReference` from its serialized envelope dict.

    Reads the five reference fields explicitly (rather than ``**data``) so
    a future additive envelope field cannot break construction.
    """
    return ArtifactReference(
        path=data["path"],
        content_type=data["content_type"],
        size_bytes=data["size_bytes"],
        summary=data["summary"],
        retention=data["retention"],
    )
