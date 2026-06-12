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

import ast
import json
import os
from collections.abc import Callable
from typing import Any

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.session_artifact_store import (
    ArtifactReference,
    SessionArtifactStore,
)

__all__ = ["ArtifactBridge", "FormGate", "FormRefusedError"]


class FormRefusedError(Exception):
    """A FormGate refused to emit a clearly-wrong deliverable (ADR-035 §4).

    The refusal channel for the detect-and-refuse escalation: no shipped
    gate raises it (the default gate is pass-through), but the channel
    pre-exists so installing the escalated gate touches only the seam —
    the Terminal already degrades this to a dispatch-failure completion
    (FC-57's zero-Terminal-edits criterion).
    """


FormGate = Callable[["str | bytes", "str | None", "str | None"], "str | bytes"]
"""The FormGate seam's contract (ADR-035 §Decision 4, FC-57; ADR-041 §Decision 2).

Evaluates marshalled content against the destination before emission, given
both the destination tool (``write``/``edit``/``bash``) and the full
destination path (``src/foo/bar.py``). A gate may *refuse* (raise) or apply
the defined conservative normalization; it never paraphrases, summarizes, or
attempts multi-fence extraction (Spike χ F-χ.1 rejects that path as fragile)
— form-gating composes with, and does not weaken, the fidelity contract
(FC-49). The destination path is the full path, not a pre-extracted
extension, so the same seam serves any future extension-keyed check
(ADR-041 §Decision 2).
"""


def _passthrough_form_gate(
    content: str | bytes,
    destination_tool: str | None,
    destination_path: str | None,
) -> str | bytes:
    """The default FormGate — identity.

    The form contract's primary mechanism is the boundary directive
    (FC-53/54), not bridge-side shaping; under the directive the spike
    probes produced zero stray fences (χ-P3/P4/P5), so no normalization
    ships by default (LB-5 resolved pass-through — defensive dead code
    is debt). The detect-and-refuse escalation installs at this seam on
    PLAY evidence with zero Terminal edits.
    """
    return content


# --- SPIKE π (Cycle 7 loop-back #8) — env-gated; REVERT at spike close -------
# Candidate detect-and-refuse FormGate (ADR-035 §4 escalation): validate the
# deliverable parses as what its destination path claims. Default-off; active
# only under LLMORC_SPIKE_PI_GATE=parse. Threaded `destination_path` is the
# seam extension the parse-check needs (the BUILD seed if the spike grounds it).
def _spike_pi_form_check(content: str | bytes, destination_path: str | None) -> None:
    if os.environ.get("LLMORC_SPIKE_PI_GATE") != "parse":
        return
    if not isinstance(content, str) or not destination_path:
        return
    ext = os.path.splitext(destination_path)[1].lower()
    if ext == ".py":
        try:
            ast.parse(content)
        except SyntaxError as exc:
            raise FormRefusedError(
                f"{destination_path}: not valid Python ({exc.msg})"
            ) from exc
    elif ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise FormRefusedError(
                f"{destination_path}: not valid JSON ({exc})"
            ) from exc


# --- end SPIKE π -------------------------------------------------------------


class ArtifactBridge:
    """Marshals a dispatch deliverable into client tool-call content.

    Constructed with the Session Artifact Store the substrate-routed
    deliverables were written to; the store's ``read_deliverable``
    accessor (the read-side API added for WP-LB-D) is the only
    dependency. An optional ``form_gate`` installs at the FormGate seam
    (default pass-through).
    """

    def __init__(
        self,
        store: SessionArtifactStore,
        *,
        form_gate: FormGate | None = None,
    ) -> None:
        self._store = store
        self._form_gate: FormGate = form_gate or _passthrough_form_gate

    def marshal(
        self,
        envelope: DispatchEnvelope,
        *,
        destination_tool: str | None = None,
        destination_path: str | None = None,
    ) -> str | bytes:
        """Return the deliverable content for a client tool-call argument.

        Reads from the Session Artifact Store for substrate-routed
        deliverables (``envelope.artifacts`` populated) and from
        ``envelope.primary`` for inline ones (the artifact-vs-inline
        branch the bridge owns), then evaluates the content at the
        FormGate for the ``destination_tool`` (FC-57; default
        pass-through). Raises
        :class:`~llm_orc.agentic.session_artifact_store.ArtifactNotFoundError`
        (from ``read_deliverable``) when a substrate reference does not
        resolve — the Terminal degrades that to a dispatch-failure
        completion (system-design.agents.md §Client-Tool-Action Terminal
        error handling).
        """
        reference = _artifact_reference(envelope)
        if reference is None:
            content: str | bytes = envelope.primary
        else:
            content = self._store.read_deliverable(reference)
        _spike_pi_form_check(content, destination_path)  # SPIKE π — revert at close
        return self._form_gate(content, destination_tool, destination_path)


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
