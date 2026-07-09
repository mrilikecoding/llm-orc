"""Typed ``DispatchEnvelope`` shared type (Cycle 6 WP-D, ADR-024).

Per ``docs/serving.md`` §Shared type:
``DispatchEnvelope`` — *new in Cycle 6 per ADR-024*. Not a module — a
shared type with the same cross-cutting status as
``LlmOrcStructuralError`` (Cycle 4 shared base class). Defined alongside
the agentic-layer modules so the closed-five-tool dispatch surface and
the Orchestrator-Context Event Sink can both consume the typed shape
without a circular dependency.

Architectural drivers:

* ADR-024 — the typed envelope contract; the field set, the
  ``diagnostics``-not-``metadata`` naming, the advisory-schema posture.
* Spike α (research log ``essays/research-logs/cycle-6-spike-alpha-envelope-survey.md``)
  — candidate B (additive typed fields) selected as the lowest-disruption
  typed-field path; ``structured`` and ``errors[]`` surfaced as
  load-bearing distinct from ``primary``.
* Spike β (research log
  ``essays/research-logs/cycle-6-spike-beta-composition-predictability.md``)
  — the schema validation is advisory at dispatch time because
  output-spec drift's actual mechanism is the orchestrator's
  ``input.data`` override, not synthesizer non-compliance.
* Cycle 6 DECIDE snapshot Finding 2 (advisory carry-forward) —
  ``diagnostics.dispatch_id`` is one of three surfaces sharing the
  single-source-of-truth ``dispatch_id`` value (event stream,
  envelope diagnostics, ADR-025 artifact path's ``<dispatch_id>``
  segment).

Producer site: Orchestrator Tool Dispatch's ``invoke_ensemble`` returns
an envelope on every successful dispatch (attached to the existing
``ToolCallSuccess`` wrapper via the ``envelope`` field — same additive
pattern as WP-C's ``dispatch_id``). The closed-five-tool framework's
uniform ``ToolCallResult`` return type is preserved; the envelope is the
typed structural contract on the ``invoke_ensemble`` slot.

Consumer sites: Orchestrator Runtime (envelope flows through to the
ReAct loop's tool-result observation), Orchestrator-Context Event Sink
(reads ``diagnostics.dispatch_id`` to compose the structured observation
at turn boundaries — same correlation identifier as WP-C events), and
the serving-layer chat-completion response. ``dataclasses.asdict``
serializes cleanly to JSON for the chat-completion response shape.

WP-D scope: the typed envelope is the **return shape** of
``invoke_ensemble`` for the success path; the ``output_schema:``
per-ensemble declaration populates ``structured`` advisorily when the
synthesizer's response parses as JSON. WP-E extends to substrate-routed
dispatches where ``primary`` becomes a summary line and ``artifacts[0]``
carries the typed ``ArtifactReference``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "DispatchEnvelope",
    "EnvelopeStatus",
]


EnvelopeStatus = Literal["success", "error", "timeout", "partial"]
"""Dispatch outcome category per ADR-024 §Decision.

``success`` — ensemble completed and produced a valid deliverable.
``error`` — dispatch failed (typed error in ``errors[0]``).
``timeout`` — dispatch exceeded ``timeout_seconds``.
``partial`` — some stages completed, others errored (per-stage errors
in ``errors[]``).
"""


@dataclass(frozen=True)
class DispatchEnvelope:
    """Response shape for ``invoke_ensemble`` dispatches.

    Per ADR-024's per-capability dispatch contract; specifies the shape
    of the capability ensemble's output the orchestrator returns to the
    skill framework as the chat-completion response. Frozen so consumers
    can rely on the envelope's identity across the ReAct loop's
    observation surface.

    Field-name ``diagnostics`` (not ``metadata``) aligns with the Cycle
    6 MODEL vocabulary entry on Common I/O envelope. The
    ``execution.json`` artifact retains ``metadata`` at the artifact
    layer (the rename is at the envelope layer only — Cycle 7+
    artifact-shape ADR territory).

    ``output_schema:`` is advisory at dispatch time per spike β's
    finding that output-spec drift's mechanism is the orchestrator's
    ``input.data`` override rather than synthesizer non-compliance —
    enforcement at the synthesizer would catch the wrong thing.
    """

    status: EnvelopeStatus
    """Dispatch outcome category. See :data:`EnvelopeStatus`."""

    primary: str
    """The canonical deliverable, human-readable.

    For inline-response ensembles (WP-D default for every dispatch
    until WP-E ships substrate-routing), this carries the summarized
    deliverable content directly. For substrate-routed ensembles per
    ADR-025 (WP-E), this becomes a summary line referencing
    ``artifacts[0]``. Always a string; consumers do not parse this
    field structurally — they parse :attr:`structured` or read
    :attr:`artifacts`.
    """

    structured: dict[str, Any] | None = None
    """Optional typed payload.

    Present when the dispatched ensemble's YAML declares
    ``output_schema:`` AND the synthesizer's response parses as JSON
    matching the schema shape. ``None`` when no schema is declared OR
    when the synthesizer's response is not JSON-parseable. Schema
    validation is advisory at dispatch time — the schema's value is
    enabling downstream consumers (other ensembles in a composition;
    the orchestrator's reasoning surface) to parse the structured
    payload predictably when present, not enforcing format compliance
    at the synthesizer agent's output.
    """

    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Operator-readable dispatch diagnostics.

    Populated by Orchestrator Tool Dispatch's envelope construction
    step from the dispatch's emitted events (queried via
    :meth:`DispatchEventSubstrate.events_for`). Typical fields:
    ``ensemble``, ``dispatch_id`` (the ADR-023 correlation identifier),
    ``duration_seconds`` (from :class:`DispatchTiming`),
    ``model_profile``, ``tier``, ``topaz_skill``,
    ``calibration_verdict``, ``audit_findings`` (list of
    :class:`AuditDiagnostic` entries, possibly empty). The default
    ``{}`` accommodates legacy / no-substrate paths during the Cycle 6
    transition; production envelopes always carry a populated dict.
    """

    errors: list[dict[str, Any]] | None = None
    """Optional per-stage errors.

    Present when :attr:`status` is ``error``, ``partial``, or when a
    successful dispatch surfaced non-fatal errors. Each entry:
    ``stage`` (agent name or pipeline phase), ``error_type`` (typed
    error name per ADR-015 / ADR-017 vocabulary), ``message``,
    ``recoverable`` (bool).
    """

    artifacts: list[dict[str, Any]] | None = None
    """Optional artifact references (WP-E populates per ADR-025).

    For substrate-routed capability ensembles (WP-E default),
    ``artifacts[0]`` carries the typed reference to the deliverable
    written by Session Artifact Store: ``{path, content_type,
    size_bytes, summary, retention}``. ``None`` (the WP-D default for
    inline-response dispatches) means the deliverable is inline at
    :attr:`primary`. System ensembles remain inline; capability
    ensembles substrate-route once WP-E ships.
    """
