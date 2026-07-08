"""Tests for the ``DispatchEnvelope`` shared type (Cycle 6 WP-D, ADR-024).

Per ``docs/agentic-serving/system-design.agents.md`` §Shared type:
``DispatchEnvelope`` — *new in Cycle 6 per ADR-024*. The envelope is the
typed return shape of ``invoke_ensemble`` consumed by the Runtime and the
skill framework's chat-completion response. Lives alongside
``LlmOrcStructuralError`` as a cross-cutting shared type.

Per ADR-024 §Decision: ``{status, primary, structured?, diagnostics,
errors?, artifacts?}``; field name ``diagnostics`` (not ``metadata``);
optional ``output_schema:`` per-ensemble declaration is advisory at
dispatch time. The structural shape lives here; the construction site
(``invoke_ensemble``) is exercised in
``test_orchestrator_tool_dispatch.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from llm_orc.models.dispatch_envelope import DispatchEnvelope


class TestEnvelopeShape:
    """The frozen dataclass carries the six fields ADR-024 specifies."""

    def test_envelope_carries_status_and_primary(self) -> None:
        envelope = DispatchEnvelope(status="success", primary="summary line")

        assert envelope.status == "success"
        assert envelope.primary == "summary line"

    def test_diagnostics_defaults_to_empty_dict(self) -> None:
        envelope = DispatchEnvelope(status="success", primary="x")

        assert envelope.diagnostics == {}

    def test_optional_fields_default_none(self) -> None:
        envelope = DispatchEnvelope(status="success", primary="x")

        assert envelope.structured is None
        assert envelope.errors is None
        assert envelope.artifacts is None

    def test_envelope_is_frozen(self) -> None:
        envelope = DispatchEnvelope(status="success", primary="x")

        with pytest.raises(dataclasses.FrozenInstanceError):
            envelope.primary = "mutated"  # type: ignore[misc]

    def test_envelope_carries_all_fields_when_populated(self) -> None:
        envelope = DispatchEnvelope(
            status="success",
            primary="Summary line referencing the deliverable",
            structured={"claims": [{"text": "x", "label": "established"}]},
            diagnostics={
                "ensemble": "claim-extractor",
                "dispatch_id": "session-dispatch-0001",
                "duration_seconds": 1.23,
                "model_profile": "agentic-tier-cheap-general",
                "tier": "cheap",
                "topaz_skill": "factual_knowledge",
                "calibration_verdict": "proceed",
                "audit_findings": [],
            },
            errors=None,
            artifacts=None,
        )

        assert envelope.structured == {
            "claims": [{"text": "x", "label": "established"}]
        }
        assert envelope.diagnostics["dispatch_id"] == "session-dispatch-0001"
        assert envelope.diagnostics["calibration_verdict"] == "proceed"

    def test_envelope_supports_per_stage_errors(self) -> None:
        envelope = DispatchEnvelope(
            status="partial",
            primary="Partial dispatch — see errors",
            errors=[
                {
                    "stage": "extractor",
                    "error_type": "invocation_failed",
                    "message": "timeout after 60s",
                    "recoverable": True,
                }
            ],
        )

        assert envelope.status == "partial"
        assert envelope.errors is not None
        assert envelope.errors[0]["stage"] == "extractor"


class TestEnvelopeSerializesCleanly:
    """The envelope is consumed by JSON-serializing layers downstream.

    ``dataclasses.asdict`` produces nested ``dict``s suitable for
    ``json.dumps`` — no custom encoder required. WP-D's value-add for
    downstream consumers (orchestrator-context sink, skill-framework
    responses) depends on this property.
    """

    def test_asdict_returns_nested_dict(self) -> None:
        envelope = DispatchEnvelope(
            status="success",
            primary="x",
            structured={"k": "v"},
            diagnostics={"dispatch_id": "s-dispatch-0001"},
        )

        payload = dataclasses.asdict(envelope)

        assert payload == {
            "status": "success",
            "primary": "x",
            "structured": {"k": "v"},
            "diagnostics": {"dispatch_id": "s-dispatch-0001"},
            "errors": None,
            "artifacts": None,
        }

    def test_asdict_serializes_lists(self) -> None:
        envelope = DispatchEnvelope(
            status="success",
            primary="x",
            artifacts=[
                {
                    "path": "agentic-sessions/s/dispatch-0001/out.py",
                    "content_type": "application/python",
                    "size_bytes": 1247,
                    "summary": "Class CircularBuffer; 24 lines.",
                    "retention": "session",
                }
            ],
        )

        payload = dataclasses.asdict(envelope)

        assert payload["artifacts"] is not None
        assert payload["artifacts"][0]["content_type"] == "application/python"
