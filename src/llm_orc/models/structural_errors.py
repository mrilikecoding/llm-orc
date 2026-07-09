"""Shared structural-error base class for the orchestrator pipeline.

Cycle 4's eight typed-error surfaces (``tool_call_rejected_per_model``,
``phantom_tool_call``, ``compaction_layer_4_failure``,
``write_gate_rejection``, ``calibration_abstain``, ``escalation_bypass``,
``missing_skill_metadata``, ``malformed_signal``) derive from
:class:`LlmOrcStructuralError` per ADR-017 §"Shared typed-error base
class" and FC-17.

The base class carries four common fields so operator surfaces and
downstream recovery logic can treat the eight surfaces uniformly:

* ``error_kind`` — discriminator for the structural-error surface
* ``dispatch_context`` — orchestrator turn / session identifiers
* ``recovery_action_required`` — one of four orchestrator-side or
  operator-side recovery dispositions
* ``operator_diagnostic`` — operator-readable description of the
  structural condition

The fourth ``recovery_action_required`` value
(``operator_intervention_required``) distinguishes errors the
orchestrator cannot recover from (e.g., ADR-012 Layer 4 circuit-breaker
trip; ADR-016 mechanism (d) severe-drift fail-safe) from
orchestrator-recoverable errors per round-2 argument-audit finding
2026-05-06.

The precedent for the typed-error pattern is commit ``9f86d0b``
(``ToolCallingNotSupportedError``), which is migrated as the first
concrete subclass in :mod:`llm_orc.models.base`.
"""

from typing import Any, Literal

RecoveryAction = Literal[
    "reformulate",
    "escalate",
    "abstain",
    "operator_intervention_required",
]


class LlmOrcStructuralError(Exception):
    """Base class for the eight Cycle 4 structural-error surfaces (FC-17)."""

    def __init__(
        self,
        message: str,
        *,
        error_kind: str,
        recovery_action_required: RecoveryAction,
        dispatch_context: dict[str, Any] | None = None,
        operator_diagnostic: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.dispatch_context: dict[str, Any] = (
            dispatch_context if dispatch_context is not None else {}
        )
        self.recovery_action_required: RecoveryAction = recovery_action_required
        self.operator_diagnostic = (
            operator_diagnostic if operator_diagnostic is not None else message
        )
