"""Tests for the shared structural-error base class (ADR-017, WP-A4).

LlmOrcStructuralError is the base class binding the eight Cycle 4
typed-error surfaces (FC-17). It carries four common fields and
preserves the message-pass-through behavior of the precedent class
``ToolCallingNotSupportedError`` (commit ``9f86d0b``).
"""

import pytest

from llm_orc.agentic.calibration_gate import CalibrationAbstainError
from llm_orc.agentic.calibration_signal_channel import MalformedSignalError
from llm_orc.agentic.tier_router import (
    EscalationBypassError,
    MissingSkillMetadataError,
)
from llm_orc.models.base import ToolCallingNotSupportedError
from llm_orc.models.structural_errors import LlmOrcStructuralError


class TestLlmOrcStructuralError:
    """Base class behavior — the four common fields and the raise/catch path."""

    def test_instantiates_with_four_common_fields(self) -> None:
        """All four fields per FC-17 are accessible on the instance."""
        error = LlmOrcStructuralError(
            "something structural went wrong",
            error_kind="phantom_tool_call",
            dispatch_context={"session_id": "s1", "turn": 3},
            recovery_action_required="reformulate",
            operator_diagnostic="orchestrator claimed tool call without structure",
        )

        assert error.error_kind == "phantom_tool_call"
        assert error.dispatch_context == {"session_id": "s1", "turn": 3}
        assert error.recovery_action_required == "reformulate"
        assert (
            error.operator_diagnostic
            == "orchestrator claimed tool call without structure"
        )

    def test_is_an_exception_and_preserves_message(self) -> None:
        """Raise/catch behavior + ``str(e)`` reads the message arg."""
        with pytest.raises(LlmOrcStructuralError, match="something structural"):
            raise LlmOrcStructuralError(
                "something structural went wrong",
                error_kind="phantom_tool_call",
                recovery_action_required="reformulate",
            )

    def test_dispatch_context_defaults_to_empty_dict(self) -> None:
        """Producers that lack context still satisfy FC-17's four-field shape."""
        error = LlmOrcStructuralError(
            "no context available",
            error_kind="malformed_signal",
            recovery_action_required="reformulate",
        )

        assert error.dispatch_context == {}

    def test_operator_diagnostic_defaults_to_message(self) -> None:
        """When no separate diagnostic is provided, the message serves both roles."""
        error = LlmOrcStructuralError(
            "compaction layer 4 failure",
            error_kind="compaction_layer_4_failure",
            recovery_action_required="operator_intervention_required",
        )

        assert error.operator_diagnostic == "compaction layer 4 failure"

    @pytest.mark.parametrize(
        "recovery_action",
        ["reformulate", "escalate", "abstain", "operator_intervention_required"],
    )
    def test_accepts_each_recovery_action_literal_value(
        self, recovery_action: str
    ) -> None:
        """Per ADR-017 §"Shared typed-error base class", the literal set is 4."""
        error = LlmOrcStructuralError(
            "test",
            error_kind="test_kind",
            recovery_action_required=recovery_action,  # type: ignore[arg-type]
        )

        assert error.recovery_action_required == recovery_action


class TestToolCallingNotSupportedErrorAsConcreteSubclass:
    """``ToolCallingNotSupportedError`` (commit ``9f86d0b``) is the first
    concrete subclass per ADR-017 §"Shared typed-error base class"."""

    def test_is_subclass_of_structural_error_base(self) -> None:
        assert issubclass(ToolCallingNotSupportedError, LlmOrcStructuralError)

    def test_fixes_error_kind_to_tool_call_rejected_per_model(self) -> None:
        """The error_kind discriminator is set by construction; not caller-supplied."""
        error = ToolCallingNotSupportedError("model X does not support tool calling")

        assert error.error_kind == "tool_call_rejected_per_model"

    def test_fixes_recovery_action_to_reformulate(self) -> None:
        """Per FC-17 ``error_kind`` table — orchestrator-recoverable surface."""
        error = ToolCallingNotSupportedError("model X does not support tool calling")

        assert error.recovery_action_required == "reformulate"

    def test_preserves_message_for_existing_callers(self) -> None:
        """Pre-existing call sites (``raise ToolCallingNotSupportedError(msg)``)
        continue to work — ``str(e)`` and ``match=`` still see the message."""
        with pytest.raises(
            ToolCallingNotSupportedError, match="does not support tool calling"
        ):
            raise ToolCallingNotSupportedError("model X does not support tool calling")


class TestCalibrationAbstainErrorAsConcreteSubclass:
    """``CalibrationAbstainError`` is the fifth concrete subclass per FC-17
    (after ``ToolCallingNotSupportedError``, ``PhantomToolCallError``,
    ``WriteGateRejectionError``, ``CompactionLayer4FailureError``)."""

    def test_is_subclass_of_structural_error_base(self) -> None:
        assert issubclass(CalibrationAbstainError, LlmOrcStructuralError)

    def test_fixes_error_kind_to_calibration_abstain(self) -> None:
        """The error_kind discriminator is set by construction per ADR-014."""
        error = CalibrationAbstainError(
            "entropy collapse exceeded 1.5σ threshold",
            session_id="s1",
            ensemble_name="composed-a",
            criterion="entropy_collapse",
        )

        assert error.error_kind == "calibration_abstain"

    def test_fixes_recovery_action_to_abstain(self) -> None:
        """Per ADR-014 §Decision and the four-value Literal in
        ``structural_errors``: the orchestrator must take a different
        action (reformulate, dispatch elsewhere, or abstain entirely).
        """
        error = CalibrationAbstainError(
            "entropy collapse exceeded 1.5σ threshold",
            session_id="s1",
            ensemble_name="composed-a",
            criterion="entropy_collapse",
        )

        assert error.recovery_action_required == "abstain"

    def test_dispatch_context_carries_calibration_metadata(self) -> None:
        """``dispatch_context`` is the operator-readable structural cause."""
        error = CalibrationAbstainError(
            "post-hoc hard failure (non-recoverable)",
            session_id="s1",
            ensemble_name="composed-a",
            criterion="post_hoc_hard_failure",
        )

        assert error.dispatch_context == {
            "session_id": "s1",
            "ensemble_name": "composed-a",
            "criterion": "post_hoc_hard_failure",
        }

    def test_preserves_message_for_raise_catch_path(self) -> None:
        with pytest.raises(CalibrationAbstainError, match="severe drift"):
            raise CalibrationAbstainError(
                "severe drift verdict from ADR-016 mechanism (d)",
                session_id="s2",
                ensemble_name="composed-b",
                criterion="severe_drift",
            )


class TestEscalationBypassErrorAsConcreteSubclass:
    """``EscalationBypassError`` is the sixth concrete subclass per FC-17.

    Raised by the Tier-Escalation Router (WP-G4-1) when the calibration
    verdict is Abstain. Per ADR-015 §Router logic, Abstain bypasses
    routing entirely — the orchestrator must reformulate, dispatch to
    a different ensemble, or abandon the task.
    """

    def test_is_subclass_of_structural_error_base(self) -> None:
        assert issubclass(EscalationBypassError, LlmOrcStructuralError)

    def test_fixes_error_kind_to_escalation_bypass(self) -> None:
        """The error_kind discriminator is set by construction per ADR-015."""
        error = EscalationBypassError(
            "Abstain verdict — routing bypassed",
            ensemble_name="composed-a",
        )

        assert error.error_kind == "escalation_bypass"

    def test_fixes_recovery_action_to_reformulate(self) -> None:
        """Per ADR-015 §Router logic: the orchestrator must reformulate."""
        error = EscalationBypassError(
            "Abstain verdict — routing bypassed",
            ensemble_name="composed-a",
        )

        assert error.recovery_action_required == "reformulate"

    def test_dispatch_context_carries_router_metadata(self) -> None:
        """``dispatch_context`` records the ensemble and session causing the bypass."""
        error = EscalationBypassError(
            "Abstain verdict — routing bypassed",
            ensemble_name="composed-a",
            session_id="s-77",
        )

        assert error.dispatch_context == {
            "session_id": "s-77",
            "ensemble_name": "composed-a",
        }

    def test_preserves_message_for_raise_catch_path(self) -> None:
        with pytest.raises(EscalationBypassError, match="routing bypassed"):
            raise EscalationBypassError(
                "Abstain verdict — routing bypassed",
                ensemble_name="composed-a",
            )


class TestMissingSkillMetadataErrorAsConcreteSubclass:
    """``MissingSkillMetadataError`` is the seventh concrete subclass per FC-17.

    Raised by the Tier-Escalation Router when a dispatched ensemble's
    YAML configuration does not declare a ``topaz_skill`` metadata
    field. Per ADR-015 §Per-skill role profiling, every ensemble in
    the library must declare its primary Topaz skill.
    """

    def test_is_subclass_of_structural_error_base(self) -> None:
        assert issubclass(MissingSkillMetadataError, LlmOrcStructuralError)

    def test_fixes_error_kind_to_missing_skill_metadata(self) -> None:
        error = MissingSkillMetadataError(
            "ensemble lacks topaz_skill metadata",
            ensemble_name="untagged-ensemble",
        )

        assert error.error_kind == "missing_skill_metadata"

    def test_fixes_recovery_action_to_reformulate(self) -> None:
        """Per ADR-015 §Per-skill role profiling: the orchestrator must
        dispatch to a different ensemble (one with the metadata field)
        or abandon the task; operator correction is the durable fix."""
        error = MissingSkillMetadataError(
            "ensemble lacks topaz_skill metadata",
            ensemble_name="untagged-ensemble",
        )

        assert error.recovery_action_required == "reformulate"

    def test_dispatch_context_carries_ensemble_name(self) -> None:
        error = MissingSkillMetadataError(
            "ensemble lacks topaz_skill metadata",
            ensemble_name="untagged-ensemble",
            session_id="s-77",
        )

        assert error.dispatch_context == {
            "session_id": "s-77",
            "ensemble_name": "untagged-ensemble",
        }

    def test_preserves_message_for_raise_catch_path(self) -> None:
        with pytest.raises(MissingSkillMetadataError, match="topaz_skill"):
            raise MissingSkillMetadataError(
                "ensemble lacks topaz_skill metadata",
                ensemble_name="untagged-ensemble",
            )


class TestMalformedSignalErrorAsConcreteSubclass:
    """``MalformedSignalError`` is the eighth and final FC-17 subclass.

    Raised by the Calibration Signal Channel (WP-H4, ADR-016) at the
    L0 → L1 boundary when an incoming signal fails schema validation.
    Per ADR-016 §"Mechanism (e)" the error is *internal* — the L0
    caller catches and drops the signal; the error never propagates
    to the orchestrator's reasoning surface. ``recovery_action_
    required`` is ``reformulate`` to align with the FC-17 family
    shape; the recovery path is exercised only by tests because the
    operative call sites catch the error.

    FC-17 coverage reaches 8 of 8 at WP-H4 close:
    ``tool_call_rejected_per_model``, ``phantom_tool_call``,
    ``write_gate_rejection``, ``compaction_layer_4_failure``,
    ``calibration_abstain``, ``escalation_bypass``,
    ``missing_skill_metadata``, ``malformed_signal``.
    """

    def test_is_subclass_of_structural_error_base(self) -> None:
        assert issubclass(MalformedSignalError, LlmOrcStructuralError)

    def test_fixes_error_kind_to_malformed_signal(self) -> None:
        error = MalformedSignalError(
            "signal missing required dispatch_success field",
            ensemble_name="ens-A",
            field_name="dispatch_success",
        )

        assert error.error_kind == "malformed_signal"

    def test_fixes_recovery_action_to_reformulate(self) -> None:
        error = MalformedSignalError(
            "signal missing required dispatch_success field",
            ensemble_name="ens-A",
        )

        assert error.recovery_action_required == "reformulate"

    def test_dispatch_context_carries_field_and_ensemble_metadata(self) -> None:
        error = MalformedSignalError(
            "signal field 'recent_token_entropy' must be numeric",
            ensemble_name="ens-A",
            field_name="recent_token_entropy",
        )

        assert error.dispatch_context == {
            "ensemble_name": "ens-A",
            "field_name": "recent_token_entropy",
        }

    def test_omits_ensemble_name_from_context_when_unavailable(self) -> None:
        """Tolerates ``None`` ensemble_name (top-level shape errors)."""
        error = MalformedSignalError(
            "calibration signal must be a CalibrationSignal or Mapping",
            ensemble_name=None,
        )

        assert "ensemble_name" not in error.dispatch_context

    def test_preserves_message_for_raise_catch_path(self) -> None:
        with pytest.raises(MalformedSignalError, match="missing required"):
            raise MalformedSignalError(
                "signal missing required ensemble_name field",
                ensemble_name=None,
                field_name="ensemble_name",
            )
