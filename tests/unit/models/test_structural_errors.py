"""Tests for the shared structural-error base class (ADR-017, WP-A4).

LlmOrcStructuralError is the base class binding the eight Cycle 4
typed-error surfaces (FC-17). It carries four common fields and
preserves the message-pass-through behavior of the precedent class
``ToolCallingNotSupportedError`` (commit ``9f86d0b``).
"""

import pytest

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
