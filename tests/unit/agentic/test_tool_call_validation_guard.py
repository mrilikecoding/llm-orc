"""Tests for the tool-call structural validation guard (ADR-017, WP-C4).

Covers the four ADR-017 scenarios from
``docs/agentic-serving/scenarios.md`` §Tool-Call Structural Validation
Guard:

1. Match — prose claim plus tool-call structure passes validation.
2. Mismatch — prose claim without tool-call structure produces a
   ``phantom_tool_call`` typed error.
3. Future-intent patterns are not flagged (conservative
   false-positive discipline; rejected alternative (f)).
4. Pattern set is operator-extensible at deployment configuration.

The guard is the structural-validation interposition on the
orchestrator's response per system-design.agents.md §Orchestrator Tool
Dispatch (extended in Cycle 4 per ADR-017): scans response text for
assertion patterns; cross-checks against emitted tool-call structures;
returns a ``PhantomToolCallError`` on mismatch.
"""

from __future__ import annotations

import pytest

from llm_orc.agentic.tool_call_validation_guard import (
    DEFAULT_ASSERTION_PATTERNS,
    PhantomToolCallError,
    scan_response_for_phantom_claims,
)
from llm_orc.models.structural_errors import LlmOrcStructuralError


class TestPhantomToolCallError:
    """Typed-error shape (FC-17 — fourth concrete subclass)."""

    def test_is_subclass_of_structural_error(self) -> None:
        assert issubclass(PhantomToolCallError, LlmOrcStructuralError)

    def test_fixes_error_kind_to_phantom_tool_call(self) -> None:
        error = PhantomToolCallError(
            "orchestrator claimed tool call without structure",
            detected_prose_claim="the tool returned X",
            emitted_tool_call_names=(),
            dispatch_context={"session_id": "s1", "turn": 3},
        )

        assert error.error_kind == "phantom_tool_call"

    def test_fixes_recovery_action_to_reformulate(self) -> None:
        error = PhantomToolCallError(
            "phantom claim",
            detected_prose_claim="the tool returned X",
            emitted_tool_call_names=(),
        )

        assert error.recovery_action_required == "reformulate"

    def test_carries_detected_prose_claim_in_dispatch_context(self) -> None:
        """Per ADR-017 §Rejection — the detected prose substring is recorded."""
        error = PhantomToolCallError(
            "phantom claim",
            detected_prose_claim="the result is displayed above",
            emitted_tool_call_names=(),
        )

        assert error.dispatch_context["detected_prose_claim"] == (
            "the result is displayed above"
        )

    def test_carries_emitted_tool_call_names_in_dispatch_context(self) -> None:
        """Per ADR-017 §Rejection — the list of emitted tool-call structures
        (zero or more) is recorded so partial-mismatch context is preserved."""
        error = PhantomToolCallError(
            "phantom claim with partial structure",
            detected_prose_claim="I called `invoke_ensemble` and the result was X",
            emitted_tool_call_names=("compose_ensemble",),
        )

        assert error.dispatch_context["emitted_tool_call_names"] == (
            "compose_ensemble",
        )

    def test_merges_extra_dispatch_context(self) -> None:
        """Producers add session/turn context alongside the prose+structures shape."""
        error = PhantomToolCallError(
            "phantom claim",
            detected_prose_claim="the tool returned X",
            emitted_tool_call_names=(),
            dispatch_context={"session_id": "s1", "turn": 7},
        )

        assert error.dispatch_context == {
            "detected_prose_claim": "the tool returned X",
            "emitted_tool_call_names": (),
            "session_id": "s1",
            "turn": 7,
        }


class TestDefaultAssertionPatterns:
    """Conservative false-positive discipline (ADR-017 rejected (c) and (f))."""

    @pytest.mark.parametrize(
        "assertion_prose",
        [
            "the tool returned the answer",
            "the tool call returned the answer",
            "I called invoke_ensemble and the result was X",
            "the result of invoke_ensemble is displayed above",
            "as the observation above shows, X happened",
            "after running invoke_ensemble, I have the answer",
            "after invoking invoke_ensemble, I have the answer",
        ],
    )
    def test_default_patterns_flag_documented_assertion_phrasings(
        self, assertion_prose: str
    ) -> None:
        """ADR-017 §Detection pattern set must match its own documented examples."""
        result = scan_response_for_phantom_claims(
            assertion_prose, emitted_tool_call_names=()
        )

        assert isinstance(result, PhantomToolCallError)

    @pytest.mark.parametrize(
        "future_intent_prose",
        [
            "I will call invoke_ensemble next",
            "I am going to invoke invoke_ensemble",
            "Let me call invoke_ensemble",
        ],
    )
    def test_default_patterns_do_not_flag_future_intent_phrasings(
        self, future_intent_prose: str
    ) -> None:
        """Per ADR-017 rejected alternative (f) and §Detection conservative discipline.

        Future-intent patterns are pre-call narration that may legitimately
        be followed by an actual tool-call structure on the next turn;
        flagging them would produce operator-visible session disruption.
        """
        result = scan_response_for_phantom_claims(
            future_intent_prose, emitted_tool_call_names=()
        )

        assert result is None

    def test_default_pattern_set_is_a_non_empty_tuple(self) -> None:
        """The pattern set is exposed for operator inspection (per ADR-017
        operator-extensibility) and BUILD-time review."""
        assert isinstance(DEFAULT_ASSERTION_PATTERNS, tuple)
        assert len(DEFAULT_ASSERTION_PATTERNS) > 0


class TestScanResponseMatchScenario:
    """Scenario 1 — Match: prose claim plus tool-call structure passes."""

    def test_assertion_prose_with_emitted_tool_call_returns_none(self) -> None:
        """Prose 'I called invoke_ensemble and the tool returned ...' plus an
        emitted ``invoke_ensemble`` structure passes — the structural
        correspondence holds."""
        result = scan_response_for_phantom_claims(
            "I called invoke_ensemble and the tool returned the answer",
            emitted_tool_call_names=("invoke_ensemble",),
        )

        assert result is None

    def test_assertion_prose_with_any_emitted_tool_call_returns_none(self) -> None:
        """A non-empty tool-call set is the structural anchor — the minimum
        viable discipline does not require per-name correspondence beyond the
        conservative pattern set."""
        result = scan_response_for_phantom_claims(
            "the tool returned the answer",
            emitted_tool_call_names=("compose_ensemble",),
        )

        assert result is None


class TestScanResponseMismatchScenario:
    """Scenario 2 — Mismatch: phantom_tool_call error produced."""

    def test_assertion_prose_with_zero_tool_calls_returns_phantom_error(
        self,
    ) -> None:
        """ADR-017 §Validation §Mismatch — the response contains the prose
        claim but no corresponding tool-call structure."""
        spike_phrasing = "The tool call has been made and the result is displayed above"
        result = scan_response_for_phantom_claims(
            spike_phrasing, emitted_tool_call_names=()
        )

        assert isinstance(result, PhantomToolCallError)
        assert result.error_kind == "phantom_tool_call"
        assert result.dispatch_context["emitted_tool_call_names"] == ()
        detected = result.dispatch_context["detected_prose_claim"]
        assert detected
        assert detected.lower() in spike_phrasing.lower()

    def test_mismatch_carries_dispatch_context_through_to_error(self) -> None:
        """Caller-supplied dispatch context (session id, turn) flows into
        the produced PhantomToolCallError so operators can trace the
        rejection."""
        result = scan_response_for_phantom_claims(
            "The tool call has been made",
            emitted_tool_call_names=(),
            dispatch_context={"session_id": "s-42", "turn": 9},
        )

        assert isinstance(result, PhantomToolCallError)
        assert result.dispatch_context["session_id"] == "s-42"
        assert result.dispatch_context["turn"] == 9


class TestScanResponseFutureIntentScenario:
    """Scenario 3 — Future-intent patterns are not flagged."""

    def test_future_intent_prose_with_zero_tool_calls_returns_none(self) -> None:
        """Pre-call narration without an actual structure is acceptable — the
        next turn may emit the call."""
        result = scan_response_for_phantom_claims(
            "I will call invoke_ensemble next to get the answer",
            emitted_tool_call_names=(),
        )

        assert result is None

    def test_empty_response_text_returns_none(self) -> None:
        """A response with no content and no tool-calls is not a phantom — it's
        just a stop turn."""
        result = scan_response_for_phantom_claims("", emitted_tool_call_names=())

        assert result is None

    def test_unrelated_prose_returns_none(self) -> None:
        """Prose with no assertion pattern and no tool-call is not a phantom."""
        result = scan_response_for_phantom_claims(
            "Let me think about how to approach this task.",
            emitted_tool_call_names=(),
        )

        assert result is None


class TestScanResponseOperatorExtensibility:
    """Scenario 4 — Pattern set is operator-extensible at deployment configuration."""

    def test_operator_added_pattern_triggers_phantom_error(self) -> None:
        """Operators extend the pattern set via ``extra_patterns``; an
        operator-added regex that matches the response text and no
        corresponding tool-call structure produces a phantom_tool_call error."""
        operator_pattern = r"the deployment-specific phrasing X happened"
        result = scan_response_for_phantom_claims(
            "the deployment-specific phrasing X happened, finished",
            emitted_tool_call_names=(),
            extra_patterns=(operator_pattern,),
        )

        assert isinstance(result, PhantomToolCallError)
        assert result.error_kind == "phantom_tool_call"

    def test_operator_pattern_alongside_default_set(self) -> None:
        """Operator patterns are appended to the default set (per ADR-017
        §Minimal default pattern set with operator-extension surface)."""
        operator_pattern = r"site-specific phantom phrasing"
        # Default pattern still works
        default_result = scan_response_for_phantom_claims(
            "the tool returned X",
            emitted_tool_call_names=(),
            extra_patterns=(operator_pattern,),
        )

        # Operator pattern still works
        operator_result = scan_response_for_phantom_claims(
            "site-specific phantom phrasing happened",
            emitted_tool_call_names=(),
            extra_patterns=(operator_pattern,),
        )

        assert isinstance(default_result, PhantomToolCallError)
        assert isinstance(operator_result, PhantomToolCallError)

    def test_empty_extra_patterns_falls_back_to_defaults(self) -> None:
        """Operators who do not extend the pattern set still get default
        scanning."""
        result = scan_response_for_phantom_claims(
            "the tool returned X",
            emitted_tool_call_names=(),
            extra_patterns=(),
        )

        assert isinstance(result, PhantomToolCallError)
