"""Tests for the Budget Controller module.

Per `docs/agentic-serving/system-design.md` §Budget Controller (L1) and
§Integration Contracts (Orchestrator Runtime → Budget Controller).
Enforces AS-3: Budget enforcement is a control-plane concern, checked
at each ReAct iteration boundary regardless of what the orchestrator
LLM decides.
"""

from llm_orc.agentic.budget_controller import (
    BudgetCheckExhausted,
    BudgetCheckPass,
    BudgetController,
)


class TestBudgetPermitsIteration:
    """The Budget Controller answers one question per iteration:

    is the next ReAct iteration permitted under current Session state?
    """

    def test_check_passes_when_both_below_limits(self) -> None:
        controller = BudgetController(turn_limit=10, token_limit=1000)

        result = controller.check(turn_count=3, token_spend=250)

        assert isinstance(result, BudgetCheckPass)

    def test_check_reports_turn_exhaustion_when_turn_count_reaches_limit(
        self,
    ) -> None:
        """Boundary semantics: the check fires at ``>=``, not ``>``.

        A Session that has completed ``turn_limit`` iterations already
        exhausted its Budget — the next iteration would be over the cap.
        """
        controller = BudgetController(turn_limit=5, token_limit=1000)

        result = controller.check(turn_count=5, token_spend=250)

        assert isinstance(result, BudgetCheckExhausted)
        assert result.reason == "turn_limit"

    def test_check_reports_token_exhaustion_when_token_spend_reaches_limit(
        self,
    ) -> None:
        controller = BudgetController(turn_limit=10, token_limit=1000)

        result = controller.check(turn_count=3, token_spend=1000)

        assert isinstance(result, BudgetCheckExhausted)
        assert result.reason == "token_limit"

    def test_turn_limit_is_reported_first_when_both_limits_exceeded(
        self,
    ) -> None:
        """Deterministic reporting: turn precedes token when both fail.

        The Runtime uses ``reason`` to shape the exhaustion message. Both
        being exhausted is a rare race (turn increments per iteration;
        token increments per LLM response within the iteration), but the
        reporting order is fixed so the message is predictable.
        """
        controller = BudgetController(turn_limit=5, token_limit=1000)

        result = controller.check(turn_count=5, token_spend=1000)

        assert isinstance(result, BudgetCheckExhausted)
        assert result.reason == "turn_limit"

    def test_exhaustion_result_carries_state_and_limits(self) -> None:
        """The Runtime shapes its response from these fields without
        re-reading Session state or configuration."""
        controller = BudgetController(turn_limit=5, token_limit=1000)

        result = controller.check(turn_count=5, token_spend=275)

        assert isinstance(result, BudgetCheckExhausted)
        assert result.turn_count == 5
        assert result.token_spend == 275
        assert result.turn_limit == 5
        assert result.token_limit == 1000
