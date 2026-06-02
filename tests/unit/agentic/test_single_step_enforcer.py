"""Tests for the Single-Step Enforcer (Cycle 7 loop-back WP-LB-B, ADR-033).

The enforcer is the framework's grounding guarantee: it truncates the
seat-filler's proposed batch of client tool calls to a single action per
assistant turn (the D2 batch-truncation technique — the only candidate
with direct Spike τ′ evidence, model-independent, stateless). Scenario
from ``docs/agentic-serving/scenarios.md`` §"Single-action-per-turn
enforcement truncates a driver batch" (FC-43).
"""

from __future__ import annotations

from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall


def _write(call_id: str, path: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="write",
        arguments_json=f'{{"filePath": "{path}"}}',
    )


class TestSingleStepEnforcer:
    """FC-43 — at most one client tool call is dispatched per assistant turn."""

    def test_truncates_batch_to_first_action(self) -> None:
        proposed = [_write("a", "one.py"), _write("b", "two.py")]

        enforced = SingleStepEnforcer().enforce(proposed)

        assert enforced.action == _write("a", "one.py")
        assert enforced.truncated is True
        assert enforced.dropped == 1

    def test_single_action_is_not_truncated(self) -> None:
        proposed = [_write("a", "one.py")]

        enforced = SingleStepEnforcer().enforce(proposed)

        assert enforced.action == _write("a", "one.py")
        assert enforced.truncated is False
        assert enforced.dropped == 0

    def test_empty_batch_yields_no_action(self) -> None:
        enforced = SingleStepEnforcer().enforce([])

        assert enforced.action is None
        assert enforced.truncated is False
        assert enforced.dropped == 0
