"""Unit tests for WS-8 aggregate metrics (#131; deterministic).

Rounds/wall-clock/cost — the mechanical metrics named in
`docs/serving-roadmap.md`'s WS-8 section that need no battery-specific
judgment. Run with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

from benchmarks.agentic_serving import metrics
from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn


class TestRoundsConsumed:
    def test_counts_the_tool_calls_in_one_turn(self) -> None:
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            tool_calls=(ToolCall(name="read"), ToolCall(name="write")),
        )
        assert metrics.rounds_consumed(turn) == 2

    def test_zero_when_no_tool_calls(self) -> None:
        turn = Turn(index=1, prompt="p", assistant_text="a")
        assert metrics.rounds_consumed(turn) == 0

    def test_total_rounds_sums_across_turns(self) -> None:
        transcript = Transcript(
            arm="serve",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    tool_calls=(ToolCall(name="write"),),
                ),
                Turn(
                    index=2,
                    prompt="p",
                    assistant_text="a",
                    tool_calls=(ToolCall(name="read"), ToolCall(name="write")),
                ),
            ),
        )
        assert metrics.total_rounds(transcript) == 3


class TestTotalWallSeconds:
    def test_sums_observed_wall_clock(self) -> None:
        transcript = Transcript(
            arm="serve",
            turns=(
                Turn(index=1, prompt="p", assistant_text="a", wall_seconds=12.5),
                Turn(index=2, prompt="p", assistant_text="a", wall_seconds=7.5),
            ),
        )
        assert metrics.total_wall_seconds(transcript) == 20.0

    def test_untimed_turns_contribute_zero(self) -> None:
        transcript = Transcript(
            arm="serve", turns=(Turn(index=1, prompt="p", assistant_text="a"),)
        )
        assert metrics.total_wall_seconds(transcript) == 0.0


_SONNET_5 = metrics.Pricing(input_per_mtok=3.00, output_per_mtok=15.00)


class TestTurnCost:
    def test_computes_dollar_cost_from_token_counts(self) -> None:
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert metrics.turn_cost(turn, _SONNET_5) == 18.00

    def test_none_when_the_arm_has_no_token_counts(self) -> None:
        """Arm 0 (the serve) has no per-token billing — local inference
        isn't billed per token, so the turn contributes no cost data."""
        turn = Turn(index=1, prompt="p", assistant_text="a")
        assert metrics.turn_cost(turn, _SONNET_5) is None


class TestTotalCost:
    def test_sums_cost_across_priced_turns(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=500_000,
                    output_tokens=0,
                ),
                Turn(
                    index=2,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=500_000,
                    output_tokens=0,
                ),
            ),
        )
        assert metrics.total_cost(transcript, _SONNET_5) == 3.00

    def test_unpriced_turns_contribute_zero(self) -> None:
        """Arm 0 turns (no token counts) don't error the aggregate — the
        serve's marginal cost is $0, per WS-8."""
        transcript = Transcript(
            arm="serve", turns=(Turn(index=1, prompt="p", assistant_text="a"),)
        )
        assert metrics.total_cost(transcript, _SONNET_5) == 0.0


class TestCostPerSolvedTurn:
    def test_divides_total_cost_by_solved_count(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1_000_000,
                    output_tokens=0,
                ),
            ),
        )
        assert (
            metrics.cost_per_solved_turn(transcript, _SONNET_5, solved_count=3) == 1.00
        )

    def test_none_when_nothing_solved(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1_000_000,
                    output_tokens=0,
                ),
            ),
        )
        assert (
            metrics.cost_per_solved_turn(transcript, _SONNET_5, solved_count=0) is None
        )
