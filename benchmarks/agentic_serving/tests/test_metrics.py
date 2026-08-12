"""Unit tests for WS-8 aggregate metrics (#131; deterministic).

Rounds/wall-clock/cost — the mechanical metrics named in
`docs/serving-roadmap.md`'s WS-8 section that need no battery-specific
judgment. Run with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

import pytest

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
_SONNET_5_WITH_CACHE = metrics.Pricing(
    input_per_mtok=3.00,
    output_per_mtok=15.00,
    cache_creation_per_mtok=3.75,
    cache_read_per_mtok=0.30,
)


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

    def test_adds_cache_cost_when_pricing_has_cache_rates(self) -> None:
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        assert metrics.turn_cost(turn, _SONNET_5_WITH_CACHE) == pytest.approx(4.05)

    def test_excludes_cache_cost_when_pricing_has_no_cache_rates(self) -> None:
        # The fresh input/output figure is unchanged: a caller with no cache
        # rates gets a smaller, deliberately excl-cache number, never a
        # crash and never a silently-wrong inflated-looking total.
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        assert metrics.turn_cost(turn, _SONNET_5) == 18.00


class TestTurnCostExcludesCache:
    def test_true_when_cache_tokens_present_but_rate_missing(self) -> None:
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=500,
        )
        assert metrics.turn_cost_excludes_cache(turn, _SONNET_5) is True

    def test_false_when_cache_tokens_present_and_rate_given(self) -> None:
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=500,
        )
        assert metrics.turn_cost_excludes_cache(turn, _SONNET_5_WITH_CACHE) is False

    def test_false_when_no_cache_tokens_at_all(self) -> None:
        turn = Turn(
            index=1, prompt="p", assistant_text="a", input_tokens=1, output_tokens=1
        )
        assert metrics.turn_cost_excludes_cache(turn, _SONNET_5) is False

    def test_false_when_cache_tokens_are_exactly_zero(self) -> None:
        # Zero is a real, reported count -- there is nothing to exclude, so
        # this must not read the same as "cache accounting unavailable".
        turn = Turn(
            index=1,
            prompt="p",
            assistant_text="a",
            input_tokens=1,
            output_tokens=1,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        assert metrics.turn_cost_excludes_cache(turn, _SONNET_5) is False


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


class TestTotalCostExcludesCache:
    def test_true_if_any_turn_excludes_cache(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1,
                    output_tokens=1,
                ),
                Turn(
                    index=2,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=500,
                ),
            ),
        )
        assert metrics.total_cost_excludes_cache(transcript, _SONNET_5) is True

    def test_false_when_no_turn_carries_unpriced_cache_tokens(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(Turn(index=1, prompt="p", assistant_text="a", input_tokens=1),),
        )
        assert metrics.total_cost_excludes_cache(transcript, _SONNET_5) is False


class TestCacheTokenTotals:
    def test_sums_cache_creation_and_read_across_turns(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    cache_creation_tokens=100,
                    cache_read_tokens=10,
                ),
                Turn(
                    index=2,
                    prompt="p",
                    assistant_text="a",
                    cache_creation_tokens=200,
                    cache_read_tokens=20,
                ),
            ),
        )
        assert metrics.total_cache_creation_tokens(transcript) == 300
        assert metrics.total_cache_read_tokens(transcript) == 30

    def test_turns_with_no_cache_tokens_contribute_zero(self) -> None:
        transcript = Transcript(
            arm="serve", turns=(Turn(index=1, prompt="p", assistant_text="a"),)
        )
        assert metrics.total_cache_creation_tokens(transcript) == 0
        assert metrics.total_cache_read_tokens(transcript) == 0


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


class TestCostPerSolvedTurnExcludesCache:
    """MINOR 1 (round 3): cost_per_solved_turn's numerator is total_cost, so
    it needs the same lower-bound companion, not a bare unqualified float."""

    def test_true_when_the_underlying_total_cost_excludes_cache(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=500,
                ),
            ),
        )
        assert (
            metrics.cost_per_solved_turn_excludes_cache(
                transcript, _SONNET_5, solved_count=1
            )
            is True
        )

    def test_false_when_pricing_has_the_cache_rates(self) -> None:
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=500,
                ),
            ),
        )
        assert (
            metrics.cost_per_solved_turn_excludes_cache(
                transcript, _SONNET_5_WITH_CACHE, solved_count=1
            )
            is False
        )

    def test_none_when_nothing_solved(self) -> None:
        # Mirrors cost_per_solved_turn's own None -- there's no figure to
        # qualify.
        transcript = Transcript(
            arm="sonnet-5",
            turns=(
                Turn(
                    index=1,
                    prompt="p",
                    assistant_text="a",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=500,
                ),
            ),
        )
        assert (
            metrics.cost_per_solved_turn_excludes_cache(
                transcript, _SONNET_5, solved_count=0
            )
            is None
        )
