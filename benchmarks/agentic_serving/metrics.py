"""WS-8 aggregate metrics (#131) — pure functions over the transcript IR
(:mod:`benchmarks.agentic_serving.transcript`).

Dishonesty/verification classification lives in
:mod:`benchmarks.agentic_serving.honesty`; this module is the remaining
mechanical metrics named in `docs/serving-roadmap.md`'s WS-8 section: wall-
clock per turn, cost per solved turn, rounds/retries consumed. Deterministic
and CI-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.agentic_serving.transcript import Transcript, Turn


@dataclass(frozen=True)
class Pricing:
    """Per-million-token pricing for one model (input/output, USD)."""

    input_per_mtok: float
    output_per_mtok: float


def rounds_consumed(turn: Turn) -> int:
    """Tool-call rounds spent on one turn — a direct count, no judgment
    (WS-8's rounds/retries-consumed metric)."""
    return len(turn.tool_calls)


def total_rounds(transcript: Transcript) -> int:
    """Rounds consumed across the whole battery run."""
    return sum(rounds_consumed(turn) for turn in transcript.turns)


def total_wall_seconds(transcript: Transcript) -> float:
    """Sum of observed per-turn wall-clock (WS-8's wall-clock-per-turn
    metric, aggregated); turns with no timing contribute 0."""
    return sum(turn.wall_seconds or 0.0 for turn in transcript.turns)


def turn_cost(turn: Turn, pricing: Pricing) -> float | None:
    """Dollar cost of one turn, or ``None`` when the arm carries no token
    counts (Arm 0, the serve — local inference isn't billed per token; its
    marginal cost is $0 by construction, not merely unmeasured, so callers
    that need a number should treat ``None`` as $0, which :func:`total_cost`
    already does)."""
    if turn.input_tokens is None or turn.output_tokens is None:
        return None
    return (
        turn.input_tokens / 1_000_000 * pricing.input_per_mtok
        + turn.output_tokens / 1_000_000 * pricing.output_per_mtok
    )


def total_cost(transcript: Transcript, pricing: Pricing) -> float:
    """Total dollar cost across the battery run; a turn with no token
    counts contributes $0 (Arm 0's structural free marginal cost)."""
    return sum(turn_cost(turn, pricing) or 0.0 for turn in transcript.turns)


def cost_per_solved_turn(
    transcript: Transcript, pricing: Pricing, solved_count: int
) -> float | None:
    """WS-8's cost-per-solved-turn metric: total cost divided by turns
    marked solved. ``None`` when nothing was solved — a missing-data case,
    not a $0 cost."""
    if solved_count <= 0:
        return None
    return total_cost(transcript, pricing) / solved_count
