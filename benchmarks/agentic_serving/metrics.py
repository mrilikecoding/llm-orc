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
    """Per-million-token pricing for one model (USD).

    ``cache_creation_per_mtok`` / ``cache_read_per_mtok`` are OPTIONAL: cache
    tokens bill at provider- and cache-TTL-specific multiples of the fresh
    rate (Anthropic's are roughly 1.25x write / 0.1x read, but that varies
    by provider and is not safe to assume here), so no default is invented.
    A caller that omits them gets a cost figure that structurally EXCLUDES
    cache-token cost — see :func:`turn_cost_excludes_cache`.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_creation_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None


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
    already does).

    Cache tokens are priced when ``pricing`` has a rate for them; when a
    turn carries cache tokens but the matching rate is ``None``, that
    portion is silently left out of the figure here — the arithmetic can't
    express "I don't know", so callers that need to know whether THIS
    number is the true total or a lower bound must check
    :func:`turn_cost_excludes_cache` (or :func:`total_cost_excludes_cache`
    for the aggregate) rather than trust the float alone.
    """
    if turn.input_tokens is None or turn.output_tokens is None:
        return None
    cost = (
        turn.input_tokens / 1_000_000 * pricing.input_per_mtok
        + turn.output_tokens / 1_000_000 * pricing.output_per_mtok
    )
    if turn.cache_creation_tokens and pricing.cache_creation_per_mtok is not None:
        cost += turn.cache_creation_tokens / 1_000_000 * pricing.cache_creation_per_mtok
    if turn.cache_read_tokens and pricing.cache_read_per_mtok is not None:
        cost += turn.cache_read_tokens / 1_000_000 * pricing.cache_read_per_mtok
    return cost


def turn_cost_excludes_cache(turn: Turn, pricing: Pricing) -> bool:
    """True when ``turn`` reports a positive count of cache-creation or
    cache-read tokens that ``pricing`` has no rate for — :func:`turn_cost`'s
    figure then structurally EXCLUDES that cost, a lower bound rather than
    the true total. A turn with no cache tokens (``None`` or exactly ``0``)
    has nothing to exclude, so this is ``False`` regardless of ``pricing``.
    """
    if turn.cache_creation_tokens and pricing.cache_creation_per_mtok is None:
        return True
    if turn.cache_read_tokens and pricing.cache_read_per_mtok is None:
        return True
    return False


def total_cost(transcript: Transcript, pricing: Pricing) -> float:
    """Total dollar cost across the battery run; a turn with no token
    counts contributes $0 (Arm 0's structural free marginal cost)."""
    return sum(turn_cost(turn, pricing) or 0.0 for turn in transcript.turns)


def total_cost_excludes_cache(transcript: Transcript, pricing: Pricing) -> bool:
    """True when ANY turn's cost figure excludes cache-token cost (see
    :func:`turn_cost_excludes_cache`) — the aggregate ``total_cost`` is then
    a lower bound on the run's true cost, not the true total."""
    return any(turn_cost_excludes_cache(turn, pricing) for turn in transcript.turns)


def total_cache_creation_tokens(transcript: Transcript) -> int:
    """Cache-creation tokens observed across the run; a turn with no cache
    accounting (``None``) contributes 0."""
    return sum(turn.cache_creation_tokens or 0 for turn in transcript.turns)


def total_cache_read_tokens(transcript: Transcript) -> int:
    """Cache-read tokens observed across the run; a turn with no cache
    accounting (``None``) contributes 0."""
    return sum(turn.cache_read_tokens or 0 for turn in transcript.turns)


def cost_per_solved_turn(
    transcript: Transcript, pricing: Pricing, solved_count: int
) -> float | None:
    """WS-8's cost-per-solved-turn metric: total cost divided by turns
    marked solved. ``None`` when nothing was solved — a missing-data case,
    not a $0 cost."""
    if solved_count <= 0:
        return None
    return total_cost(transcript, pricing) / solved_count
