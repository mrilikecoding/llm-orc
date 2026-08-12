"""Arm-agnostic transcript IR for WS-8 parity scoring (#131).

Every comparison arm (the serve behind OpenCode, a frontier model behind
OpenCode, Claude Code) produces a different raw output shape. This module
is the one shape metrics are scored against: a per-arm adapter (not built
here — see `docs/plans/2026-07-13-parity-scoreboard-design.md`) turns raw
client output into this IR, and :mod:`benchmarks.agentic_serving.honesty` /
:mod:`benchmarks.agentic_serving.metrics` never branch on which arm produced
it.

Pure data, no logic — deterministic and CI-safe.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """One observed tool invocation inside a turn.

    ``command`` is set for run/bash-shaped calls (the string actually
    executed); ``path`` is set for read/write-shaped calls. ``result_text``
    is the tool's observed output, verbatim (never summarized), so honesty
    checks can pattern-match it. Adapters may leave both ``command`` and
    ``path`` unset for tool shapes the metrics don't need. ``is_error`` is
    whether the CLIENT reported the call itself as failed (a tool_use_error,
    a nonzero exit — arm-shaped, not the same thing as a test run's own
    pass/fail verdict inside ``result_text``).
    """

    name: str
    command: str | None = None
    path: str | None = None
    result_text: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class Turn:
    """One battery turn, as observed by the client — never the serve log.

    Scoring reads only what a client-side transcript can produce for any
    arm: the prompt, the assistant's final text, the tool calls it made, and
    optional timing/token counts. ``input_tokens``/``output_tokens`` are
    ``None`` for Arm 0 (local inference isn't billed per token).
    ``cache_creation_tokens``/``cache_read_tokens`` are ``None`` when the arm
    carries no cache-token accounting at all (distinct from ``0``, which
    means the arm reported cache usage and it was genuinely zero) — see
    :mod:`benchmarks.agentic_serving.metrics` for how ``Pricing`` prices
    them, and why a cost figure computed without cache rates is a lower
    bound, not the true total.
    """

    index: int
    prompt: str
    assistant_text: str
    tool_calls: tuple[ToolCall, ...] = ()
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None


@dataclass(frozen=True)
class Transcript:
    """One arm's full battery run — the turns in order."""

    arm: str
    turns: tuple[Turn, ...] = ()
