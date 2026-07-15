"""opencode ``--format json`` -> Transcript IR adapter (#131).

The one arm-specific seam in WS-8 scoring: turns OpenCode's raw JSONL event
stream (one stream per ``opencode run`` invocation = one battery turn) into
the arm-agnostic IR (:mod:`benchmarks.agentic_serving.transcript`) that the
scorer reads. The scorer never branches on arm; this adapter is where the
OpenCode-specific shape is absorbed.

Schema pinned by real captures (``docs/plans/2026-07-13-opencode-run-
captures/``): JSONL, one event per line keyed by ``type`` — ``step_start``
(envelope, ``timestamp``), ``text`` (``part.text``), ``tool_use``
(``part.tool`` / ``part.state.input`` / ``part.state.output``), ``step_finish``
(``part.tokens`` / ``part.cost``). See
``docs/plans/2026-07-14-opencode-ir-adapter-design.md``. Deterministic, pure.
"""

from __future__ import annotations

import json
from typing import Any

from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn

# Tool names whose observed command drives the honesty verification metric.
_RUN_TOOLS = ("bash", "run")


def parse_events(jsonl_text: str) -> list[dict[str, Any]]:
    """Split an ``opencode --format json`` stream into event dicts, one per
    non-blank line.

    Unparseable lines are DROPPED rather than raised. A turn killed mid-write
    (the shape a ``timeout`` SIGTERM produces) leaves a half-written final line,
    and propagating ``JSONDecodeError`` from here would take down scoring for the
    whole battery instead of the one dead turn. The turn's death stays visible
    where it belongs: a nonzero code in ``exits.tsv``, its ``turn-NN.err``, and
    the scorer's own ``missing_turns`` when nothing parseable survived.
    """
    events: list[dict[str, Any]] = []
    for raw in jsonl_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _tool_call(part: dict[str, Any]) -> ToolCall:
    name = str(part.get("tool", ""))
    state = part.get("state", {}) or {}
    args = state.get("input", {}) or {}
    output = state.get("output", "")
    command = args.get("command") if name in _RUN_TOOLS else None
    path = args.get("filePath") or args.get("pattern") or args.get("path")
    # Captured tool outputs are plain strings; str() is a fallback if a paid
    # arm ever emits structured output (not expected — revisit on capture).
    return ToolCall(
        name=name,
        command=command,
        path=path,
        result_text="" if output is None else str(output),
    )


def turn_from_events(events: list[dict[str, Any]], *, index: int, prompt: str) -> Turn:
    """Build one :class:`Turn` from a turn's ordered opencode events."""
    texts: list[str] = []
    tool_events: dict[object, dict[str, Any]] = {}
    input_sum = 0
    output_sum = 0
    timestamps: list[int] = []

    for event in events:
        timestamp = event.get("timestamp")
        if isinstance(timestamp, (int, float)):
            timestamps.append(int(timestamp))
        part = event.get("part", {}) or {}
        etype = event.get("type")
        if etype == "text":
            texts.append(str(part.get("text", "")))
        elif etype == "tool_use":
            # Dedup by callID keeping the terminal state, so a paid stream
            # that emits pending -> completed for one call counts as ONE
            # round (rounds_consumed). Insertion order = execution order.
            # A keyless event gets a unique sentinel so it can never collide
            # with a real callID string.
            call_id: object = part.get("callID") or object()
            tool_events[call_id] = part
        elif etype == "step_finish":
            tokens = part.get("tokens", {}) or {}
            input_sum += int(tokens.get("input", 0) or 0)
            # Reasoning bills at the OUTPUT rate (Anthropic), so fold it into
            # output. Cache-read/write tokens are EXCLUDED: they bill at
            # 0.1x/1.25x rates a flat `Pricing` can't express, and opencode's
            # paid-path token shape isn't captured yet. So cost here is
            # FRESH-token cost — a lower bound on a cache-heavy paid turn;
            # close it in Arc D with a real paid capture (grow IR/Pricing for
            # cache). Documented, pinned by test, not silently dropped.
            output_sum += int(tokens.get("output", 0) or 0) + int(
                tokens.get("reasoning", 0) or 0
            )

    tool_calls = [_tool_call(part) for part in tool_events.values()]

    # Zero tokens is the local unbilled (Arm-0) signal: map to None so
    # metrics.turn_cost returns None and the arm is $0 by construction.
    if input_sum == 0 and output_sum == 0:
        input_tokens: int | None = None
        output_tokens: int | None = None
    else:
        input_tokens = input_sum
        output_tokens = output_sum

    wall_seconds: float | None = None
    if len(timestamps) >= 2:
        wall_seconds = (max(timestamps) - min(timestamps)) / 1000.0

    return Turn(
        index=index,
        prompt=prompt,
        assistant_text="\n".join(texts),
        tool_calls=tuple(tool_calls),
        wall_seconds=wall_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def turn_from_jsonl(jsonl_text: str, *, index: int, prompt: str) -> Turn:
    """Parse an opencode JSONL stream and build one :class:`Turn`."""
    return turn_from_events(parse_events(jsonl_text), index=index, prompt=prompt)


def transcript_from_runs(arm: str, runs: list[tuple[str, str]]) -> Transcript:
    """Assemble a :class:`Transcript` from ordered ``(prompt, jsonl_text)``
    runs, one per battery turn, numbered from 1."""
    turns = tuple(
        turn_from_jsonl(jsonl_text, index=i, prompt=prompt)
        for i, (prompt, jsonl_text) in enumerate(runs, start=1)
    )
    return Transcript(arm=arm, turns=turns)
