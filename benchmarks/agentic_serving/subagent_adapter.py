"""Claude Code subagent transcript -> Transcript IR adapter (#131, Arm 2).

Arm 2 drives the battery through ONE continuing Claude Code subagent
conversation (practitioner decision 2026-07-15), so the raw shape is a single
``subagents/agent-<id>.jsonl`` holding ALL turns — unlike OpenCode's
one-stream-per-turn. Schema pinned by a real capture
(``docs/plans/2026-07-17-arm2-subagent-captures/``): JSONL events keyed by
``type`` — ``user`` (an injected prompt when ``message.content`` is a STRING;
a tool result when it is a list of ``tool_result`` blocks), ``assistant``
(``message.content`` blocks: ``text`` / ``thinking`` / ``tool_use``), and
``attachment`` (harness metadata, skipped). Tool results pair to calls by
``tool_use_id``. Deterministic, pure.

Two traps the capture surfaced, both handled here:

- Streaming emits several events sharing one ``message.id`` with EVOLVING
  usage; summing per event would multiply-count tokens. Usage dedupes by
  ``message.id``, last event wins (the OpenCode adapter's callID lesson, one
  format over).
- An UNMAPPED tool name would silently score zero shipped (the roadmap's
  named failure mode), so unknown names raise instead of passing through.

Cache tokens are EXCLUDED from the sums for cross-arm consistency with the
OpenCode adapter's documented fresh-token lower bound — even though this
format carries the full cache split; revisit both together if IR/Pricing
grow cache fields.

Continuation prompts arrive wrapped by the harness ("The coordinator sent a
message while you were working:\\n<prompt>"). The wrapper is stripped so the
IR prompt matches the battery prompt; the framing difference is a DECLARED
Arm-2 construct confound, published with the table.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn

# Observed Claude Code tool names -> the IR names the metrics already read
# (lowercase, matching the OpenCode adapter's output). Extend DELIBERATELY on
# a new capture; never let an unknown name pass silently.
_TOOL_NAME_MAP = {
    "Bash": "bash",
    "Write": "write",
    "Read": "read",
    "Edit": "edit",
    "Glob": "glob",
    "Grep": "grep",
    "TodoWrite": "todowrite",
}

_RUN_TOOLS = ("bash",)

_COORDINATOR_PREFIX = "The coordinator sent a message while you were working:\n"
_COORDINATOR_SUFFIX = "\n\nAddress this before completing your current task."


class SubagentAdapterError(ValueError):
    """Raised when the raw stream cannot be mapped without guessing."""


def parse_events_counting_drops(
    jsonl_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Split a subagent JSONL stream into event dicts, one per non-blank line.

    Unparseable lines are dropped-and-counted for the same reason as the
    OpenCode adapter: a process killed mid-write leaves a half line, and one
    dead line must not take down scoring for the run — while a schema change
    must not silently shrink event counts either. Callers that care about
    instrument integrity read the count.
    """
    events: list[dict[str, Any]] = []
    dropped = 0
    for raw in jsonl_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            dropped += 1
    return events, dropped


def _prompt_of(event: dict[str, Any]) -> str | None:
    """The injected prompt when this user event IS a turn boundary, else
    None. String content = a prompt; list content = tool results."""
    if event.get("type") != "user":
        return None
    content = (event.get("message") or {}).get("content")
    if not isinstance(content, str):
        return None
    if content.startswith(_COORDINATOR_PREFIX):
        content = content[len(_COORDINATOR_PREFIX) :]
        if content.endswith(_COORDINATOR_SUFFIX):
            content = content[: -len(_COORDINATOR_SUFFIX)]
    return content


def split_turns(
    events: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Segment one conversation's events into ``(prompt, turn_events)``.

    Events before the first prompt (none observed in captures) would be
    unattributable to a turn; their existence raises rather than being
    silently dropped or misfiled into turn 1.
    """
    turns: list[tuple[str, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] | None = None
    for event in events:
        prompt = _prompt_of(event)
        if prompt is not None:
            current = []
            turns.append((prompt, current))
            continue
        if event.get("type") == "attachment":
            continue
        if current is None:
            raise SubagentAdapterError(
                "event precedes the first prompt; cannot attribute to a turn"
            )
        current.append(event)
    return turns


def _result_text(content: Any) -> str:
    """A tool_result body, verbatim: plain string, or text blocks joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "" if content is None else str(content)


def _timestamp_seconds(event: dict[str, Any]) -> float | None:
    raw = event.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _tool_call_of(block: dict[str, Any]) -> ToolCall:
    """Map one ``tool_use`` block into the IR, refusing unknown names."""
    raw_name = str(block.get("name", ""))
    name = _TOOL_NAME_MAP.get(raw_name)
    if name is None:
        raise SubagentAdapterError(
            f"unmapped tool name {raw_name!r}: extend _TOOL_NAME_MAP from a "
            "real capture instead of letting it score as zero shipped"
        )
    args = block.get("input") or {}
    return ToolCall(
        name=name,
        command=args.get("command") if name in _RUN_TOOLS else None,
        path=args.get("file_path") or args.get("pattern") or args.get("path"),
    )


def _absorb_assistant(
    message: dict[str, Any],
    texts: list[str],
    calls: dict[str, ToolCall],
    usage_by_message: dict[str, tuple[int, int]],
) -> None:
    usage = message.get("usage") or {}
    message_id = str(message.get("id", "")) or f"_anon_{len(usage_by_message)}"
    # Last event for a message id wins: streaming repeats the id with
    # evolving (cumulative) usage, so the terminal snapshot is the real
    # count and summing per event would multiply-count.
    usage_by_message[message_id] = (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
    )
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            calls[str(block.get("id", ""))] = _tool_call_of(block)


def _absorb_results(content: Any, calls: dict[str, ToolCall]) -> None:
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        call_id = str(block.get("tool_use_id", ""))
        call = calls.get(call_id)
        if call is not None:
            calls[call_id] = ToolCall(
                name=call.name,
                command=call.command,
                path=call.path,
                result_text=_result_text(block.get("content")),
            )


def turn_from_events(events: list[dict[str, Any]], *, index: int, prompt: str) -> Turn:
    """Build one :class:`Turn` from one turn's ordered subagent events."""
    texts: list[str] = []
    calls: dict[str, ToolCall] = {}  # tool_use id -> call, insertion-ordered
    usage_by_message: dict[str, tuple[int, int]] = {}  # message.id -> (in, out)
    timestamps: list[float] = []

    for event in events:
        ts = _timestamp_seconds(event)
        if ts is not None:
            timestamps.append(ts)
        message = event.get("message") or {}
        if event.get("type") == "assistant":
            _absorb_assistant(message, texts, calls, usage_by_message)
        elif event.get("type") == "user":
            _absorb_results(message.get("content"), calls)

    input_sum = sum(pair[0] for pair in usage_by_message.values())
    output_sum = sum(pair[1] for pair in usage_by_message.values())
    wall_seconds: float | None = None
    if len(timestamps) >= 2:
        wall_seconds = max(timestamps) - min(timestamps)

    return Turn(
        index=index,
        prompt=prompt,
        assistant_text="\n".join(texts),
        tool_calls=tuple(calls.values()),
        wall_seconds=wall_seconds,
        input_tokens=input_sum if (input_sum or output_sum) else None,
        output_tokens=output_sum if (input_sum or output_sum) else None,
    )


def transcript_from_jsonl(
    arm: str, jsonl_text: str, *, expected_prompts: list[str] | None = None
) -> Transcript:
    """Assemble a :class:`Transcript` from ONE subagent conversation's JSONL.

    ``expected_prompts``, when given, is verified against the observed
    prompts (order and count): a driver that dropped or duplicated a turn
    must fail here, not read as a short-but-valid run.
    """
    events, _ = parse_events_counting_drops(jsonl_text)
    segmented = split_turns(events)
    if expected_prompts is not None:
        observed = [prompt for prompt, _ in segmented]
        if observed != list(expected_prompts):
            raise SubagentAdapterError(
                f"observed prompts diverge from the battery: "
                f"{len(observed)} observed vs {len(expected_prompts)} expected"
            )
    turns = tuple(
        turn_from_events(turn_events, index=i, prompt=prompt)
        for i, (prompt, turn_events) in enumerate(segmented, start=1)
    )
    return Transcript(arm=arm, turns=turns)
