"""Claude Code subagent transcript -> Transcript IR adapter (#131).

The Arm-2 counterpart to :mod:`benchmarks.agentic_serving.opencode_adapter`:
turns a captured Claude Code subagent transcript (JSONL) into the same
arm-agnostic IR (:mod:`benchmarks.agentic_serving.transcript`) so
``score_run.tally_oracles`` runs unchanged across arms. The scorer never
branches on arm; this adapter is where the subagent-specific shape is
absorbed.

Schema pinned by real captures (``docs/plans/2026-07-15-arm2-runs/`` — two
committed 13-turn runs, haiku and sonnet; probed and documented at
``docs/plans/2026-07-17-arm2-subagent-captures/README.md``). One run is ONE
continuing conversation logged to a SINGLE JSONL file (unlike opencode's
one-file-per-turn), so this module also splits a run into turns: each
injected prompt lands as a ``user``-typed event whose ``message.content`` is
a plain string (a real turn's tool RESULTS also arrive as ``user``-typed
events, but always with a LIST ``message.content`` of ``tool_result``
blocks — never a bare string — so the two never collide).

Top-level event ``type`` is one of ``user`` / ``assistant`` / ``attachment``.
``assistant`` message content blocks are ``thinking`` (skipped — no scored
content), ``tool_use`` (``{id, name, input}``), or ``text``. Tool names
observed in both committed runs: ``Write`` ``{file_path, content}``,
``Bash`` ``{command, description}``, ``Read`` ``{file_path}``, and ``Edit``
``{file_path, old_string, new_string, replace_all}``. **Edit is a
discrepancy**: the 2026-07-17 capture README's schema section names only
Write/Bash/Read, but both real runs use Edit too (5/5 haiku, 4/4 sonnet Edit
calls) — resolved here in the captures' favor, since the adapter must build
against what the transcripts actually contain, not the doc summarizing them.

Every observed tool name is mapped to its lowercase IR name (``bash``,
``read``, ``write``, ``edit``) so the shared, arm-blind
:mod:`benchmarks.agentic_serving.honesty` module (which keys test-command
detection on lowercase ``"bash"``) works unchanged across arms. Per the
roadmap's WS-8 card, an unrecognized tool name — or any other event/content
shape outside what's enumerated above — RAISES rather than silently mapping
to an empty turn: a silently-dropped stream would read as honest restraint
in the shipped/oracle tally, which is exactly the bias-toward-comparator
failure this instrument exists to avoid.

Usage/token accounting: multiple JSONL lines share one assistant
``message.id`` (streaming increments — a growing usage snapshot per content
block emitted), so usage is deduped by ``message.id``, keeping only the
LAST (terminal, cumulative) usage seen for each id, then summed across the
turn's distinct ids — never summed per line, which would double- (or
sextuple-) count. ``cache_creation_input_tokens`` / ``cache_read_input_tokens``
are excluded from ``input_tokens``: the shared ``metrics.Pricing`` table is
flat input/output per-mtok with no cache-rate slot, the same documented
limitation ``opencode_adapter`` carries, so cost here is fresh-token cost on
this arm too — keeping the two arms comparable rather than under-costing
only OpenCode.

Deterministic, pure.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn

# Real tool names -> lowercase IR names, demonstrated in BOTH committed runs
# (see module docstring for the Edit discrepancy). Anything else raises.
_TOOL_NAMES = {
    "Write": "write",
    "Read": "read",
    "Bash": "bash",
    "Edit": "edit",
}

# Tool names whose input carries the actually-executed command.
_RUN_TOOLS = ("bash",)

_KNOWN_EVENT_TYPES = ("user", "assistant", "attachment")
_KNOWN_ASSISTANT_BLOCKS = ("thinking", "tool_use", "text")
_KNOWN_USER_BLOCKS = ("tool_result",)


def parse_events(jsonl_text: str) -> list[dict[str, Any]]:
    """Split a captured subagent transcript into event dicts, one per
    non-blank line.

    Unparseable lines are DROPPED rather than raised, mirroring
    ``opencode_adapter.parse_events``: a subagent process killed mid-write
    leaves a half-written final line, and propagating ``JSONDecodeError``
    from here would take down scoring for the whole 13-turn run instead of
    the one dead turn.
    """
    events, _ = parse_events_counting_drops(jsonl_text)
    return events


def parse_events_counting_drops(
    jsonl_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """:func:`parse_events`, plus how many non-blank lines failed to parse."""
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


def _is_turn_boundary(event: dict[str, Any]) -> bool:
    """True for the injected prompt that starts a new turn: a ``user``-typed
    event whose ``message.content`` is a plain string. A turn's own tool
    results also arrive as ``user``-typed events, but their content is
    always a LIST of ``tool_result`` blocks, never a bare string."""
    if event.get("type") != "user":
        return False
    content = event.get("message", {}).get("content")
    return isinstance(content, str)


def split_turns(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split one run's full ordered event stream into per-turn slices at
    each injected-prompt boundary. A stream with no boundary (empty, or
    truncated before the first prompt survived parsing) splits into zero
    turns."""
    boundaries = [i for i, event in enumerate(events) if _is_turn_boundary(event)]
    if not boundaries:
        return []
    ends = boundaries[1:] + [len(events)]
    return [events[start:end] for start, end in zip(boundaries, ends, strict=True)]


def _tool_ir_name(raw_name: str) -> str:
    try:
        return _TOOL_NAMES[raw_name]
    except KeyError:
        raise ValueError(
            f"unmapped subagent tool {raw_name!r} (known: {sorted(_TOOL_NAMES)})"
        ) from None


def _tool_call(name: str, args: dict[str, Any], result_text: str) -> ToolCall:
    command = args.get("command") if name in _RUN_TOOLS else None
    path = args.get("file_path")
    return ToolCall(name=name, command=command, path=path, result_text=result_text)


class _TurnState:
    """Mutable accumulator for one turn's events, filled in event order by
    :func:`_apply_assistant_event` / :func:`_apply_user_event` and read by
    :func:`turn_from_events`. Split out purely to keep each event handler's
    branching small (the project's complexity gate)."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.tool_names: dict[str, str] = {}
        self.tool_args: dict[str, dict[str, Any]] = {}
        self.tool_order: list[str] = []
        self.tool_results: dict[str, str] = {}
        self.usage_by_message: dict[str, dict[str, Any]] = {}


def _apply_assistant_event(event: dict[str, Any], state: _TurnState) -> None:
    message = event.get("message", {}) or {}
    message_id = message.get("id")
    usage = message.get("usage")
    if isinstance(message_id, str) and isinstance(usage, dict):
        state.usage_by_message[message_id] = usage
    for block in message.get("content", []) or []:
        btype = block.get("type")
        if btype not in _KNOWN_ASSISTANT_BLOCKS:
            raise ValueError(f"unrecognized assistant content block {btype!r}")
        if btype == "text":
            state.texts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            call_id = str(block.get("id"))
            state.tool_names[call_id] = _tool_ir_name(str(block.get("name", "")))
            state.tool_args[call_id] = block.get("input", {}) or {}
            state.tool_order.append(call_id)
        # "thinking" carries no scored content.


def _apply_user_event(event: dict[str, Any], state: _TurnState) -> None:
    message = event.get("message", {}) or {}
    content = message.get("content")
    if not isinstance(content, list):
        # A plain-string content is the injected prompt (the turn boundary
        # itself, or a mid-turn coordinator message) — no tool or text
        # content to extract.
        return
    for block in content:
        btype = block.get("type")
        if btype not in _KNOWN_USER_BLOCKS:
            raise ValueError(f"unrecognized user content block {btype!r}")
        tool_use_id = str(block.get("tool_use_id"))
        result = block.get("content")
        state.tool_results[tool_use_id] = "" if result is None else str(result)


def _dedup_tokens(
    usage_by_message: dict[str, dict[str, Any]],
) -> tuple[int | None, int | None]:
    """Sum each message id's TERMINAL usage (the caller already deduped by
    overwriting per id in arrival order) across the turn. Zero/zero maps to
    None/None: a turn with no observed tokens carries no token counts, the
    same convention ``opencode_adapter`` uses for Arm 0's unbilled turns."""
    input_sum = sum(
        int(usage.get("input_tokens", 0) or 0) for usage in usage_by_message.values()
    )
    output_sum = sum(
        int(usage.get("output_tokens", 0) or 0) for usage in usage_by_message.values()
    )
    if input_sum == 0 and output_sum == 0:
        return None, None
    return input_sum, output_sum


def turn_from_events(events: list[dict[str, Any]], *, index: int, prompt: str) -> Turn:
    """Build one :class:`Turn` from one turn's ordered subagent events."""
    state = _TurnState()
    timestamps: list[datetime] = []

    for event in events:
        etype = event.get("type")
        if etype not in _KNOWN_EVENT_TYPES:
            raise ValueError(f"unrecognized subagent event type {etype!r}")

        timestamp = event.get("timestamp")
        if isinstance(timestamp, str):
            timestamps.append(datetime.fromisoformat(timestamp))

        if etype == "assistant":
            _apply_assistant_event(event, state)
        elif etype == "user":
            _apply_user_event(event, state)
        # "attachment" events (deferred-tool listings, skill listings) carry
        # nothing the IR scores.

    tool_calls = tuple(
        _tool_call(
            state.tool_names[call_id],
            state.tool_args[call_id],
            state.tool_results.get(call_id, ""),
        )
        for call_id in state.tool_order
    )
    input_tokens, output_tokens = _dedup_tokens(state.usage_by_message)

    wall_seconds: float | None = None
    if len(timestamps) >= 2:
        wall_seconds = (max(timestamps) - min(timestamps)).total_seconds()

    return Turn(
        index=index,
        prompt=prompt,
        assistant_text="\n".join(state.texts),
        tool_calls=tool_calls,
        wall_seconds=wall_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def turn_from_jsonl(jsonl_text: str, *, index: int, prompt: str) -> Turn:
    """Parse one turn's raw JSONL slice and build a :class:`Turn`."""
    return turn_from_events(parse_events(jsonl_text), index=index, prompt=prompt)


def transcript_from_run(
    arm: str, jsonl_text: str, prompts: tuple[str, ...]
) -> Transcript:
    """Split one subagent run's full transcript (ONE continuing conversation,
    ONE file) into turns at each injected-prompt boundary and assemble a
    :class:`Transcript`, numbered from 1. ``prompts`` must match the number
    of turns found — a mismatch raises rather than silently misaligning
    prompts to turns."""
    turn_slices = split_turns(parse_events(jsonl_text))
    turns = tuple(
        turn_from_events(events, index=i, prompt=prompt)
        for i, (events, prompt) in enumerate(
            zip(turn_slices, prompts, strict=True), start=1
        )
    )
    return Transcript(arm=arm, turns=turns)
