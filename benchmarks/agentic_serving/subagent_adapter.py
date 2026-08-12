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
one-file-per-turn), so this module also splits a run into turns.

Turn boundaries (capture README finding 4): the PRIMARY rule is the first
occurrence of a new ``promptId`` on a ``user``-typed event. A ``user``-typed
event with a plain-STRING ``message.content`` is kept as an independent
CROSS-CHECK, because it is what every real injected prompt looks like too —
but it is also what an in-turn interruption notice
("[Request interrupted by user]") looks like, and that notice REUSES the
current turn's promptId rather than starting a new one. When the two rules
disagree, :func:`split_turns` raises rather than guessing: silently
following the string rule alone would either misalign every later turn
against its oracle, or — if the phantom boundary lands at the end of a
transcript — invent an extra completed turn and MASK a genuine death. That
phantom class is unrepresentable today, by design, until it needs handling.

Top-level event ``type`` is one of ``user`` / ``assistant`` / ``attachment``.
``assistant`` message content blocks are ``thinking`` (skipped — no scored
content), ``tool_use`` (``{id, name, input}``), or ``text``; a missing
``message``, a null ``content``, or a string ``content`` on an assistant
event all RAISE (the first two used to silently produce an empty turn, the
third used to escape as a bare ``AttributeError`` from iterating a string's
characters instead of this module's own ``ValueError``). Tool names observed
in both committed runs: ``Write`` ``{file_path, content}``, ``Bash``
``{command, description}``, ``Read`` ``{file_path}``, and ``Edit``
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
failure this instrument exists to avoid. The same applies at TOOL
granularity: a ``tool_result`` whose ``tool_use_id`` matches no observed
call (orphaned), a ``tool_use`` that never gets a linked ``tool_result``
(unlinked — the realistic shape a dead-mid-tool-call subagent leaves), or a
``tool_result`` with non-string ``content`` (documented as verbatim
elsewhere; a structured value would otherwise get silently ``str()``-repr'd)
all RAISE rather than producing a ``ToolCall`` with an empty or mangled
``result_text`` that a downstream honesty check could misread as "verified,
but nothing observed" instead of "this stream is unsafe to score."

Usage/token accounting: multiple JSONL lines share one assistant
``message.id`` (streaming increments — a growing usage snapshot per content
block emitted), so usage is deduped by ``message.id``, keeping only the
LAST (terminal, cumulative) usage seen for each id, then summed across the
turn's distinct ids — never summed per line, which would double- (or
sextuple-) count. Cache tokens (``cache_creation_input_tokens`` /
``cache_read_input_tokens``) are captured on the Turn, never discarded:
across haiku-run1 they dwarf the fresh-token counts (122 summed fresh input
tokens against roughly 1.34M cache tokens for the whole run), so excluding
them from cost — the previous design — priced the run at roughly a fifth of
its real API cost, and specifically favors whichever arm retains the
longest conversation context (this arm's continuing-conversation construct),
not a comparator-neutral simplification. Pricing them is
:mod:`benchmarks.agentic_serving.metrics`'s job: ``Pricing`` carries OPTIONAL
cache rate fields, and a caller that omits them gets a cost figure flagged
as a lower bound (``metrics.turn_cost_excludes_cache``), never a silently
short number.

Wall-clock zero point (documented next to ``opencode_adapter``'s matching
note): a turn's ``wall_seconds`` starts at the first ASSISTANT event's
timestamp, not the injected-prompt boundary. The boundary event's own
timestamp includes prompt-arrival / coordinator-dispatch latency that
``opencode_adapter``'s per-turn stream never logs in the first place (its
first event already IS generation start) — starting from assistant activity
keeps the two arms' wall-clock definitions comparable instead of inflating
this arm's numbers by the (measured) 20-25% that latency otherwise adds.

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


def _string_content_boundary_indices(events: list[dict[str, Any]]) -> list[int]:
    """CROSS-CHECK for :func:`_promptid_boundary_indices`: a ``user``-typed
    event with a plain-STRING ``message.content``. True for every real
    injected prompt, but ALSO true for a mid-turn interruption notice that
    reuses the current turn's promptId — see the module docstring."""
    return [
        i
        for i, event in enumerate(events)
        if event.get("type") == "user"
        and isinstance(event.get("message", {}).get("content"), str)
    ]


def _promptid_boundary_indices(events: list[dict[str, Any]]) -> list[int]:
    """The PRIMARY boundary rule (capture README finding 4): a new turn
    starts at the first ``user``-typed event to carry a given ``promptId``.
    A tool-result event, or an interruption notice, reuses the CURRENT
    turn's promptId and is never a boundary under this rule, unlike the
    string-content rule above."""
    seen: set[str] = set()
    boundaries: list[int] = []
    for i, event in enumerate(events):
        if event.get("type") != "user":
            continue
        prompt_id = event.get("promptId")
        if not isinstance(prompt_id, str):
            continue
        if prompt_id not in seen:
            seen.add(prompt_id)
            boundaries.append(i)
    return boundaries


def split_turns(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split one run's full ordered event stream into per-turn slices at
    each injected-prompt boundary. A stream with no boundary (empty, or
    truncated before the first prompt survived parsing) splits into zero
    turns.

    Raises when the promptId-based rule and the string-content cross-check
    disagree (see the module docstring), or when any event precedes the
    first boundary — nothing should exist before the first injected prompt,
    and silently dropping such events would hide a schema surprise the same
    way silently mis-splitting would.
    """
    by_promptid = _promptid_boundary_indices(events)
    by_string = _string_content_boundary_indices(events)
    if by_promptid != by_string:
        raise ValueError(
            "turn-boundary rules disagree: promptId-based boundaries "
            f"{by_promptid} != string-content boundaries {by_string} -- "
            "refusing to guess which is right (see split_turns docstring)"
        )
    boundaries = by_promptid
    if not boundaries:
        return []
    if boundaries[0] != 0:
        raise ValueError(
            f"{boundaries[0]} event(s) precede the first turn boundary at "
            f"index {boundaries[0]}"
        )
    ends = boundaries[1:] + [len(events)]
    return [events[start:end] for start, end in zip(boundaries, ends, strict=True)]


def _tool_ir_name(raw_name: str) -> str:
    try:
        return _TOOL_NAMES[raw_name]
    except KeyError:
        raise ValueError(
            f"unmapped subagent tool {raw_name!r} (known: {sorted(_TOOL_NAMES)})"
        ) from None


def _tool_call(
    name: str, args: dict[str, Any], result_text: str, *, is_error: bool
) -> ToolCall:
    command = args.get("command") if name in _RUN_TOOLS else None
    path = args.get("file_path")
    return ToolCall(
        name=name,
        command=command,
        path=path,
        result_text=result_text,
        is_error=is_error,
    )


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
        self.tool_errors: dict[str, bool] = {}
        self.usage_by_message: dict[str, dict[str, Any]] = {}


def _apply_assistant_event(event: dict[str, Any], state: _TurnState) -> None:
    message = event.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"assistant event has no message object: {message!r}")
    message_id = message.get("id")
    usage = message.get("usage")
    if isinstance(message_id, str) and isinstance(usage, dict):
        state.usage_by_message[message_id] = usage
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError(f"assistant message.content must be a list, got {content!r}")
    for block in content:
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
    if isinstance(content, str):
        # The injected prompt (the turn boundary itself, or an interruption
        # notice reusing the current promptId) -- no tool or text content.
        return
    if not isinstance(content, list):
        raise ValueError(
            f"user message.content must be a string or list, got {content!r}"
        )
    for block in content:
        btype = block.get("type")
        if btype not in _KNOWN_USER_BLOCKS:
            raise ValueError(f"unrecognized user content block {btype!r}")
        tool_use_id = str(block.get("tool_use_id"))
        if tool_use_id not in state.tool_names:
            raise ValueError(
                f"orphaned tool_result for unknown tool_use_id {tool_use_id!r}"
            )
        result = block.get("content")
        if not isinstance(result, str):
            raise ValueError(f"tool_result content must be a string, got {result!r}")
        state.tool_results[tool_use_id] = result
        state.tool_errors[tool_use_id] = bool(block.get("is_error", False))


def _dedup_usage(
    usage_by_message: dict[str, dict[str, Any]],
) -> tuple[int | None, int | None, int | None, int | None]:
    """Sum each message id's TERMINAL usage (the caller already deduped by
    overwriting per id in arrival order) across the turn.

    ``input``/``output`` map zero/zero to None/None: a turn with no observed
    tokens carries no token counts, the same convention ``opencode_adapter``
    uses for Arm 0's unbilled turns. Cache counts follow a SEPARATE rule:
    they are ``None`` only when no assistant message contributed usage data
    at all (an untimed/empty turn), never collapsed to ``None`` just because
    they happen to be genuinely zero — a turn that reports zero cache usage
    is a different fact than a turn with no cache accounting whatsoever.
    """
    if not usage_by_message:
        return None, None, None, None
    input_sum = sum(
        int(usage.get("input_tokens", 0) or 0) for usage in usage_by_message.values()
    )
    output_sum = sum(
        int(usage.get("output_tokens", 0) or 0) for usage in usage_by_message.values()
    )
    cache_creation_sum = sum(
        int(usage.get("cache_creation_input_tokens", 0) or 0)
        for usage in usage_by_message.values()
    )
    cache_read_sum = sum(
        int(usage.get("cache_read_input_tokens", 0) or 0)
        for usage in usage_by_message.values()
    )
    if input_sum == 0 and output_sum == 0:
        return None, None, cache_creation_sum, cache_read_sum
    return input_sum, output_sum, cache_creation_sum, cache_read_sum


def turn_from_events(events: list[dict[str, Any]], *, index: int, prompt: str) -> Turn:
    """Build one :class:`Turn` from one turn's ordered subagent events."""
    state = _TurnState()
    all_timestamps: list[datetime] = []
    assistant_timestamps: list[datetime] = []

    for event in events:
        etype = event.get("type")
        if etype not in _KNOWN_EVENT_TYPES:
            raise ValueError(f"unrecognized subagent event type {etype!r}")

        timestamp = event.get("timestamp")
        if isinstance(timestamp, str):
            parsed = datetime.fromisoformat(timestamp)
            all_timestamps.append(parsed)
            if etype == "assistant":
                assistant_timestamps.append(parsed)

        if etype == "assistant":
            _apply_assistant_event(event, state)
        elif etype == "user":
            _apply_user_event(event, state)
        # "attachment" events (deferred-tool listings, skill listings) carry
        # nothing the IR scores.

    unlinked = [
        call_id for call_id in state.tool_order if call_id not in state.tool_results
    ]
    if unlinked:
        raise ValueError(
            f"unlinked tool_use call(s), no tool_result observed: {unlinked!r}"
        )

    tool_calls = tuple(
        _tool_call(
            state.tool_names[call_id],
            state.tool_args[call_id],
            state.tool_results[call_id],
            is_error=state.tool_errors[call_id],
        )
        for call_id in state.tool_order
    )
    input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = (
        _dedup_usage(state.usage_by_message)
    )

    wall_seconds: float | None = None
    if assistant_timestamps and all_timestamps:
        wall_seconds = (max(all_timestamps) - min(assistant_timestamps)).total_seconds()

    return Turn(
        index=index,
        prompt=prompt,
        assistant_text="\n".join(state.texts),
        tool_calls=tool_calls,
        wall_seconds=wall_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
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
