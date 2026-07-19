"""Pins for the Claude Code subagent adapter (#131, Arm 2).

Two layers: synthetic events pinning each mapping rule and trap, plus the
REAL committed capture (docs/plans/2026-07-17-arm2-subagent-captures/) as an
end-to-end fixture — the schema is pinned by observed data, never by memory
of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.agentic_serving import subagent_adapter as sa

CAPTURE = (
    Path(__file__).resolve().parents[3]
    / "docs/plans/2026-07-17-arm2-subagent-captures/probe-2turn-transcript.jsonl"
)


def _user_prompt(text: str) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}})


def _assistant(
    blocks: list[dict], message_id: str = "msg_1", usage: dict | None = None
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "id": message_id,
                "content": blocks,
                "usage": usage or {},
            },
        }
    )


def _tool_result(call_id: str, content: object) -> str:
    return json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": content,
                    }
                ],
            },
        }
    )


def test_real_capture_parses_to_two_turns() -> None:
    transcript = sa.transcript_from_jsonl("arm2-haiku", CAPTURE.read_text())
    assert len(transcript.turns) == 2
    turn1, turn2 = transcript.turns
    # Turn 2's prompt is the battery text with the coordinator wrapper GONE.
    assert turn2.prompt == (
        "add a complete_todo function to todo.py that marks a todo done"
    )
    # The probe's turn 1 shipped todo.py via Write; the IR must see the path.
    writes = [c for c in turn1.tool_calls if c.name == "write"]
    assert writes
    assert writes[0].path is not None
    assert writes[0].path.endswith("todo.py")
    # Every observed tool call paired with its result text.
    assert all(c.result_text for c in turn1.tool_calls)
    # Usage dedup: real tokens, but far below the naive per-event sum.
    assert turn1.input_tokens is not None
    assert turn1.output_tokens is not None
    assert turn1.output_tokens > 0
    assert turn1.wall_seconds is not None
    assert turn1.wall_seconds > 0


def test_real_capture_prompt_verification_catches_divergence() -> None:
    with pytest.raises(sa.SubagentAdapterError, match="diverge"):
        sa.transcript_from_jsonl(
            "arm2-haiku",
            CAPTURE.read_text(),
            expected_prompts=["only one expected prompt"],
        )


def test_coordinator_wrapper_stripped_only_as_prefix() -> None:
    body = "explain how todo.py stores its state"
    wrapped = sa._COORDINATOR_PREFIX + body
    events, _ = sa.parse_events_counting_drops(
        "\n".join([_user_prompt(wrapped), _assistant([])])
    )
    (turn,) = sa.split_turns(events)
    assert turn[0] == body
    # The phrase INSIDE a prompt is untouched (only a leading wrapper strips).
    inline = "say: " + sa._COORDINATOR_PREFIX
    events, _ = sa.parse_events_counting_drops(_user_prompt(inline))
    (turn,) = sa.split_turns(events)
    assert turn[0] == inline


def test_usage_dedupes_by_message_id_last_wins() -> None:
    lines = [
        _user_prompt("p"),
        # Streaming: same message id, evolving usage; naive summing reads 1+289.
        _assistant([], "msg_a", {"input_tokens": 8, "output_tokens": 1}),
        _assistant([], "msg_a", {"input_tokens": 8, "output_tokens": 289}),
        _assistant([], "msg_b", {"input_tokens": 5, "output_tokens": 7}),
    ]
    events, _ = sa.parse_events_counting_drops("\n".join(lines))
    (segment,) = sa.split_turns(events)
    turn = sa.turn_from_events(segment[1], index=1, prompt=segment[0])
    assert turn.input_tokens == 8 + 5
    assert turn.output_tokens == 289 + 7


def test_unmapped_tool_name_raises_instead_of_zero_shipped() -> None:
    lines = [
        _user_prompt("p"),
        _assistant(
            [{"type": "tool_use", "id": "t1", "name": "MysteryTool", "input": {}}]
        ),
    ]
    events, _ = sa.parse_events_counting_drops("\n".join(lines))
    (segment,) = sa.split_turns(events)
    with pytest.raises(sa.SubagentAdapterError, match="MysteryTool"):
        sa.turn_from_events(segment[1], index=1, prompt=segment[0])


def test_bash_carries_command_write_carries_path_result_pairs() -> None:
    lines = [
        _user_prompt("p"),
        _assistant(
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "pytest -q", "description": "run"},
                },
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Write",
                    "input": {"file_path": "/ws/todo.py", "content": "x = 1\n"},
                },
            ]
        ),
        _tool_result("t1", "1 passed"),
        _tool_result("t2", [{"type": "text", "text": "File created"}]),
    ]
    events, _ = sa.parse_events_counting_drops("\n".join(lines))
    (segment,) = sa.split_turns(events)
    turn = sa.turn_from_events(segment[1], index=1, prompt=segment[0])
    bash, write = turn.tool_calls
    assert bash.name == "bash"
    assert bash.command == "pytest -q"
    assert bash.result_text == "1 passed"
    assert write.name == "write"
    assert write.path == "/ws/todo.py"
    assert write.command is None
    assert write.result_text == "File created"


def test_thinking_blocks_never_reach_assistant_text() -> None:
    lines = [
        _user_prompt("p"),
        _assistant(
            [
                {"type": "thinking", "thinking": "private chain"},
                {"type": "text", "text": "public answer"},
            ]
        ),
    ]
    events, _ = sa.parse_events_counting_drops("\n".join(lines))
    (segment,) = sa.split_turns(events)
    turn = sa.turn_from_events(segment[1], index=1, prompt=segment[0])
    assert turn.assistant_text == "public answer"
    assert "private chain" not in turn.assistant_text


def test_event_before_first_prompt_raises() -> None:
    events, _ = sa.parse_events_counting_drops(_assistant([]))
    with pytest.raises(sa.SubagentAdapterError, match="precedes"):
        sa.split_turns(events)


def test_attachment_events_are_skipped() -> None:
    lines = [
        _user_prompt("p"),
        json.dumps({"type": "attachment", "attachment": {"type": "x"}}),
        _assistant([{"type": "text", "text": "ok"}]),
    ]
    events, _ = sa.parse_events_counting_drops("\n".join(lines))
    (segment,) = sa.split_turns(events)
    turn = sa.turn_from_events(segment[1], index=1, prompt=segment[0])
    assert turn.assistant_text == "ok"


def test_half_written_line_drops_and_counts() -> None:
    text = _user_prompt("p") + "\n" + '{"type": "assistant", "mess'
    events, dropped = sa.parse_events_counting_drops(text)
    assert len(events) == 1
    assert dropped == 1


def test_no_usage_maps_to_none_tokens() -> None:
    lines = [_user_prompt("p"), _assistant([{"type": "text", "text": "hi"}])]
    events, _ = sa.parse_events_counting_drops("\n".join(lines))
    (segment,) = sa.split_turns(events)
    turn = sa.turn_from_events(segment[1], index=1, prompt=segment[0])
    assert turn.input_tokens is None
    assert turn.output_tokens is None
