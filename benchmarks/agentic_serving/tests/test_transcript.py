"""Unit tests for the arm-agnostic transcript IR (#131 WS-8; deterministic).

Run with the llm_orc coverage gate disabled (the benchmark is not llm_orc):
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn


class TestToolCall:
    def test_defaults_are_empty(self) -> None:
        call = ToolCall(name="read")
        assert call.command is None
        assert call.path is None
        assert call.result_text == ""

    def test_carries_command_and_result(self) -> None:
        call = ToolCall(name="bash", command="pytest -q", result_text="3 passed")
        assert call.command == "pytest -q"
        assert call.result_text == "3 passed"


class TestTurn:
    def test_defaults_have_no_tool_calls_or_timing(self) -> None:
        turn = Turn(index=1, prompt="build it", assistant_text="done")
        assert turn.tool_calls == ()
        assert turn.wall_seconds is None
        assert turn.input_tokens is None
        assert turn.output_tokens is None

    def test_carries_tool_calls_in_order(self) -> None:
        first = ToolCall(name="read", path="test_buggy.py")
        second = ToolCall(name="write", path="buggy.py")
        turn = Turn(
            index=13,
            prompt="fix the bug",
            assistant_text="fixed",
            tool_calls=(first, second),
        )
        assert turn.tool_calls == (first, second)


class TestTranscript:
    def test_defaults_to_no_turns(self) -> None:
        transcript = Transcript(arm="serve")
        assert transcript.turns == ()

    def test_carries_arm_name_and_turns(self) -> None:
        turn = Turn(index=1, prompt="p", assistant_text="a")
        transcript = Transcript(arm="haiku-4.5", turns=(turn,))
        assert transcript.arm == "haiku-4.5"
        assert transcript.turns == (turn,)
