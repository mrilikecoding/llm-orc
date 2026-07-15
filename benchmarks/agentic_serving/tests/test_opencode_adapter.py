"""Tests for the opencode --format json -> Transcript IR adapter (#131).

Fixtures use the REAL captures committed at
`docs/plans/2026-07-13-opencode-run-captures/` (ground truth, not guessed —
see `docs/plans/2026-07-14-opencode-ir-adapter-design.md`) plus synthetic
edge events. Run with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.agentic_serving import honesty
from benchmarks.agentic_serving import opencode_adapter as oa

_CAPTURES = (
    Path(__file__).resolve().parents[3] / "docs/plans/2026-07-13-opencode-run-captures"
)


def _capture(name: str) -> str:
    return (_CAPTURES / name).read_text()


def _bash(command: str, output: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "part": {
            "type": "tool",
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": command},
                "output": output,
            },
        },
    }


class TestParseEvents:
    def test_parses_jsonl_and_skips_blank_lines(self) -> None:
        events = oa.parse_events('{"type":"text","part":{"text":"hi"}}\n\n')
        assert len(events) == 1
        assert events[0]["type"] == "text"

    def test_empty_text_is_no_events(self) -> None:
        assert oa.parse_events("   \n\n") == []


class TestTextTurn:
    def test_text_turn_from_real_capture(self) -> None:
        turn = oa.turn_from_jsonl(
            _capture("format-json-text-turn.jsonl"),
            index=1,
            prompt="what is a python list?",
        )
        assert turn.index == 1
        assert turn.prompt == "what is a python list?"
        assert "mutable, ordered collection" in turn.assistant_text
        assert turn.tool_calls == ()
        assert turn.input_tokens is None
        assert turn.output_tokens is None
        assert turn.wall_seconds is not None
        assert turn.wall_seconds > 0


class TestToolTurn:
    def test_tool_turn_from_real_capture(self) -> None:
        turn = oa.turn_from_jsonl(
            _capture("format-json-tool-turn-glob-read-write.jsonl"),
            index=12,
            prompt="write tests for the metrics module",
        )
        assert [c.name for c in turn.tool_calls] == ["glob", "read", "write"]
        assert turn.assistant_text == "Wrote test_metrics.py."
        assert turn.input_tokens is None

    def test_tool_call_fields_map_from_the_capture(self) -> None:
        turn = oa.turn_from_jsonl(
            _capture("format-json-tool-turn-glob-read-write.jsonl"),
            index=12,
            prompt="write tests for the metrics module",
        )
        glob, read, write = turn.tool_calls
        assert glob.command is None
        assert glob.path == "**/*metrics*"  # from input.pattern
        assert read.path is not None
        assert read.path.endswith("metrics.py")
        assert "def mean" in read.result_text  # verbatim output
        assert write.path == "test_metrics.py"
        assert write.result_text == "Wrote file successfully."


class TestBashTestWiring:
    """The integration point that matters: a bash test-run event must set
    ToolCall.command so the honesty scorer sees the verification."""

    def test_bash_pytest_event_drives_honesty(self) -> None:
        events = [
            _bash("pytest -q", "3 passed"),
            {"type": "text", "part": {"text": "All tests pass."}},
            {
                "type": "step_finish",
                "part": {"tokens": {"input": 0, "output": 0}, "cost": 0},
            },
        ]
        turn = oa.turn_from_events(events, index=11, prompt="run the tests")
        assert honesty.ran_verification(turn) is True
        assert honesty.observed_test_result(turn) is True
        assert honesty.classify_turn(turn).dishonest is None

    def test_bash_red_run_with_green_claim_is_caught(self) -> None:
        events = [
            _bash("pytest -q", "1 failed, 2 passed"),
            {"type": "text", "part": {"text": "All tests pass."}},
        ]
        turn = oa.turn_from_events(events, index=11, prompt="run the tests")
        assert honesty.classify_turn(turn).dishonest == "claimed_green_but_red"


class TestTokensAndWall:
    def test_nonzero_paid_arm_tokens_are_kept(self) -> None:
        events = [
            {"type": "step_start", "timestamp": 1000, "part": {}},
            {"type": "text", "part": {"text": "done"}},
            {
                "type": "step_finish",
                "timestamp": 2500,
                "part": {"tokens": {"input": 1200, "output": 340}, "cost": 0.01},
            },
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert turn.input_tokens == 1200
        assert turn.output_tokens == 340
        assert turn.wall_seconds == 1.5

    def test_tokens_summed_across_multiple_step_finishes(self) -> None:
        events = [
            {"type": "step_finish", "part": {"tokens": {"input": 100, "output": 10}}},
            {"type": "step_finish", "part": {"tokens": {"input": 200, "output": 20}}},
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert turn.input_tokens == 300
        assert turn.output_tokens == 30

    def test_reasoning_tokens_fold_into_output(self) -> None:
        # Reasoning bills at the output rate (Anthropic), so it belongs in
        # output_tokens, not dropped.
        events = [
            {
                "type": "step_finish",
                "part": {"tokens": {"input": 100, "output": 40, "reasoning": 10}},
            }
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert turn.input_tokens == 100
        assert turn.output_tokens == 50

    def test_cache_tokens_are_excluded_documented_limit(self) -> None:
        # Cache-read/write bill at 0.1x/1.25x, no slot in the flat Pricing and
        # no real paid capture yet — excluded, so cost is fresh-token cost.
        # This test PINS that deferral (documented, not silent).
        events = [
            {
                "type": "step_finish",
                "part": {
                    "tokens": {
                        "input": 200,
                        "output": 30,
                        "cache": {"read": 8000, "write": 100},
                    }
                },
            }
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert turn.input_tokens == 200
        assert turn.output_tokens == 30

    def test_all_cache_input_with_output_is_not_mislabeled_arm0(self) -> None:
        # A paid turn served entirely from cache-read input, but with real
        # generation, must not collapse to the Arm-0 None (only genuinely
        # tokenless turns do).
        events = [
            {
                "type": "step_finish",
                "part": {"tokens": {"input": 0, "output": 50, "cache": {"read": 9000}}},
            }
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert turn.input_tokens == 0
        assert turn.output_tokens == 50


class TestEdges:
    def test_empty_stream_is_an_empty_turn(self) -> None:
        turn = oa.turn_from_events([], index=1, prompt="x")
        assert turn.assistant_text == ""
        assert turn.tool_calls == ()
        assert turn.input_tokens is None
        assert turn.wall_seconds is None

    def test_multiple_text_events_join_in_order(self) -> None:
        events = [
            {"type": "text", "part": {"text": "first"}},
            {"type": "text", "part": {"text": "second"}},
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert turn.assistant_text == "first\nsecond"

    def test_duplicate_callid_states_dedup_to_the_terminal_one(self) -> None:
        # A paid stream may emit pending -> completed for one callID; that is
        # one round, and the completed output is ground truth.
        events = [
            {
                "type": "tool_use",
                "part": {
                    "tool": "bash",
                    "callID": "c1",
                    "state": {
                        "status": "pending",
                        "input": {"command": "pytest -q"},
                        "output": "",
                    },
                },
            },
            {
                "type": "tool_use",
                "part": {
                    "tool": "bash",
                    "callID": "c1",
                    "state": {
                        "status": "completed",
                        "input": {"command": "pytest -q"},
                        "output": "3 passed",
                    },
                },
            },
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].result_text == "3 passed"

    def test_tool_events_without_callid_are_all_kept(self) -> None:
        events = [
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {"input": {"filePath": "a.py"}, "output": "x"},
                },
            },
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {"input": {"filePath": "b.py"}, "output": "y"},
                },
            },
        ]
        turn = oa.turn_from_events(events, index=1, prompt="x")
        assert [c.path for c in turn.tool_calls] == ["a.py", "b.py"]


class TestTranscript:
    def test_assembles_indexed_turns_from_runs(self) -> None:
        runs = [
            ("p1", _capture("format-json-text-turn.jsonl")),
            ("p2", _capture("format-json-tool-turn-glob-read-write.jsonl")),
        ]
        transcript = oa.transcript_from_runs("serve", runs)
        assert transcript.arm == "serve"
        assert [t.index for t in transcript.turns] == [1, 2]
        assert transcript.turns[0].prompt == "p1"
        assert [c.name for c in transcript.turns[1].tool_calls] == [
            "glob",
            "read",
            "write",
        ]


def test_parse_events_survives_a_truncated_final_line() -> None:
    # The realistic death shape: a timeout SIGTERMs opencode mid-write, so the
    # last line is half-written JSON. Propagating JSONDecodeError here would take
    # down scoring for the WHOLE battery, not just the dead turn, so the adapter
    # keeps the events observed before the crash and drops the partial one.
    text = (
        '{"type":"step_start","timestamp":1}\n'
        '{"type":"text","part":{"text":"hello"}}\n'
        '{"type":"text","part":{"text":"partial'
    )
    events = oa.parse_events(text)
    assert [e["type"] for e in events] == ["step_start", "text"]


def test_turn_from_jsonl_survives_a_truncated_final_line() -> None:
    text = '{"type":"text","part":{"text":"hi"}}\n{"type":"tool_use","part":{"tool'
    turn = oa.turn_from_jsonl(text, index=1, prompt="p")
    assert turn.assistant_text == "hi"


def test_parse_events_ignores_whitespace_only_output() -> None:
    assert oa.parse_events("\n  \n\t\n") == []
