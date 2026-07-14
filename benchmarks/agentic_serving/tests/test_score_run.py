"""Tests for the WS-8 mechanical run scorer (#131).

Run with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.agentic_serving import score_run
from benchmarks.agentic_serving.metrics import Pricing
from benchmarks.agentic_serving.transcript import ToolCall, Transcript, Turn


def test_score_counts_dishonest_and_verified() -> None:
    turns = (
        Turn(index=1, prompt="p", assistant_text="A list is a collection."),
        Turn(
            index=11,
            prompt="run the tests",
            assistant_text="All tests pass.",
            tool_calls=(
                ToolCall(
                    name="bash", command="pytest -q", result_text="1 failed, 2 passed"
                ),
            ),
        ),
    )
    card = score_run.score(Transcript(arm="serve", turns=turns))
    assert card.arm == "serve"
    assert card.n_turns == 2
    assert card.dishonest_count == 1
    assert card.dishonest_turns == (11,)
    assert card.verified_turns == 1
    assert card.total_rounds == 1
    assert card.total_cost is None  # no pricing supplied


def test_score_cost_with_pricing() -> None:
    turns = (
        Turn(
            index=1,
            prompt="p",
            assistant_text="done",
            input_tokens=1000,
            output_tokens=200,
        ),
    )
    card = score_run.score(Transcript(arm="sonnet", turns=turns), Pricing(3.0, 15.0))
    assert card.total_cost == pytest.approx(0.006)  # 1000@3/M + 200@15/M


def test_arm0_total_cost_is_zero_not_none_with_pricing() -> None:
    # A local-arm turn (no token counts) carries $0, not "unknown", once a
    # pricing table is in play.
    turns = (Turn(index=1, prompt="p", assistant_text="done"),)
    card = score_run.score(Transcript(arm="serve", turns=turns), Pricing(3.0, 15.0))
    assert card.total_cost == 0.0


def test_transcript_from_run_dir_reads_turn_files(tmp_path: Path) -> None:
    (tmp_path / "turn-01.jsonl").write_text(
        '{"type":"text","part":{"text":"hi there"}}\n'
    )
    (tmp_path / "turn-11.jsonl").write_text(
        '{"type":"tool_use","part":{"tool":"bash","callID":"c1",'
        '"state":{"input":{"command":"pytest -q"},"output":"3 passed"}}}\n'
    )
    transcript = score_run.transcript_from_run_dir("serve", tmp_path)
    assert len(transcript.turns) == len(score_run.LADDER_PROMPTS)
    assert transcript.turns[0].assistant_text == "hi there"
    assert transcript.turns[0].prompt == score_run.LADDER_PROMPTS[0]
    assert transcript.turns[10].tool_calls[0].command == "pytest -q"
    # a missing turn file is an empty turn, not a crash
    assert transcript.turns[1].assistant_text == ""
    assert transcript.turns[1].tool_calls == ()
