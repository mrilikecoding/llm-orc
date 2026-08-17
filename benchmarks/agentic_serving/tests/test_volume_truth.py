"""Pins for the #138 shared truth capture.

Truth capture is the arm-blind substrate: it judges what reached disk,
identically for every arm. One implementation, shared by every driver,
so the substrate cannot drift between arms (the ladder's rule).
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agentic_serving.tests.test_volume_oracles import (
    CORRECT,
    TEACHING_TO_THE_TEST,
)
from benchmarks.agentic_serving.transcript import ToolCall, Turn
from benchmarks.agentic_serving.volume_fixture import write_fixture
from benchmarks.agentic_serving.volume_score import Verification, score_level
from benchmarks.agentic_serving.volume_truth import capture_truth


def test_untouched_workspace_shows_nothing_shipped(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=2)
    truth = capture_truth(workspace, level=2, exit_code=0)
    assert truth["manifest"] == truth["baseline_manifest"]
    assert truth["seeded_rc"] == {"ledger": 1, "qty": 1}
    assert truth["oracles"] == {"ledger": False, "qty": False}


def test_correct_fixes_are_recorded_green_end_to_end(tmp_path: Path) -> None:
    """The composition pin: fixture -> arm edit -> truth -> score. This is
    the only place the three pieces are proven to agree."""
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=3)
    for module in ("ledger", "qty", "window"):
        (workspace / f"{module}.py").write_text(CORRECT[module])

    truth = capture_truth(workspace, level=3, exit_code=0)
    assert truth["seeded_rc"] == {"ledger": 0, "qty": 0, "window": 0}
    assert truth["oracles"] == {"ledger": True, "qty": True, "window": True}

    score = score_level(
        truth,
        Turn(
            index=1,
            prompt="fix the bugs",
            assistant_text="fixed",
            tool_calls=(
                ToolCall(name="bash", command="pytest -q", result_text="6 passed"),
            ),
        ),
    )
    assert score.shipped == 3
    assert score.shipped_correct == 3
    assert score.shipped_broken == 0
    assert score.verification is Verification.RAN_GREEN


def test_a_hardcode_reaches_the_scorer_as_shipped_broken(tmp_path: Path) -> None:
    """The instrument's reason for existing, proven end to end: a fix that
    satisfies the visible seeded test lands in the shipped-broken cell,
    not the correct one."""
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=2)
    (workspace / "ledger.py").write_text(CORRECT["ledger"])
    (workspace / "qty.py").write_text(TEACHING_TO_THE_TEST["qty"])

    truth = capture_truth(workspace, level=2, exit_code=0)
    assert truth["seeded_rc"]["qty"] == 0
    assert truth["oracles"]["qty"] is False

    score = score_level(truth, Turn(index=1, prompt="p", assistant_text="done"))
    outcomes = {module.module: module for module in score.modules}
    assert outcomes["qty"].shipped is True
    assert outcomes["qty"].correct is False
    assert score.shipped_broken == 1


def test_capture_never_writes_into_the_live_workspace(tmp_path: Path) -> None:
    """Probes and truth pytest CALL arm-authored code, which has side
    effects. Anything they leave behind would be attributed to the arm in
    the manifest, so they run in a throwaway copy."""
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=1)
    (workspace / "ledger.py").write_text(
        CORRECT["ledger"] + '\nopen("side-effect.txt", "w").write("x")\n'
    )
    before = sorted(p.name for p in workspace.iterdir())
    capture_truth(workspace, level=1, exit_code=0)
    assert sorted(p.name for p in workspace.iterdir()) == before


def test_exit_code_rides_through_for_the_censoring_channel(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=1)
    truth = capture_truth(workspace, level=1, exit_code=124)
    assert truth["exit_code"] == 124
    assert score_level(truth, Turn(1, "p", "")).timeout_censored is True


def test_truth_is_written_as_json_when_an_out_dir_is_given(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=1)
    out = tmp_path / "out"
    out.mkdir()
    truth = capture_truth(workspace, level=1, exit_code=0, out_dir=out)
    written = json.loads((out / "truth-L1.json").read_text())
    assert written == truth
