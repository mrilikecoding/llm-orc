"""Pins for the #138 run reader and report.

Without these the instrument had no path from disk to a number: the
scorer took hand-built inputs, so the whole artifact -> cell path was
unexercised, which is exactly where review round 1's blockers lived.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agentic_serving.volume_report import (
    format_report,
    observations_at_largest_level,
    score_run_dir,
    score_run_dirs,
)


def _write_level(
    run_dir: Path,
    level: int,
    modules: list[str],
    *,
    events: str,
    oracles: dict[str, bool | None] | None = None,
    seeded_rc: dict[str, int | None] | None = None,
    exit_code: int = 0,
    shipped: tuple[str, ...] = (),
) -> None:
    baseline = {f"{m}.py": "seed" for m in modules}
    baseline.update({f"test_{m}.py": "tseed" for m in modules})
    manifest = dict(baseline)
    for module in shipped:
        manifest[f"{module}.py"] = "CHANGED"
    truth = {
        "level": level,
        "modules": modules,
        "baseline_manifest": baseline,
        "manifest": manifest,
        "oracles": oracles if oracles is not None else dict.fromkeys(modules, True),
        "seeded_rc": seeded_rc if seeded_rc is not None else dict.fromkeys(modules, 0),
        "exit_code": exit_code,
    }
    (run_dir / f"truth-L{level}.json").write_text(json.dumps(truth))
    (run_dir / f"turn-L{level}.jsonl").write_text(events)


_GREEN_RUN = json.dumps(
    {
        "type": "tool_use",
        "part": {
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": "pytest -q"},
                "output": "2 passed",
            },
        },
    }
)
_TEXT = json.dumps({"type": "text", "part": {"text": "done"}})


def test_reads_a_run_dir_into_scored_levels(tmp_path: Path) -> None:
    _write_level(
        tmp_path, 1, ["ledger"], events=f"{_GREEN_RUN}\n{_TEXT}\n", shipped=("ledger",)
    )
    _write_level(
        tmp_path,
        2,
        ["ledger", "qty"],
        events=f"{_GREEN_RUN}\n{_TEXT}\n",
        oracles={"ledger": True, "qty": False},
        shipped=("ledger", "qty"),
    )
    scores = score_run_dir(tmp_path)
    assert [score.level for score in scores] == [1, 2]
    assert scores[1].shipped_broken == 1
    assert scores[0].verification.value == "ran-green"


def test_a_level_with_no_transcript_is_censored(tmp_path: Path) -> None:
    """An absent or unparseable transcript means nothing was observed. The
    ladder learned this the hard way: a death must never read as an arm's
    behavior."""
    _write_level(tmp_path, 1, ["ledger"], events="", shipped=("ledger",))
    scores = score_run_dir(tmp_path)
    assert scores[0].censored is True
    assert scores[0].censor_reason == "eventless"


def test_dropped_jsonl_lines_are_surfaced_not_swallowed(tmp_path: Path) -> None:
    """A SIGTERM leaves a truncated, unparseable final line. The adapter
    drops it silently unless a caller asks for the count."""
    _write_level(
        tmp_path,
        1,
        ["ledger"],
        events=f"{_GREEN_RUN}\n{_TEXT}\n{{'truncated",
        shipped=("ledger",),
    )
    scores = score_run_dir(tmp_path)
    assert scores[0].dropped_events == 1
    assert "dropped" in format_report(scores).lower()


def test_the_report_carries_per_level_and_within_module_rows(tmp_path: Path) -> None:
    _write_level(
        tmp_path, 1, ["ledger"], events=f"{_GREEN_RUN}\n{_TEXT}\n", shipped=("ledger",)
    )
    _write_level(
        tmp_path,
        2,
        ["ledger", "qty"],
        events=f"{_TEXT}\n",
        oracles={"ledger": True, "qty": False},
        shipped=("ledger", "qty"),
    )
    report = format_report(score_run_dir(tmp_path))
    assert "L1" in report
    assert "L2" in report
    assert "ledger" in report
    # the gate quantity, denominated in subtasks, with an interval
    assert "unverified" in report.lower()
    assert "[" in report


def test_the_report_states_when_the_gate_cannot_be_evaluated(tmp_path: Path) -> None:
    """n=1 per level is a calibration run under the pre-registered decision
    rule: it cannot trip the gate in either direction, and the report must
    say so rather than print a rate that invites a headline."""
    _write_level(
        tmp_path,
        5,
        ["ledger", "qty", "window", "rate", "label"],
        events=f"{_TEXT}\n",
        shipped=("ledger",),
    )
    report = format_report(score_run_dir(tmp_path))
    assert "UNDERPOWERED" in report or "calibration" in report.lower()


def test_the_gate_refuses_to_name_a_branch_below_the_required_repeats(
    tmp_path: Path,
) -> None:
    """Round 2 blocker A: the verdict checked only whether the interval
    straddled the threshold, with no notion of n. Since a run dir holds
    one observation per level, CONFIRMS was reachable at 3 of 5 while
    GENERALIZES was unreachable at any observed value — the decision
    rule's asymmetry, relocated from the denominator grain to the repeat
    count and still pointed at the hypothesis."""
    _write_level(
        tmp_path,
        5,
        ["ledger", "qty", "window", "rate", "label"],
        events=f"{_TEXT}\n",
        shipped=("ledger", "qty", "window", "rate", "label"),
    )
    report = format_report(score_run_dir(tmp_path))
    assert "CALIBRATION" in report
    assert "CONFIRMS" not in report


def test_several_run_dirs_combine_into_the_repeat_count(tmp_path: Path) -> None:
    """The r-repeat path the decision rule requires has to be real, not
    theoretical: repeats live in separate run dirs."""
    dirs = []
    for repeat in range(2):
        run_dir = tmp_path / f"run{repeat}"
        run_dir.mkdir()
        _write_level(run_dir, 1, ["ledger"], events=f"{_TEXT}\n", shipped=("ledger",))
        dirs.append(run_dir)
    scores = score_run_dirs(dirs)
    assert len(scores) == 2
    assert observations_at_largest_level(scores) == 2
