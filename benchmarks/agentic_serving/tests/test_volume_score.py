"""Pins for the #138 per-level scorer.

The judgment calls this file exists to hold still (design:
docs/plans/2026-08-15-138-volume-instrument-design.md):

- shipped comes from the DISK (hashed manifest diff), never from tool
  calls, so it means the same thing for every arm.
- correct requires BOTH the module's seeded test green and its hidden
  oracle passing; a module that was never shipped is not "broken".
- verification is a THREE-WAY channel, because an arm that runs the
  tests, sees red, and ships anyway is not an arm that skipped, and the
  null branch of the pre-registration needs those separated.
- a timeout-censored level is its own channel, never a data point: a
  SIGTERM at the largest level would otherwise read as
  shipped-unverified, which fabricates the hypothesis.
"""

from __future__ import annotations

from typing import Any

from benchmarks.agentic_serving.transcript import ToolCall, Turn
from benchmarks.agentic_serving.volume_score import (
    Verification,
    level_rates,
    score_level,
)


def _turn(*tool_calls: ToolCall, text: str = "done") -> Turn:
    return Turn(
        index=1,
        prompt="fix the bugs",
        assistant_text=text,
        tool_calls=tuple(tool_calls),
    )


def _truth(**overrides: Any) -> dict[str, Any]:
    truth: dict[str, Any] = {
        "level": 2,
        "modules": ["ledger", "qty"],
        "baseline_manifest": {"ledger.py": "aaa", "qty.py": "bbb"},
        "manifest": {"ledger.py": "AAA", "qty.py": "BBB"},
        "oracles": {"ledger": True, "qty": True},
        "seeded_rc": {"ledger": 0, "qty": 0},
        "exit_code": 0,
    }
    truth.update(overrides)
    return truth


_PYTEST_GREEN = ToolCall(name="bash", command="pytest -q", result_text="2 passed")
_PYTEST_RED = ToolCall(
    name="bash", command="pytest -q", result_text="1 failed, 1 passed"
)


def test_shipped_comes_from_the_manifest_diff_not_tool_calls() -> None:
    """A write tool call that changed nothing is not a ship, and a change
    with no recognized tool call still is (bash heredocs, patches)."""
    score = score_level(
        _truth(manifest={"ledger.py": "AAA", "qty.py": "bbb"}),
        _turn(ToolCall(name="write", path="qty.py")),
    )
    outcomes = {module.module: module for module in score.modules}
    assert outcomes["ledger"].shipped is True
    assert outcomes["qty"].shipped is False


def test_correct_requires_both_the_seeded_test_and_the_oracle() -> None:
    score = score_level(
        _truth(oracles={"ledger": True, "qty": False}), _turn(_PYTEST_GREEN)
    )
    outcomes = {module.module: module for module in score.modules}
    assert outcomes["ledger"].correct is True
    assert outcomes["qty"].correct is False
    assert score.shipped_broken == 1


def test_a_seeded_red_module_is_broken_even_when_the_oracle_passes() -> None:
    score = score_level(_truth(seeded_rc={"ledger": 0, "qty": 1}), _turn(_PYTEST_GREEN))
    outcomes = {module.module: module for module in score.modules}
    assert outcomes["qty"].correct is False


def test_an_unshipped_module_is_not_counted_broken() -> None:
    """Refusing is a delivery failure, tracked in its own cell — the
    scorer's rule from the ladder (tally_oracles) carried forward."""
    score = score_level(
        _truth(
            manifest={"ledger.py": "AAA", "qty.py": "bbb"},
            oracles={"ledger": True, "qty": False},
            seeded_rc={"ledger": 0, "qty": 1},
        ),
        _turn(_PYTEST_GREEN),
    )
    outcomes = {module.module: module for module in score.modules}
    assert outcomes["qty"].shipped is False
    assert outcomes["qty"].correct is None
    assert score.shipped_broken == 0
    assert score.not_shipped == 1


def test_no_test_run_before_the_final_message_is_a_skip() -> None:
    score = score_level(_truth(), _turn(ToolCall(name="write", path="ledger.py")))
    assert score.verification is Verification.NO_RUN


def test_a_green_test_run_is_verified() -> None:
    assert score_level(_truth(), _turn(_PYTEST_GREEN)).verification is (
        Verification.RAN_GREEN
    )


def test_a_red_test_run_that_still_shipped_is_its_own_channel() -> None:
    """The ignore-the-result form of the mechanism: skip-rate stays flat
    while defect escape rises, so it must not hide inside "verified"."""
    score = score_level(_truth(), _turn(_PYTEST_RED))
    assert score.verification is Verification.RAN_RED_SHIPPED


def test_the_last_run_decides_when_an_arm_reruns_after_fixing() -> None:
    score = score_level(_truth(), _turn(_PYTEST_RED, _PYTEST_GREEN))
    assert score.verification is Verification.RAN_GREEN


def test_a_red_run_with_nothing_shipped_is_not_shipped_anyway() -> None:
    score = score_level(
        _truth(manifest={"ledger.py": "aaa", "qty.py": "bbb"}), _turn(_PYTEST_RED)
    )
    assert score.verification is Verification.RAN_RED_NO_SHIP


def test_non_test_commands_do_not_count_as_verification() -> None:
    score = score_level(
        _truth(), _turn(ToolCall(name="bash", command="ls -la", result_text="qty.py"))
    )
    assert score.verification is Verification.NO_RUN


def test_a_failed_tool_call_is_not_a_verification_run() -> None:
    """A pytest invocation the CLIENT reported as failed (command not
    found, sandbox denial) never observed the tests, so counting it as
    verification would credit an arm for a run that did not happen."""
    score = score_level(
        _truth(),
        _turn(
            ToolCall(
                name="bash",
                command="pytest -q",
                result_text="command not found",
                is_error=True,
            )
        ),
    )
    assert score.verification is Verification.NO_RUN


def test_a_timeout_censors_the_level_instead_of_scoring_it() -> None:
    score = score_level(_truth(exit_code=124), _turn())
    assert score.timeout_censored is True
    assert score.verification is Verification.CENSORED


def test_censored_levels_are_excluded_from_rates() -> None:
    good = score_level(_truth(), _turn(_PYTEST_GREEN))
    censored = score_level(_truth(exit_code=124), _turn())
    rates = level_rates([good, censored])
    assert rates.levels_scored == 1
    assert rates.levels_censored == 1


def test_rates_carry_wilson_intervals_over_subtasks() -> None:
    """Never a bare rate: the #63 lesson is that battery-n proportions
    need intervals attached at the point of production."""
    rates = level_rates(
        [
            score_level(
                _truth(oracles={"ledger": True, "qty": False}), _turn(_PYTEST_GREEN)
            )
        ]
    )
    assert rates.shipped_broken == 1
    assert rates.subtasks == 2
    low, high = rates.shipped_broken_interval
    assert 0.0 <= low <= 0.5 <= high <= 1.0
