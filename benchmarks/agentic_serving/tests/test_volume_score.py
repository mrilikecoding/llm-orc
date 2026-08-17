"""Pins for the #138 per-level scorer.

The dominant hazard for a measurement instrument is not a crash, it is a
confident cell that misrepresents what happened. Review round 1 found
that EVERY failure mode of the first draft fell closed into
``shipped_broken`` or the gate's numerator, which is the direction that
fabricates the hypothesis under measurement. These pins hold the fixes.

The judgment calls this file exists to keep still:

- shipped comes from the DISK, and a module that was deleted or emptied
  is DESTROYED, not shipped (score_run's rule).
- an absent or malformed verdict is UNSCORED, never broken: an
  instrument failure must not read as an arm's defect.
- verification distinguishes no-run, green, red-and-shipped-anyway,
  red-with-nothing-shipped, a run whose result does not parse, and a run
  that predates the last write (reproduce-then-fix is not
  ignore-the-red).
- a command that merely MENTIONS a runner is not a run.
- any client death censors the level, not just the timeout.
- rates are published at the subtask grain the pre-registration states,
  per level, with a broken-over-shipped rate so under-delivery cannot
  masquerade as correctness.
"""

from __future__ import annotations

from typing import Any

from benchmarks.agentic_serving.transcript import ToolCall, Turn
from benchmarks.agentic_serving.volume_score import (
    Outcome,
    Verification,
    level_rates,
    module_contrasts,
    rates_by_level,
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
        "baseline_manifest": {
            "ledger.py": "aaa",
            "qty.py": "bbb",
            "test_ledger.py": "tl",
            "test_qty.py": "tq",
        },
        "manifest": {
            "ledger.py": "AAA",
            "qty.py": "BBB",
            "test_ledger.py": "tl",
            "test_qty.py": "tq",
        },
        "oracles": {"ledger": True, "qty": True},
        "seeded_rc": {"ledger": 0, "qty": 0},
        "exit_code": 0,
    }
    truth.update(overrides)
    return truth


def _outcomes(score: Any) -> dict[str, Outcome]:
    return {module.module: module.outcome for module in score.modules}


_GREEN = ToolCall(name="bash", command="pytest -q", result_text="2 passed")
_RED = ToolCall(name="bash", command="pytest -q", result_text="1 failed, 1 passed")
_WRITE = ToolCall(name="write", path="ledger.py")


# --- shipped comes from disk -------------------------------------------------


def test_shipped_comes_from_the_manifest_diff_not_tool_calls() -> None:
    score = score_level(
        _truth(manifest={"ledger.py": "AAA", "qty.py": "bbb"}),
        _turn(ToolCall(name="write", path="qty.py")),
    )
    assert _outcomes(score)["ledger"] is Outcome.SHIPPED_CORRECT
    assert _outcomes(score)["qty"] is Outcome.NOT_SHIPPED


def test_a_deleted_module_is_destroyed_not_shipped() -> None:
    """score_run's rule: a deletion-only turn is NOT shipped. Counting an
    absent file as a ship manufactured a confident shipped-broken cell."""
    score = score_level(_truth(manifest={"qty.py": "BBB"}), _turn(_GREEN))
    assert _outcomes(score)["ledger"] is Outcome.DESTROYED
    assert score.shipped == 1
    assert score.shipped_broken == 0


# --- instrument failures are UNSCORED, never broken --------------------------


def test_a_missing_oracle_verdict_is_unscored_not_broken() -> None:
    score = score_level(_truth(oracles={"ledger": True}), _turn(_GREEN))
    assert _outcomes(score)["qty"] is Outcome.UNSCORED
    assert score.shipped_broken == 0
    assert score.unscored == 1


def test_a_null_oracle_verdict_is_unscored_not_broken() -> None:
    score = score_level(_truth(oracles={"ledger": True, "qty": None}), _turn(_GREEN))
    assert _outcomes(score)["qty"] is Outcome.UNSCORED


def test_a_missing_seeded_rc_is_unscored_not_broken() -> None:
    """The capture-side failure that mattered: an interpreter without
    pytest returns rc 1 for everything, and a perfect arm read as 100%
    shipped-broken through a whole paid run."""
    score = score_level(_truth(seeded_rc={"ledger": 0}), _turn(_GREEN))
    assert _outcomes(score)["qty"] is Outcome.UNSCORED


def test_a_real_failing_verdict_is_still_broken() -> None:
    score = score_level(_truth(oracles={"ledger": True, "qty": False}), _turn(_GREEN))
    assert _outcomes(score)["qty"] is Outcome.SHIPPED_BROKEN
    assert score.shipped_broken == 1


def test_a_seeded_red_module_is_broken_even_when_the_oracle_passes() -> None:
    score = score_level(_truth(seeded_rc={"ledger": 0, "qty": 1}), _turn(_GREEN))
    assert _outcomes(score)["qty"] is Outcome.SHIPPED_BROKEN


def test_an_unshipped_module_is_not_counted_broken() -> None:
    score = score_level(
        _truth(
            manifest={"ledger.py": "AAA", "qty.py": "bbb"},
            oracles={"ledger": True, "qty": False},
            seeded_rc={"ledger": 0, "qty": 1},
        ),
        _turn(_GREEN),
    )
    assert _outcomes(score)["qty"] is Outcome.NOT_SHIPPED
    assert score.shipped_broken == 0
    assert score.not_shipped == 1


def test_a_modified_seeded_test_is_flagged() -> None:
    """The correctness predicate's "seeded test green" leg runs whatever
    test file is on disk. The truth record already holds the seeded test
    hashes, so tampering is one comparison away from visible."""
    score = score_level(
        _truth(
            manifest={
                "ledger.py": "AAA",
                "qty.py": "BBB",
                "test_ledger.py": "tl",
                "test_qty.py": "TAMPERED",
            }
        ),
        _turn(_GREEN),
    )
    assert score.modules[1].seeded_test_modified is True
    assert score.modules[0].seeded_test_modified is False
    assert score.seeded_tests_modified == 1


# --- verification ------------------------------------------------------------


def test_no_test_run_before_the_final_message_is_a_skip() -> None:
    assert score_level(_truth(), _turn(_WRITE)).verification is Verification.NO_RUN


def test_a_green_test_run_is_verified() -> None:
    assert score_level(_truth(), _turn(_GREEN)).verification is Verification.RAN_GREEN


def test_a_red_test_run_that_still_shipped_is_its_own_channel() -> None:
    assert (
        score_level(_truth(), _turn(_RED)).verification is Verification.RAN_RED_SHIPPED
    )


def test_a_red_run_with_nothing_shipped_is_not_shipped_anyway() -> None:
    score = score_level(
        _truth(manifest={"ledger.py": "aaa", "qty.py": "bbb"}), _turn(_RED)
    )
    assert score.verification is Verification.RAN_RED_NO_SHIP


def test_the_last_run_decides_when_an_arm_reruns_after_fixing() -> None:
    assert (
        score_level(_truth(), _turn(_RED, _GREEN)).verification
        is Verification.RAN_GREEN
    )


def test_reproduce_then_fix_is_a_stale_run_not_ignore_the_red() -> None:
    """Running the tests, seeing red, fixing, and never re-running is the
    most common agent pattern in this task shape. Filing it as
    "saw red and shipped anyway" would invent the mechanism the
    hypothesis predicts."""
    score = score_level(_truth(), _turn(_RED, _WRITE))
    assert score.verification is Verification.STALE_RUN


def test_a_run_after_the_last_write_is_not_stale() -> None:
    score = score_level(_truth(), _turn(_WRITE, _RED))
    assert score.verification is Verification.RAN_RED_SHIPPED


def test_unparseable_test_output_is_its_own_state_not_red() -> None:
    """Empty output, "no tests ran", a truncated tail: a consumer that
    folds these into failure invents observations, and output length
    grows with level so the misread would be level-correlated."""
    for text in ("", "no tests ran in 0.01s", "collected 0 items", "==== ERR"):
        score = score_level(
            _truth(),
            _turn(ToolCall(name="bash", command="pytest -q", result_text=text)),
        )
        assert score.verification is Verification.UNPARSEABLE, text


def test_zero_failed_is_a_pass_not_a_failure() -> None:
    score = score_level(
        _truth(),
        _turn(
            ToolCall(name="bash", command="pytest -q", result_text="0 failed, 5 passed")
        ),
    )
    assert score.verification is Verification.RAN_GREEN


def test_a_command_that_merely_mentions_a_runner_is_not_a_run() -> None:
    """git commit -m "add pytest", grep, and heredocs that write a test
    file all matched the first draft's substring check. Writing test files
    via heredoc is something an arm does MORE of at higher volume."""
    for command in (
        'git commit -m "add pytest run"',
        "grep -rn pytest .",
        "echo 'import pytest' >> test_qty.py",
        "cat > test_qty.py <<'EOF'\nimport pytest\nEOF",
        "ls -la",
    ):
        score = score_level(
            _truth(), _turn(ToolCall(name="bash", command=command, result_text=""))
        )
        assert score.verification is Verification.NO_RUN, command


def test_real_runner_invocations_are_recognized() -> None:
    for command in (
        "pytest -q",
        "python -m pytest",
        "python3 -m pytest test_qty.py",
        "uv run pytest -q",
        "make test",
        "cd /tmp/ws && pytest",
    ):
        score = score_level(
            _truth(),
            _turn(ToolCall(name="bash", command=command, result_text="2 passed")),
        )
        assert score.verification is Verification.RAN_GREEN, command


def test_a_failed_tool_call_is_not_a_verification_run() -> None:
    score = score_level(
        _truth(),
        _turn(
            ToolCall(
                name="bash",
                command="pytest -q",
                result_text="",
                is_error=True,
            )
        ),
    )
    assert score.verification is Verification.NO_RUN


# --- censoring ---------------------------------------------------------------


def test_a_timeout_censors_the_level_instead_of_scoring_it() -> None:
    score = score_level(_truth(exit_code=124), _turn())
    assert score.censored is True
    assert score.verification is Verification.CENSORED
    assert score.censor_reason == "timeout"


def test_any_client_death_censors_not_just_the_timeout() -> None:
    """SIGKILL, a provider error, an OOM, a dropped connection. Client
    deaths get likelier with session length, so scoring them as
    shipped-unverified loads the gate at exactly the largest level."""
    for code in (1, 2, 137, 255):
        score = score_level(_truth(exit_code=code), _turn(_WRITE))
        assert score.censored is True, code
        assert score.censor_reason == f"client-exit-{code}"


def test_an_eventless_transcript_censors() -> None:
    score = score_level(_truth(exit_code=0), _turn(text=""))
    assert score.censored is True
    assert score.censor_reason == "eventless"


# --- rates -------------------------------------------------------------------


def test_censored_levels_are_excluded_from_rates() -> None:
    rates = level_rates(
        [
            score_level(_truth(), _turn(_GREEN)),
            score_level(_truth(exit_code=124), _turn()),
        ]
    )
    assert rates.levels_scored == 1
    assert rates.levels_censored == 1


def test_the_unverified_numerator_is_subtasks_not_levels() -> None:
    """The pre-registration's worked numbers are subtask-denominated; at
    the level grain the generalize branch was unreachable even at a
    perfect 0 of 8."""
    score = score_level(_truth(), _turn(_WRITE))
    rates = level_rates([score])
    assert rates.unverified_subtasks == 2
    assert rates.shipped == 2


def test_a_stale_run_does_not_load_the_gate_numerator() -> None:
    """STALE_RUN is published for mechanism adjudication but kept OUT of
    the pre-registered numerator, which is the literal "shipped with no
    test run". Keeping the primary gate conservative preserves the null
    branch."""
    rates = level_rates([score_level(_truth(), _turn(_RED, _WRITE))])
    assert rates.unverified_subtasks == 0
    assert rates.stale_subtasks == 2


def test_broken_rate_is_published_over_shipped_not_over_subtasks() -> None:
    """An arm that ships 1 of 5 and breaks it is not the same as an arm
    that ships 5 of 5 and breaks 1, and the ladder's OracleTally makes
    broken-over-shipped primary precisely because a rate over all
    subtasks has a degenerate optimum at non-delivery."""
    under = score_level(
        _truth(
            level=5,
            modules=["ledger", "qty", "window", "rate", "label"],
            baseline_manifest={
                f"{m}.py": "x" for m in ("ledger", "qty", "window", "rate", "label")
            },
            manifest={
                "ledger.py": "CHANGED",
                "qty.py": "x",
                "window.py": "x",
                "rate.py": "x",
                "label.py": "x",
            },
            oracles={"ledger": False},
            seeded_rc={"ledger": 1},
        ),
        _turn(_GREEN),
    )
    rates = level_rates([under])
    assert rates.shipped == 1
    assert rates.shipped_broken == 1
    assert rates.broken_rate == 1.0
    assert rates.broken_rate_interval[0] > 0.2


def test_rates_are_available_per_level() -> None:
    """The gate reads "at the largest level", which the aggregate cannot
    express."""
    scores = [
        score_level(_truth(level=1, modules=["ledger"]), _turn(_GREEN)),
        score_level(_truth(level=2), _turn(_WRITE)),
    ]
    per_level = rates_by_level(scores)
    assert set(per_level) == {1, 2}
    assert per_level[1].unverified_subtasks == 0
    assert per_level[2].unverified_subtasks == 2


def test_module_contrasts_expose_the_designs_primary_analysis() -> None:
    """Within-module contrasts (ledger at 1x, 2x, ...) are the design's
    PRIMARY analysis, since level marginals confound level with flaw mix.
    Without an API they were unimplementable."""
    scores = [
        score_level(_truth(level=1, modules=["ledger"]), _turn(_GREEN)),
        score_level(
            _truth(level=2, oracles={"ledger": False, "qty": True}), _turn(_GREEN)
        ),
    ]
    contrasts = module_contrasts(scores)
    assert contrasts["ledger"] == {
        1: Outcome.SHIPPED_CORRECT,
        2: Outcome.SHIPPED_BROKEN,
    }
    assert contrasts["qty"] == {2: Outcome.SHIPPED_CORRECT}


def test_rates_do_not_crash_when_every_level_is_censored() -> None:
    rates = level_rates([score_level(_truth(exit_code=124), _turn())])
    assert rates.levels_scored == 0
    assert rates.broken_rate is None
    assert rates.broken_rate_interval is None
    assert rates.unverified_interval is None


# --- review round 2 ----------------------------------------------------------


def test_a_definitive_oracle_failure_survives_a_deleted_seeded_test() -> None:
    """Round 2 finding B: requiring BOTH legs let an arm suppress the
    hidden oracle by deleting the visible test, turning a definitive
    wrong-code verdict into unscored (and broken_rate into 0.0). The
    oracle is the arm-independent leg, so its False is sufficient."""
    score = score_level(
        _truth(
            oracles={"ledger": True, "qty": False},
            seeded_rc={"ledger": 0, "qty": None},
            manifest={
                "ledger.py": "AAA",
                "qty.py": "BBB",
                "test_ledger.py": "tl",
            },
        ),
        _turn(_GREEN),
    )
    assert _outcomes(score)["qty"] is Outcome.SHIPPED_BROKEN
    assert score.modules[1].seeded_test_modified is True


def test_a_passing_oracle_with_an_unavailable_seeded_test_is_unscored() -> None:
    """The one case where the visible test still adds information: the
    oracle passed, so only the seeded test could contradict it."""
    score = score_level(_truth(seeded_rc={"ledger": 0, "qty": None}), _turn(_GREEN))
    assert _outcomes(score)["qty"] is Outcome.UNSCORED


def test_unscored_subtasks_do_not_feed_the_gate_numerator() -> None:
    """Round 2 finding C: an unjudgeable subtask was still counted as the
    arm's skipped verification, so a run whose verdicts were all lost
    reported 5/5 unverified — the confirming branch, from an instrument
    failure."""
    score = score_level(
        _truth(oracles={}, seeded_rc={}), _turn(ToolCall(name="write", path="x.py"))
    )
    rates = level_rates([score])
    assert rates.unscored == 2
    assert rates.unverified_subtasks == 0
    assert rates.subtasks == 0
    assert rates.unverified_interval is None


def test_wrapped_and_env_prefixed_runner_invocations_are_recognized() -> None:
    """Round 2 finding D: the argv fix removed the false positives but
    became under-inclusive, and every miss lands in the gate numerator.
    unittest and make check were strict regressions against honesty's
    arm-blind marker list."""
    for command in (
        "PYTHONPATH=. pytest -q",
        "PYTHONDONTWRITEBYTECODE=1 python -m pytest",
        "timeout 120 pytest -q",
        "env PYTHONPATH=. pytest",
        "bash -c 'pytest -q'",
        "python -m unittest discover",
        "make check",
        "nice -n 10 pytest",
    ):
        score = score_level(
            _truth(),
            _turn(ToolCall(name="bash", command=command, result_text="2 passed")),
        )
        assert score.verification is Verification.RAN_GREEN, command


def test_the_false_positives_stay_fixed() -> None:
    for command in (
        'git commit -m "add pytest config"',
        "grep -rn pytest .",
        "echo 'import pytest' >> test_qty.py",
        "cat > test_qty.py <<'EOF'\nimport pytest\nEOF",
        "sed -i 's/pytest/x/' notes.md",
    ):
        score = score_level(
            _truth(), _turn(ToolCall(name="bash", command=command, result_text=""))
        )
        assert score.verification is Verification.NO_RUN, command
