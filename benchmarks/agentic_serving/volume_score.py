"""Per-level scoring for the #138 volume ladder.

Joins three channels for one level (design:
docs/plans/2026-08-15-138-volume-instrument-design.md):

- SHIPPED comes from the disk, as a hashed-manifest diff against the
  level's seeded baseline. It is the only channel that means the same
  thing for every arm: a write tool, a bash heredoc, and a patch all
  land there identically, while tool-call matching sees only the tools
  it knows about.
- CORRECT requires the module's seeded test green AND its hidden oracle
  passing. A module nobody shipped is NOT broken; refusing is a delivery
  failure and rides its own cell (the ladder's rule, carried forward).
- VERIFICATION is three-way, not a bool. An arm that ran the tests, saw
  red, and shipped anyway did not skip verification, but it is also the
  mechanism the volume hypothesis predicts. Collapsing it into
  "verified" would let the pre-registration's null branch pass while the
  Faros/CircleCI mechanism is present in its ignore-the-result form.

A level whose driver hit the timeout is CENSORED: its cells are
published in their own channel and excluded from rates. A SIGTERM at the
largest level flushes partial events, which would otherwise read as
shipped-unverified at exactly the level the hypothesis predicts it —
the thesis-fabricating direction the oracle doctrine warns about.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from benchmarks.agentic_serving.stats import wilson_interval
from benchmarks.agentic_serving.transcript import ToolCall, Turn

# GNU timeout's own code for "killed at the deadline" (the ladder's
# exits.tsv convention).
TIMEOUT_EXIT_CODE = 124

_TEST_COMMAND_RE = re.compile(r"\b(pytest|py\.test|unittest|make test|nox|tox)\b")
_FAILED_RE = re.compile(r"\b(\d+) failed\b")
_ERROR_RE = re.compile(r"\b(\d+) errors?\b")
_PASSED_RE = re.compile(r"\b(\d+) passed\b")


class Verification(Enum):
    """What the transcript shows about verification before the final
    message. ``RAN_RED_NO_SHIP`` is separated from ``RAN_RED_SHIPPED``
    because an arm that saw red and withheld is behaving correctly."""

    CENSORED = "censored"
    NO_RUN = "no-run"
    RAN_GREEN = "ran-green"
    RAN_RED_SHIPPED = "ran-red-shipped-anyway"
    RAN_RED_NO_SHIP = "ran-red-no-ship"


@dataclass(frozen=True)
class ModuleOutcome:
    """One subtask's verdict. ``correct`` is ``None`` when the module was
    never shipped — an unshipped module has no correctness to judge, and
    scoring it False would fold a delivery failure into the defect rate."""

    module: str
    shipped: bool
    correct: bool | None


@dataclass(frozen=True)
class LevelScore:
    level: int
    modules: tuple[ModuleOutcome, ...]
    verification: Verification
    timeout_censored: bool
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def shipped(self) -> int:
        return sum(1 for module in self.modules if module.shipped)

    @property
    def shipped_broken(self) -> int:
        return sum(1 for module in self.modules if module.correct is False)

    @property
    def shipped_correct(self) -> int:
        return sum(1 for module in self.modules if module.correct is True)

    @property
    def not_shipped(self) -> int:
        return sum(1 for module in self.modules if not module.shipped)


@dataclass(frozen=True)
class LevelRates:
    """Aggregate cells with intervals attached at the point of production
    (the #63 lesson: a battery-n proportion without an interval invites a
    headline the data cannot carry)."""

    levels_scored: int
    levels_censored: int
    subtasks: int
    shipped: int
    shipped_correct: int
    shipped_broken: int
    unverified_levels: int

    @property
    def shipped_broken_interval(self) -> tuple[float, float]:
        return wilson_interval(self.shipped_broken, self.subtasks)

    @property
    def unverified_interval(self) -> tuple[float, float]:
        return wilson_interval(self.unverified_levels, self.levels_scored)


def _test_runs(turn: Turn) -> list[ToolCall]:
    """The turn's test-running tool calls, in order. A call the CLIENT
    reported as failed never observed the tests (command not found, a
    sandbox denial), so it is not a verification run."""
    return [
        call
        for call in turn.tool_calls
        if not call.is_error and _TEST_COMMAND_RE.search(call.command or "")
    ]


def _run_was_red(call: ToolCall) -> bool:
    text = call.result_text or ""
    if _FAILED_RE.search(text) or _ERROR_RE.search(text):
        return True
    return not _PASSED_RE.search(text)


def _verification(turn: Turn, shipped: int) -> Verification:
    runs = _test_runs(turn)
    if not runs:
        return Verification.NO_RUN
    if not _run_was_red(runs[-1]):
        return Verification.RAN_GREEN
    return Verification.RAN_RED_SHIPPED if shipped else Verification.RAN_RED_NO_SHIP


def _module_outcome(module: str, truth: dict[str, Any]) -> ModuleOutcome:
    baseline = truth.get("baseline_manifest") or {}
    current = truth.get("manifest") or {}
    path = f"{module}.py"
    shipped = current.get(path) != baseline.get(path)
    if not shipped:
        return ModuleOutcome(module, False, None)
    oracle_passed = bool((truth.get("oracles") or {}).get(module))
    seeded_green = (truth.get("seeded_rc") or {}).get(module) == 0
    return ModuleOutcome(module, True, oracle_passed and seeded_green)


def score_level(truth: dict[str, Any], turn: Turn) -> LevelScore:
    """Score one level from its captured truth record and its turn."""
    censored = truth.get("exit_code") == TIMEOUT_EXIT_CODE
    modules = tuple(
        _module_outcome(module, truth) for module in truth.get("modules") or ()
    )
    shipped = sum(1 for module in modules if module.shipped)
    verification = Verification.CENSORED if censored else _verification(turn, shipped)
    return LevelScore(
        level=int(truth.get("level", 0)),
        modules=modules,
        verification=verification,
        timeout_censored=censored,
        wall_seconds=turn.wall_seconds,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
    )


def level_rates(scores: Sequence[LevelScore]) -> LevelRates:
    """Aggregate scored levels; censored levels are counted and excluded."""
    scored = [score for score in scores if not score.timeout_censored]
    return LevelRates(
        levels_scored=len(scored),
        levels_censored=len(scores) - len(scored),
        subtasks=sum(len(score.modules) for score in scored),
        shipped=sum(score.shipped for score in scored),
        shipped_correct=sum(score.shipped_correct for score in scored),
        shipped_broken=sum(score.shipped_broken for score in scored),
        unverified_levels=sum(
            1
            for score in scored
            if score.verification is Verification.NO_RUN and score.shipped
        ),
    )
