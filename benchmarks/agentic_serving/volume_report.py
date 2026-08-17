"""Read a #138 volume-ladder run directory and report it.

The path from artifacts to cells: ``truth-L<n>.json`` (the arm-blind
disk truth) joined to ``turn-L<n>.jsonl`` (the client transcript) through
:mod:`volume_score`. Without this module the scorer only ever saw
hand-built inputs, so the whole disk-to-number path went unexercised —
which is where review round 1's blockers lived.

The report states what the decision rule says about the numbers it
prints. At calibration n the gate cannot be tripped in either direction,
and a bare rate at that n invites exactly the headline the #63 slice
showed the data cannot carry, so the report says UNDERPOWERED itself
rather than leaving that to a reader.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from benchmarks.agentic_serving import opencode_adapter as oa
from benchmarks.agentic_serving.volume_fixture import VOLUME_PROMPTS
from benchmarks.agentic_serving.volume_score import (
    LevelScore,
    level_rates,
    module_contrasts,
    rates_by_level,
)

# The pre-registered threshold and repeat count (design: docs/plans/
# 2026-08-15-138-volume-instrument-design.md, Decision rule).
UNVERIFIED_THRESHOLD = 0.15
REQUIRED_REPEATS = 8


@dataclass(frozen=True)
class RunLevelScore(LevelScore):
    """A scored level plus how much of its transcript failed to parse.
    A dropped line is usually the truncated tail a SIGTERM leaves, so the
    count belongs in the report rather than in a silent except."""

    dropped_events: int = 0


def _turn_for(run_dir: Path, level: int) -> tuple[object, int]:
    path = run_dir / f"turn-L{level}.jsonl"
    text = path.read_text() if path.exists() else ""
    events, dropped = oa.parse_events_counting_drops(text)
    turn = oa.turn_from_events(
        events, index=level, prompt=VOLUME_PROMPTS.get(level, "")
    )
    return turn, dropped


def score_run_dir(run_dir: str | Path) -> list[RunLevelScore]:
    """Every level the run recorded, in level order."""
    from benchmarks.agentic_serving.volume_score import score_level

    directory = Path(run_dir)
    scores: list[RunLevelScore] = []
    for path in sorted(directory.glob("truth-L*.json")):
        truth = json.loads(path.read_text())
        level = int(truth.get("level", 0))
        turn, dropped = _turn_for(directory, level)
        base = score_level(truth, turn)  # type: ignore[arg-type]
        scores.append(
            replace(
                RunLevelScore(**base.__dict__),
                dropped_events=dropped,
            )
        )
    return sorted(scores, key=lambda score: score.level)


def score_run_dirs(run_dirs: Sequence[str | Path]) -> list[RunLevelScore]:
    """Every level from several run dirs — repeats of the same level live
    in separate dirs, so the decision rule's r-repeat path needs this to
    be a real code path rather than a described one."""
    scores: list[RunLevelScore] = []
    for run_dir in run_dirs:
        scores.extend(score_run_dir(run_dir))
    return sorted(scores, key=lambda score: score.level)


def observations_at_largest_level(scores: Sequence[RunLevelScore]) -> int:
    """How many scored observations the largest level actually has. The
    gate's n, which the verdict must know before naming a branch."""
    scored = [score for score in scores if not score.censored]
    if not scored:
        return 0
    largest = max(score.level for score in scored)
    return sum(1 for score in scored if score.level == largest)


def _interval(bounds: tuple[float, float] | None) -> str:
    if bounds is None:
        return "n/a"
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def format_report(scores: Sequence[RunLevelScore]) -> str:
    """A per-level table, the within-module contrasts the design names as
    its PRIMARY analysis, and the gate quantity with its interval."""
    lines: list[str] = ["level  subtasks  shipped  correct  broken  verification"]
    for score in scores:
        lines.append(
            f"L{score.level:<5} {len(score.modules):>8} {score.shipped:>8} "
            f"{score.shipped_correct:>8} {score.shipped_broken:>7}  "
            f"{score.verification.value}"
            + (f"  ({score.censor_reason})" if score.censored else "")
        )

    dropped = sum(score.dropped_events for score in scores)
    lines.append(f"\ndropped transcript lines: {dropped}")
    tampered = sum(score.seeded_tests_modified for score in scores)
    if tampered:
        lines.append(f"seeded tests modified by the arm: {tampered}")

    lines.append("\nwithin-module contrasts (PRIMARY: level marginals confound")
    lines.append("level with flaw mix, since rate and label appear only at L5)")
    for module, by_level in sorted(module_contrasts(scores).items()):
        cells = "  ".join(
            f"L{level}={outcome.value}" for level, outcome in sorted(by_level.items())
        )
        lines.append(f"  {module:<8} {cells}")

    lines.append("\nper level")
    for level, rates in rates_by_level(scores).items():
        lines.append(
            f"  L{level}: broken/shipped="
            f"{'n/a' if rates.broken_rate is None else f'{rates.broken_rate:.3f}'} "
            f"{_interval(rates.broken_rate_interval)}  "
            f"unverified={rates.unverified_subtasks}/{rates.subtasks} "
            f"{_interval(rates.unverified_interval)}"
        )

    overall = level_rates(scores)
    lines.append(
        f"\noverall: levels scored {overall.levels_scored}, "
        f"censored {overall.levels_censored}, unscored subtasks "
        f"{overall.unscored}, destroyed {overall.destroyed}, "
        f"stale-run subtasks {overall.stale_subtasks}, "
        f"unparseable levels {overall.unparseable_levels}"
    )
    lines.append(_gate_verdict(scores))
    return "\n".join(lines)


def _gate_verdict(scores: Sequence[RunLevelScore]) -> str:
    """What the pre-registered decision rule permits saying about THIS
    run. The rule is interval-vs-threshold at the largest level; anything
    that does not separate is UNDERPOWERED, which is a reportable
    outcome, not a failure."""
    scored = [score for score in scores if not score.censored]
    if not scored:
        return "\nGATE: no scored levels — nothing to evaluate."
    largest = max(score.level for score in scored)
    rates = rates_by_level(scored).get(largest)
    if rates is None or not rates.subtasks:
        return "\nGATE: the largest level has no scored subtasks."
    bounds = rates.unverified_interval
    if bounds is None:
        return "\nGATE: no interval at the largest level."
    observations = observations_at_largest_level(scored)
    if observations < REQUIRED_REPEATS:
        # The decision rule forbids naming a branch below r repeats, and
        # the asymmetry is the reason: at one observation per level the
        # confirming branch is reachable on an observed value while the
        # generalizing branch is not reachable at all. Refusing here keeps
        # that asymmetry from becoming a printed verdict.
        return (
            f"\nGATE at L{largest}: shipped-unverified "
            f"{rates.unverified_subtasks}/{rates.subtasks} {_interval(bounds)} "
            f"-> CALIBRATION (n={observations} per level, the decision rule "
            f"requires r={REQUIRED_REPEATS}; no branch may be named at this n "
            f"in either direction)"
        )
    if bounds[0] > UNVERIFIED_THRESHOLD:
        verdict = "CONFIRMS (interval entirely above the threshold)"
    elif bounds[1] < UNVERIFIED_THRESHOLD:
        verdict = "GENERALIZES (interval entirely below the threshold)"
    else:
        verdict = (
            "UNDERPOWERED (interval straddles the threshold; at calibration n "
            "this is the expected outcome and the gate cannot be tripped)"
        )
    return (
        f"\nGATE at L{largest}: shipped-unverified "
        f"{rates.unverified_subtasks}/{rates.subtasks} {_interval(bounds)} "
        f"vs {UNVERIFIED_THRESHOLD:.2f} -> {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(format_report(score_run_dir(args.run_dir)))


if __name__ == "__main__":
    main()
