#!/usr/bin/env python3
"""WP-D8 Grounding Reframe (follow-up) — isolate criteria-richness from domain knowledge.

The first pass (drive_reframe.py) found the judge's coverage catch on the leap-year
fixture (d) SURVIVED criteria thinning: the model knows the century rule from world
knowledge, so stripping it from the requirement did not blind the judge. That is a
CONFOUND -- the canonical fixtures don't isolate "criteria richness" from "domain
familiarity".

This follow-up controls the confound with a NOVEL, arbitrary business rule the model
cannot know a priori (a made-up late-fee escalation). Same structure as fixture d:
buggy code + non-trivial tests that miss the buggy input.

  d_leap_control  -> canonical concept (model KNOWS the omitted rule)
  e_novel_rule    -> arbitrary rule (model CANNOT know the omitted rule)

Prediction: on e, RICH criteria let the judge demand coverage of the escalation;
THIN criteria leave the judge with no signal (no criteria, no world knowledge), so
the buggy code ships. That is the honest thin-criteria degradation ADR-048 §2 names,
isolated from the domain-knowledge rescue that masked it on d.

Run: uv run python scratch/spike-q2-grounded-accept/drive_reframe_novel.py [N]
"""

import asyncio
import json
import sys
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
GATE_YAML = PROJECT / "ensembles" / "q2-accept-gate.yaml"

# --- d: canonical control (model knows the century rule) ---
BUGGY_LEAP = "def is_leap_year(y):\n    return y % 4 == 0\n"
INCOMPLETE_LEAP_TESTS = (
    "def test_2020():\n    assert is_leap_year(2020) is True\n"
    "def test_2021():\n    assert is_leap_year(2021) is False\n"
    "def test_2024():\n    assert is_leap_year(2024) is True\n"
)
REQ_LEAP_RICH = (
    "Write is_leap_year(y): a year is a leap year if divisible by 4, EXCEPT "
    "century years (divisible by 100), which are leap only if also divisible "
    "by 400. So 1900 is NOT a leap year but 2000 IS."
)
REQ_LEAP_THIN = "Write is_leap_year(y) that returns True if y is a leap year, else False."

# --- e: NOVEL arbitrary rule the model cannot know a priori ---
# Correct: 0 if not late; $5 flat for 1..10 days; $5 + $1/day beyond 10 after that.
# Buggy: flat $5 for anything late -- ignores the >10-day escalation.
BUGGY_LATE_FEE = (
    "def late_fee(days_late):\n"
    "    if days_late <= 0:\n"
    "        return 0\n"
    "    return 5\n"
)
# Non-trivial tests, all PASS under the buggy code, none exercises days_late > 10
# (12 -> correct 7, buggy 5): the omitted-input coverage gap, same shape as d.
INCOMPLETE_LATE_FEE_TESTS = (
    "def test_not_late():\n    assert late_fee(0) == 0\n"
    "def test_within_grace():\n    assert late_fee(5) == 5\n"
    "def test_boundary():\n    assert late_fee(10) == 5\n"
)
REQ_LATE_RICH = (
    "Write late_fee(days_late): return 0 if days_late <= 0. For 1 to 10 days "
    "late, the fee is $5 flat. For MORE than 10 days late, the fee is $5 plus "
    "$1 for each day beyond 10. So 12 days late costs $5 + $2 = $7."
)
REQ_LATE_THIN = (
    "Write late_fee(days_late) that returns the late fee for a payment that is "
    "days_late days late."
)

FIXTURES = {
    "d_leap_control": {
        "code": BUGGY_LEAP,
        "tests": INCOMPLETE_LEAP_TESTS,
        "req": {"rich": REQ_LEAP_RICH, "thin": REQ_LEAP_THIN},
        "note": "canonical: model KNOWS the century rule",
    },
    "e_novel_rule": {
        "code": BUGGY_LATE_FEE,
        "tests": INCOMPLETE_LATE_FEE_TESTS,
        "req": {"rich": REQ_LATE_RICH, "thin": REQ_LATE_THIN},
        "note": "novel: model CANNOT know the >10-day escalation",
    },
}


def _parse(resp: str) -> dict:
    try:
        obj = json.loads(resp)
        return obj if isinstance(obj, dict) else {"raw": resp}
    except (json.JSONDecodeError, TypeError):
        return {"raw": resp}


async def _run_once(executor, cfg, requirement: str, code: str, tests: str) -> dict:
    fixture = {"requirement": requirement, "code": code, "tests": tests}
    result = await executor.execute(cfg, json.dumps(fixture))
    results = result.get("results", {})
    gate = _parse(results.get("gate", {}).get("response", ""))
    judge = _parse(results.get("judge", {}).get("response", ""))
    return {
        "accept": gate.get("accept"),
        "tests_pass": gate.get("tests_pass"),
        "tests_adequate": gate.get("tests_adequate"),
        "judge_reason": judge.get("reason", judge.get("raw", "")),
    }


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cfg = EnsembleLoader().load_from_file(str(GATE_YAML))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print(f"WP-D8 Grounding Reframe follow-up: confound control, N={n} per cell\n")
    print("The buggy code passes its incomplete tests in BOTH fixtures (executor green).")
    print("So the gate rejects ONLY if the judge flags the coverage gap. The question:")
    print("does thinning the requirement blind the judge when world knowledge cannot")
    print("rescue it (e) the way it did for the canonical concept (d)?\n")

    for fname, fx in FIXTURES.items():
        print(f"--- {fname}  ({fx['note']}) ---")
        for cond in ("rich", "thin"):
            req = fx["req"][cond]
            rejects = 0
            reasons: list[str] = []
            for _ in range(n):
                r = await _run_once(executor, cfg, req, fx["code"], fx["tests"])
                if r["accept"] is False:
                    rejects += 1
                reasons.append(str(r["judge_reason"])[:150])
            verdict = "CATCHES gap" if rejects >= (n + 1) // 2 else "MISSES gap (buggy ships)"
            print(f"  [{cond:4s}] reject={rejects}/{n}  -> judge {verdict}")
            print(f"        e.g. {reasons[0]!r}")
        print()

    print("=== READING ===")
    print("d rich vs thin similar  -> world knowledge, not criteria, drove the catch.")
    print("e rich CATCHES / e thin MISSES -> criteria richness is the real variable,")
    print("  isolated: thin criteria on a NOVEL rule degrades the judge to triviality-")
    print("  only, and the executor is the sole remaining anchor (ADR-048 s2, grounded).")


if __name__ == "__main__":
    asyncio.run(main())
