#!/usr/bin/env python3
"""WP-D8 Grounding Reframe — thinned-criteria rerun of the Q2 accept gate.

Owed at WP-D8 entry (DECIDE->ARCHITECT gate + susceptibility snapshot): rerun
ADR-048's grounding-spike fixtures with the acceptance criteria withheld/thinned
and measure what the isolated judge adds OVER the deterministic executor on
thin-criteria turns, before wiring the gate unconditional (default-on-for-build).

Controlled manipulation: hold {code, tests} constant, vary ONLY the requirement
richness (rich = the original explicit spec; thin = the bare task with the
discriminating detail stripped). N repeats per (condition x fixture) to see the
8B judge's stochastic consistency (a named ADR-048 Conditional-Acceptance target).

The judge's MARGINAL value over the executor = fixtures where the executor passes
(tests_pass=True) but the gate still rejects (judge INADEQUATE): that is b (trivial
tests) and d (coverage gap). Question: which of those survive criteria thinning?

Run: uv run python scratch/spike-q2-grounded-accept/drive_reframe.py [N]
"""

import asyncio
import json
import sys
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
GATE_YAML = PROJECT / "ensembles" / "q2-accept-gate.yaml"

# --- code + tests held CONSTANT across conditions (only the requirement varies) ---
CORRECT_CODE = "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\n"
WRONG_CODE = "def celsius_to_fahrenheit(c):\n    return c * 9 / 5\n"  # missing + 32
REAL_TESTS = (
    "def test_freezing():\n    assert celsius_to_fahrenheit(0) == 32\n"
    "def test_boiling():\n    assert celsius_to_fahrenheit(100) == 212\n"
    "def test_neg40():\n    assert celsius_to_fahrenheit(-40) == -40\n"
)
TRIVIAL_TESTS = (
    "def test_exists():\n    assert celsius_to_fahrenheit is not None\n"
    "def test_callable():\n    assert callable(celsius_to_fahrenheit)\n"
)
BUGGY_LEAP = "def is_leap_year(y):\n    return y % 4 == 0\n"  # omits century rule
INCOMPLETE_LEAP_TESTS = (
    "def test_2020():\n    assert is_leap_year(2020) is True\n"
    "def test_2021():\n    assert is_leap_year(2021) is False\n"
    "def test_2024():\n    assert is_leap_year(2024) is True\n"
)

# --- the ONLY manipulated variable: requirement richness ---
REQ_CELSIUS_RICH = "Write celsius_to_fahrenheit(c) that returns c * 9/5 + 32."
REQ_CELSIUS_THIN = "Write celsius_to_fahrenheit(c) that converts Celsius to Fahrenheit."
REQ_LEAP_RICH = (
    "Write is_leap_year(y): a year is a leap year if divisible by 4, EXCEPT "
    "century years (divisible by 100), which are leap only if also divisible "
    "by 400. So 1900 is NOT a leap year but 2000 IS."
)
REQ_LEAP_THIN = "Write is_leap_year(y) that returns True if y is a leap year, else False."

# fixture -> (code, tests, {condition: requirement}), plus the ground-truth
# executor outcome (criteria-independent) and whether the judge SHOULD reject.
FIXTURES = {
    "a_correct": {
        "code": CORRECT_CODE,
        "tests": REAL_TESTS,
        "req": {"rich": REQ_CELSIUS_RICH, "thin": REQ_CELSIUS_THIN},
        "executor_pass": True,
        "should_reject": False,  # both signals green -> accept
    },
    "b_gamed_trivial_tests": {
        "code": CORRECT_CODE,
        "tests": TRIVIAL_TESTS,
        "req": {"rich": REQ_CELSIUS_RICH, "thin": REQ_CELSIUS_THIN},
        "executor_pass": True,
        "should_reject": True,  # JUDGE-ONLY catch (executor fooled) -- criteria-independent?
    },
    "c_wrong_code": {
        "code": WRONG_CODE,
        "tests": REAL_TESTS,
        "req": {"rich": REQ_CELSIUS_RICH, "thin": REQ_CELSIUS_THIN},
        "executor_pass": False,
        "should_reject": True,  # EXECUTOR catch -- judge verdict moot for the gate
    },
    "d_false_adequate": {
        "code": BUGGY_LEAP,
        "tests": INCOMPLETE_LEAP_TESTS,
        "req": {"rich": REQ_LEAP_RICH, "thin": REQ_LEAP_THIN},
        "executor_pass": True,
        "should_reject": True,  # JUDGE-ONLY coverage catch -- criteria-DEPENDENT?
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

    print(f"WP-D8 Grounding Reframe: thinned-criteria rerun, N={n} per cell\n")
    print("Legend: reject_rate = fraction of runs the gate rejected (accept=False).")
    print("        The judge's marginal catch = executor-passes fixtures (b,d) the")
    print("        gate still rejects. b tests the ANTI-GAMING catch (should survive")
    print("        thinning); d tests the COVERAGE catch (predicted to degrade).\n")

    # rows: (fixture, condition) -> aggregate
    summary: dict[tuple[str, str], dict] = {}
    for fname, fx in FIXTURES.items():
        for cond in ("rich", "thin"):
            req = fx["req"][cond]
            rejects = 0
            adequate_false = 0
            sample_reason = ""
            for i in range(n):
                r = await _run_once(executor, cfg, req, fx["code"], fx["tests"])
                if r["accept"] is False:
                    rejects += 1
                if r["tests_adequate"] is False:
                    adequate_false += 1
                    if not sample_reason:
                        sample_reason = str(r["judge_reason"])[:160]
                elif not sample_reason:
                    sample_reason = "[adequate] " + str(r["judge_reason"])[:150]
            summary[(fname, cond)] = {
                "reject_rate": rejects / n,
                "judge_inadequate_rate": adequate_false / n,
                "executor_pass": fx["executor_pass"],
                "should_reject": fx["should_reject"],
                "sample_reason": sample_reason,
            }
            print(
                f"[{fname:22s} {cond:4s}] reject={rejects}/{n}  "
                f"judge_inadequate={adequate_false}/{n}  "
                f"exec_pass={fx['executor_pass']}  should_reject={fx['should_reject']}"
            )
            print(f"    e.g. judge: {sample_reason!r}")
        print()

    # --- the load-bearing comparison ---
    print("=== JUDGE MARGINAL VALUE (executor-passes fixtures only) ===")
    for fname in ("b_gamed_trivial_tests", "d_false_adequate"):
        rich = summary[(fname, "rich")]
        thin = summary[(fname, "thin")]
        print(
            f"{fname}: rich reject={rich['reject_rate']:.0%}  "
            f"thin reject={thin['reject_rate']:.0%}  "
            f"(judge is the ONLY signal here; executor passes)"
        )
    print()
    print("=== FALSE-REJECT WATCH (a_correct: judge should NOT reject) ===")
    for cond in ("rich", "thin"):
        a = summary[("a_correct", cond)]
        print(f"a_correct {cond}: reject={a['reject_rate']:.0%} (false-reject if > 0)")


if __name__ == "__main__":
    asyncio.run(main())
