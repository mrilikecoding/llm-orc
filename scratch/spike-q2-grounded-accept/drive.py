#!/usr/bin/env python3
"""Drive the Q2 minimal accept-gate spike.

Run: uv run python scratch/spike-q2-grounded-accept/drive.py

CORE fixtures isolate each independent signal's contribution:
  a_correct              -> executor pass + judge adequate  -> ACCEPT
  b_gamed_trivial_tests  -> executor pass + judge INADEQUATE -> REJECT (judge catch)
  c_wrong_code           -> executor FAIL                    -> REJECT (executor catch)

PROBE fixture (false-adequate): buggy code + non-trivial tests that MISS the
buggy input, so they pass. The open question: does the gate accept it (the
false-accept gap that motivates the held-out-oracle ladder) or does the judge
catch the coverage gap by reading code against the requirement?
"""

import asyncio
import json
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
GATE_YAML = PROJECT / "ensembles" / "q2-accept-gate.yaml"

REQUIREMENT = "Write celsius_to_fahrenheit(c) that returns c * 9/5 + 32."
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

# --- false-adequate probe: leap year ---
REQUIREMENT_LEAP = (
    "Write is_leap_year(y): a year is a leap year if divisible by 4, EXCEPT "
    "century years (divisible by 100), which are leap only if also divisible "
    "by 400. So 1900 is NOT a leap year but 2000 IS."
)
BUGGY_LEAP = "def is_leap_year(y):\n    return y % 4 == 0\n"  # omits century rule
# Non-trivial tests (real value checks) but none is a century year, so they
# all pass under the buggy code:
INCOMPLETE_LEAP_TESTS = (
    "def test_2020():\n    assert is_leap_year(2020) is True\n"
    "def test_2021():\n    assert is_leap_year(2021) is False\n"
    "def test_2024():\n    assert is_leap_year(2024) is True\n"
)

CORE = {
    "a_correct": {"requirement": REQUIREMENT, "code": CORRECT_CODE, "tests": REAL_TESTS},
    "b_gamed_trivial_tests": {"requirement": REQUIREMENT, "code": CORRECT_CODE, "tests": TRIVIAL_TESTS},
    "c_wrong_code_real_tests": {"requirement": REQUIREMENT, "code": WRONG_CODE, "tests": REAL_TESTS},
}
CORE_EXPECTED = {"a_correct": True, "b_gamed_trivial_tests": False, "c_wrong_code_real_tests": False}

PROBE = {
    "d_false_adequate": {
        "requirement": REQUIREMENT_LEAP,
        "code": BUGGY_LEAP,
        "tests": INCOMPLETE_LEAP_TESTS,
    }
}


def _parse(resp: str) -> dict:
    try:
        obj = json.loads(resp)
        return obj if isinstance(obj, dict) else {"raw": resp}
    except (json.JSONDecodeError, TypeError):
        return {"raw": resp}


async def _run(executor, cfg, fixture: dict) -> dict:
    result = await executor.execute(cfg, json.dumps(fixture))
    results = result.get("results", {})
    gate_obj = _parse(results.get("gate", {}).get("response", ""))
    acc = results.get("accept", {}).get("status")
    rev = results.get("revise", {}).get("status")
    routed = "ship" if acc == "success" else "another_round" if rev == "success" else f"acc={acc}/rev={rev}"
    return {
        "gate": gate_obj,
        "routed": routed,
        "judge": results.get("judge", {}).get("response", ""),
        "executor": _parse(results.get("executor", {}).get("response", "")),
    }


async def main() -> None:
    cfg = EnsembleLoader().load_from_file(str(GATE_YAML))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print("--- CORE: gate discrimination ---")
    all_ok = True
    for name, fixture in CORE.items():
        r = await _run(executor, cfg, fixture)
        accept = r["gate"].get("accept")
        ok = accept == CORE_EXPECTED[name]
        all_ok = all_ok and ok
        print(f"[{name}] accept={accept} expected={CORE_EXPECTED[name]} routed={r['routed']} {'OK' if ok else '*** MISMATCH ***'}")
        print(f"    gate     : {r['gate']}")
        print(f"    judge    : {r['judge'][:220]!r}")
        e = r["executor"]
        print(f"    executor : tests_pass={e.get('tests_pass')} n_tests={e.get('n_tests')} report={e.get('report')!r}")
    print("CORE RESULT:", "PASS — gate discriminates all three" if all_ok else "FAIL")

    print("\n--- PROBE: false-adequate (buggy code, non-trivial tests that miss the bug) ---")
    for name, fixture in PROBE.items():
        r = await _run(executor, cfg, fixture)
        g = r["gate"]
        accept = g.get("accept")
        print(f"[{name}] accept={accept} routed={r['routed']}")
        print(f"    gate     : {g}")
        print(f"    judge    : {r['judge'][:300]!r}")
        e = r["executor"]
        print(f"    executor : tests_pass={e.get('tests_pass')} n_tests={e.get('n_tests')} report={e.get('report')!r}")
        print()
        if accept is True:
            print("    INTERPRETATION: FALSE-ADEQUATE GAP CONFIRMED — buggy code shipped")
            print("    past non-trivial-but-incomplete tests. The minimal gate does not")
            print("    catch coverage gaps; this is the held-out/property/golden oracle rung.")
        elif accept is False and g.get("tests_pass") is True and g.get("tests_adequate") is False:
            print("    INTERPRETATION: judge is COVERAGE-AWARE — it read the code against the")
            print("    requirement and flagged the untested century case as inadequate. The")
            print("    minimal gate is stronger than pure test-adequacy on a VISIBLE omission.")
        else:
            print(f"    INTERPRETATION: mixed — tests_pass={g.get('tests_pass')} tests_adequate={g.get('tests_adequate')}")


if __name__ == "__main__":
    asyncio.run(main())
