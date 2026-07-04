#!/usr/bin/env python3
"""WP-D8 Grounding Reframe (final) — strip the last tell: opaque naming.

Passes 1-2 found the judge's coverage catch survives criteria thinning via THREE
criteria-independent sources: world knowledge (leap year), naming/signature semantics
(late_fee(days_late) should vary), and test-shape structure (constant-return tests
look suspicious). Each fixture carried a reconstructable "tell".

This fixture strips all of them. An OPAQUE spec: three output bands with ARBITRARY
thresholds, a name ("categorize") that does not telegraph how many bands or where
the cuts are, and buggy code that drops the middle band. Tests cover only the two
surviving bands with plausible real-value assertions. The omitted middle band is
reconstructable ONLY from an explicit requirement -- not from the concept, the name,
or the test shape.

Prediction: rich CATCHES (requirement names the mid band); thin MISSES (no source
for "a mid band should exist") -> buggy code ships. If this holds, it locates the
real thin-criteria degradation floor ADR-048 s2 names, isolated from every rescue.

Run: uv run python scratch/spike-q2-grounded-accept/drive_reframe_opaque.py [N]
"""

import asyncio
import json
import sys
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
GATE_YAML = PROJECT / "ensembles" / "q2-accept-gate.yaml"

# Buggy: only two bands (drops 'mid'); 50..79 wrongly returns 'high'.
BUGGY_CATEGORIZE = (
    "def categorize(n):\n"
    "    if n < 50:\n"
    "        return 'low'\n"
    "    return 'high'\n"
)
# Real-value tests, all PASS under the buggy code, none in the missing mid band
# (65 -> correct 'mid', buggy 'high'):
INCOMPLETE_CATEGORIZE_TESTS = (
    "def test_small():\n    assert categorize(10) == 'low'\n"
    "def test_low_edge():\n    assert categorize(40) == 'low'\n"
    "def test_big():\n    assert categorize(90) == 'high'\n"
    "def test_high_edge():\n    assert categorize(85) == 'high'\n"
)
REQ_RICH = (
    "Write categorize(n): return 'low' if n < 50, 'mid' if 50 <= n < 80, and "
    "'high' if n >= 80."
)
REQ_THIN = "Write categorize(n) that categorizes the number n."

FIXTURE = {
    "code": BUGGY_CATEGORIZE,
    "tests": INCOMPLETE_CATEGORIZE_TESTS,
    "req": {"rich": REQ_RICH, "thin": REQ_THIN},
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
        "tests_adequate": gate.get("tests_adequate"),
        "judge_reason": judge.get("reason", judge.get("raw", "")),
    }


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cfg = EnsembleLoader().load_from_file(str(GATE_YAML))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print(f"WP-D8 Grounding Reframe final: opaque fixture, N={n} per cell\n")
    print("Buggy code passes its incomplete tests (executor green). Gate rejects")
    print("only if the judge flags the missing 'mid' band. No tell survives thinning:")
    print("the mid band is knowable ONLY from an explicit requirement.\n")

    for cond in ("rich", "thin"):
        req = FIXTURE["req"][cond]
        rejects = 0
        reasons: list[str] = []
        for _ in range(n):
            r = await _run_once(executor, cfg, req, FIXTURE["code"], FIXTURE["tests"])
            if r["accept"] is False:
                rejects += 1
            reasons.append(str(r["judge_reason"])[:160])
        verdict = "CATCHES gap" if rejects >= (n + 1) // 2 else "MISSES gap (BUGGY SHIPS)"
        print(f"[{cond:4s}] reject={rejects}/{n}  -> judge {verdict}")
        for rs in reasons:
            print(f"      {rs!r}")
        print()

    print("=== READING ===")
    print("rich CATCHES / thin MISSES -> the degradation floor is located: with every")
    print("  reconstructable tell stripped, thin criteria blind the judge and the")
    print("  executor is the sole anchor. Confirms ADR-048 s2 in its TRUE narrow form:")
    print("  degradation needs thin criteria AND no world-knowledge/naming/test-shape tell.")
    print("thin also CATCHES -> the judge is even more robust than s2 assumed; default-on")
    print("  stands a fortiori and the 'weakens to runs-and-non-trivially-tested' floor")
    print("  is narrower than the ADR states.")


if __name__ == "__main__":
    asyncio.run(main())
