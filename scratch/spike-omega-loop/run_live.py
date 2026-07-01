#!/usr/bin/env python3
"""Spike Ω-loop (LIVE / PAID) — confirm a real architect converges under the loop.

The paid arm of engine handoff #2. Same declarative bounded loop the
deterministic arm validated, but the body's architect is the REAL frontier model
(qwen3.6-plus via paid OpenCode Go). Spends frontier tokens: one architect call
per iteration, up to max_iterations (3). Any outcome is informative:

  - converge iteration 1 -> architect got it right first try.
  - converge iteration 2-3 -> the gate caught a real frontier incoherence and the
    loop repaired it (the strongest validation; this is the exact failure mode
    that motivated the coherence gate on 2026-06-30).
  - exhaust -> the bound protected against a stuck frontier; honest failure.

Usage:
    uv run python scratch/spike-omega-loop/run_live.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / ".llm-orc"
TOP = PROJECT_DIR / "ensembles" / "spike-omega-loop-live.yaml"
OUT = Path(__file__).resolve().parent / "live_result.json"

CALC_TASK = (
    "Build a small arithmetic expression calculator as flat Python modules "
    "plus a CLI, a test, and a README: a tokenizer that turns an expression "
    "string into a list of tokens; a parser that builds an AST from tokens "
    "(importing the tokenizer); an evaluator that computes a numeric result "
    "from the AST (importing the parser); an argparse CLI that reads an "
    "expression argument and prints the result (importing the evaluator); a "
    "test for the evaluator end to end; and Markdown docs. Support + - * / "
    "and parentheses. Modules import each other by bare module name."
)


def _loop_outcome(result: dict[str, Any]) -> dict[str, Any]:
    node = result.get("results", {}).get("repair-loop", {})
    response = node.get("response", "") if isinstance(node, dict) else ""
    return json.loads(response)


async def run() -> None:
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)
    config = EnsembleLoader().load_from_file(str(TOP))

    print("[Ω-loop LIVE] driving the real frontier architect through the loop...")
    t0 = time.perf_counter()
    result = await executor.execute(config, CALC_TASK)
    elapsed = time.perf_counter() - t0

    outcome = _loop_outcome(result)
    output = outcome.get("output", {})
    files = [d.get("file") for d in output.get("contract", [])]

    print("\n==== Ω-loop LIVE result ====")
    print(f"terminated : {outcome.get('terminated')!r}")
    print(f"iterations : {outcome.get('iterations')}")
    print(f"ok         : {output.get('ok')}")
    print(f"elapsed    : {elapsed:.0f}s")
    print(f"contract   : {files}")
    reasons = output.get("reasons", [])
    if reasons:
        print("final reasons (gate complaints at termination):")
        for r in reasons:
            print(f"   - {r}")

    OUT.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "outcome": outcome}, indent=2))
    print(f"\nsaved -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    asyncio.run(run())
