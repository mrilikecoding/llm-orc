#!/usr/bin/env python3
"""Ω-P3 (LIVE) — run the real serving flow and report the execution gate.

Runs a live ensemble end to end and prints the resolve-contract loop outcome (if
present) plus the real score node's execution-gate result, saving the full result
to disk. Two ensembles:

    spike-omega-p3-fanbuild-live  -> local-only de-risk (no paid architect)
    spike-omega-p3-full-live      -> full PAID flow (frontier architect + builders)

Usage:
    uv run python scratch/spike-omega-p3/run_live_full.py [ensemble-name]
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
import sys

TOP = sys.argv[1] if len(sys.argv) > 1 else "spike-omega-p3-full-live"
OUT = Path(__file__).resolve().parent / f"{TOP}_result.json"
TASK = (
    "Build a small arithmetic expression calculator as flat Python modules "
    "plus a CLI, a test, and a README: a tokenizer that turns an expression "
    "string into a list of tokens; a parser that builds an AST from tokens "
    "(importing the tokenizer); an evaluator that computes a numeric result "
    "from the AST (importing the parser); an argparse CLI that reads an "
    "expression argument and prints the result (importing the evaluator); a "
    "test for the evaluator end to end; and Markdown docs. Support + - * / "
    "and parentheses. Modules import each other by bare module name."
)


def _resp(node: Any) -> str:
    return node.get("response", "") if isinstance(node, dict) else ""


async def run() -> None:
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)
    config = EnsembleLoader().load_from_file(str(PROJECT_DIR / "ensembles" / f"{TOP}.yaml"))
    print(f"[Ω-P3 LIVE] running {TOP} ...")
    t0 = time.perf_counter()
    result = await executor.execute(config, TASK)
    elapsed = time.perf_counter() - t0
    r = result.get("results", {}) if isinstance(result, dict) else {}

    print(f"\n==== {TOP}  ({elapsed:.0f}s) ====")
    print("stage status:", {n: r.get(n, {}).get("status") for n in r if "[" not in n})

    if "resolve-contract" in r:
        rc = json.loads(_resp(r["resolve-contract"]))
        print(
            f"resolve-contract (frontier loop): terminated={rc['terminated']} "
            f"iterations={rc['iterations']} ok={rc['output']['ok']}"
        )

    # Per-instance build status (diagnose fan-out failures).
    build_instances = {n: r[n].get("status") for n in r if n.startswith("build[")}
    if build_instances:
        print("build instances:", build_instances)

    score_raw = _resp(r["score"])
    try:
        score = json.loads(score_raw)
        print(f"\nfiles built: {score['count']}   test files: {score.get('test_files')}")
        for b in score["built"]:
            print(f"   {str(b['file']):16} tier={str(b['tier']):6} bytes={b['bytes']}")
        print(f"\nEXECUTION GATE: {score['execution']}")
    except (json.JSONDecodeError, ValueError, TypeError):
        print("\nscore response (raw):", score_raw)

    # Dump the full results dict for diagnosis (statuses + raw responses).
    debug = {n: {"status": r[n].get("status"), "response": _resp(r[n])} for n in r}
    DEBUG = OUT.with_name(f"{TOP}_debug.json")
    DEBUG.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "results": debug}, indent=2))
    OUT.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "score_raw": score_raw}, indent=2))
    print(f"\nsaved -> {OUT.relative_to(REPO)} + {DEBUG.name}")


if __name__ == "__main__":
    asyncio.run(run())
