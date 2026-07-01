#!/usr/bin/env python3
"""Ω-P3 (Phase 3) — the full agentic serving flow as ONE declarative ensemble.

Free structural spike (engine handoff #1): drive the whole flow
(resolve-contract loop -> plan -> fan-out build -> score) with the deterministic
architect + stand-in builders, to show it composes from shipped primitives with
no Python driver and to find where the declarative vocabulary runs out.

Usage:
    uv run python scratch/spike-omega-p3/run_p3.py [top-ensemble-name]
        default: spike-omega-p3-full   (also: spike-omega-p3-fanbuild)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / ".llm-orc"
ENS = PROJECT_DIR / "ensembles"
TOP = sys.argv[1] if len(sys.argv) > 1 else "spike-omega-p3-full"
TASK = "Build a small arithmetic expression calculator (flat modules + CLI + test + README)."


def _resp(node: Any) -> Any:
    return node.get("response", "") if isinstance(node, dict) else ""


async def run() -> None:
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)
    config = EnsembleLoader().load_from_file(str(ENS / f"{TOP}.yaml"))
    result = await executor.execute(config, TASK)
    r = result.get("results", {}) if isinstance(result, dict) else {}

    print(f"==== {TOP} ====")
    print("stage status:", {n: r.get(n, {}).get("status") for n in r if "[" not in n})

    if "resolve-contract" in r:
        rc = json.loads(_resp(r["resolve-contract"]))
        print(
            f"resolve-contract (loop): terminated={rc['terminated']} "
            f"iterations={rc['iterations']} ok={rc['output']['ok']}"
        )

    if "score" in r:
        score = json.loads(_resp(r["score"]))
        print(f"score: {score['count']} files built")
        for b in score["built"]:
            print(
                f"   {str(b['file']):16} tier={str(b['tier']):6} "
                f"fired={b['fired']:12} content={b['has_content']}"
            )
    else:
        for name, node in r.items():
            head = _resp(node)
            head = head if isinstance(head, str) else json.dumps(head)
            print(f"[{node.get('status')}] {name}: {head[:160]}")


if __name__ == "__main__":
    asyncio.run(run())
