#!/usr/bin/env python3
"""Spike Ω-loop — validate the bounded-loop primitive as a real declarative flow.

Drives the engine's shipped `loop:` combinator over the Ω-E architect-coherence
repair cycle, expressed entirely as config (two script nodes + a body ensemble +
a loop node) instead of the Python resolve_contract driver. A deterministic
fixture-replay architect stands in for the frontier architect, so the LOOP
MECHANICS — carry threading, `until` termination, and bound termination — are
validated WITHOUT spending frontier tokens. (The live frontier-architect
convergence run is the separate, paid follow-up.)

Two arms:
  converge -> incoherent contract on iteration 1; the gate's feedback is carried
              to iteration 2, where the architect emits a coherent contract; the
              loop stops via `until` at iteration 2 (within the bound of 3).
  exhaust  -> never-coherent contract; `until` never holds, so the loop stops
              "exhausted" at the bound (3), reporting the last failure honestly.

Usage:
    uv run python scratch/spike-omega-loop/validate_loop.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / ".llm-orc"
ENS = PROJECT_DIR / "ensembles"
CALC_TASK = "Build a small arithmetic expression calculator as flat Python modules."

_LOADER = EnsembleLoader()


def _loop_outcome(result: dict[str, Any]) -> dict[str, Any]:
    """The repair-loop node's LoopOutcome: {output, iterations, terminated}."""
    node = result.get("results", {}).get("repair-loop", {})
    response = node.get("response", "") if isinstance(node, dict) else ""
    return json.loads(response)


async def _run(executor: Any, yaml_name: str) -> dict[str, Any]:
    config = _LOADER.load_from_file(str(ENS / yaml_name))
    result = await executor.execute(config, CALC_TASK)
    return _loop_outcome(result)


def _check(
    label: str,
    outcome: dict[str, Any],
    *,
    terminated: str,
    iterations: int,
    ok: bool,
) -> bool:
    got_term = outcome.get("terminated")
    got_iter = outcome.get("iterations")
    got_ok = outcome.get("output", {}).get("ok")
    passed = got_term == terminated and got_iter == iterations and got_ok == ok
    print(
        f"[{label}] terminated={got_term!r} iterations={got_iter} ok={got_ok}  "
        f"expected terminated={terminated!r} iterations={iterations} ok={ok}  "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return passed


async def run() -> None:
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)

    converge = await _run(executor, "spike-omega-loop-converge.yaml")
    exhaust = await _run(executor, "spike-omega-loop-exhaust.yaml")

    print("\n==== Ω-loop validation (declarative bounded loop over the Ω-E gate) ====")
    ok1 = _check("converge", converge, terminated="until", iterations=2, ok=True)
    ok2 = _check("exhaust", exhaust, terminated="exhausted", iterations=3, ok=False)

    print("\n--- converge: final contract files ---")
    files = [d["file"] for d in converge.get("output", {}).get("contract", [])]
    print(files)
    print("--- exhaust: reasons reported at the bound ---")
    print(f"{len(exhaust.get('output', {}).get('reasons', []))} reasons (non-empty = honest failure)")

    print(f"\nRESULT: {'ALL PASS' if ok1 and ok2 else 'FAILURE'}")
    if not (ok1 and ok2):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run())
