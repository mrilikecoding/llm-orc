#!/usr/bin/env python3
"""Live-model drive for the dynamic-dispatch serve skeleton (32GB rig, $0 local).

Runs the classify -> seat(dispatch) -> marshal skeleton against the REAL local
8b model seats, exercising the two open spike criteria with real models:
  - swap-ability (#2): the same skeleton fills the seat with dd-seat-code-solo
    (single model) then dd-seat-code-verified (coder + reviewer), zero skeleton
    change.
  - generality / non-build turn (#3): the same skeleton routes an explain turn
    to dd-seat-explain (prose), not a code seat.

This drives the ensemble directly through the real EnsembleExecutor. Real
opencode-run transport is a2-proven separately (spike-omega-serve) and is not
re-spiked here; to drive through opencode, serve dd-serve-skeleton behind the
Ω-serve harness (see README).

Models must be pulled in Ollama first (the profiles resolve to local qwen3).
Prints each turn's routed seat, the marshal outcome, and elapsed seconds so the
per-turn latency (Tension 23) is visible.

Run:  uv run python scratch/spike-dynamic-dispatch/drive_skeleton.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / ".llm-orc"
SKELETON = PROJECT_DIR / "ensembles" / "dd-serve-skeleton.yaml"
_LOADER = EnsembleLoader()

_PRIME = "Write a Python function is_prime(n) that returns True iff n is prime."
_EXPLAIN = "Explain what a Python list comprehension is, briefly."


def _turns(fast: bool) -> list[tuple[str, dict[str, Any]]]:
    """The same three turns, pointed at thinking-on or thinking-off seats.

    `fast` routes to the `-fast` seat variants (qwen3 /no_think); the skeleton
    is identical, only the seat strategy the classifier names differs.
    """
    suffix = "-fast" if fast else ""
    return [
        (
            "build -> solo seat (strategy A)",
            {"task": _PRIME, "seat": f"dd-seat-code-solo{suffix}", "file": "is_prime.py"},
        ),
        (
            "build -> verified seat (strategy B, structurally different)",
            {
                "task": _PRIME,
                "seat": f"dd-seat-code-verified{suffix}",
                "file": "is_prime.py",
            },
        ),
        (
            "non-build -> explain seat (generality)",
            {"task": _EXPLAIN, "seat": f"dd-seat-explain{suffix}"},
        ),
    ]


async def _run(turn: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    config = _LOADER.load_from_file(str(SKELETON))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)
    start = time.time()
    result = await executor.execute(config, json.dumps(turn))
    elapsed = time.time() - start
    classify = result["results"].get("classify", {}).get("response", "{}")
    target = json.loads(classify).get("target", "?")
    outcome: dict[str, Any] = json.loads(result["results"]["marshal"]["response"])
    return target, outcome, elapsed


async def main() -> None:
    fast = "--fast" in sys.argv[1:]
    label = "OFF (options.think=false)" if fast else "ON (default)"
    print(f"thinking mode: {label}")
    for label, turn in _turns(fast):
        target, outcome, elapsed = await _run(turn)
        print(f"\n=== {label} ===")
        print(f"routed seat: {target}   ({elapsed:.1f}s)")
        if outcome.get("finish"):
            print("outcome: FINISH (prose)")
            print(outcome.get("content", "")[:600])
        else:
            print(f"outcome: WRITE {outcome.get('file')}")
            print(outcome.get("content", "")[:600])


if __name__ == "__main__":
    asyncio.run(main())
