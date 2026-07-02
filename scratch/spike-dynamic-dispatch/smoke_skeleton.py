#!/usr/bin/env python3
"""No-model smoke for the dynamic-dispatch serve skeleton.

Proves the declarative classify -> seat(dispatch) -> marshal wiring end to end
without spending model tokens: deterministic echo seats stand in for the real
model seats, so this exercises routing, the dispatch primitive, input_key
selection, swap-at-zero-skeleton-change, and marshal shaping. The live drive
(drive_skeleton.py) swaps the echo seats for the real model seats.

Run:  uv run python scratch/spike-dynamic-dispatch/smoke_skeleton.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / ".llm-orc"
SKELETON = PROJECT_DIR / "ensembles" / "dd-serve-skeleton.yaml"
CLASSIFY = PROJECT_DIR / "scripts" / "spike-dynamic-dispatch" / "classify.py"
_LOADER = EnsembleLoader()


async def _run(turn: dict[str, Any]) -> dict[str, Any]:
    config = _LOADER.load_from_file(str(SKELETON))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)
    result = await executor.execute(config, json.dumps(turn))
    outcome: dict[str, Any] = json.loads(result["results"]["marshal"]["response"])
    return outcome


def _classify_target(turn: dict[str, Any]) -> str:
    envelope = json.dumps({"input_data": json.dumps(turn)})
    out = subprocess.run(
        [sys.executable, str(CLASSIFY)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    target: str = json.loads(out)["target"]
    return target


async def main() -> int:
    failures: list[str] = []

    # 1. build turn routed to the solo seat -> file write
    solo = await _run({"task": "build a thing", "seat": "dd-echo-solo"})
    if solo.get("finish") is not False or "SOLO-CODE" not in solo.get("content", ""):
        failures.append(f"solo: {solo}")

    # 2. swap to the structurally different verified seat, zero skeleton change
    verified = await _run({"task": "build a thing", "seat": "dd-echo-verified"})
    if verified.get("finish") is not False or "VERIFIED-CODE" not in verified.get(
        "content", ""
    ):
        failures.append(f"verified: {verified}")

    # 3. non-build turn routed to an explain seat -> prose finish
    explain = await _run({"task": "explain the thing", "seat": "dd-echo-explain"})
    if explain.get("finish") is not True or "EXPLAIN-PROSE" not in explain.get(
        "content", ""
    ):
        failures.append(f"explain: {explain}")

    # 4. deterministic default routing (no override)
    if _classify_target({"task": "explain foo.py"}) != "dd-seat-explain":
        failures.append("default explain routing did not reach dd-seat-explain")
    if _classify_target({"task": "build a cli"}) != "dd-seat-code-solo":
        failures.append("default build routing did not reach dd-seat-code-solo")

    if failures:
        print("SMOKE FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print(
        "SMOKE OK: the dispatch skeleton routes build / swap / non-build "
        "correctly (classify -> seat -> marshal), no models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
