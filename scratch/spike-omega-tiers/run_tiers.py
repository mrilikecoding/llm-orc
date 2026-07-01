#!/usr/bin/env python3
"""Spike Ω-tiers — contract-first (E) + per-sub-task model-tier routing.

The frontier architect emits the frozen contract AND a tier per deliverable
(the smallest local model it judges can implement that file). The build routes
each file to its tier's coder; the contract gate enforces correctness; recovery
retries within-tier. Measures: does tier-routed build still produce a running
package, what did the architect assign, and per-tier first-attempt success
(the first data toward the model-ceiling questions).

Usage:
    uv run python scratch/spike-omega-tiers/run_tiers.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spike-omega-e"))
from run_e import (  # noqa: E402
    TASK, build_dispatch_input, clean_content, contract_gate,
    extract_terminal, parse_json_obj, score,
)

ENS = Path(__file__).resolve().parents[2] / ".llm-orc" / "ensembles"
ARCHITECT_YAML = ENS / "spike-omega-tiers" / "architect-tiers.yaml"
PROJECT_DIR = ENS.parent
OUT_DIR = Path(__file__).resolve().parent / "out"
MAX_RETRIES = 2

TIER_CAP = {
    "micro": ENS / "spike-omega-tiers" / "code-generator-micro.yaml",      # qwen3:0.6b
    "small": ENS / "spike-omega-tiers" / "code-generator-small.yaml",      # qwen3:1.7b
    "cheap": ENS / "spike-omega" / "code-generator-omega.yaml",            # qwen3:8b
    "standard": ENS / "spike-omega-tiers" / "code-generator-standard.yaml",  # qwen3:14b
}
PROSE_CAP = ENS / "spike-omega-dispatch" / "prose-generator-omega.yaml"     # qwen3:8b

_LOADER = EnsembleLoader()
_ARCHITECT = _LOADER.load_from_file(str(ARCHITECT_YAML))


TIER_ORDER = ["micro", "small", "cheap", "standard"]


async def build_file(executor, d: dict, contract: list[dict]) -> tuple[str, bool, int, str]:
    """Build with escalation: start at the assigned tier, climb on give-up.

    Returns (content, ok, total_attempts, success_tier). The success_tier is
    the minimum tier that cleared the contract gate — the ceiling data point.
    """
    if d["kind"] == "markdown_doc":
        tiers = ["cheap"]  # prose at cheap; docs don't escalate
    else:
        start = d.get("tier", "cheap")
        start = start if start in TIER_ORDER else "cheap"
        tiers = TIER_ORDER[TIER_ORDER.index(start):]
    attempts_total, last = 0, ""
    for tier in tiers:
        cap_path = str(PROSE_CAP if d["kind"] == "markdown_doc" else TIER_CAP[tier])
        cap = _LOADER.load_from_file(cap_path)
        hint = ""
        for attempt in range(MAX_RETRIES + 1):
            attempts_total += 1
            di = build_dispatch_input(d, contract, hint if attempt > 0 else "")
            content = extract_terminal(await executor.execute(cap, di))
            ok, why, cleaned = contract_gate(content, d, contract)
            last = cleaned
            if ok:
                return cleaned, True, attempts_total, tier
            hint = why
        # tier exhausted → escalate to the next tier up
    return last, False, attempts_total, tiers[-1]


async def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)

    print("[Ω-tiers] architect (frontier) emitting contract + tier assignments...")
    t0 = time.perf_counter()
    ar = await executor.execute(_ARCHITECT, TASK)
    raw = (ar.get("results", {}).get("architect", {}).get("response", "")
           if isinstance(ar, dict) else "")
    contract = parse_json_obj(raw).get("deliverables", [])
    arch_s = time.perf_counter() - t0
    if not contract:
        print(f"[Ω-tiers] no contract. raw:\n{raw[:400]}")
        return
    assignments = {d["file"]: d.get("tier", "?") for d in contract}
    print(f"[Ω-tiers] contract ({arch_s:.0f}s). tier assignments:")
    for f, t in assignments.items():
        print(f"    {f}: {t}")
    (OUT_DIR / "_contract.json").write_text(json.dumps(contract, indent=2))

    per_tier: dict[str, dict] = {}
    times: list[float] = []
    for d in contract:
        assigned = "cheap(prose)" if d["kind"] == "markdown_doc" else d.get("tier", "cheap")
        t1 = time.perf_counter()
        content, ok, attempts, tier = await build_file(executor, d, contract)
        el = time.perf_counter() - t1
        times.append(el)
        (OUT_DIR / d["file"]).write_text(content)
        rec = per_tier.setdefault(tier, {"files": 0, "first_try": 0, "gaveup": 0})
        rec["files"] += 1
        rec["first_try"] += 1 if (ok and attempts == 1) else 0
        rec["gaveup"] += 0 if ok else 1
        escal = "" if tier == assigned else f" (escalated from {assigned})"
        print(f"  [Ω-tiers] {d['file']} assigned={assigned} succeeded@{tier}{escal} "
              f"{'ok' if ok else 'GAVEUP'} in {attempts} attempt(s), {el:.0f}s")

    result = {"arm": "tiers", "architect_s": round(arch_s, 1),
              "build_s": round(sum(times), 1), "tier_assignments": assignments,
              "per_tier": per_tier, "score": score(OUT_DIR, contract)}
    print("\n==== RESULT ====")
    print(json.dumps(result, indent=2))
    (OUT_DIR / "_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
