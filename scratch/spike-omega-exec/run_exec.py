#!/usr/bin/env python3
"""Spike Ω-exec — execution gate as a stage + escalation cascade.

Contract-first (E) build at the CHEAP tier for everything, then an EXECUTION
GATE (run the package's tests) as a stage. On failure the adapter attributes
the culprit file(s) and escalates ONLY those up the code ladder
(cheap 8b → standard 14b → frontier qwen3.6-plus), rebuilds, and re-runs the
tests, until they pass or the ladder is exhausted.

This composes strategy E (frozen contract) + strategy C (cascade) with a
DETERMINISTIC, EXECUTIONAL escalation trigger ("the test failed") instead of
the literature's usual self-uncertainty trigger. It directly tests the thesis:
frontier-quality (a running, correct package) at minimal frontier tokens
(escalate only what the tests prove is broken).

Default task is `calc` (its 8b parser is logically broken — the gate should
catch it and escalate to fix it).

Usage:
    uv run python scratch/spike-omega-exec/run_exec.py [calc|todo]
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spike-omega-e"))
from run_e import (  # noqa: E402
    TASKS, build_dispatch_input, contract_gate, extract_terminal, parse_json_obj,
)

ENS = Path(__file__).resolve().parents[2] / ".llm-orc" / "ensembles"
ARCHITECT_YAML = ENS / "spike-omega-e" / "architect.yaml"
PROSE_CAP = ENS / "spike-omega-dispatch" / "prose-generator-omega.yaml"
PROJECT_DIR = ENS.parent
OUT_DIR = Path(__file__).resolve().parent / "out"
MAX_RETRIES = 1          # within-tier structural retries
MAX_EXEC_ROUNDS = 4      # execution-gate escalation rounds

CODE_LADDER = ["cheap", "standard", "frontier"]
TIER_CAP = {
    "cheap": ENS / "spike-omega" / "code-generator-omega.yaml",            # qwen3:8b
    "standard": ENS / "spike-omega-tiers" / "code-generator-standard.yaml",  # qwen3:14b
    "frontier": ENS / "spike-omega-tiers" / "code-generator-frontier.yaml",  # qwen3.6-plus
}

_LOADER = EnsembleLoader()
_ARCHITECT = _LOADER.load_from_file(str(ARCHITECT_YAML))


def next_tier(tier: str) -> str | None:
    i = CODE_LADDER.index(tier)
    return CODE_LADDER[i + 1] if i + 1 < len(CODE_LADDER) else None


async def build_at(executor, d: dict, contract: list[dict], tier: str,
                   extra_hint: str = "") -> tuple[str, bool]:
    cap_path = str(PROSE_CAP if d["kind"] == "markdown_doc" else TIER_CAP[tier])
    cap = _LOADER.load_from_file(cap_path)
    hint, last = extra_hint, ""
    for attempt in range(MAX_RETRIES + 1):
        use_hint = hint if (attempt > 0 or extra_hint) else ""
        di = build_dispatch_input(d, contract, use_hint)
        content = extract_terminal(await executor.execute(cap, di))
        ok, why, cleaned = contract_gate(content, d, contract)
        last = cleaned
        if ok:
            return cleaned, True
        hint = why
    return last, False


def run_tests(out_dir: Path, contract: list[dict]) -> tuple[bool, str]:
    test_files = [d["file"] for d in contract if d["file"].startswith("test_")]
    if not test_files:
        return True, "no tests"
    tmp = Path(tempfile.mkdtemp(prefix="omega_exec_"))
    try:
        for d in contract:
            src = out_dir / d["file"]
            if src.exists():
                shutil.copy(src, tmp / d["file"])
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files],
            cwd=tmp, capture_output=True, text=True)
        out = r.stdout + r.stderr
        return r.returncode == 0, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pick_culprits(out: str, contract: list[dict], tiers: dict[str, str]) -> list[str]:
    code = [d["file"] for d in contract
            if not d["file"].startswith("test_") and d["kind"] != "markdown_doc"]
    escalatable = [f for f in code if next_tier(tiers[f]) is not None]
    # exception-style failure: a code file named in the traceback → minimal
    named = [f for f in escalatable if f in out]
    if named:
        return named
    # assertion-style failure (no code frame): escalate all not-yet-top code files
    return escalatable


async def run() -> None:
    task_name = sys.argv[1] if len(sys.argv) > 1 else "calc"
    task = TASKS.get(task_name, TASKS["calc"])
    out_dir = OUT_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*"):
        if p.is_file():
            p.unlink()
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)

    print(f"[Ω-exec] architect (frontier) for task '{task_name}'...")
    ar = await executor.execute(_ARCHITECT, task)
    raw = (ar.get("results", {}).get("architect", {}).get("response", "")
           if isinstance(ar, dict) else "")
    contract = parse_json_obj(raw).get("deliverables", [])
    if not contract:
        print(f"[Ω-exec] no contract:\n{raw[:300]}")
        return
    print(f"[Ω-exec] contract: {[d['file'] for d in contract]}")

    tiers = {d["file"]: "cheap" for d in contract}
    t0 = time.perf_counter()
    print("[Ω-exec] initial build — all CHEAP (8b)...")
    for d in contract:
        content, ok = await build_at(executor, d, contract, "cheap")
        (out_dir / d["file"]).write_text(content)
        print(f"    {d['file']}: cheap {'ok' if ok else 'gate-GAVEUP'}")

    rounds = 0
    passed, out = run_tests(out_dir, contract)
    print(f"[Ω-exec] exec gate (all cheap): {'PASS' if passed else 'FAIL'}")
    while not passed and rounds < MAX_EXEC_ROUNDS:
        rounds += 1
        culprits = pick_culprits(out, contract, tiers)
        if not culprits:
            print("[Ω-exec] nothing left to escalate; stopping.")
            break
        fail_tail = out.strip().splitlines()[-1:] or [""]
        for f in culprits:
            nt = next_tier(tiers[f])
            tiers[f] = nt
            d = next(dd for dd in contract if dd["file"] == f)
            content, ok = await build_at(executor, d, contract, nt,
                                         extra_hint=f"the package test failed: {fail_tail[0]}")
            (out_dir / f).write_text(content)
            print(f"  [Ω-exec] round {rounds}: escalated {f} -> {nt} ({'ok' if ok else 'gate-GAVEUP'})")
        passed, out = run_tests(out_dir, contract)
        print(f"[Ω-exec] exec gate (round {rounds}): {'PASS' if passed else 'FAIL'}")

    elapsed = time.perf_counter() - t0
    frontier_files = [f for f, t in tiers.items() if t == "frontier"]
    result = {
        "task": task_name, "passed": passed, "escalation_rounds": rounds,
        "elapsed_s": round(elapsed, 1), "final_tiers": tiers,
        "frontier_files": frontier_files,
        "frontier_token_files": len(frontier_files),
    }
    print("\n==== RESULT ====")
    print(json.dumps(result, indent=2))
    if not passed:
        print("---- last test output ----")
        print(out.strip()[-600:])
    (out_dir / "_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
