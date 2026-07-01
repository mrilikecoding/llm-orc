#!/usr/bin/env python3
"""Spike Ω-dispatch harness — adapter-mediated dynamic dispatch.

The §6 primitive-4 test. The engine cannot resolve a runtime-chosen
`ensemble:` target (it reads a static YAML string). So the turn splits:
  1. DECIDE ensemble (parse -> plan -> score) picks the capability by
     reading the library, and emits the capability's file path.
  2. The ADAPTER (this harness) loads the chosen capability by path and
     invokes it — the dispatcher block, demarcated below, whose line
     count is the §4b 5a-vs-5b signal (~30 lines => keep it adapter-side).
  3. Validate + write + advance the substrate; recover on form failure.

Two-deliverable task (converters.py + README.md) so both routes fire:
.py -> code-generator-omega, .md -> prose-generator-omega.

Usage:
    uv run python scratch/spike-omega-dispatch/run_dispatch.py
"""

from __future__ import annotations

import asyncio
import ast
import json
import re
import time
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

DECIDE_YAML = (
    Path(__file__).resolve().parents[2]
    / ".llm-orc" / "ensembles" / "spike-omega-dispatch" / "agent-turn-dispatch.yaml"
)
OUT_DIR = Path(__file__).resolve().parent / "out"
SUBSTRATE_PATH = OUT_DIR / "session_state.json"

TASK = (
    "Create a small temperature library:\n"
    "  - converters.py: def celsius_to_fahrenheit(c), fahrenheit_to_celsius(f), "
    "celsius_to_kelvin(c). Use float math. No imports.\n"
    "  - README.md: brief Markdown documentation for converters.py, naming the "
    "real functions.\n"
)
PLAN_QUEUE = ["converters.py", "README.md"]
MAX_RETRIES = 2
MAX_TURNS = 8


def write_initial_substrate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "produced").mkdir(exist_ok=True)
    for p in (OUT_DIR / "produced").glob("*"):
        p.unlink()
    SUBSTRATE_PATH.write_text(json.dumps({
        "task": TASK, "requested": list(PLAN_QUEUE), "produced": [],
        "plan_queue": list(PLAN_QUEUE), "remaining_anchor": "",
    }, indent=2))


def load_state() -> dict:
    return json.loads(SUBSTRATE_PATH.read_text())


def extract_terminal_content(cap_result: dict) -> str:
    """Pull the deliverable from the capability's execution result."""
    results = cap_result.get("results", {}) if isinstance(cap_result, dict) else {}
    if not results:
        return ""
    terminal = list(results.keys())[-1]
    node = results.get(terminal, {})
    content = node.get("response", "") if isinstance(node, dict) else ""
    return content if isinstance(content, str) else ""


def clean_and_validate(content: str, file_path: str) -> tuple[bool, str, str]:
    """Minimal inline gate (fence-strip + ast.parse for .py, non-empty for .md).

    The dispatch spike isn't re-testing the gate (that is the deferred
    gate-strength question); this just confirms the dispatched output is
    usable enough to write and to drive the recovery path.
    """
    stripped = content.strip()
    if "```" in stripped:
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    if not stripped:
        return False, "", "empty content"
    if file_path.endswith(".py"):
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            return False, stripped, f"ast.parse: {e}"
    return True, stripped, ""


def update_substrate(file_path: str) -> None:
    state = load_state()
    if file_path not in state["produced"]:
        state["produced"].append(file_path)
    state["plan_queue"] = [p for p in state["plan_queue"] if p != file_path]
    state["remaining_anchor"] = (
        f"Produce {state['plan_queue'][0]} next." if state["plan_queue"]
        else "All deliverables produced."
    )
    SUBSTRATE_PATH.write_text(json.dumps(state, indent=2))


async def run() -> None:
    write_initial_substrate()
    project_dir = DECIDE_YAML.parents[2]
    loader = EnsembleLoader()
    decide_config = loader.load_from_file(str(DECIDE_YAML))
    executor = ExecutorFactory.create_root_executor(project_dir=project_dir)
    print(f"[Ω-disp] loaded decide ensemble: {decide_config.name}")

    per_turn: list[float] = []
    routing: dict[str, str] = {}
    retry_counts: dict[str, int] = {}
    turn = 0

    while turn < MAX_TURNS:
        state = load_state()
        if not state["plan_queue"]:
            print("[Ω-disp] queue drained.")
            break
        target = state["plan_queue"][0]
        retries = retry_counts.get(target, 0)
        turn += 1
        last_tool_result = (
            f"PRODUCTION REJECTED {target}: {retry_counts.get(target + '__err', '')}"
            if retries > 0 else ""
        )
        print(f"\n[Ω-disp] === turn {turn} === target={target} "
              f"(attempt {retries + 1}/{MAX_RETRIES + 1})")

        start = time.perf_counter()
        request = json.dumps({
            "task": TASK, "substrate_path": str(SUBSTRATE_PATH),
            "last_tool_result": last_tool_result,
        })
        decide_result = await executor.execute(decide_config, request)
        score_raw = (
            decide_result.get("results", {}).get("score", {}).get("response", "")
            if isinstance(decide_result, dict) else ""
        )
        try:
            decision = json.loads(score_raw)
        except json.JSONDecodeError:
            print(f"       score did not emit JSON: {score_raw[:200]}")
            break
        cap_name = decision.get("capability_name")
        cap_path = decision.get("capability_path")
        dispatch_input = decision.get("dispatch_input", "")
        print(f"       score: {target} -> {cap_name}  scores={decision.get('scores')}")
        routing[target] = cap_name
        if not cap_path:
            print("       score produced no capability_path; stopping.")
            break

        # ===== DISPATCHER (adapter-mediated dynamic dispatch) =====
        # The engine can't resolve a runtime-chosen ensemble target, so the
        # adapter does it: load the score-chosen capability BY PATH (which
        # also sidesteps the engine's non-recursive name resolver) and
        # invoke it. This block is the §4b 5a-vs-5b measurement.
        cap_config = loader.load_from_file(cap_path)
        cap_result = await executor.execute(cap_config, dispatch_input)
        content = extract_terminal_content(cap_result)
        # ===== END DISPATCHER =====

        elapsed = time.perf_counter() - start
        per_turn.append(elapsed)

        ok, cleaned, err = clean_and_validate(content, target)
        if not ok:
            print(f"       {elapsed:.1f}s → {cap_name} produced INVALID: {err}")
            if retries < MAX_RETRIES:
                retry_counts[target] = retries + 1
                retry_counts[target + "__err"] = err
                print(f"       → adapter retry {retries + 1}/{MAX_RETRIES}")
                continue
            print(f"       → gave up on {target} after {MAX_RETRIES} retries")
            update_substrate(target)  # drop from queue (not added to produced on give-up handled below)
            state = load_state()
            if target in state["produced"]:
                state["produced"].remove(target)
                SUBSTRATE_PATH.write_text(json.dumps(state, indent=2))
            retry_counts.pop(target, None)
            retry_counts.pop(target + "__err", None)
            continue

        dest = OUT_DIR / "produced" / target
        dest.write_text(cleaned)
        update_substrate(target)
        retry_counts.pop(target, None)
        retry_counts.pop(target + "__err", None)
        print(f"       {elapsed:.1f}s → {cap_name} wrote {target} ({len(cleaned)} bytes)")

    print(f"\n[Ω-disp] session: {turn} turns, total {sum(per_turn):.1f}s, "
          f"per-turn {[f'{t:.0f}' for t in per_turn]}s")
    print(f"[Ω-disp] routing decisions: {json.dumps(routing, indent=2)}")
    final = load_state()
    print(f"[Ω-disp] final produced: {final['produced']}")
    print(f"[Ω-disp] produced/ on disk: {sorted(p.name for p in (OUT_DIR / 'produced').glob('*'))}")


if __name__ == "__main__":
    asyncio.run(run())
