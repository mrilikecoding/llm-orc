#!/usr/bin/env python3
"""Spike Ω-4 prerequisite-verification harness.

Verifies the two Ω-4 prerequisites on the 3-file task that exposed the
gaps in omega-2b / Ω-dispatch:

  Prereq 1 — strengthened gate: a deterministic, destination-aware
  structural check driven by per-deliverable expectations (a CLI must
  import the sibling + use argparse; a doc must be Markdown not Python;
  a module must define the named functions). Replaces the ast.parse-only
  gate that over-reported correctness.

  Prereq 2 — grounding fix: score.py injects real sibling signatures into
  the producer input, so the producer can't invent API names.

Flow per file: decide ensemble (parse->plan->score) -> adapter dispatch to
the score-chosen capability -> structural gate -> recover on failure (the
gate's specific error becomes the retry hint) -> write -> advance.

This is the prereq check, NOT Ω-4 proper. Ω-4 proper adds the long-horizon
task + the bespoke and frontier arms.

Usage:
    uv run python scratch/spike-omega-4/run_omega4.py
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
    / ".llm-orc" / "ensembles" / "spike-omega-4" / "agent-turn-omega4.yaml"
)
OUT_DIR = Path(__file__).resolve().parent / "out"
SUBSTRATE_PATH = OUT_DIR / "session_state.json"

TASK = (
    "Create a small temperature library with three files:\n"
    "  - converters.py: def celsius_to_fahrenheit(c), fahrenheit_to_celsius(f), "
    "celsius_to_kelvin(c). Use float math. No imports.\n"
    "  - cli.py: an argparse CLI that imports converters and exposes the three "
    "conversions, calling converters' functions (do not re-implement them).\n"
    "  - README.md: Markdown documentation for converters.py and cli.py, naming "
    "the real functions.\n"
)

DELIVERABLES = [
    {"file": "converters.py", "kind": "python_module",
     "must_define": ["celsius_to_fahrenheit", "fahrenheit_to_celsius", "celsius_to_kelvin"]},
    {"file": "cli.py", "kind": "python_cli",
     "must_import": ["converters"],
     "must_reference": ["celsius_to_fahrenheit", "fahrenheit_to_celsius", "celsius_to_kelvin"],
     "must_define": ["main"]},
    {"file": "README.md", "kind": "markdown_doc",
     "must_mention": ["converters.py", "cli.py", "celsius_to_fahrenheit",
                      "fahrenheit_to_celsius", "celsius_to_kelvin"]},
]
MAX_RETRIES = 2
MAX_TURNS = 12


def write_initial_substrate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "produced").mkdir(exist_ok=True)
    for p in (OUT_DIR / "produced").glob("*"):
        p.unlink()
    queue = [d["file"] for d in DELIVERABLES]
    SUBSTRATE_PATH.write_text(json.dumps({
        "task": TASK, "requested": list(queue), "produced": [],
        "plan_queue": list(queue), "remaining_anchor": "",
        "expectations": {d["file"]: d for d in DELIVERABLES},
    }, indent=2))


def load_state() -> dict:
    return json.loads(SUBSTRATE_PATH.read_text())


def update_substrate(file_path: str, produced_ok: bool) -> None:
    state = load_state()
    if produced_ok and file_path not in state["produced"]:
        state["produced"].append(file_path)
    state["plan_queue"] = [p for p in state["plan_queue"] if p != file_path]
    state["remaining_anchor"] = (
        f"Produce {state['plan_queue'][0]} next." if state["plan_queue"]
        else "All deliverables resolved."
    )
    SUBSTRATE_PATH.write_text(json.dumps(state, indent=2))


def extract_terminal_content(cap_result: dict) -> str:
    results = cap_result.get("results", {}) if isinstance(cap_result, dict) else {}
    if not results:
        return ""
    node = results.get(list(results.keys())[-1], {})
    content = node.get("response", "") if isinstance(node, dict) else ""
    return content if isinstance(content, str) else ""


def check_deliverable(content: str, exp: dict) -> tuple[bool, str, str]:
    """Strengthened, deterministic, destination-aware structural gate."""
    stripped = content.strip()
    if "```" in stripped:
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    if not stripped:
        return False, "", "empty content"

    kind = exp.get("kind")
    if kind in ("python_module", "python_cli"):
        try:
            tree = ast.parse(stripped)
        except SyntaxError as e:
            return False, stripped, f"ast.parse: {e}"
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        imported: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module.split(".")[0])
        missing_def = [d for d in exp.get("must_define", []) if d not in defined]
        if missing_def:
            return False, stripped, f"missing required function(s): {missing_def}"
        missing_imp = [m for m in exp.get("must_import", []) if m not in imported]
        if missing_imp:
            return False, stripped, (
                f"must import sibling module(s) {missing_imp} and call their "
                "functions, do NOT re-implement them inline"
            )
        missing_ref = [r for r in exp.get("must_reference", []) if r not in stripped]
        if missing_ref:
            return False, stripped, f"missing reference(s) to sibling APIs: {missing_ref}"
        if kind == "python_cli" and "argparse" not in imported:
            return False, stripped, "a CLI must import and use argparse"
        return True, stripped, ""

    if kind == "markdown_doc":
        try:
            t = ast.parse(stripped)
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Import, ast.ImportFrom, ast.ClassDef))
                   for n in t.body):
                return False, stripped, (
                    "this must be Markdown documentation, not Python source code"
                )
        except SyntaxError:
            pass  # not valid Python => fine for Markdown
        missing = [m for m in exp.get("must_mention", []) if m not in stripped]
        if missing:
            return False, stripped, f"documentation must mention: {missing}"
        if not any(ln.lstrip().startswith("#") for ln in stripped.splitlines()):
            return False, stripped, "Markdown doc needs at least one heading (# ...)"
        return True, stripped, ""

    return True, stripped, ""


async def run() -> None:
    write_initial_substrate()
    project_dir = DECIDE_YAML.parents[2]
    loader = EnsembleLoader()
    decide_config = loader.load_from_file(str(DECIDE_YAML))
    executor = ExecutorFactory.create_root_executor(project_dir=project_dir)
    print(f"[Ω-4] loaded decide ensemble: {decide_config.name}")

    per_turn: list[float] = []
    routing: dict[str, str] = {}
    outcomes: dict[str, str] = {}
    retry_counts: dict[str, int] = {}
    turn = 0

    while turn < MAX_TURNS:
        state = load_state()
        if not state["plan_queue"]:
            print("[Ω-4] queue drained.")
            break
        target = state["plan_queue"][0]
        exp = state["expectations"][target]
        retries = retry_counts.get(target, 0)
        turn += 1
        last_tool_result = (
            f"PRODUCTION REJECTED {target}: {retry_counts.get(target + '__err', '')}"
            if retries > 0 else ""
        )
        print(f"\n[Ω-4] === turn {turn} === target={target} "
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
            print(f"       score not JSON: {score_raw[:160]}")
            break
        cap_name = decision.get("capability_name")
        cap_path = decision.get("capability_path")
        dispatch_input = decision.get("dispatch_input", "")
        routing[target] = cap_name
        if not cap_path:
            print("       no capability_path; stopping.")
            break

        cap_config = loader.load_from_file(cap_path)
        cap_result = await executor.execute(cap_config, dispatch_input)
        content = extract_terminal_content(cap_result)
        elapsed = time.perf_counter() - start
        per_turn.append(elapsed)

        ok, cleaned, err = check_deliverable(content, exp)
        print(f"       {elapsed:.1f}s → {cap_name} | gate {'PASS' if ok else 'FAIL'}"
              f"{'' if ok else ': ' + err}")
        if not ok:
            if retries < MAX_RETRIES:
                retry_counts[target] = retries + 1
                retry_counts[target + "__err"] = err
                print(f"       → adapter retry {retries + 1}/{MAX_RETRIES}")
                continue
            print(f"       → gave up on {target} after {MAX_RETRIES} retries")
            outcomes[target] = f"gave-up: {err}"
            update_substrate(target, produced_ok=False)
            retry_counts.pop(target, None)
            retry_counts.pop(target + "__err", None)
            continue

        (OUT_DIR / "produced" / target).write_text(cleaned)
        update_substrate(target, produced_ok=True)
        outcomes[target] = "correct"
        retry_counts.pop(target, None)
        retry_counts.pop(target + "__err", None)
        print(f"       wrote {target} ({len(cleaned)} bytes)")

    print(f"\n[Ω-4] session: {turn} turns, total {sum(per_turn):.1f}s, "
          f"per-turn {[f'{t:.0f}' for t in per_turn]}s")
    print(f"[Ω-4] routing: {json.dumps(routing)}")
    print(f"[Ω-4] per-file outcomes (against structural gate): {json.dumps(outcomes, indent=2)}")
    correct = sum(1 for v in outcomes.values() if v == "correct")
    print(f"[Ω-4] correctness: {correct}/{len(DELIVERABLES)} deliverables pass the structural gate")
    final = load_state()
    print(f"[Ω-4] final produced: {final['produced']}")


if __name__ == "__main__":
    asyncio.run(run())
