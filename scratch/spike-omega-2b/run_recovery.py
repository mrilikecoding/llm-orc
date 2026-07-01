#!/usr/bin/env python3
"""Spike Ω-2b harness — multi-turn ensemble with adapter-side recovery.

Extends the Ω-2 harness with the one behavior Ω-2 lacked: a retry loop.
The harness IS the between-turn adapter. Each turn:
  1. Load the substrate (session_state.json); take plan_queue[0] as the
     target file.
  2. Invoke the ensemble with {task, substrate_path, last_tool_result}.
     last_tool_result carries the rejection feedback on a retry, "" first.
  3. Read marshal output:
     - tool_call → simulate OpenCode write; marshal already advanced the
       substrate; clear this file's retry count; continue.
     - finish_reason=stop "validate failed ..." → recoverable. If retries
       remain (MAX_RETRIES=2), re-invoke the SAME file with the ast error
       as last_tool_result. Else give up on this file: drop it from the
       queue (NOT added to produced) and move on, so we can measure how
       many of the three files converge.
     - finish_reason=stop "All ... produced" → done.

The point: does adapter-side retry (ADR-041 self-healing, relocated
between turns) make the 3-file task converge where the Ω-2 form did not?
No engine primitive added.

Usage:
    uv run python scratch/spike-omega-2b/run_recovery.py
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

ENSEMBLE_YAML = (
    Path(__file__).resolve().parents[2]
    / ".llm-orc"
    / "ensembles"
    / "spike-omega-2b"
    / "agent-turn-omega2b.yaml"
)

OUT_DIR = Path(__file__).resolve().parent / "out"
SUBSTRATE_PATH = OUT_DIR / "session_state.json"

TASK = (
    "Create a small temperature library with three Python files:\n"
    "  - converters.py: def celsius_to_fahrenheit(c), fahrenheit_to_celsius(f), "
    "celsius_to_kelvin(c). Use float math. No imports.\n"
    "  - cli.py: a small argparse CLI exposing `convert c-to-f`, `convert f-to-c`, "
    "`convert c-to-k` subcommands, calling converters.py functions.\n"
    "  - README.md: brief Markdown documentation for both converters.py and cli.py, "
    "naming the real functions.\n"
)

PLAN_QUEUE = ["converters.py", "cli.py", "README.md"]

MAX_RETRIES = 2          # adapter-side retries per file before giving up
MAX_TURNS = 14           # safety: 3 files x (1 + 2 retries) + slack


def write_initial_substrate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "produced").mkdir(exist_ok=True)
    for p in (OUT_DIR / "produced").glob("*"):
        p.unlink()
    state = {
        "task": TASK,
        "requested": list(PLAN_QUEUE),
        "produced": [],
        "plan_queue": list(PLAN_QUEUE),
        "remaining_anchor": "",
    }
    SUBSTRATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"[Ω-2b] wrote initial substrate; requested: {PLAN_QUEUE}")


def load_state() -> dict:
    return json.loads(SUBSTRATE_PATH.read_text())


def drop_from_queue(file_path: str) -> None:
    """Give-up path: remove a file from plan_queue without marking produced."""
    state = load_state()
    state["plan_queue"] = [p for p in state.get("plan_queue", []) if p != file_path]
    state["remaining_anchor"] = (
        f"Produce {state['plan_queue'][0]} next."
        if state["plan_queue"]
        else "Gave up on remaining; queue drained."
    )
    SUBSTRATE_PATH.write_text(json.dumps(state, indent=2))


def simulate_client_write(marshal_output: str, produced_dir: Path) -> Path | None:
    try:
        parsed = json.loads(marshal_output)
    except json.JSONDecodeError:
        return None
    if "tool_calls" not in parsed:
        return None
    tc = parsed["tool_calls"][0]
    args = json.loads(tc["function"]["arguments"])
    file_path = args.get("filePath")
    content = args.get("content", "")
    if not file_path:
        return None
    dest = produced_dir / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest


def extract_validate_error(content: str) -> str:
    # "marshal: validate failed for converters.py: ast.parse: <error>"
    m = re.search(r"validate failed for [^:]+:\s*(.*)", content)
    return m.group(1).strip() if m else content.strip()


def check_cross_file_coherence() -> dict:
    pdir = OUT_DIR / "produced"

    def read(name: str) -> str:
        p = pdir / name
        return p.read_text() if p.exists() else ""

    converters_src, cli_src, readme_src = (
        read("converters.py"),
        read("cli.py"),
        read("README.md"),
    )
    result: dict = {"produced_files": [], "converters_apis": [], "cli": {}, "readme": {}}

    if converters_src:
        try:
            tree = ast.parse(converters_src)
            result["converters_apis"] = [
                n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            ]
            result["produced_files"].append("converters.py")
        except SyntaxError as e:
            result["converters_apis"] = [f"SYNTAX_ERROR: {e}"]

    if cli_src:
        result["produced_files"].append("cli.py")
        referencing = [f for f in result["converters_apis"] if f and f in cli_src]
        inventing = [
            f for f in re.findall(r"converters\.(\w+)", cli_src)
            if f not in result["converters_apis"]
        ]
        result["cli"] = {
            "references_known_apis": referencing,
            "invented_apis": inventing,
            "imports_converters": bool(re.search(r"(from|import)\s+converters", cli_src)),
        }

    if readme_src:
        result["produced_files"].append("README.md")
        result["readme"] = {
            "references_known_apis": [
                f for f in result["converters_apis"] if f and f in readme_src
            ],
            "mentions_converters_py": "converters.py" in readme_src,
            "mentions_cli_py": "cli.py" in readme_src,
        }
    return result


async def run() -> None:
    write_initial_substrate()
    project_dir = ENSEMBLE_YAML.parents[2]

    loader = EnsembleLoader()
    config = loader.load_from_file(str(ENSEMBLE_YAML))
    print(f"[Ω-2b] loaded ensemble: {config.name}, {len(config.agents)} agents")

    executor = ExecutorFactory.create_root_executor(project_dir=project_dir)

    per_turn: list[float] = []
    retry_counts: dict[str, int] = {}
    file_outcomes: dict[str, str] = {}
    turn = 0

    while turn < MAX_TURNS:
        state = load_state()
        queue = state.get("plan_queue", [])
        if not queue:
            print("[Ω-2b] queue drained.")
            break
        target = queue[0]
        retries = retry_counts.get(target, 0)

        turn += 1
        last_tool_result = ""
        if retries > 0:
            last_tool_result = (
                f"PRODUCTION REJECTED {target}: "
                f"{retry_counts.get(target + '__err', '')}"
            )
        print(
            f"\n[Ω-2b] === turn {turn} === target={target} "
            f"(attempt {retries + 1}/{MAX_RETRIES + 1})"
        )

        request = json.dumps(
            {"task": TASK, "substrate_path": str(SUBSTRATE_PATH), "last_tool_result": last_tool_result}
        )

        start = time.perf_counter()
        result = await executor.execute(config, request)
        elapsed = time.perf_counter() - start
        per_turn.append(elapsed)

        marshal_result = (
            result.get("results", {}).get("marshal", {}).get("response", "")
            if isinstance(result, dict)
            else ""
        )
        if not marshal_result:
            print(f"[Ω-2b] turn {turn}: no marshal output ({elapsed:.1f}s). Stopping.")
            file_outcomes[target] = "no-marshal-output"
            break

        try:
            parsed = json.loads(marshal_result)
        except json.JSONDecodeError:
            print(f"[Ω-2b] turn {turn}: marshal output not JSON. Stopping.")
            file_outcomes[target] = "marshal-not-json"
            break

        if "tool_calls" in parsed:
            args = json.loads(parsed["tool_calls"][0]["function"]["arguments"])
            dest = simulate_client_write(marshal_result, OUT_DIR / "produced")
            parses = ""
            if dest and dest.suffix == ".py":
                try:
                    ast.parse(dest.read_text())
                    parses = " (parses)"
                except SyntaxError as e:
                    parses = f" (does NOT parse: {e})"
            print(
                f"       {elapsed:.1f}s → write {args.get('filePath')} "
                f"({len(args.get('content', ''))} bytes){parses}"
            )
            retry_counts.pop(target, None)
            retry_counts.pop(target + "__err", None)
            file_outcomes[target] = f"landed{parses}"
            # Adapter stops when the substrate queue drains (real-contract
            # shape: every produced file rides a write tool_call; the
            # adapter, not the marshal, decides the session is done).
            if not load_state().get("plan_queue", []):
                print("       queue drained after terminal write; stopping.")
                break
            continue

        content = parsed.get("content", "")
        if "validate failed" in content:
            err = extract_validate_error(content)
            print(f"       {elapsed:.1f}s → validate FAILED: {err}")
            if retries < MAX_RETRIES:
                retry_counts[target] = retries + 1
                retry_counts[target + "__err"] = err
                print(f"       → adapter retry {retries + 1}/{MAX_RETRIES} for {target}")
                continue
            print(f"       → gave up on {target} after {MAX_RETRIES} retries; dropping from queue")
            file_outcomes[target] = f"gave-up-after-{MAX_RETRIES}-retries"
            drop_from_queue(target)
            retry_counts.pop(target, None)
            retry_counts.pop(target + "__err", None)
            continue

        if "produced" in content and content.strip().startswith("All"):
            print(f"       {elapsed:.1f}s → {content}")
            break

        print(f"       {elapsed:.1f}s → unexpected stop: {content}. Stopping.")
        file_outcomes[target] = "unexpected-stop"
        break

    print(
        f"\n[Ω-2b] session: {turn} turns, total {sum(per_turn):.1f}s, "
        f"per-turn {[f'{t:.0f}' for t in per_turn]}s"
    )
    print(f"[Ω-2b] per-file outcomes: {json.dumps(file_outcomes, indent=2)}")

    final = load_state()
    print(f"[Ω-2b] final produced: {final.get('produced')}")
    print(f"[Ω-2b] final plan_queue: {final.get('plan_queue')}")

    print("\n[Ω-2b] cross-file coherence:")
    print(json.dumps(check_cross_file_coherence(), indent=2))


if __name__ == "__main__":
    asyncio.run(run())
