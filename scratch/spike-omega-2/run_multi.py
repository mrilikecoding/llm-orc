#!/usr/bin/env python3
"""Spike Ω-2 harness — multi-turn ensemble with substrate state.

The harness acts as the OpenCode-side adapter for the agent-turn-omega2
ensemble. Each turn:
  1. Load the substrate (session_state.json) and the ensemble.
  2. Invoke the ensemble with {task, substrate_path, last_tool_result}.
  3. Read the marshal's output:
     - tool_call → simulate the OpenCode `write` (write file to disk),
       advance the turn counter, re-invoke.
     - finish_reason=stop → terminate the loop.

Cross-file coherence check (the §2 substrate-+-scripts bet): after the
multi-turn session, parse the produced files and verify the dependent
files (cli.py, README.md) reference real APIs from the produced
siblings — the bespoke's Finding H test.

Usage:
    uv run python scratch/spike-omega-2/run_multi.py
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
    / "spike-omega"
    / "agent-turn-omega2.yaml"
)

OUT_DIR = Path(__file__).resolve().parent / "out"
SUBSTRATE_PATH = OUT_DIR / "session_state.json"

# Three-file temperature-library task — the exact shape the bespoke
# LoopDriver handled through ADR-040 deterministic completeness + ADR-039
# content anchor. A real test of whether the ensemble form alone
# preserves coherence, with substrate replacing the bespoke's anchor.
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


def write_initial_substrate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove prior artifacts
    for p in OUT_DIR.glob("*.py"):
        p.unlink()
    if SUBSTRATE_PATH.exists():
        SUBSTRATE_PATH.unlink()
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
    print(f"[Ω-2] wrote initial substrate to {SUBSTRATE_PATH}")
    print(f"       requested: {PLAN_QUEUE}")


def simulate_client_write(marshal_output: str, produced_dir: Path) -> Path | None:
    try:
        parsed = json.loads(marshal_output)
    except json.JSONDecodeError:
        return None
    if "tool_calls" not in parsed:
        return None
    tc = parsed["tool_calls"][0]
    fn = tc["function"]
    args = json.loads(fn["arguments"])
    file_path = args.get("filePath")
    content = args.get("content", "")
    if not file_path:
        return None
    # Files land in ./produced/ so the parse stage's sibling reader sees them.
    dest = produced_dir / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest


def check_cross_file_coherence() -> dict:
    """The bespoke's Finding H test, run on the Ω-2 ensemble's output.

    Thebespoke LoopDriver passed only via ADR-039's content-anchor (sibling
    signatures in the dispatch input). Ω-2 sends full sibling content via
    the parse script's filesystem read. Check whether cli.py and
    README.md reference the real converters.py APIs (no invention).
    """
    converters_src = (OUT_DIR / "produced" / "converters.py").read_text() if (
        OUT_DIR / "produced" / "converters.py"
    ).exists() else ""
    cli_src = (OUT_DIR / "produced" / "cli.py").read_text() if (
        OUT_DIR / "produced" / "cli.py"
    ).exists() else ""
    readme_src = (OUT_DIR / "produced" / "README.md").read_text() if (
        OUT_DIR / "produced" / "README.md"
    ).exists() else ""

    result = {"produced_files": [], "converters_apis": [], "cli": {}, "readme": {}}

    if converters_src:
        # Extract defined functions from converters.py.
        try:
            tree = ast.parse(converters_src)
            funcs = [
                n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            ]
            result["converters_apis"] = funcs
            result["produced_files"].append("converters.py")
        except SyntaxError as e:
            result["converters_apis"] = [f"SYNTAX_ERROR: {e}"]

    if cli_src:
        result["produced_files"].append("cli.py")
        # Check cli.py references real converters functions
        referencing = [f for f in result["converters_apis"] if f and f in cli_src]
        inventing = re.findall(r"converters\.(\w+)", cli_src)
        inventing = [f for f in inventing if f not in result["converters_apis"]]
        result["cli"] = {
            "references_known_apis": referencing,
            "invented_apis": inventing,
            "imports_converters": bool(re.search(r"(from|import)\s+converters", cli_src)),
        }

    if readme_src:
        result["produced_files"].append("README.md")
        referenced_in_readme = [
            f for f in result["converters_apis"] if f and f in readme_src
        ]
        result["readme"] = {
            "references_known_apis": referenced_in_readme,
            "mentions_converters_py": "converters.py" in readme_src,
            "mentions_cli_py": "cli.py" in readme_src,
        }
    return result


async def run() -> None:
    write_initial_substrate()
    project_dir = ENSEMBLE_YAML.parents[2]

    loader = EnsembleLoader()
    config = loader.load_from_file(str(ENSEMBLE_YAML))
    print(f"[Ω-2] loaded ensemble: {config.name}, {len(config.agents)} agents")

    executor = ExecutorFactory.create_root_executor(project_dir=project_dir)

    per_turn_latencies: list[float] = []
    turn = 0
    MAX_TURNS = 6  # safety cap (3 deliverables + 2 retries)

    while turn < MAX_TURNS:
        turn += 1
        print(f"\n[Ω-2] === turn {turn} ===")
        request = json.dumps(
            {"task": TASK, "substrate_path": str(SUBSTRATE_PATH), "last_tool_result": ""}
        )

        start = time.perf_counter()
        result = await executor.execute(config, request)
        elapsed = time.perf_counter() - start
        per_turn_latencies.append(elapsed)

        # Show per-stage status
        if isinstance(result, dict):
            for name, res in result.get("results", {}).items():
                if isinstance(res, dict):
                    status = res.get("status", "?")
                    response = res.get("response", "")
                    preview = (str(response)[:100] if response else "<empty>")
                    print(f"       {name:<14} {status:<8} {preview}")
                    if status != "success":
                        print(f"       --- full {name} result ---")
                        print(json.dumps(res, indent=2, default=str)[:1500])
                        print(f"       --- end {name} ---")

        # Inspect marshal output
        marshal_result = (
            result.get("results", {}).get("marshal", {}).get("response", "")
            if isinstance(result, dict)
            else ""
        )

        if not marshal_result:
            print(f"[Ω-2] turn {turn}: no marshal output (latency {elapsed:.2f}s)")
            break

        try:
            parsed = json.loads(marshal_result)
        except json.JSONDecodeError:
            print(f"[Ω-2] turn {turn}: marshal output not JSON")
            break

        if "tool_calls" in parsed:
            tc = parsed["tool_calls"][0]
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            print(
                f"       latency: {elapsed:.2f}s → "
                f"tool_call write {args.get('filePath')} "
                f"({len(args.get('content', ''))} bytes)"
            )
            dest = simulate_client_write(marshal_result, OUT_DIR / "produced")
            if dest is None:
                print("       simulate_client_write returned None")
                break
            # Verify the file parses (if Python).
            if dest.suffix == ".py":
                try:
                    ast.parse(dest.read_text())
                    print(f"       wrote + parses: {dest.name}")
                except SyntaxError as e:
                    print(f"       wrote but does NOT parse: {e}")
            else:
                print(f"       wrote: {dest.name}")
            continue

        if "finish_reason" in parsed or "content" in parsed:
            print(f"       latency: {elapsed:.2f}s → {parsed}")
            break

        print(f"       unexpected marshal output: {parsed}")
        break

    print(f"\n[Ω-2] session complete: {turn} turns, total "
          f"{sum(per_turn_latencies):.2f}s, "
          f"per-turn {[f'{l:.2f}' for l in per_turn_latencies]}s")

    print("\n[Ω-2] cross-file coherence check (Findings §10.2 bet):")
    coherence = check_cross_file_coherence()
    print(json.dumps(coherence, indent=2))

    final_state = json.loads(SUBSTRATE_PATH.read_text())
    print("\n[Ω-2] final substrate state:")
    print(f"       produced: {final_state.get('produced')}")
    print(f"       plan_queue: {final_state.get('plan_queue')}")


if __name__ == "__main__":
    asyncio.run(run())