#!/usr/bin/env python3
"""WP-D8 slice-3 grounding — the full build-gated shape with real models.

Runs a build turn through test_writer -> code_writer -> gather -> executor ->
judge -> accept_gate -> envelope and prints the pipeline trace + the final
ADR-024 envelope (code deliverable + accept verdict in diagnostics). Grounds the
composed shape end-to-end before it is wired into serving (slice 4).

Run: uv run python scratch/wp-d8-accept-gate/ground_build_gated.py
"""

import asyncio
import json
import sys
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
BUILD_GATED = PROJECT / "ensembles" / "agentic-serving" / "build-gated.yaml"

CRITERIA = "Write is_prime(n): return True if n is a prime number, else False."


def _terminal(text: str) -> str:
    current = text
    for _ in range(6):
        try:
            obj = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return current
        if not isinstance(obj, dict):
            return current
        for key in ("deliverable", "output"):
            if isinstance(obj.get(key), str):
                current = obj[key]
                break
        else:
            results = obj.get("results")
            if isinstance(results, dict) and results:
                node = results[list(results.keys())[-1]]
                current = node.get("response", "") if isinstance(node, dict) else str(node)
                continue
            return current
    return current


def _node(results: dict, name: str) -> str:
    return _terminal(results.get(name, {}).get("response", "") or "")


async def main() -> None:
    cfg = EnsembleLoader().load_from_file(str(BUILD_GATED))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print(f"criteria: {CRITERIA}\n")
    result = await executor.execute(cfg, CRITERIA)
    results = result.get("results", {})

    executor_out = _node(results, "executor")
    judge_out = _node(results, "judge")
    gate_out = _node(results, "accept_gate")
    env_out = _node(results, "envelope")

    print("--- pipeline trace ---")
    for name in ("test_writer", "code_writer", "gather", "executor", "judge", "accept_gate"):
        status = results.get(name, {}).get("status", "?")
        print(f"  {name:12s} status={status}")
    print()

    try:
        gate = json.loads(gate_out)
        print(f"executor tests_pass = {json.loads(executor_out).get('tests_pass')}")
        print(f"judge tests_adequate = {json.loads(judge_out).get('tests_adequate')}")
        print(f"gate accept = {gate.get('accept')}  reason = {gate.get('reason')!r}")
    except (json.JSONDecodeError, TypeError) as err:
        print(f"(could not parse a verdict: {err})")
        print("executor:", executor_out[:200])
        print("judge:", judge_out[:200])
        print("gate:", gate_out[:200])

    print("\n--- final envelope ---")
    try:
        env = json.loads(env_out)
        print("code deliverable:\n" + env["artifacts"][0]["content"])
        print("\ndiagnostics:", json.dumps(env["diagnostics"], indent=2))
        ok = env["diagnostics"]["accept"] and env["artifacts"][0]["content"].strip()
        print("\nRESULT:", "GROUNDED — build-gated produced code + an accept verdict"
              if ok else "*** envelope missing code or verdict ***")
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        print(f"*** envelope parse failed: {err}")
        print(env_out[:400])
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
