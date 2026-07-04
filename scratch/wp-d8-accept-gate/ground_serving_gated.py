#!/usr/bin/env python3
"""WP-D8 slice-4 grounding — a build turn through the whole serving ensemble.

Drives serving.yaml directly (the harness level): classify -> resolve (maps the
code intent to build-gated) -> seat (dispatches build-gated: test_writer ->
code_writer -> gather -> executor -> judge -> accept_gate -> envelope) -> shape
(carries the verdict) -> form_gate -> emit. Confirms an accepted build turn
emits a file-write outcome carrying the code, through the real skeleton.

Run: uv run python scratch/wp-d8-accept-gate/ground_serving_gated.py
"""

import asyncio
import json
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
SERVING = PROJECT / "ensembles" / "agentic-serving" / "serving.yaml"

TURN = "write a function is_palindrome(s) that returns True if s reads the same forwards and backwards"


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
    cfg = EnsembleLoader().load_from_file(str(SERVING))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print(f"turn: {TURN}\n")
    result = await executor.execute(cfg, TURN)
    results = result.get("results", {})

    resolve = json.loads(_node(results, "resolve") or "{}")
    print(f"resolve.target = {resolve.get('target')!r}  build = {resolve.get('build')}")

    emit_raw = _node(results, "emit")
    try:
        emit = json.loads(emit_raw)
    except json.JSONDecodeError:
        print("emit (unparsed):", emit_raw[:300])
        return

    print(f"\nemit.finish = {emit.get('finish')}")
    if emit.get("finish") is False:
        print(f"emit.file = {emit.get('file')!r}")
        print("emit.content (the deliverable):\n" + str(emit.get("content", "")))
        routed = resolve.get("target") == "build-gated"
        wrote = bool(emit.get("file")) and bool(str(emit.get("content", "")).strip())
        print(
            "\nRESULT:",
            "GROUNDED — a build turn routed through build-gated and emitted a "
            "file-write on accept"
            if routed and wrote
            else "*** integration gap ***",
        )
    else:
        print("emit.content:", str(emit.get("content", ""))[:400])
        print("\n(the gate routed another round or the turn was non-build)")


if __name__ == "__main__":
    asyncio.run(main())
