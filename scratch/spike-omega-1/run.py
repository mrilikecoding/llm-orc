#!/usr/bin/env python3
"""Spike Ω-1 harness.

Invokes the agent-turn-omega1 ensemble end-to-end through the existing
ensemble engine, bypassing the HTTP serving layer. Captures per-turn
latency (the measurement design docs can't produce) and prints the
glue inventory (the ranked list of bespoke shims the spike used).

Usage:
    uv run python scratch/spike-omega-1/run.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

ENSEMBLE_YAML = (
    Path(__file__).resolve().parents[2]
    / ".llm-orc"
    / "ensembles"
    / "spike-omega"
    / "agent-turn-omega1.yaml"
)

# Single concrete task the bespoke LoopDriver already handles, per the
# doc's "the same shape the bespoke already handles" principle. One
# Python file, three functions, no dependency on siblings (turn-1 case).
TASK = (
    "Write a Python module `converters.py` with three functions: "
    "`celsius_to_fahrenheit(c)` returning fahrenheit, "
    "`fahrenheit_to_celsius(f)` returning celsius, "
    "`celsius_to_kelvin(c)` returning kelvin. "
    "Use float math. No imports needed."
)


async def run() -> None:
    project_dir = ENSEMBLE_YAML.parents[2]
    print(f"[Ω-1] project_dir: {project_dir}")
    print(f"[Ω-1] ensemble: {ENSEMBLE_YAML}")

    loader = EnsembleLoader()
    config = loader.load_from_file(str(ENSEMBLE_YAML))
    print(f"[Ω-1] loaded ensemble: {config.name}, {len(config.agents)} agents")
    for agent in config.agents:
        kind = (
            "script" if hasattr(agent, "script") and getattr(agent, "script", None)
            else "ensemble" if hasattr(agent, "ensemble") and getattr(agent, "ensemble", None)
            else "llm"
        )
        deps = agent.depends_on
        print(f"       - {agent.name:<14} {kind:<8} depends_on={deps}")

    executor = ExecutorFactory.create_root_executor(project_dir=project_dir)

    request = json.dumps({"task": TASK, "last_tool_result": ""})
    print("\n[Ω-1] request:")
    print(f"       {request[:140]}...")

    start = time.perf_counter()
    result = await executor.execute(config, request)
    elapsed = time.perf_counter() - start

    print(f"\n[Ω-1] executed in {elapsed:.2f}s")
    print("\n[Ω-1] ensemble result (top-level keys):")
    if isinstance(result, dict):
        for key in result:
            print(f"       - {key}")
        results_dict = result.get("results", {})
        if isinstance(results_dict, dict):
            print("\n[Ω-1] per-stage status:")
            for name, res in results_dict.items():
                if isinstance(res, dict):
                    status = res.get("status", "?")
                    response = res.get("response", "")
                    response_preview = str(response)[:120]
                    print(f"       {name:<14} {status:<8} {response_preview}")
                    if status != "success":
                        # Dump the full result for stage failures
                        print(f"       --- full {name} result ---")
                        print(json.dumps(res, indent=2, default=str)[:1500])
                        print(f"       --- end {name} ---")

    # The marshal stage's output is the tool_call (or finish_reason)
    marshal_result = (
        result.get("results", {}).get("marshal", {}).get("response", "")
        if isinstance(result, dict)
        else ""
    )
    print("\n[Ω-1] marshal output (the emitted chat-completions shape):")
    print(f"       {marshal_preview(marshal_result)}")

    # Simulate OpenCode executing the write tool_call, then verify the
    # landed file parses as Python. This is the Ω-1 gate condition.
    out_dir = Path(__file__).resolve().parent / "out"
    if out_dir.exists():
        for f in out_dir.glob("*.py"):
            f.unlink()
    written = simulate_client_write(marshal_result, out_dir)
    if written is not None:
        print(f"       deliverable landed at {written.relative_to(Path(__file__).resolve().parents[2])}")

    print("\n[Ω-1] glue inventory (bespoke shims used by the ensemble form):")
    print("       - parse.py            (ScriptAgentRunner subprocess; deploy in Ω-3 as substrate reader)")
    print("       - dispatch_shim.py    (static-target seam; Ω-3 makes it a capability-scorer)")
    print("       - validate.py         (ast.parse FormGate; bespoke's ADR-041 translated)")
    print("       - marshal.py          (write tool_call emission; bespoke's ArtifactBridge translated)")
    print("       - HTTP adapter         (deferred; harness acts as adapter for Ω-1)")
    print("       - dynamic dispatch     (sidestepped: ensemble target is static in YAML)")


def marshal_preview(s: str) -> str:
    """Pretty-preview the marshal output."""
    if not s:
        return "(empty)"
    try:
        parsed = json.loads(s)
        if "tool_calls" in parsed:
            tc = parsed["tool_calls"][0]
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            return (
                f"tool_call name={fn['name']} "
                f"filePath={args.get('filePath')} "
                f"content_preview={args.get('content', '')[:80]!r}"
            )
        if "content" in parsed:
            return f"finish_reason={parsed.get('finish_reason')} content={parsed['content'][:120]!r}"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return s[:200]


def simulate_client_write(marshal_output: str, dest_dir: Path) -> Path | None:
    """Simulate OpenCode executing the `write` tool call: write the
    delivered file to disk and ast.parse it as the final gate check."""
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
    dest = dest_dir / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    # The gate condition: the landed deliverable parses as a Python module.
    import ast
    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"[Ω-1] GATE FAIL: written file does not parse as Python")
        print(f"       destination: {dest}")
        print(f"       content preview: {content[:200]!r}")
        print(f"       syntax error: {e}")
        return None
    print(f"[Ω-1] GATE PASS: written file parses as Python at {dest} ({len(content)} bytes)")
    return dest


if __name__ == "__main__":
    asyncio.run(run())