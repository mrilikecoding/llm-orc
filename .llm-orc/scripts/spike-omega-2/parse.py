#!/usr/bin/env python3
"""Spike Ω-2 parse stage — substrate-aware.

Reads the incoming request and (when substrate_path is present) reads
the session_state.json from that path. Threads:
  - task (original)
  - the next deliverable from plan_queue
  - the produced files' raw content (the §2 substrate-+-scripts bet:
    cross-file coherence without ADR-039's explicit content-anchor
    primitive; the script can do file reads because scripts run with
    full filesystem access, no sandbox — verified in script_agent.py:555)

If no substrate_path is given, behaves as the Ω-1 identity parse.

Emits JSON: {
    "task": "<str>",
    "plan_input": "<structured text for the plan LLM>",
    "produced": ["<path>", ...],
    "plan_queue": ["<path>", ...],
    "substrate_path": "<abs path>"
}
"""

import json
import sys
from pathlib import Path


def main() -> None:
    raw = sys.stdin.read().strip()
    request = {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            req_str = data.get("input_data") or data.get("input") or ""
            if isinstance(req_str, str):
                try:
                    request = json.loads(req_str) if req_str else {}
                except json.JSONDecodeError:
                    request = {"task": req_str, "last_tool_result": ""}
            elif isinstance(req_str, dict):
                request = req_str
    except (json.JSONDecodeError, TypeError):
        request = {"task": raw, "last_tool_result": ""}

    substrate_path = request.get("substrate_path")
    task = request.get("task", "") or request.get("last_tool_result", "")

    if not task:
        print(json.dumps({"success": False, "error": "No task in input"}))
        return

    # If no substrate_path, behave as Ω-1 identity parse.
    if not substrate_path:
        print(
            json.dumps(
                {"task": task, "plan_input": task, "plan_queue": [], "produced": [], "next_file": ""}
            )
        )
        return

    substrate = Path(substrate_path)
    if not substrate.exists():
        print(json.dumps({"success": False, "error": f"substrate not found: {substrate}"}))
        return

    try:
        state = json.loads(substrate.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(json.dumps({"success": False, "error": f"substrate parse failed: {e}"}))
        return

    produced = state.get("produced", [])
    plan_queue = state.get("plan_queue", [])
    anchor = state.get("remaining_anchor", "")

    if not plan_queue:
        # No deliverables remaining — emit a signal so the plan stage
        # emits nothing (the termination check).
        print(
            json.dumps(
                {
                    "task": task,
                    "plan_input": "(no deliverables remaining; emit {\"file_path\": \"\", \"brief\": \"none\"})",
                    "produced": produced,
                    "plan_queue": [],
                    "next_file": "",
                    "substrate_path": substrate_path,
                    "terminal": True,
                }
            )
        )
        return

    next_file = plan_queue[0]

    # The substrate-+-scripts bet: read produced files' content to ground
    # cross-file references. The bespoke solved this with ADR-039's content
    # anchor (sibling signatures); Ω-2 uses raw file content via the parse
    # script's filesystem access. Cheaper to implement — but tests whether
    # the script-substrate path actually delivers coherence.
    siblings_text = ""
    if produced:
        sibling_dir = substrate.parent / "produced"
        if not sibling_dir.exists():
            # The harness writes produced files alongside the state file
            # under ./produced/. Fall back to ./scratch path try.
            sibling_dir = substrate.parent
        chunks = []
        for p in produced:
            p_path = sibling_dir / p
            if p_path.exists():
                try:
                    chunks.append(f"=== {p} ===\n{p_path.read_text()}")
                except OSError:
                    chunks.append(f"=== {p} (unavailable) ===")
        siblings_text = "\n\n".join(chunks)

    plan_input = (
        f"Original task: {task}\n\n"
        f"Already produced (with content, so reference real APIs):\n{siblings_text or '(none)'}\n\n"
        f"Produce this file next: {next_file}\n"
    )
    if anchor:
        plan_input += f"\nRemaining-work anchor: {anchor}\n"

    print(
        json.dumps(
            {
                "task": task,
                "plan_input": plan_input,
                "produced": produced,
                "plan_queue": plan_queue,
                "next_file": next_file,
                "substrate_path": substrate_path,
            }
        )
    )


if __name__ == "__main__":
    main()