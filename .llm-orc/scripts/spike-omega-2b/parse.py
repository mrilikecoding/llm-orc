#!/usr/bin/env python3
"""Spike Ω-2b parse stage — substrate-aware + recovery-aware.

Same as the Ω-2 parse (reads session_state.json, emits next_file +
sibling content for cross-file grounding) plus one addition: when the
adapter re-invokes after a validate failure, it puts the rejection
feedback in `last_tool_result`. This stage surfaces it as `recovery_hint`
so the dispatch-shim can append the exact ast error verbatim to the
code-generator input, bypassing the plan LLM's paraphrasing.

This is bespoke's ADR-041 self-healing re-dispatch, relocated from
in-process to between-turn (the §8 adapter boundary). No engine
primitive: the retry lives in the harness; the hint rides the existing
request fields.

Emits JSON: {
    "task": "<str>",
    "plan_input": "<structured text for the plan LLM>",
    "produced": ["<path>", ...],
    "plan_queue": ["<path>", ...],
    "next_file": "<path>",
    "recovery_hint": "<str or empty>",
    "substrate_path": "<abs path>"
}
"""

import json
import sys
from pathlib import Path


def _is_rejection(text: str) -> bool:
    return text.strip().upper().startswith("PRODUCTION REJECTED")


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
    last_tool_result = request.get("last_tool_result", "") or ""
    recovery_hint = last_tool_result if _is_rejection(last_tool_result) else ""

    if not task:
        print(json.dumps({"success": False, "error": "No task in input"}))
        return

    # If no substrate_path, behave as Ω-1 identity parse.
    if not substrate_path:
        print(
            json.dumps(
                {
                    "task": task,
                    "plan_input": task,
                    "plan_queue": [],
                    "produced": [],
                    "next_file": "",
                    "recovery_hint": recovery_hint,
                }
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
        print(
            json.dumps(
                {
                    "task": task,
                    "plan_input": "(no deliverables remaining; emit {\"file_path\": \"\", \"brief\": \"none\"})",
                    "produced": produced,
                    "plan_queue": [],
                    "next_file": "",
                    "recovery_hint": "",
                    "substrate_path": substrate_path,
                    "terminal": True,
                }
            )
        )
        return

    next_file = plan_queue[0]

    # Read produced files' content to ground cross-file references (the §2
    # substrate-+-scripts bet; scripts have full filesystem access).
    siblings_text = ""
    if produced:
        sibling_dir = substrate.parent / "produced"
        if not sibling_dir.exists():
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
    if recovery_hint:
        plan_input += (
            f"\nNOTE: the previous attempt at {next_file} was rejected by the "
            f"form gate. Keep the brief identical; the fix is purely formatting.\n"
        )

    print(
        json.dumps(
            {
                "task": task,
                "plan_input": plan_input,
                "produced": produced,
                "plan_queue": plan_queue,
                "next_file": next_file,
                "recovery_hint": recovery_hint,
                "substrate_path": substrate_path,
            }
        )
    )


if __name__ == "__main__":
    main()
