#!/usr/bin/env python3
"""Spike Ω-2b dispatch-shim stage — recovery-aware.

Same as the Ω-1/Ω-2 dispatch-shim (deterministic next_file from parse,
form directive append) plus: when parse surfaces a `recovery_hint` (the
adapter re-invoked after a validate failure), append the exact rejection
feedback to the code-generator input verbatim, so the coder sees the ast
error and the "re-emit only this file" instruction directly rather than
through the plan LLM's paraphrase.

Emits JSON: {
    "file_path": "<str>",
    "dispatch_input": "<formatted string for code-generator>",
    "brief": "<str>"
}
"""

import json
import sys


def main() -> None:
    raw = sys.stdin.read().strip()
    plan = {}
    parse_state = {}
    deterministic_next_file = None

    try:
        data = json.loads(raw)
        deps = data.get("dependencies", {}) if isinstance(data, dict) else {}

        plan_dep = deps.get("plan", {})
        response = plan_dep.get("response", "") if isinstance(plan_dep, dict) else ""
        if isinstance(response, str):
            try:
                plan = json.loads(response)
            except json.JSONDecodeError:
                import re

                m = re.search(r"\{[^{}]*\}", response)
                if m:
                    try:
                        plan = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        plan = {}
        elif isinstance(response, dict):
            plan = response

        parse_dep = deps.get("parse", {})
        parse_response = parse_dep.get("response", "") if isinstance(parse_dep, dict) else ""
        if isinstance(parse_response, str):
            try:
                parse_state = json.loads(parse_response)
            except json.JSONDecodeError:
                parse_state = {}
        elif isinstance(parse_response, dict):
            parse_state = parse_response

        next_file = parse_state.get("next_file")
        if next_file:
            deterministic_next_file = next_file
        else:
            plan_queue = parse_state.get("plan_queue") or []
            if plan_queue:
                deterministic_next_file = plan_queue[0]
    except (json.JSONDecodeError, TypeError):
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {}

    file_path = deterministic_next_file or plan.get("file_path", "output.py")
    brief = plan.get("brief", "")

    if not brief:
        print(json.dumps({"success": False, "error": "Plan produced no brief"}))
        return

    form_directive = (
        "\n\nOutput ONLY the exact raw bytes of the file. "
        "No markdown fences, no prose, no explanations, no example blocks."
    )

    recovery_hint = parse_state.get("recovery_hint", "") or ""
    recovery_block = ""
    if recovery_hint:
        recovery_block = (
            f"\n\nRETRY — the previous attempt was rejected: {recovery_hint}\n"
            "Emit the corrected file content ONLY. The first byte you emit is the "
            "first byte of the file; do not lead with shell examples, the filename, "
            "or any prose."
        )

    dispatch_input = f"Write {file_path}: {brief}{form_directive}{recovery_block}"

    print(
        json.dumps(
            {
                "file_path": file_path,
                "dispatch_input": dispatch_input,
                "brief": brief,
            }
        )
    )


if __name__ == "__main__":
    main()
