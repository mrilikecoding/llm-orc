#!/usr/bin/env python3
"""Spike Ω-1 dispatch-shim stage.

Reads the plan LLM's output {"file_path": ..., "brief": ...} and emits
the formatted input string for the static-targeted code-generator
ensemble, carrying the file_path through so the marshal stage can
recover it.

This is the §4 dispatch node faked in script form. In Ω-3 this becomes
a real capability-scorer (library reflection + embeddings + rules).
For Ω-1 the dispatch decision is fixed in YAML, so the shim only
formats.

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

        # Plan LLM output (the brief describes what to write).
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

        # Parse-stage output, if available (Ω-2 ensemble passes both). The
        # bespoke LoopDriver picks the next file deterministically from the
        # plan_queue head (the ADR-040 + ADR-038 mechanism); the LLM should
        # not choose the file. dispatch_shim overrides plan's file_path with
        # parse's next_file whenever parse supplies a substrate state.
        parse_dep = deps.get("parse", {})
        parse_response = parse_dep.get("response", "") if isinstance(parse_dep, dict) else ""
        if isinstance(parse_response, str):
            try:
                parse_state = json.loads(parse_response)
            except json.JSONDecodeError:
                parse_state = {}
        elif isinstance(parse_response, dict):
            parse_state = parse_response

        # Prefer parse's explicit next_file; fall back to plan_queue[0];
        # both are deterministic-from-substrate sources.
        next_file = parse_state.get("next_file")
        if next_file:
            deterministic_next_file = next_file
        else:
            plan_queue = parse_state.get("plan_queue") or []
            if plan_queue:
                deterministic_next_file = plan_queue[0]
    except (json.JSONDecodeError, TypeError):
        # Fall back: try the raw stdin as JSON
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {}

    # File path: prefer the parse stage's deterministic plan_queue[0] (Ω-2
    # ensemble) — the bespoke's deterministic-first principle. Fall back to
    # plan LLM's choice for the Ω-1 ensemble shape (no substrate state).
    file_path = deterministic_next_file or plan.get("file_path", "output.py")
    brief = plan.get("brief", "")

    if not brief:
        print(json.dumps({"success": False, "error": "Plan produced no brief"}))
        return

    # The ADR-035 form directive — the bespoke LoopDriver's LB-6 wording
    # translated into the ensemble form. Tells the code-generator's coder
    # (and synthesizer) that the deliverable is bare file bytes, not prose.
    form_directive = (
        "\n\nOutput ONLY the exact raw bytes of the file. "
        "No markdown fences, no prose, no explanations, no example blocks."
    )
    dispatch_input = f"Write {file_path}: {brief}{form_directive}"

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