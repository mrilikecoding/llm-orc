#!/usr/bin/env python3
"""Spike Ω-2 marshal stage — substrate-updating.

Same as the Ω-1 marshal — emit the write tool_call — plus update
session_state.json to mark the current file produced and remove it
from plan_queue. The state file is the substrate for the next turn's
parse stage. No engine primitive required: the script has full
filesystem access.

If plan_queue is empty after this turn's update, emit
finish_reason=stop (sends the harness the signal to stop looping).
Until then, emit the write tool_call so the harness simulates the
OpenCode write and re-invokes the ensemble.

Substrate state schema:
  {
    "task": "...",
    "requested": ["<path>", ...],
    "produced": ["<path>", ...],   # grows each turn
    "plan_queue": ["<path>", ...], # shrinks each turn
    "remaining_anchor": "..."
  }
"""

import json
import sys
import uuid
from pathlib import Path


def main() -> None:
    raw = sys.stdin.read().strip()

    file_path = None
    validate_result = None
    substrate_path = None
    state = None

    try:
        data = json.loads(raw)
        deps = data.get("dependencies", {}) if isinstance(data, dict) else {}

        parse_dep = deps.get("parse", {})
        if isinstance(parse_dep, dict):
            parse_response = parse_dep.get("response", "")
            if isinstance(parse_response, str):
                try:
                    parsed = json.loads(parse_response)
                    substrate_path = parsed.get("substrate_path")
                except json.JSONDecodeError:
                    pass

        dispatch_dep = deps.get("dispatch-shim", {})
        if isinstance(dispatch_dep, dict):
            response = dispatch_dep.get("response", "")
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    file_path = parsed.get("file_path")
                except json.JSONDecodeError:
                    pass

        validate_dep = deps.get("validate", {})
        if isinstance(validate_dep, dict):
            response = validate_dep.get("response", "")
            if isinstance(response, str):
                try:
                    validate_result = json.loads(response)
                except json.JSONDecodeError:
                    validate_result = {"valid": False, "error": "non-JSON validate"}
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Read current substrate state so we can update it.
    if substrate_path:
        try:
            state = json.loads(Path(substrate_path).read_text())
        except (OSError, json.JSONDecodeError):
            state = None

    if not file_path:
        print(
            json.dumps(
                {
                    "finish_reason": "stop",
                    "content": "marshal: could not extract file_path from dispatch-shim",
                }
            )
        )
        return

    if not validate_result or not validate_result.get("valid"):
        err = validate_result.get("error", "unknown") if validate_result else "missing"
        print(
            json.dumps(
                {
                    "finish_reason": "stop",
                    "content": f"marshal: validate failed for {file_path}: {err}",
                }
            )
        )
        return

    content = validate_result.get("content", "")

    # Update substrate state: mark this file produced, drop from plan_queue.
    # Dedup: if the plan stage deviated and re-chose an already-produced file,
    # this turn is a no-op for the produced set.
    existing_produced = state.get("produced", []) if state else []
    if file_path not in existing_produced:
        produced_after = existing_produced + [file_path]
    else:
        produced_after = existing_produced
    queue_after = [p for p in (state.get("plan_queue") or []) if p != file_path] if state else []
    if state is not None:
        state["produced"] = produced_after
        state["plan_queue"] = queue_after
        state["remaining_anchor"] = (
            f"Produce {queue_after[0]} next." if queue_after else "All deliverables produced."
        )
        try:
            Path(substrate_path).write_text(json.dumps(state, indent=2))
        except OSError as e:
            print(
                json.dumps(
                    {"finish_reason": "stop", "content": f"marshal: substrate write failed: {e}"}
                )
            )
            return

    # If plan_queue is now empty, signal the harness to stop.
    if not queue_after:
        print(
            json.dumps(
                {
                    "finish_reason": "stop",
                    "content": f"All {len(produced_after)} deliverables produced: {produced_after}",
                }
            )
        )
        return

    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
    print(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": json.dumps(
                                {"filePath": file_path, "content": content}
                            ),
                        },
                    }
                ]
            }
        )
    )


if __name__ == "__main__":
    main()