#!/usr/bin/env python3
"""Spike Ω-2b marshal stage — terminal-write fixed.

Same as the Ω-2 marshal (emit the write tool_call, update session_state)
with one correction: it NEVER short-circuits the terminal deliverable.
The Ω-2 marshal, on the last file, marked it produced and emitted
finish_reason=stop WITHOUT a tool_call — so the last file's validated
content was silently dropped (disk got N-1 of N files). That also
mis-models the real client contract, where every produced file rides a
write tool_call and the adapter decides to stop once the queue drains.

Here: whenever validate passes and a file_path is present, emit the
write tool_call and update the substrate. The harness (adapter) reloads
the substrate after the write and stops when plan_queue is empty. A
finish_reason=stop is emitted only when there is genuinely nothing to
produce (no file_path, e.g. a terminal turn) or validate failed (the
recovery path the adapter retries on).

Substrate schema:
  {task, requested, produced[], plan_queue[], remaining_anchor}
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
                    "content": "marshal: no file_path (terminal turn or dispatch gap)",
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

    # Update substrate: mark produced, drop from queue (dedup-safe).
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

    # Always emit the write tool_call — including the terminal deliverable.
    # The adapter reloads the substrate and stops when plan_queue is empty.
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
