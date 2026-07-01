#!/usr/bin/env python3
"""Spike Ω-1 marshal stage.

Reads the dispatch-shim's {file_path, dispatch_input, brief} and the
validate stage's {valid, content, error?}, emits the OpenAI-shaped
write tool_call that OpenCode (or any chat-completions client) would
execute locally. This is the §4 marshal script; the bespoke LoopDriver's
ArtifactBridge + ClientToolActionTerminal emission translates here.

If validate failed, emits a {finish, error} shape instead of a write
tool_call. The recovery loop in Ω-3+ would re-dispatch here; Ω-1 just
surfaces the failure.

Emits JSON: {
    "tool_calls": [
        {
            "id": "<generated>",
            "type": "function",
            "function": {"name": "write", "arguments": {"filePath": "...", "content": "..."}}
        }
    ]
    OR
    {"finish_reason": "stop", "content": "<error explanation>"}
}
"""

import json
import sys
import uuid


def main() -> None:
    raw = sys.stdin.read().strip()

    file_path = None
    validate_result = None

    try:
        data = json.loads(raw)
        # ScriptAgentInput shape: {"input_data": "...", "dependencies": {...}}
        deps = data.get("dependencies", {}) if isinstance(data, dict) else {}

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