#!/usr/bin/env python3
"""Spike Ω-smoke — standalone chat-completions adapter over agent-turn-omega1.

The §4b item-3 contract check. The production /v1/chat/completions endpoint
routes everything through the bespoke LoopDriver (ADR-043). To smoke-test
the ENSEMBLE form against a real client, this is a thin standalone adapter:
it invokes the agent-turn-omega1 ensemble (single turn) and marshals the
result into an OpenAI chat.completion body. The marshal stage already emits
OpenAI-shaped tool_calls, so this is nearly passthrough — which is the point
(the ensemble's output already fits the wire contract).

It validates the marshal's tool_call / finish_reason shape against a real
HTTP chat-completions exchange (curl, then optionally a real OpenCode run),
not the hand-shaped harness — the WP-A "validate against the real client"
concern.

Run:
    uv run python scratch/spike-omega-smoke/smoke_adapter.py [port]
Then:
    curl -s localhost:8099/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"omega1","messages":[{"role":"user","content":"<task>"}],"tools":[]}'
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

ENSEMBLE_YAML = (
    Path(__file__).resolve().parents[2]
    / ".llm-orc" / "ensembles" / "spike-omega" / "agent-turn-omega1.yaml"
)
PROJECT_DIR = ENSEMBLE_YAML.parents[2]

_LOADER = EnsembleLoader()
_CONFIG = _LOADER.load_from_file(str(ENSEMBLE_YAML))


def _task_from_messages(messages: list[dict]) -> tuple[str, str]:
    """Derive (task, last_tool_result) from the OpenAI messages[].

    Single-turn smoke: the task is the last user message; a role:tool
    message (multi-turn round-trip) becomes last_tool_result.
    """
    task = ""
    last_tool_result = ""
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user" and content:
            task = content
        elif role == "tool" and content:
            last_tool_result = content
    return task, last_tool_result


def _run_turn(task: str, last_tool_result: str) -> str:
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)
    request = json.dumps({"task": task, "last_tool_result": last_tool_result})
    result = asyncio.run(executor.execute(_CONFIG, request))
    if not isinstance(result, dict):
        return ""
    return result.get("results", {}).get("marshal", {}).get("response", "") or ""


def _build_body(marshal_raw: str, model: str) -> dict:
    try:
        parsed = json.loads(marshal_raw)
    except json.JSONDecodeError:
        parsed = {"finish_reason": "stop", "content": marshal_raw}

    message: dict = {"role": "assistant", "content": None}
    if isinstance(parsed, dict) and "tool_calls" in parsed:
        # marshal already emits {id,type,function:{name,arguments}} —
        # the OpenAI message.tool_calls shape. Pass through; content null.
        message["content"] = None
        message["tool_calls"] = parsed["tool_calls"]
        finish_reason = "tool_calls"
    else:
        message["content"] = parsed.get("content", "") if isinstance(parsed, dict) else ""
        finish_reason = parsed.get("finish_reason", "stop") if isinstance(parsed, dict) else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[smoke] " + (fmt % args) + "\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # Some clients probe /v1/models at bootstrap.
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": "omega1", "object": "model", "owned_by": "llm-orc-spike"}],
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length).decode() if length else "{}"
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return
        model = req.get("model", "omega1")
        task, last_tool_result = _task_from_messages(req.get("messages", []))
        if not task:
            self._send_json(400, {"error": "no user task in messages"})
            return
        sys.stderr.write(f"[smoke] turn: task={task[:60]!r} ...\n")
        marshal_raw = _run_turn(task, last_tool_result)
        self._send_json(200, _build_body(marshal_raw, model))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"[smoke] agent-turn-omega1 adapter on http://127.0.0.1:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
