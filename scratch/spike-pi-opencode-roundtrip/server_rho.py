#!/usr/bin/env python3
"""Spike rho - planner-driven delegation + tool_calls terminal, together.

Unlike Phase B (which hardcoded "always dispatch + always emit write"), this
server makes the delegation decision the way ADR-027 says the framework must:

  real agent turn ->
    1. invoke the routing planner (spike-cycle7-zeta-routing-planner) on the
       task extracted from OpenCode's message stream
    2. plan.action == "dispatch" -> invoke the generation ensemble, emit a
       streamed `write` tool_call carrying the deliverable (parity path)
    3. plan.action == "direct"   -> return a text completion (fallback path)
  follow-up turn (tool result present) -> closing text, loop terminates.

This answers: does the framework's planner reliably DECIDE to delegate on a
real tool-rich OpenCode request, AND does that delegated work return via the
tool_call terminal with parity? Generation routes to spike-pi-code-generator
(confirmed-free qwen3:8b) as a stand-in for the planner-named capability
ensemble; the production code-generator's undefined profile + artifact
substrate are logged findings, out of this spike's focused scope. $0.
"""
import datetime
import http.server
import json
import os
import re
import socketserver
import subprocess

PORT = int(os.environ.get("PORT", "8099"))
WORKSPACE = os.environ.get("WORKSPACE", "workspace")
PLANNER = os.environ.get("PLANNER", "spike-cycle7-zeta-routing-planner")
GENERATOR = os.environ.get("GENERATOR", "spike-pi-code-generator")
PROJECT = os.environ.get("PROJECT_DIR") or None
LOG = "requests_rho.jsonl"


def log_event(obj: dict) -> None:
    obj["ts"] = datetime.datetime.now().isoformat()
    with open(LOG, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _invoke(ensemble: str, text: str, agent: str) -> str:
    cmd = ["llm-orc", "invoke", ensemble, text, "--output-format", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT,
                       timeout=150)
    d = json.loads(p.stdout)
    return d["results"][agent]["response"]


def plan(task: str) -> dict:
    """Run the routing planner; return parsed {action, ensemble, rationale}."""
    try:
        raw = _invoke(PLANNER, task, "planner")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        return json.loads(m.group(0)) if m else {"action": "direct",
                                                 "ensemble": None,
                                                 "rationale": "unparseable plan"}
    except Exception as e:
        return {"action": "direct", "ensemble": None,
                "rationale": f"planner error: {e}"}


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"


def generate(task: str) -> str:
    try:
        return clean(_invoke(GENERATOR, task, "code-generator"))
    except Exception as e:
        return f"# spike-rho generation failed: {e}\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:  # noqa: N802
        self._body()
        log_event({"method": "GET", "path": self.path})
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "spike-model", "object": "model", "owned_by": "spike"}]})
        else:
            self._json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        raw = self._body()
        try:
            b = json.loads(raw)
        except Exception:
            b = {"_unparsed": raw.decode("utf-8", "replace")}
        msgs = b.get("messages", [])
        has_tools = bool(b.get("tools"))
        is_followup = any(m.get("role") == "tool" for m in msgs)

        if is_followup:
            log_event({"method": "POST", "stage": "followup"})
            self._sse_text("Done - the file has been created.")
            return
        if not has_tools:
            log_event({"method": "POST", "stage": "title-gen"})
            self._sse_text("ok")
            return

        task = ""
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content")
                task = c if isinstance(c, str) else json.dumps(c)
        decision = plan(task or "create hello.py that prints hello world")
        log_event({"method": "POST", "stage": "agent-turn", "task": task,
                   "plan": decision})

        if decision.get("action") == "dispatch":
            content = generate(task)
            log_event({"event": "dispatched", "named_ensemble":
                       decision.get("ensemble"),
                       "generator_used": GENERATOR, "bytes": len(content)})
            self._sse_tool_write("hello.py", content)
        else:
            log_event({"event": "direct_fallback"})
            self._sse_text("[direct completion fallback] "
                           + decision.get("rationale", ""))

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse_open(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _emit(self, delta: dict, created: int, finish=None) -> None:
        obj = {"id": "chatcmpl-spike", "object": "chat.completion.chunk",
               "created": created, "model": "spike-model",
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
        self.wfile.flush()

    def _sse_text(self, text: str) -> None:
        self._sse_open()
        created = int(datetime.datetime.now().timestamp())
        self._emit({"role": "assistant"}, created)
        self._emit({"content": text}, created)
        self._emit({}, created, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _sse_tool_write(self, file_path: str, content: str) -> None:
        self._sse_open()
        created = int(datetime.datetime.now().timestamp())
        args = json.dumps({"filePath": file_path, "content": content})
        self._emit({"role": "assistant", "content": None, "tool_calls": [
            {"index": 0, "id": "call_write_1", "type": "function",
             "function": {"name": "write", "arguments": ""}}]}, created)
        self._emit({"tool_calls": [
            {"index": 0, "function": {"arguments": args}}]}, created)
        self._emit({}, created, finish="tool_calls")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        log_event({"event": "emitted_tool_call", "tool": "write",
                   "filePath": file_path, "args_len": len(args)})

    def log_message(self, *args) -> None:
        pass


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    srv = ThreadingServer(("127.0.0.1", PORT), Handler)
    print(f"LISTENING {PORT} PLANNER={PLANNER} GENERATOR={GENERATOR}",
          flush=True)
    srv.serve_forever()
