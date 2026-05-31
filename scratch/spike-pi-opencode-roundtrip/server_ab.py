#!/usr/bin/env python3
"""Spike pi - Phase A / Phase B server.

Both phases delegate file-content generation to a real llm-orc ensemble
(spike-pi-code-generator, qwen3:8b/ollama, $0). They differ ONLY in the
response shape, which is the variable under test:

  PHASE=A  direct co-located write + text acknowledgment (necessity test)
  PHASE=B  streamed `write` tool_call OpenCode executes itself (round-trip)

On a follow-up turn (messages contain a role:"tool" result), returns closing
text so the agent loop terminates instead of re-emitting a tool_call.
"""
import datetime
import http.server
import json
import os
import re
import socketserver
import subprocess

PORT = int(os.environ.get("PORT", "8099"))
PHASE = os.environ.get("PHASE", "A").upper()
WORKSPACE = os.environ.get("WORKSPACE", "workspace")
ENSEMBLE = os.environ.get("ENSEMBLE", "spike-pi-code-generator")
PROJECT = os.environ.get("PROJECT_DIR") or None
LOG = f"requests_{PHASE}.jsonl"


def log_event(obj: dict) -> None:
    obj["ts"] = datetime.datetime.now().isoformat()
    with open(LOG, "a") as f:
        f.write(json.dumps(obj) + "\n")


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


def generate(request_text: str) -> str:
    cmd = ["llm-orc", "invoke", ENSEMBLE, request_text, "--output-format", "json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PROJECT, timeout=120)
        d = json.loads(p.stdout)
        resp = d["results"]["code-generator"]["response"]
    except Exception as e:  # surface failures as file content so they are visible
        resp = f"# spike-pi generation failed: {e}\n"
    return clean(resp)


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
        roles = [m.get("role") for m in msgs]
        has_tools = bool(b.get("tools"))
        is_followup = any(m.get("role") == "tool" for m in msgs)
        log_event({"method": "POST", "phase": PHASE, "roles": roles,
                   "has_tools": has_tools, "is_followup": is_followup, "body": b})

        if is_followup:
            self._sse_text("Done - hello.py has been created with the "
                           "requested content.")
            return
        if not has_tools:
            self._sse_text("ok")
            return

        user_text = ""
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content")
                user_text = c if isinstance(c, str) else json.dumps(c)
        content = generate(user_text or "create hello.py that prints hello world")

        if PHASE == "A":
            os.makedirs(WORKSPACE, exist_ok=True)
            path = os.path.join(WORKSPACE, "hello.py")
            with open(path, "w") as f:
                f.write(content)
            log_event({"event": "direct_write", "path": path,
                       "bytes": len(content)})
            self._sse_text("I created hello.py with the requested content "
                           "(a hello-world script).")
        else:
            self._sse_tool_write("hello.py", content)

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
    print(f"LISTENING {PORT} PHASE={PHASE}", flush=True)
    srv.serve_forever()
