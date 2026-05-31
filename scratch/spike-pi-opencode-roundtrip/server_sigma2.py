#!/usr/bin/env python3
"""Spike sigma.2 - integrated north-star pattern: cheap layer-A loop-driver +
layer-B ensemble delegation, multi-turn, through OpenCode.

Each tool-bearing turn: forward conversation + tools to a layer-A driver
(qwen3:14b via Ollama, native tool-use) for the next-action decision. If the
decision includes a `write`, delegate the file CONTENT to the code-generation
ensemble (layer B), substituting the ensemble's output into the write call.
Other tools (bash/read/edit) pass through. The driver decides when to finish
(emits text, no tool_calls). Validates whether loop-driving + ensemble
delegation compose across a multi-step task with parity. $0 (all local).
"""
import datetime
import http.server
import json
import os
import re
import socketserver
import subprocess
import urllib.request

PORT = int(os.environ.get("PORT", "8099"))
DRIVER_MODEL = os.environ.get("DRIVER_MODEL", "qwen3:14b")
OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434/v1/chat/completions")
GENERATOR = os.environ.get("GENERATOR", "spike-pi-code-generator")
PROJECT = os.environ.get("PROJECT_DIR") or None
LOG = "requests_sigma2.jsonl"


def log_event(o: dict) -> None:
    o["ts"] = datetime.datetime.now().isoformat()
    with open(LOG, "a") as f:
        f.write(json.dumps(o) + "\n")


def driver(messages: list, tools: list) -> dict:
    payload = {"model": DRIVER_MODEL, "messages": messages, "tools": tools,
               "stream": False}
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer ollama"})
    with urllib.request.urlopen(req, timeout=200) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]


def clean(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    if t.startswith("```"):
        ls = t.splitlines()
        if ls and ls[0].startswith("```"):
            ls = ls[1:]
        if ls and ls[-1].strip() == "```":
            ls = ls[:-1]
        t = "\n".join(ls)
    return t.strip() + "\n"


def ensemble_content(file_path: str, spec: str) -> str:
    task = (f"Produce the complete contents of the file {file_path}. "
            f"Requirements:\n{spec}")
    cmd = ["llm-orc", "invoke", GENERATOR, task, "--output-format", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT,
                       timeout=150)
    d = json.loads(p.stdout)
    return clean(d["results"]["code-generator"]["response"])


class Handler(http.server.BaseHTTPRequestHandler):
    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:  # noqa: N802
        self._body()
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list",
                             "data": [{"id": "spike-model", "object": "model"}]})
        else:
            self._json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        raw = self._body()
        try:
            b = json.loads(raw)
        except Exception:
            b = {}
        msgs = b.get("messages", [])
        tools = b.get("tools", [])
        if not tools:
            log_event({"stage": "title-gen"})
            self._sse_text("ok")
            return
        try:
            msg = driver(msgs, tools)
        except Exception as e:
            log_event({"stage": "driver-error", "err": str(e)})
            self._sse_text(f"driver error: {e}")
            return
        tcs = msg.get("tool_calls") or []
        if not tcs:
            log_event({"stage": "finish",
                       "text": (msg.get("content") or "")[:160]})
            self._sse_text(clean(msg.get("content") or "done"))
            return
        delegated = []
        for tc in tcs:
            fn = tc.get("function", {})
            if fn.get("name") == "write":
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    fp = args.get("filePath", "out.txt")
                    args["content"] = ensemble_content(fp, args.get("content", ""))
                    fn["arguments"] = json.dumps(args)
                    delegated.append(fp)
                except Exception as e:
                    log_event({"stage": "delegate-error", "err": str(e)})
        log_event({"stage": "turn",
                   "tool_calls": [tc.get("function", {}).get("name")
                                  for tc in tcs],
                   "delegated_writes": delegated})
        self._sse_tool_calls(tcs)

    def _json(self, code: int, p: dict) -> None:
        d = json.dumps(p).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(d)))
        self.end_headers()
        self.wfile.write(d)

    def _sse_open(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _emit(self, delta: dict, created: int, finish=None) -> None:
        o = {"id": "chatcmpl-spike", "object": "chat.completion.chunk",
             "created": created, "model": "spike-model",
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode())
        self.wfile.flush()

    def _sse_text(self, text: str) -> None:
        self._sse_open()
        c = int(datetime.datetime.now().timestamp())
        self._emit({"role": "assistant"}, c)
        self._emit({"content": text}, c)
        self._emit({}, c, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _sse_tool_calls(self, tcs: list) -> None:
        self._sse_open()
        c = int(datetime.datetime.now().timestamp())
        shells = [{"index": i, "id": tc.get("id") or f"call_{i}",
                   "type": "function",
                   "function": {"name": tc["function"]["name"],
                                "arguments": ""}}
                  for i, tc in enumerate(tcs)]
        self._emit({"role": "assistant", "content": None,
                    "tool_calls": shells}, c)
        for i, tc in enumerate(tcs):
            self._emit({"tool_calls": [
                {"index": i,
                 "function": {"arguments": tc["function"]["arguments"]}}]}, c)
        self._emit({}, c, finish="tool_calls")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a) -> None:
        pass


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    srv = ThreadingServer(("127.0.0.1", PORT), Handler)
    print(f"LISTENING {PORT} DRIVER={DRIVER_MODEL} GEN={GENERATOR}", flush=True)
    srv.serve_forever()
