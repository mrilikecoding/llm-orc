#!/usr/bin/env python3
"""Spike tau - grounded-loop FALSIFICATION probe (OQ #27 axis 1 only).

A thin passthrough loop-driver: forward conversation + tools to a cheap local
driver (qwen3:14b via Ollama, native tool-use), emit whatever tool_calls it
returns UNCHANGED (no ensemble delegation - this isolates the driver's own
loop behavior). Logs the tool_calls emitted per turn so batched-vs-stepped is
directly visible.

The task is deliberately UN-BATCHABLE: a later write depends on a random value
the driver can only know by observing an earlier bash result. If the driver
slips from grounded per-turn stepping into ungrounded batch planning (the OQ
#27 axis-1 discriminating failure - "emits a multi-step batch whose later steps
presuppose earlier outputs it never observed"), the verification step fails.

A clean PASS is an AXIS-1 PASS ONLY. Axis 2 (sequential-composition error
accumulation over a long horizon) is a BUILD-phase target the short probe
cannot settle. $0 (all local Ollama).
"""
import datetime
import http.server
import json
import os
import re
import socketserver
import urllib.request

PORT = int(os.environ.get("PORT", "8099"))
DRIVER_MODEL = os.environ.get("DRIVER_MODEL", "qwen3:14b")
OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434/v1/chat/completions")
# SINGLE_STEP=1 -> framework structurally forces one action per turn: any batch
# the driver emits is truncated to its first tool call, so OpenCode must execute
# + return that result before the driver can decide the next action. Tests
# whether grounding can be STRUCTURALLY ENFORCED (the tau' mitigation probe).
SINGLE_STEP = os.environ.get("SINGLE_STEP", "0") == "1"
LOG = "requests_tau.jsonl"


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
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]


def clean(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip() or "done"


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
        # how many prior assistant tool-call turns are in history (turn index)
        prior_tool_turns = sum(
            1 for m in msgs
            if m.get("role") == "assistant" and m.get("tool_calls"))
        # log what the driver OBSERVED this turn (prior tool results in history)
        observed = []
        for m in msgs:
            if m.get("role") == "tool":
                c = m.get("content")
                c = c if isinstance(c, str) else json.dumps(c)
                observed.append(c[:300])
        if observed:
            log_event({"stage": "observed", "turn_index": prior_tool_turns,
                       "tool_results": observed})
        try:
            msg = driver(msgs, tools)
        except Exception as e:
            log_event({"stage": "driver-error", "err": str(e)})
            self._sse_text(f"driver error: {e}")
            return
        tcs = msg.get("tool_calls") or []
        if not tcs:
            log_event({"stage": "finish", "turn_index": prior_tool_turns,
                       "text": (msg.get("content") or "")[:200]})
            self._sse_text(clean(msg.get("content") or "done"))
            return
        if SINGLE_STEP and len(tcs) > 1:
            log_event({"stage": "truncated", "turn_index": prior_tool_turns,
                       "from_n": len(tcs), "to_n": 1,
                       "dropped": [tc.get("function", {}).get("name")
                                   for tc in tcs[1:]]})
            tcs = tcs[:1]
        summary = []
        for tc in tcs:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"_raw": fn.get("arguments")}
            summary.append({"name": fn.get("name"),
                            "args": json.dumps(args)[:200]})
        log_event({"stage": "turn", "turn_index": prior_tool_turns,
                   "n_tool_calls": len(tcs), "calls": summary})
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
    print(f"LISTENING {PORT} DRIVER={DRIVER_MODEL} (passthrough, no delegation)",
          flush=True)
    srv.serve_forever()
