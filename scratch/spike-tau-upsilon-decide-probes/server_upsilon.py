#!/usr/bin/env python3
"""Spike upsilon - WRAPPER-shaped probe (OQ #26).

The missing evidence shape. Every multi-turn run the loop-back produced (sigma.2)
is callee-shaped: the loop-driver called a BARE ensemble for per-turn generation.
This probe runs the wrapper reading instead - the FULL ADR-027 pipeline
(plan -> dispatch -> synthesize) runs as the per-turn generation subroutine,
UNDER a layer-A loop-driver, on the same task sigma.2 ran. So the wrapper-vs-callee
fork can be compared on real evidence rather than settling by callee-skew default.

Per tool-bearing turn: forward conversation + tools to the layer-A driver
(qwen3:14b). If it emits a `write`, generate the file content by running the
whole pipeline:
  1. plan      - zeta routing-planner reads the generation task -> {action, ensemble}
  2. dispatch  - invoke the planner-named ensemble (code-generator stand-in) -> raw output
  3. synthesize- epsilon response-synthesizer reads (REQUEST + PLAN + DISPATCH RESULTS)
                 -> the deliverable, marshalled into the write content.
Other tools (bash/read/edit) pass through. Per-stage wall-clock is logged so
latency compounding across turns (a named OQ #26 discriminating criterion) is
measurable against sigma.2's single-call-per-write callee baseline. $0 (all local).
"""
import datetime
import http.server
import json
import os
import re
import socketserver
import subprocess
import time
import urllib.request

PORT = int(os.environ.get("PORT", "8099"))
DRIVER_MODEL = os.environ.get("DRIVER_MODEL", "qwen3:14b")
OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434/v1/chat/completions")
PLANNER = os.environ.get("PLANNER", "spike-cycle7-zeta-routing-planner")
SYNTH = os.environ.get("SYNTH", "spike-cycle7-epsilon-response-synthesizer")
GEN_ENSEMBLE = os.environ.get("GEN_ENSEMBLE", "spike-pi-code-generator")
PROJECT = os.environ.get("PROJECT_DIR") or None
LOG = "requests_upsilon.jsonl"


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


def _invoke(ensemble: str, text: str, agent: str) -> str:
    cmd = ["llm-orc", "invoke", ensemble, text, "--output-format", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT,
                       timeout=200)
    d = json.loads(p.stdout)
    return d["results"][agent]["response"]


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


def pipeline_content(file_path: str, spec: str) -> str:
    """The WRAPPER subroutine: full plan -> dispatch -> synthesize per write."""
    task = (f"Produce the complete contents of the file {file_path}. "
            f"Requirements:\n{spec}")
    stages = {}

    # 1. plan
    t0 = time.time()
    try:
        raw = _invoke(PLANNER, task, "planner")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        plan = json.loads(m.group(0)) if m else {
            "action": "direct", "ensemble": None, "rationale": "unparseable"}
    except Exception as e:
        plan = {"action": "direct", "ensemble": None,
                "rationale": f"planner error: {e}"}
    stages["plan_s"] = round(time.time() - t0, 1)

    # 2. dispatch
    t0 = time.time()
    if plan.get("action") == "dispatch":
        try:
            dispatched = clean(_invoke(GEN_ENSEMBLE, task, "code-generator"))
        except Exception as e:
            dispatched = f"# generation failed: {e}\n"
        dispatched_ens = plan.get("ensemble") or "code-generator"
    else:
        dispatched = ""
        dispatched_ens = "none"
    stages["dispatch_s"] = round(time.time() - t0, 1)

    # 3. synthesize
    t0 = time.time()
    synth_input = (
        f"ORIGINAL REQUEST\n{task}\n\n"
        f"PLAN\nDispatched: {dispatched_ens}\nPlanned-but-not-run: none\n\n"
        f"DISPATCH RESULTS\n[{dispatched_ens}]\n{dispatched}\n")
    try:
        synthesized = _invoke(SYNTH, synth_input, "synthesizer")
        content = clean(synthesized)
    except Exception as e:
        synthesized = ""
        content = f"# synth failed: {e}\n"
    stages["synth_s"] = round(time.time() - t0, 1)

    log_event({"stage": "pipeline", "file": file_path, "plan": plan,
               "stages_s": stages,
               "total_s": round(sum(stages.values()), 1),
               "dispatched_len": len(dispatched),
               "synth_len": len(synthesized) if plan.get("action") == "dispatch"
               else 0,
               "final_len": len(content),
               "content_head": content[:120],
               # did the synthesizer wrap the raw code in chat prose?
               "synth_differs_from_dispatch": clean(dispatched).strip()
               != content.strip() if dispatched else None})
    return content


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
                       "text": (msg.get("content") or "")[:200]})
            self._sse_text(clean(msg.get("content") or "done"))
            return
        wrapped = []
        for tc in tcs:
            fn = tc.get("function", {})
            if fn.get("name") == "write":
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    fp = args.get("filePath", "out.txt")
                    args["content"] = pipeline_content(fp, args.get("content", ""))
                    fn["arguments"] = json.dumps(args)
                    wrapped.append(fp)
                except Exception as e:
                    log_event({"stage": "wrap-error", "err": str(e)})
        log_event({"stage": "turn", "n_tool_calls": len(tcs),
                   "tool_calls": [tc.get("function", {}).get("name")
                                  for tc in tcs],
                   "pipeline_writes": wrapped})
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
    print(f"LISTENING {PORT} DRIVER={DRIVER_MODEL} PIPELINE=plan->dispatch->synth",
          flush=True)
    srv.serve_forever()
