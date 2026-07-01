#!/usr/bin/env python3
"""Spike Ω-serve — the ensemble served to OpenCode as a transparent model.

Answers (a): can the ensemble-only architecture serve OpenCode transparently,
multi-turn, so OpenCode drives a task to completion as if talking to one model?

A standalone /v1/chat/completions server. Behind the endpoint:
  - turn 1 (no session substrate yet): a DECOMPOSE ensemble turns the task
    into an ordered deliverable list -> substrate plan_queue.
  - each request: produce the plan_queue head via the omega-4 flow
    (decide -> adapter-mediated dispatch -> internal recovery against a light
    structural gate), write a copy to the session's produced/ dir for
    cross-file grounding, advance the substrate, return ONE write tool_call.
  - queue empty: return finish_reason=stop.

State lives in a per-session substrate file (keyed by the first user message),
threaded across OpenCode's turns. No engine primitive; the adapter is the
irreducible serving glue (§8). Recovery is WITHIN a request so each response
is a validated tool_call, exactly what a good single model would return.

Run:
    uv run python scratch/spike-omega-serve/serve_ensemble.py [port]
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

ENS = Path(__file__).resolve().parents[2] / ".llm-orc" / "ensembles"
DECOMPOSE_YAML = ENS / "spike-omega-serve" / "decompose-8b.yaml"
DECIDE_YAML = ENS / "spike-omega-4" / "agent-turn-omega4-8b.yaml"
PROJECT_DIR = ENS.parent
OUT_ROOT = Path(__file__).resolve().parent / "serve_out"
MAX_RETRIES = 2

_LOADER = EnsembleLoader()
_DECOMPOSE = _LOADER.load_from_file(str(DECOMPOSE_YAML))
_DECIDE = _LOADER.load_from_file(str(DECIDE_YAML))


def clean_content(content: str, is_code: bool) -> str:
    s = content.strip()
    if is_code and s.startswith("```"):
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    return s


def light_gate(content: str, kind: str) -> tuple[bool, str, str]:
    """General, kind-based structural gate (no per-task expectations)."""
    is_code = kind in ("python_module", "python_cli")
    s = clean_content(content, is_code)
    if not s:
        return False, "empty content", s
    if is_code:
        try:
            ast.parse(s)
        except SyntaxError as e:
            return False, f"python syntax error: {e}", s
        return True, "ok", s
    if kind == "markdown_doc":
        try:
            t = ast.parse(s)
            if any(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom,
                                  ast.ClassDef)) for n in t.body):
                return False, "must be Markdown prose, not Python source", s
        except SyntaxError:
            pass
        if not any(ln.lstrip().startswith("#") for ln in s.splitlines()):
            return False, "Markdown doc needs a heading (# ...)", s
        return True, "ok", s
    return True, "ok", s


def _first_user(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return m["content"]
    return ""


def _aux_reply(messages: list[dict]) -> str:
    """A plain-text reply for OpenCode's toolless meta calls (title/summary).

    Uses the last user message (the actual subject) to produce a short title;
    never drives the build pipeline.
    """
    subject = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            subject = m["content"].strip().strip('"')
            break
    words = subject.split()
    return " ".join(words[:6]) if words else "Task"


def _extract_terminal(result: dict) -> str:
    results = result.get("results", {}) if isinstance(result, dict) else {}
    if not results:
        return ""
    node = results[list(results.keys())[-1]]
    return node.get("response", "") if isinstance(node, dict) else ""


def _parse_json_obj(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


async def _decompose(executor, task: str) -> list[dict]:
    res = await executor.execute(_DECOMPOSE, task)
    resp = (res.get("results", {}).get("decomposer", {}).get("response", "")
            if isinstance(res, dict) else "")
    data = _parse_json_obj(resp)
    delivs = data.get("deliverables", [])
    out = []
    for d in delivs:
        if isinstance(d, dict) and d.get("file"):
            out.append({"file": d["file"], "kind": d.get("kind", "other")})
    return out


async def _produce(executor, task: str, substrate: Path, target: str,
                   kind: str) -> tuple[str, bool, int]:
    """Produce one file with within-request recovery. Returns (content, ok, attempts)."""
    hint = ""
    last = ""
    for attempt in range(MAX_RETRIES + 1):
        ltr = hint if attempt > 0 else ""
        req = json.dumps({"task": task, "substrate_path": str(substrate),
                          "last_tool_result": ltr})
        dec = await executor.execute(_DECIDE, req)
        score = (dec.get("results", {}).get("score", {}).get("response", "")
                 if isinstance(dec, dict) else "")
        decision = _parse_json_obj(score)
        cap_path, di = decision.get("capability_path"), decision.get("dispatch_input", "")
        if not cap_path:
            return "", False, attempt + 1
        cap = _LOADER.load_from_file(cap_path)
        capres = await executor.execute(cap, di)
        content = _extract_terminal(capres)
        ok, why, cleaned = light_gate(content, kind)
        last = cleaned
        if ok:
            return cleaned, True, attempt + 1
        hint = f"PRODUCTION REJECTED {target}: {why}. Re-emit the corrected file only."
    return last, False, MAX_RETRIES + 1


async def handle_async(messages: list[dict]) -> dict:
    task = _first_user(messages)
    if not task:
        return {"finish": True, "content": "no task in messages"}
    sid = hashlib.sha256(task.encode()).hexdigest()[:16]
    sdir = OUT_ROOT / sid
    produced_dir = sdir / "produced"
    substrate = sdir / "session_state.json"
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)

    if not substrate.exists():
        produced_dir.mkdir(parents=True, exist_ok=True)
        delivs = await _decompose(executor, task)
        if not delivs:
            return {"finish": True, "content": "decomposition produced no deliverables"}
        substrate.write_text(json.dumps({
            "task": task, "requested": [d["file"] for d in delivs], "produced": [],
            "plan_queue": [d["file"] for d in delivs],
            "kinds": {d["file"]: d["kind"] for d in delivs},
            "remaining_anchor": "",
        }, indent=2))
        sys.stderr.write(f"[serve {sid}] decomposed -> {[d['file'] for d in delivs]}\n")

    state = json.loads(substrate.read_text())
    if not state["plan_queue"]:
        return {"finish": True,
                "content": f"All {len(state['produced'])} deliverables produced: "
                           f"{', '.join(state['produced'])}"}

    target = state["plan_queue"][0]
    kind = state.get("kinds", {}).get(target, "other")
    content, ok, attempts = await _produce(executor, task, substrate, target, kind)
    # write a copy for cross-file grounding; advance optimistically
    (produced_dir / target).parent.mkdir(parents=True, exist_ok=True)
    (produced_dir / target).write_text(content)
    state["produced"].append(target)
    state["plan_queue"] = state["plan_queue"][1:]
    substrate.write_text(json.dumps(state, indent=2))
    sys.stderr.write(f"[serve {sid}] turn -> {target} ({kind}) "
                     f"{'ok' if ok else 'GAVEUP'} after {attempts} attempt(s), "
                     f"{len(content)}B; remaining {state['plan_queue']}\n")
    return {"finish": False, "file": target, "content": content}


def _write_tool_call(outcome: dict) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
            "function": {"name": "write", "arguments": json.dumps(
                {"filePath": outcome["file"], "content": outcome["content"]})}}


def _completion_body(outcome: dict, model: str) -> dict:
    if outcome.get("finish"):
        message = {"role": "assistant", "content": outcome.get("content", "Done.")}
        finish_reason = "stop"
    else:
        message = {"role": "assistant", "content": None,
                   "tool_calls": [_write_tool_call(outcome)]}
        finish_reason = "tool_calls"
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _stream_frames(outcome: dict, model: str) -> list[dict]:
    """OpenAI chat.completion.chunk frames — real OpenCode drives via streaming."""
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = [frame({"role": "assistant"}, None)]
    if outcome.get("finish"):
        frames.append(frame({"content": outcome.get("content", "Done.")}, None))
        frames.append(frame({}, "stop"))
    else:
        tc = _write_tool_call(outcome)
        tc_delta = {"index": 0, "id": tc["id"], "type": "function",
                    "function": tc["function"]}
        frames.append(frame({"tool_calls": [tc_delta]}, None))
        frames.append(frame({}, "tool_calls"))
    return frames


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: ANN001, ANN002
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, frames: list[dict]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers()
        for fr in frames:
            self.wfile.write(f"data: {json.dumps(fr)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": "ensemble-agent", "object": "model", "owned_by": "llm-orc-spike"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("content-length", 0))
        try:
            req = json.loads(self.rfile.read(n).decode() if n else "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        stream = bool(req.get("stream"))
        msgs = req.get("messages", [])
        shape = [
            {"role": m.get("role"),
             "content_type": type(m.get("content")).__name__,
             "snippet": (m.get("content") if isinstance(m.get("content"), str)
                         else json.dumps(m.get("content")))[:120]}
            for m in msgs
        ]
        tools = req.get("tools", [])
        sys.stderr.write(
            f"[serve] request stream={stream} model={req.get('model')} "
            f"messages={len(msgs)} tools={len(tools)}\n")
        sys.stderr.write(f"[serve]   shape={json.dumps(shape)}\n")

        # OpenCode makes auxiliary NON-agent calls (session title generation,
        # summarization) with NO tools. A model serving OpenCode must answer
        # those as plain text, NOT drive the build pipeline. Discriminate on the
        # presence of tools: the coding loop provides tools; meta calls do not.
        if not tools:
            title = _aux_reply(msgs)
            sys.stderr.write(f"[serve]   -> aux (toolless) reply: {title!r}\n")
            outcome = {"finish": True, "content": title}
            model = req.get("model", "ensemble-agent")
            self._send_sse(_stream_frames(outcome, model)) if stream \
                else self._send(200, _completion_body(outcome, model))
            return

        try:
            outcome = asyncio.run(handle_async(msgs))
        except Exception as e:  # noqa: BLE001 — surface as 500 for the client
            sys.stderr.write(f"[serve] ERROR: {e!r}\n")
            self._send(500, {"error": "internal", "detail": str(e)[:200]})
            return
        model = req.get("model", "ensemble-agent")
        if stream:
            self._send_sse(_stream_frames(outcome, model))
        else:
            self._send(200, _completion_body(outcome, model))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"[serve] ensemble-agent on http://127.0.0.1:{port} "
                     f"(decompose={_DECOMPOSE.name}, decide={_DECIDE.name})\n")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
