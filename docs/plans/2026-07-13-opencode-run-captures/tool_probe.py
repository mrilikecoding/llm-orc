"""Throwaway OpenAI-compatible SSE endpoint that emits ONE arbitrary tool_call
and dumps opencode's follow-up (the tool result). Generalizes grep_probe.

Usage: python3 tool_probe.py <out_dir> <port> <tool_name> <arguments_json>
"""

import itertools
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT_DIR, PORT = sys.argv[1], int(sys.argv[2])
TOOL_NAME, TOOL_ARGS = sys.argv[3], sys.argv[4]
counter = itertools.count(1)

MODELS = {"object": "list", "data": [{"id": "agentic", "object": "model", "owned_by": "probe"}]}
TOOL_CALL = {
    "index": 0, "id": "call_probe_1", "type": "function",
    "function": {"name": TOOL_NAME, "arguments": TOOL_ARGS},
}


def _sse(o): return ("data: " + json.dumps(o) + "\n\n").encode()


def _chunk(delta, finish=None):
    return {"id": "c", "object": "chat.completion.chunk", "created": 0,
            "model": "agentic", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            b = json.dumps(MODELS).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = next(counter)
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        text = raw.decode("utf-8", "replace")
        try: data = json.loads(raw)
        except Exception: data = {"_unparsed": text}
        with open(f"{OUT_DIR}/probe-req-{n:02d}.json", "w") as f:
            json.dump(data, f, indent=2)
        done = any(isinstance(m, dict) and m.get("role") == "tool"
                   for m in (data.get("messages", []) if isinstance(data, dict) else [])) \
            or "call_probe_1" in text
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if done:
            self.wfile.write(_sse(_chunk({"role": "assistant", "content": "captured"})))
            self.wfile.write(_sse(_chunk({}, finish="stop")))
        else:
            self.wfile.write(_sse(_chunk({"role": "assistant", "tool_calls": [TOOL_CALL]})))
            self.wfile.write(_sse(_chunk({}, finish="tool_calls")))
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


print(f"tool_probe on :{PORT} emitting {TOOL_NAME}({TOOL_ARGS})", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
