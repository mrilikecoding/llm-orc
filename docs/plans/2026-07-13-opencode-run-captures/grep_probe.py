"""Throwaway OpenAI-compatible SSE endpoint that emits ONE grep tool_call,
then dumps opencode's follow-up request (which carries the real grep tool
output) so we can pin grep's wire result format for the grep->read normalizer.

Usage: python3 grep_probe.py <out_dir> <port>
"""

import itertools
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
counter = itertools.count(1)

MODELS = {
    "object": "list",
    "data": [{"id": "agentic", "object": "model", "created": 0, "owned_by": "probe"}],
}
GREP_CALL = {
    "index": 0,
    "id": "call_grep_1",
    "type": "function",
    "function": {"name": "grep", "arguments": json.dumps({"pattern": "def mean"})},
}


def _sse(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-probe",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "agentic",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: ANN001
        pass

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/v1/models"):
            body = json.dumps(MODELS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        n = next(counter)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        text = raw.decode("utf-8", "replace")
        try:
            data = json.loads(raw)
        except Exception:
            data = {"_unparsed": text}
        with open(f"{OUT_DIR}/probe-req-{n:02d}.json", "w") as f:
            json.dump(data, f, indent=2)
        msgs = data.get("messages", []) if isinstance(data, dict) else []
        has_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in msgs
        ) or ("call_grep_1" in text)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if has_tool_result:
            self.wfile.write(_sse(_chunk({"role": "assistant", "content": "captured"})))
            self.wfile.write(_sse(_chunk({}, finish="stop")))
        else:
            self.wfile.write(
                _sse(_chunk({"role": "assistant", "tool_calls": [GREP_CALL]}))
            )
            self.wfile.write(_sse(_chunk({}, finish="tool_calls")))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    print(f"grep probe on :{PORT}, dumping to {OUT_DIR}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
