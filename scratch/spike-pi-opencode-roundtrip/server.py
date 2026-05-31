#!/usr/bin/env python3
"""Spike pi - minimal OpenAI-compatible logging server.

Phase 0: log every request OpenCode sends (especially the `tools` array and
its exact write-tool schema), return a trivial assistant text message so the
turn ends cleanly. No ensemble, no tool_calls - pure reconnaissance.
"""
import datetime
import http.server
import json
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
LOG = "requests.jsonl"


def log_event(obj: dict) -> None:
    obj["ts"] = datetime.datetime.now().isoformat()
    with open(LOG, "a") as f:
        f.write(json.dumps(obj) + "\n")


class Handler(http.server.BaseHTTPRequestHandler):
    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802
        self._read_body()
        log_event({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "spike-model", "object": "model", "owned_by": "spike"}]})
        else:
            self._json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        raw = self._read_body()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"_unparsed": raw.decode("utf-8", "replace")}
        log_event({"method": "POST", "path": self.path,
                   "headers": dict(self.headers), "body": parsed})
        text = "Phase 0 observe: your request was logged."
        if bool(parsed.get("stream")):
            self._sse_text(text)
        else:
            self._json(200, self._completion_text(text))

    def _completion_text(self, text: str) -> dict:
        return {
            "id": "chatcmpl-spike", "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": "spike-model",
            "choices": [{"index": 0, "message": {"role": "assistant",
                        "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        }

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse_text(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        created = int(datetime.datetime.now().timestamp())

        def chunk(delta: dict, finish=None) -> None:
            obj = {"id": "chatcmpl-spike", "object": "chat.completion.chunk",
                   "created": created, "model": "spike-model",
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant"})
        chunk({"content": text})
        chunk({}, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args) -> None:  # silence default stderr logging
        pass


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    srv = ThreadingServer(("127.0.0.1", PORT), Handler)
    print(f"LISTENING {PORT}", flush=True)
    srv.serve_forever()
