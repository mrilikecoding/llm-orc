"""Spike ψ.0 — dump-and-forward HTTP proxy for request-byte capture.

Captures every POST body verbatim to a numbered file, forwards upstream,
and streams the response through while teeing it to disk (so SSE turns
reach the client incrementally — OpenCode's turns run minutes).

  hop 1 (client):      127.0.0.1:8766 -> 127.0.0.1:8765   (OpenCode -> llm-orc serve)
  hop 2 (seat-filler): 127.0.0.1:11435 -> 127.0.0.1:11434 (serve -> Ollama)

Usage: python capture_proxy.py <listen_port> <target_port> <dump_dir>
"""

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

LISTEN_PORT = int(sys.argv[1])
TARGET_PORT = int(sys.argv[2])
DUMP_DIR = Path(sys.argv[3])
DUMP_DIR.mkdir(parents=True, exist_ok=True)

_counter = {"n": 0}
_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


async def relay(request: Request) -> StreamingResponse:
    body = await request.body()
    _counter["n"] += 1
    seq = _counter["n"]

    stem = f"{LISTEN_PORT}-{seq:03d}"
    (DUMP_DIR / f"req-{stem}.meta.json").write_text(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "headers": dict(request.headers),
            },
            indent=2,
        )
    )
    (DUMP_DIR / f"req-{stem}.json").write_bytes(body)

    target = f"http://127.0.0.1:{TARGET_PORT}{request.url.path}"
    if request.url.query:
        target += f"?{request.url.query}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    upstream_req = _client.build_request(
        request.method, target, content=body, headers=headers
    )
    upstream = await _client.send(upstream_req, stream=True)

    resp_path = DUMP_DIR / f"resp-{stem}.bin"

    async def tee() -> AsyncIterator[bytes]:
        with resp_path.open("wb") as sink:
            async for chunk in upstream.aiter_bytes():
                sink.write(chunk)
                sink.flush()
                yield chunk
        await upstream.aclose()

    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")
    }
    return StreamingResponse(
        tee(), status_code=upstream.status_code, headers=resp_headers
    )


app = Starlette(
    routes=[Route("/{path:path}", relay, methods=["GET", "POST", "PUT", "DELETE"])]
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=LISTEN_PORT, log_level="warning")
