#!/usr/bin/env python3
"""Direct Ollama probe: is thinking the latency cost, and does disabling it work?

Hits Ollama's /api/chat directly (bypassing llm-orc) with the same coding prompt
under three conditions, and reports the output-token count and generation time
for each:

  1. default            — qwen3 thinking on (baseline)
  2. think=false param   — Ollama's native thinking toggle (top-level field)
  3. /no_think in prompt — the soft switch the seat variants used

`eval_count` is output tokens generated; `eval_duration` is time spent
generating them. If (2) or (3) sharply cut eval_count vs (1), thinking IS the
cost and that lever works — and wiring the native `think` param into
`src/llm_orc/models/ollama.py` becomes a real speed fix. If eval_count is flat,
thinking is not the bottleneck (raw throughput is).

Run:  uv run python scratch/spike-dynamic-dispatch/probe_thinking.py
"""

from __future__ import annotations

import json
import time
import urllib.request

URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
PROMPT = "Write a Python function is_prime(n). Output only the code, no prose."


def _call(body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"content-type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 (localhost)
        payload = json.loads(resp.read().decode())
    payload["_wall_s"] = time.time() - start
    return payload


def _row(label: str, r: dict) -> str:
    msg = r.get("message", {})
    has_thinking = bool(msg.get("thinking"))
    eval_count = r.get("eval_count", 0)
    eval_ns = r.get("eval_duration", 0) or 0
    tok_per_s = (eval_count / (eval_ns / 1e9)) if eval_ns else 0.0
    return (
        f"{label:<22} wall={r['_wall_s']:6.1f}s  out_tokens={eval_count:<5} "
        f"gen={eval_ns / 1e9:6.1f}s  {tok_per_s:5.1f} tok/s  "
        f"thinking_field={'yes' if has_thinking else 'no'}"
    )


def main() -> None:
    base = {"model": MODEL, "stream": False}
    user = {"role": "user", "content": PROMPT}
    nothink = {"role": "user", "content": PROMPT + " /no_think"}

    variants = [
        ("1 default", {**base, "messages": [user]}),
        ("2 think=false param", {**base, "messages": [user], "think": False}),
        ("3 /no_think prompt", {**base, "messages": [nothink]}),
    ]

    print(f"model={MODEL}\n")
    for label, body in variants:
        try:
            print(_row(label, _call(body)))
        except Exception as exc:  # noqa: BLE001 — surface probe errors inline
            print(f"{label:<22} ERROR: {exc!r}")


if __name__ == "__main__":
    main()
