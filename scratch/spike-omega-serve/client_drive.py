#!/usr/bin/env python3
"""Spike Ω-serve — a faithful OpenCode-contract client driving the ensemble.

Proves (a1): a generic client that owns the loop (send messages+tools ->
one tool_call -> execute -> re-prompt) drives the ensemble serving endpoint
multi-turn to completion, transparently — exactly the OpenCode contract.
This is the automatable stand-in for real `opencode run`; real OpenCode is
the gold-standard (a2) the user drives.

Usage (server must be running on :8099):
    uv run python scratch/spike-omega-serve/client_drive.py
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spike-omega-4"))
from omega4_compare import SYSTEM, TASK_PROMPT, TOOLS, score_arm  # noqa: E402

BASE = "http://127.0.0.1:8099/v1/chat/completions"
WORKSPACE = Path(__file__).resolve().parent / "client_workspace"


def run() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for p in WORKSPACE.glob("*"):
        if p.is_file():
            p.unlink()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TASK_PROMPT}]
    produced: list[str] = []
    turns, t0 = 0, time.perf_counter()
    while turns < 20:
        turns += 1
        r = httpx.post(BASE, json={"model": "ensemble-agent", "messages": messages,
                                   "tools": TOOLS}, timeout=900)
        if r.status_code != 200:
            print(f"  turn {turns}: HTTP {r.status_code} {r.text[:160]}")
            break
        ch = r.json()["choices"][0]
        msg = ch["message"]
        if ch["finish_reason"] == "tool_calls":
            tc = msg["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            fp, body = args["filePath"], args.get("content", "")
            (WORKSPACE / fp).parent.mkdir(parents=True, exist_ok=True)
            (WORKSPACE / fp).write_text(body)
            produced.append(fp)
            parses = ""
            if fp.endswith(".py"):
                try:
                    ast.parse(body)
                    parses = " (parses)"
                except SyntaxError:
                    parses = " (NO PARSE)"
            print(f"  turn {turns}: client executed write {fp} ({len(body)}B){parses}")
            messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": f"Wrote {fp}. Continue."})
        else:
            print(f"  turn {turns}: finish -> {(msg.get('content') or '')[:80]!r}")
            break
    elapsed = time.perf_counter() - t0
    print(f"\n[client] {turns} turns, {elapsed:.1f}s, produced: {produced}")
    print("[client] structural score (same gate as Ω-4):")
    print(json.dumps(score_arm(WORKSPACE), indent=2))


if __name__ == "__main__":
    run()
