"""Trial 3 — diagnostic: are tools even visible to the orchestrator?

Trials 1 and 2 showed zero tool calls and aggressive fabrication.
Diagnostic question: does the model see the tool schema at all? If yes,
the failure is dispatch-decision (the cheap orchestrator chooses to
fabricate rather than dispatch). If no, the failure is upstream (tool
schema not being passed by the runtime, or being stripped at the
provider boundary).

Sends a meta-question about tools + a forcing instruction. If the model
calls list_ensembles, the schema is reaching it. If not, we have a
provider-side or runtime-side issue.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import urllib.request


HERE = Path(__file__).resolve().parent
TRIALS = HERE / "trials"
SERVE_URL = "http://127.0.0.1:8765/v1/chat/completions"

DIAGNOSTIC_PROMPT = (
    "Call list_ensembles right now. Output nothing else. Just the tool "
    "call. No prose, no explanation, no narration. Tool call only."
)


def call_serve(trial_id: str, prompt: str) -> dict:
    body = {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    req = urllib.request.Request(
        SERVE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[{time.strftime('%H:%M:%S')}] {trial_id}: posting")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = time.monotonic() - t0
            print(
                f"[{time.strftime('%H:%M:%S')}] {trial_id}: HTTP "
                f"{resp.status} in {elapsed:.1f}s, {len(raw)} bytes"
            )
            return {
                "elapsed_seconds": elapsed,
                "http_status": resp.status,
                "raw_body": raw,
                "parsed": json.loads(raw),
            }
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {trial_id}: error: {e!r}")
        return {"elapsed_seconds": elapsed, "error": repr(e), "raw_body": None, "parsed": None}


def main() -> int:
    trial_id = sys.argv[1] if len(sys.argv) > 1 else "trial-3-diagnostic"
    result = call_serve(trial_id, DIAGNOSTIC_PROMPT)
    out_dir = TRIALS / trial_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_response.json").write_text(
        result.get("raw_body") or "", encoding="utf-8"
    )
    parsed = result.get("parsed") or {}
    choices = parsed.get("choices", [])
    msg = (choices[0] if choices else {}).get("message", {}) if choices else {}
    tool_calls = msg.get("tool_calls", []) or []
    summary = {
        "elapsed_seconds": result.get("elapsed_seconds"),
        "http_status": result.get("http_status"),
        "finish_reason": (choices[0] if choices else {}).get("finish_reason"),
        "n_tool_calls": len(tool_calls),
        "tool_call_names": [tc.get("function", {}).get("name") for tc in tool_calls],
        "tool_call_args": [tc.get("function", {}).get("arguments") for tc in tool_calls],
        "final_content": msg.get("content") or "",
    }
    (out_dir / "trial_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
