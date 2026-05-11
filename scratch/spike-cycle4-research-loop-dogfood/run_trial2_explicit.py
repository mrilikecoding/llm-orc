"""Trial 2 — explicit dispatch instruction.

Tests whether the cheap orchestrator dispatches an ensemble when the
prompt names the tool and ensemble explicitly. If trial 1 failed because
the orchestrator interpreted the question as content-to-answer rather
than work-to-dispatch, this trial removes that ambiguity. If it still
fails to dispatch, the limitation is in the tool-calling decision, not
the prompt phrasing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error


HERE = Path(__file__).resolve().parent
TRIALS = HERE / "trials"

SERVE_URL = "http://127.0.0.1:8765/v1/chat/completions"

EXPLICIT_PROMPT = (
    "Use your tools. First call list_ensembles to see what's available. "
    "Then call invoke_ensemble with name='spike-cycle4-research-loop' "
    "and input='What conditions make pre-specifiable routing (class-c "
    "decomposition) reliable at the cheap-orchestrator tier vs. when "
    "does it degrade?'. Then integrate that ensemble's actual returned "
    "result (do not invent or paraphrase a result you did not receive) "
    "into a research-log entry with sections Question, Findings, "
    "Tensions. If a tool call fails, report the failure verbatim — do "
    "not fabricate a successful result."
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
                "trial_id": trial_id,
                "elapsed_seconds": elapsed,
                "http_status": resp.status,
                "raw_body": raw,
                "parsed": json.loads(raw),
            }
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] {trial_id}: exception in "
            f"{elapsed:.1f}s: {e!r}"
        )
        return {
            "trial_id": trial_id,
            "elapsed_seconds": elapsed,
            "error": repr(e),
            "raw_body": None,
            "parsed": None,
        }


def annotate(parsed: dict | None) -> dict:
    if not parsed:
        return {"error": "no parsed body"}
    choices = parsed.get("choices", [])
    if not choices:
        return {"error": "no choices"}
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", []) or []
    return {
        "finish_reason": choices[0].get("finish_reason"),
        "n_tool_calls": len(tool_calls),
        "tool_call_names": [
            tc.get("function", {}).get("name") for tc in tool_calls
        ],
        "tool_call_args_preview": [
            (tc.get("function", {}).get("arguments") or "")[:300]
            for tc in tool_calls
        ],
        "final_content_len": len(message.get("content") or ""),
        "final_content_preview": (message.get("content") or "")[:600],
        "usage": parsed.get("usage", {}),
    }


def main() -> int:
    trial_id = sys.argv[1] if len(sys.argv) > 1 else "trial-2-explicit"
    result = call_serve(trial_id, EXPLICIT_PROMPT)
    annotated = annotate(result.get("parsed"))
    out_dir = TRIALS / trial_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_response.json").write_text(
        result.get("raw_body") or "", encoding="utf-8"
    )
    (out_dir / "trial_summary.json").write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "elapsed_seconds": result.get("elapsed_seconds"),
                "http_status": result.get("http_status"),
                "error": result.get("error"),
                "trajectory": annotated,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.get("parsed"):
        message = result["parsed"]["choices"][0].get("message", {})
        (out_dir / "final_content.txt").write_text(
            message.get("content") or "", encoding="utf-8"
        )
    print(json.dumps(annotated, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
