"""Cycle 4 Wave 3.A spike — research-loop dogfood trial.

Drives the running ``llm-orc serve`` instance via OpenAI-compatible
``/v1/chat/completions`` (the deployment shape under test). Sends one
research-loop user prompt and records the orchestrator's full trajectory:
its tool calls (closed five-tool surface), the ensemble dispatches it
chooses, the summarizer-harness's outputs, and the final assistant
content.

Records timing (TTFT, total wall-clock), an annotated trajectory, and
the raw response body for later inspection.
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
TRIALS.mkdir(parents=True, exist_ok=True)

SERVE_URL = "http://127.0.0.1:8765/v1/chat/completions"

# The bounded research question. Narrow enough to complete in 5-10 min.
# Chosen to be a question the cheap orchestrator can plausibly route via
# `list_ensembles` (to discover what's available) and `invoke_ensemble`
# (to dispatch the research-loop ensemble we just authored). The question
# is meaningful for the cycle's own work — it asks about a property the
# cycle's synthesis treats as load-bearing.
RESEARCH_QUESTION = (
    "I need to drive one bounded research-loop iteration on this focused "
    "question: 'What conditions make pre-specifiable routing (class-c "
    "decomposition) reliable at the cheap-orchestrator tier vs. when does "
    "it degrade?' Use the available ensemble library if it has a research "
    "ensemble that fits — discover it via list_ensembles, dispatch it via "
    "invoke_ensemble, then integrate the result into a recognizable "
    "research-log-style entry (Question / Findings / Tensions). Keep it "
    "tight; do not pad."
)


def call_serve(trial_id: str, *, stream: bool = False) -> dict:
    """POST to /v1/chat/completions and return parsed response + timing."""
    body = {
        "model": "orchestrator",  # model field is informational; serve uses
                                  # the configured orchestrator profile.
        "messages": [
            {"role": "user", "content": RESEARCH_QUESTION},
        ],
        "stream": stream,
    }
    req = urllib.request.Request(
        SERVE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[{time.strftime('%H:%M:%S')}] {trial_id}: posting to {SERVE_URL}")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = time.monotonic() - t0
            print(
                f"[{time.strftime('%H:%M:%S')}] {trial_id}: "
                f"HTTP {resp.status} in {elapsed:.1f}s, {len(raw)} bytes"
            )
            return {
                "trial_id": trial_id,
                "elapsed_seconds": elapsed,
                "http_status": resp.status,
                "raw_body": raw,
                "parsed": json.loads(raw),
            }
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0
        body_bytes = e.read()
        print(
            f"[{time.strftime('%H:%M:%S')}] {trial_id}: "
            f"HTTPError {e.code} in {elapsed:.1f}s"
        )
        return {
            "trial_id": trial_id,
            "elapsed_seconds": elapsed,
            "http_status": e.code,
            "raw_body": body_bytes.decode("utf-8", errors="replace"),
            "parsed": None,
            "error": str(e),
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] {trial_id}: "
            f"exception in {elapsed:.1f}s: {e!r}"
        )
        return {
            "trial_id": trial_id,
            "elapsed_seconds": elapsed,
            "http_status": None,
            "raw_body": None,
            "parsed": None,
            "error": repr(e),
        }


def annotate_trajectory(parsed: dict) -> dict:
    """Pull tool-call trajectory and final content out of the response."""
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
        "tool_call_args": [
            tc.get("function", {}).get("arguments") for tc in tool_calls
        ],
        "final_content_len": len(message.get("content") or ""),
        "final_content_preview": (message.get("content") or "")[:500],
        "usage": parsed.get("usage", {}),
    }


def main() -> int:
    trial_id = sys.argv[1] if len(sys.argv) > 1 else "trial-1"
    result = call_serve(trial_id)
    annotated = annotate_trajectory(result.get("parsed") or {})
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
    print(json.dumps(annotated, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
