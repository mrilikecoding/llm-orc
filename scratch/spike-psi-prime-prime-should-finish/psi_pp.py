"""Spike ψ″ — should-finish trailing shape (Finding F attribution).

Pre-registered in the ψ research log (§Spike ψ″, 2026-06-04). Replay
harness = ψ′ infrastructure: the composed seat-filler request goes
straight to Ollama (the framework ↔ seat-filler hop), qwen3:14b, $0.

Context amendment (recorded): the pre-registration named the WP-LB-I
run's first write pair; the ψ capture's trailing request
(req-11435-004.json) is used instead — it is the same shape in real
seat-filler-bound bytes (OpenCode system prompt + user task + THREE
completed assistant/tool write pairs, no new task) and is the
methods-reviewed ψ′ substrate. Strictly stronger "work complete" tail.

Arms (n=10 each):
  E1  tail + standalone trailing C3 guidance (production form as landed)
  E2  tail, no guidance appended (control — isolates the guidance)
  E3  conditional (only if E1 suppressed AND E2 finishes): guidance +
      completion clause

Measurement: finish := response with zero tool calls. more_work := any
tool call (invoke_ensemble or client tool); the called tool is recorded.

Usage: python psi_pp.py E1|E2|E3 [n]
"""

import copy
import json
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
CAPTURE = (
    HERE.parent / "spike-psi-delegation-rate" / "capture" / "req-11435-004.json"
)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"

COMPLETION_CLAUSE = (
    " If the requested work is already complete, respond with a brief "
    "summary instead of calling tools."
)

MID_TASK_ASK = (
    '"Write a python module string_utils.py with a function that reverses '
    "the word order of a string, and a test_string_utils.py with unit "
    'tests for it."'
)


def _base() -> tuple[dict, str]:
    """The captured trailing request sans the old system guidance.

    Returns (request, guidance_text). messages[0] of the capture is the
    pre-ADR-036 framework system guidance — its content is byte-identical
    to the landed ``_DELEGATION_GUIDANCE``; only placement moved.
    """
    req = copy.deepcopy(json.loads(CAPTURE.read_text()))
    guidance = req["messages"][0]["content"]
    req["messages"] = req["messages"][1:]
    return req, guidance


def _mid_task_base() -> tuple[dict, str]:
    """E4 context: two-deliverable task, ONE write pair, work remaining.

    Constructed-adjacent per the pre-registration: the user task text is
    edited and the tail truncated to one assistant/tool pair; system
    prompt, structure, and tools stay the captured bytes.
    """
    req, guidance = _base()
    req["messages"][1]["content"] = MID_TASK_ASK
    # messages: [system, user, asst, tool, asst, tool, asst, tool]
    req["messages"] = req["messages"][:4]
    return req, guidance


def build(arm: str) -> dict:
    if arm in ("E4a", "E4b"):
        req, guidance = _mid_task_base()
        if arm == "E4b":
            req["messages"].append({"role": "user", "content": guidance})
        return req
    req, guidance = _base()
    if arm == "E1":
        req["messages"].append({"role": "user", "content": guidance})
    elif arm == "E2":
        pass  # control: no guidance appended
    elif arm == "E3":
        req["messages"].append(
            {"role": "user", "content": guidance + COMPLETION_CLAUSE}
        )
    else:
        raise SystemExit(f"unknown arm {arm}")
    return req


def classify(data: dict) -> dict:
    msg = data["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {
            "finish": True,
            "tool": None,
            "content_head": (msg.get("content") or "")[:200],
        }
    fn = calls[0]["function"]
    return {
        "finish": False,
        "tool": fn.get("name"),
        "args_head": (fn.get("arguments") or "")[:200],
    }


def main() -> None:
    arm = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    req = build(arm)
    records = []
    finishes = 0
    for i in range(n):
        t0 = time.time()
        resp = httpx.post(OLLAMA, json=req, timeout=300.0)
        resp.raise_for_status()
        verdict = classify(resp.json())
        verdict["run"] = i + 1
        verdict["elapsed_s"] = round(time.time() - t0, 1)
        records.append(verdict)
        finishes += verdict["finish"]
        print(f"{arm} run {i + 1}/{n}: "
              f"{'FINISH' if verdict['finish'] else 'tool=' + str(verdict['tool'])} "
              f"({verdict['elapsed_s']}s)")
    out = RESULTS / f"{arm}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\n{arm}: finish {finishes}/{n} — recorded {out}")


if __name__ == "__main__":
    main()
