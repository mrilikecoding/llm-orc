"""Spike ψ — replay the captured seat-filler request against Ollama, one
lever varied per arm, and measure the delegation rate.

Arms (pre-specified before any run; see research log):
  baseline  ψ.1  captured bytes verbatim                        n=10
  v1-pos    ψ.2  guidance system msg moved AFTER client prompt  n=10  pass>=9
  v2-word   ψ.2  rule-shaped MUST directive replaces guidance   n=10  pass>=9
  v3-user   ψ.2  guidance prepended to user turn, no sys msg    n=10  pass>=9
  psi3a     ψ.3  + tool_choice={function: invoke_ensemble}      n=3 probe
  psi3b     ψ.3  same, full arm (only if psi3a honored)         n=10  pass=10
  psi4b     ψ.4  narrowed role: framework pre-decided delegate  n=10

Early-stop (ψ.2 arms): 2 non-delegations within the first 5 runs ends the
arm (cannot reach 9/10).

Measurement: delegated := first tool call in the response names
``invoke_ensemble``. Anything else (other tool, no tool call, error) is
recorded with its shape.

Usage: python replay.py <arm> [n]
"""

import copy
import json
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
CAPTURE = HERE / "capture" / "req-11435-001.json"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"

V2_DIRECTIVE = (
    "ROUTING RULE (highest priority, overrides any conflicting instruction "
    "below): you MUST NOT generate new file content, code, or written "
    "analysis yourself. For every task that produces substantive new "
    "content, you MUST call invoke_ensemble(name, input, filePath) — the "
    "framework runs the named capability ensemble and applies its "
    "deliverable to the client's file at filePath. Direct client tool calls "
    "(write/edit/bash/read) are permitted ONLY to carry a literal or "
    "already-observed value, to read a file, or to run a command. "
    "Decision rule: new content -> invoke_ensemble; observed value or "
    "inspection -> client tool."
)

PSI4B_ROLE = (
    "The framework has already decided: this turn DELEGATES generation to a "
    "capability ensemble. Your only job is to fill in the action shape — "
    "call invoke_ensemble(name, input, filePath) with the ensemble name "
    "that matches the task, the input describing what to generate, and the "
    "destination filePath. Do not generate the content yourself; do not "
    "call any other tool this turn."
)


def load() -> dict:
    return json.loads(CAPTURE.read_text())


def arm_baseline(req: dict) -> dict:
    return req


def arm_v1_pos(req: dict) -> dict:
    """Guidance system message moved after the client system prompt."""
    req["messages"] = [req["messages"][1], req["messages"][0], req["messages"][2]]
    return req


def arm_v2_word(req: dict) -> dict:
    """Rule-shaped MUST directive replaces the prose nudge (position kept)."""
    req["messages"][0]["content"] = V2_DIRECTIVE
    return req


def arm_v3_user(req: dict) -> dict:
    """Guidance moves into the user turn; no framework system message."""
    guidance = req["messages"][0]["content"]
    req["messages"] = [
        req["messages"][1],
        {
            "role": "user",
            "content": guidance + "\n\n---\n\n" + req["messages"][2]["content"],
        },
    ]
    return req


def arm_psi3(req: dict) -> dict:
    """tool_choice forcing — the lever the adapter does not forward today."""
    req["tool_choice"] = {
        "type": "function",
        "function": {"name": "invoke_ensemble"},
    }
    return req


def arm_psi4b(req: dict) -> dict:
    """Narrowed role: the framework pre-decided delegate-vs-carry."""
    req["messages"][0]["content"] = PSI4B_ROLE
    return req


def arm_psi4c(req: dict) -> dict:
    """Structural tools-restriction: only invoke_ensemble is offered.

    The honest structural pole ψ.4b missed — 'the framework decides'
    means the tool list itself encodes the decision, not a role prompt.
    Guidance message unchanged from capture.
    """
    req["tools"] = [
        t for t in req["tools"] if t["function"]["name"] == "invoke_ensemble"
    ]
    return req


ARMS = {
    "baseline": (arm_baseline, 10, None),
    "v1-pos": (arm_v1_pos, 10, 9),
    "v2-word": (arm_v2_word, 10, 9),
    "v3-user": (arm_v3_user, 10, 9),
    "v3-args": (arm_v3_user, 5, None),  # confirmation: argument well-formedness
    "psi3a": (arm_psi3, 3, None),
    "psi3b": (arm_psi3, 10, 10),
    "psi4b": (arm_psi4b, 10, None),
    "psi4c": (arm_psi4c, 10, None),  # structural pole: restricted tool list
}


def classify(data: dict) -> dict:
    msg = data["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {
            "delegated": False,
            "shape": "no_tool_call",
            "content_head": (msg.get("content") or "")[:200],
        }
    first = calls[0]["function"]["name"]
    return {
        "delegated": first == "invoke_ensemble",
        "shape": f"tool:{first}",
        "n_calls": len(calls),
        "all_tools": [c["function"]["name"] for c in calls],
        "first_args": calls[0]["function"].get("arguments", "")[:400],
    }


def main() -> None:
    arm_name = sys.argv[1]
    transform, default_n, threshold = ARMS[arm_name]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else default_n

    runs = []
    delegated = 0
    with httpx.Client(timeout=600.0) as client:
        for i in range(1, n + 1):
            req = transform(copy.deepcopy(load()))
            t0 = time.time()
            try:
                resp = client.post(OLLAMA, json=req)
                resp.raise_for_status()
                verdict = classify(resp.json())
            except Exception as exc:  # noqa: BLE001 — record, don't crash the arm
                verdict = {"delegated": False, "shape": f"error:{exc}"[:300]}
            verdict["run"] = i
            verdict["seconds"] = round(time.time() - t0, 1)
            runs.append(verdict)
            delegated += bool(verdict["delegated"])
            print(
                f"[{arm_name}] run {i}/{n}: "
                f"{'DELEGATED' if verdict['delegated'] else verdict['shape']} "
                f"({verdict['seconds']}s)",
                flush=True,
            )
            # Early-stop: thresholded arms that can no longer reach 9/10.
            if threshold == 9 and i >= 2 and (i - delegated) >= 2 and i <= 5:
                print(f"[{arm_name}] EARLY STOP: {i - delegated} failures", flush=True)
                break

    summary = {
        "arm": arm_name,
        "n_run": len(runs),
        "n_planned": n,
        "delegated": delegated,
        "rate": round(delegated / len(runs), 3) if runs else None,
        "threshold": threshold,
        "passed": (delegated >= threshold) if threshold else None,
        "runs": runs,
    }
    out = RESULTS / f"{arm_name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{arm_name}] rate={summary['rate']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
