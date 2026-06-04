"""Spike ψ′ — V3 confirmation under varied circumstances.

Post-methods-review design (see research log §Spike ψ′ and
housekeeping/audits/research-methods-spike-psi-prime.md). All arms use V3
composition: the delegation guidance lives in a user turn; no framework
system message. delegated := first tool call names ``invoke_ensemble``.

Arms:
  A1..A5  phrasing/complexity generalization  n=5 each  aggregate >=23/25
  B1..B3  carry-side preservation             n=5 each  <=1/15 false-deleg.
          (B3 secondary check: write content carries the literal verbatim)
  B4      repair-shaped boundary              n=5       informative
  C1..C3  multi-turn attachment variants      n=5 each  >=4/5 informative
  D1..D2  model portability                   n=5 each  directional

Usage: python psi_prime.py <arm>      (e.g. A1, B3, C2, D1)
"""

import copy
import json
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
FIRST_TURN = HERE / "capture" / "req-11435-001.json"
TRAILING = HERE / "capture" / "req-11435-004.json"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"

B3_PAYLOAD = "hello world, this is a literal payload"

A_TASKS = {
    "A1": "Create a file called csv_helper.py that loads a CSV file and computes the mean of each numeric column.",
    "A2": "I need a small utility for parsing ISO dates from log lines. Put it in date_parse.py.",
    "A3": "Can you make a quick script that renames files in a folder using a regex pattern? Call it rename_files.py.",
    "A4": "Add a fibonacci(n) function in fib.py.",
    # A5 per methods review P1-B: multi-instruction complexity.
    "A5": (
        "Create a config_loader.py module that reads settings from a JSON "
        "file, validates that the required keys host, port, and timeout are "
        "present, falls back to documented defaults for missing optional "
        "keys, and logs a warning for unknown keys. Include a usage example "
        "in the module docstring."
    ),
}

B_TASKS = {
    "B1": "Read string_utils.py and explain what it does.",
    "B2": "Run ls -la and tell me what files exist.",
    "B3": f"Write exactly this to notes.txt: {B3_PAYLOAD}",
    # B4 per methods review P2-A: psi.4a a01 boundary, informative.
    "B4": "Fix the bug in string_utils.py where count_vowels misses uppercase vowels.",
}

C_TASK = "Now also write unit tests for string_utils.py in test_string_utils.py."

D_MODELS = {"D1": "qwen3.5:9b", "D2": "mistral-nemo:12b"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _guidance(req: dict) -> str:
    """The captured delegation guidance (msg[0], system)."""
    return req["messages"][0]["content"]


def _v3_first_turn(task: str, model: str | None = None) -> dict:
    """V3 composition on the captured first-turn request, task replaced."""
    req = copy.deepcopy(_load(FIRST_TURN))
    guidance = _guidance(req)
    req["messages"] = [
        req["messages"][1],  # client system prompt, alone in the system region
        {"role": "user", "content": guidance + "\n\n---\n\n" + task},
    ]
    if model:
        req["model"] = model
    return req


def _trailing_base() -> tuple[dict, str]:
    """Captured trailing-turn context (3 assistant/tool pairs) sans guidance."""
    req = copy.deepcopy(_load(TRAILING))
    guidance = _guidance(req)
    req["messages"] = req["messages"][1:]  # drop the system guidance message
    return req, guidance


def arm_A(key: str) -> dict:
    return _v3_first_turn(A_TASKS[key])


def arm_B(key: str) -> dict:
    return _v3_first_turn(B_TASKS[key])


def arm_C1() -> dict:
    """Guidance merged into the NEW trailing user message."""
    req, guidance = _trailing_base()
    req["messages"].append(
        {"role": "user", "content": guidance + "\n\n---\n\n" + C_TASK}
    )
    return req


def arm_C2() -> dict:
    """Guidance merged into the FIRST user message only."""
    req, guidance = _trailing_base()
    first_user = next(m for m in req["messages"] if m["role"] == "user")
    first_user["content"] = guidance + "\n\n---\n\n" + first_user["content"]
    req["messages"].append({"role": "user", "content": C_TASK})
    return req


def arm_C3() -> dict:
    """Production form: clean task message, then guidance as its own
    trailing user-role message (methods review P2-C + P3-B)."""
    req, guidance = _trailing_base()
    req["messages"].append({"role": "user", "content": C_TASK})
    req["messages"].append({"role": "user", "content": guidance})
    return req


def arm_D(key: str) -> dict:
    return _v3_first_turn(A_TASKS["A1"], model=D_MODELS[key])


def build(arm: str) -> dict:
    if arm in A_TASKS:
        return arm_A(arm)
    if arm in B_TASKS:
        return arm_B(arm)
    if arm == "C1":
        return arm_C1()
    if arm == "C2":
        return arm_C2()
    if arm == "C3":
        return arm_C3()
    if arm in D_MODELS:
        return arm_D(arm)
    raise SystemExit(f"unknown arm {arm}")


def classify(arm: str, data: dict) -> dict:
    msg = data["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {
            "delegated": False,
            "shape": "no_tool_call",
            "content_head": (msg.get("content") or "")[:200],
        }
    fn = calls[0]["function"]
    args = fn.get("arguments", "")
    verdict = {
        "delegated": fn["name"] == "invoke_ensemble",
        "shape": f"tool:{fn['name']}",
        "n_calls": len(calls),
        "all_tools": [c["function"]["name"] for c in calls],
        "first_args": args[:400],
    }
    # Methods review P1-A: B3 verbatim-carry secondary measurement.
    if arm == "B3" and fn["name"] == "write":
        verdict["payload_verbatim"] = B3_PAYLOAD in args
    return verdict


def main() -> None:
    arm = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    runs = []
    delegated = 0
    with httpx.Client(timeout=600.0) as client:
        for i in range(1, n + 1):
            req = build(arm)
            t0 = time.time()
            try:
                resp = client.post(OLLAMA, json=req)
                resp.raise_for_status()
                verdict = classify(arm, resp.json())
            except Exception as exc:  # noqa: BLE001 — record, don't crash
                verdict = {"delegated": False, "shape": f"error:{exc}"[:300]}
            verdict["run"] = i
            verdict["seconds"] = round(time.time() - t0, 1)
            runs.append(verdict)
            delegated += bool(verdict["delegated"])
            extra = (
                ""
                if "payload_verbatim" not in verdict
                else f" verbatim={verdict['payload_verbatim']}"
            )
            print(
                f"[{arm}] run {i}/{n}: "
                f"{'DELEGATED' if verdict['delegated'] else verdict['shape']}"
                f"{extra} ({verdict['seconds']}s)",
                flush=True,
            )

    summary = {
        "arm": arm,
        "n_run": len(runs),
        "delegated": delegated,
        "rate": round(delegated / len(runs), 3),
        "runs": runs,
    }
    out = RESULTS / f"psi-prime-{arm}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{arm}] rate={summary['rate']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
