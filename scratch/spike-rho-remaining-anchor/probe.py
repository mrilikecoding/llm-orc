"""Spike ρ — remaining-work anchor (Cycle 7 loop-back #6, Finding G).

Pre-registered + methods-reviewed (all P1/P2/P3 applied):
docs/agentic-serving/essays/research-logs/cycle-7-spike-rho-remaining-work-anchor.md
docs/agentic-serving/housekeeping/audits/research-methods-spike-rho.md

Measures the ADR-037 amendment (route the judge's "what remains" forward to
anchor call 2). Composition through the landed code path
(compose_judgment_message / _seat_filler_messages / _delegation_tools); the
anchor in the production arms is the judge's ACTUAL statement, not a hardcoded
string (the rung-1 gap this spike closes).

Arms (qwen3:14b via Ollama /v1, $0 local):
  rho1     — judge remaining-naming: run the real judgment call, record verdict
             + the remaining-work statement (adjudicated offline against the
             three-level standard).
  rho2     — call 2 anchored with the judge's statement only (minimal form).
  rho2_imp — call 2 anchored with the judge's statement + "Produce that next."
  control  — call 2 anchored with a content-neutral trailing perturbation
             (delegation standard re-stated; no remaining-work content).

rho2 / rho2_imp / control each consume a judge statement from a paired rho1
trial (run rho1 first; its statements feed the others), so the production
composition is end-to-end. control uses a fixed neutral string (no judge
statement needed).

Usage: python probe.py rho1 B2|B3 [n]
       python probe.py rho2 B2|B3 [n]        (reads statements_<base>.json)
       python probe.py rho2_imp B2|B3 [n]
       python probe.py control B2 [n]
"""

import copy
import json
import re
import sys
from pathlib import Path

import httpx

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.loop_driver import (
    LoopDriver,
    compose_judgment_message,
    parse_verdict,
    strip_verdict,
)
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer

HERE = Path(__file__).parent
OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:14b"
CAPS = frozenset({"code-generator"})

JUDGE_SYSTEM = (
    "You review the state of an automated coding session. Your only job is "
    "to judge whether the user's requested work has been completed, based "
    "on the action record. Do not perform any work yourself."
)

IMPERATIVE = " Produce that next."
NEUTRAL = (
    "Remember: delegate generation to a capability ensemble rather than "
    "writing inline yourself."
)

B2_TASK = (
    "Write a python module string_utils.py with a function that reverses the "
    "word order of a string, and a test_string_utils.py with unit tests for it."
)
# B3 = theta E4' task text verbatim (P3-A)
B3_TASK = (
    "Write a python module string_utils.py with a function that reverses the "
    "word order of a string, a number_utils.py with a function that formats "
    "integers with thousands separators, and a test_string_utils.py with unit "
    "tests for the string module."
)

# Action records the framework would hold (the production digest join input).
RECORDS = {
    "B2": [("write", "string_utils.py", "Wrote file successfully.")],
    "B3": [
        ("write", "string_utils.py", "Wrote file successfully."),
        ("write", "number_utils.py", "Wrote file successfully."),
    ],
}
TASK = {"B2": B2_TASK, "B3": B3_TASK}
# the unproduced deliverable per base (adjudication ground truth)
UNPRODUCED = {"B2": "test_string_utils.py", "B3": "test_string_utils.py"}
PRODUCED = {
    "B2": ["string_utils.py"],
    "B3": ["string_utils.py", "number_utils.py"],
}


class _Null:
    async def generate_with_tools(self, *, messages, tools):  # pragma: no cover
        raise NotImplementedError

    async def generate_response(self, message, role_prompt):  # pragma: no cover
        raise NotImplementedError


def _driver() -> LoopDriver:
    return LoopDriver(
        seat_filler=_Null(),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_Null(),
        action_record=SessionActionRecord(),
        judgment_seat=_Null(),
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=CAPS,
    )


def _action_record(base: str) -> SessionActionRecord:
    rec = SessionActionRecord()
    for kind, path, result in RECORDS[base]:
        rec.record_action("rho", action_kind=kind, target_path=path)
        rec.join_result("rho", result)
    return rec


def _file_done_context(base: str) -> SessionContext:
    client_tools = [
        {"type": "function", "function": {"name": n}}
        for n in ("write", "edit", "read", "bash")
    ]
    msgs = [
        ChatMessage(role="system", content="You are opencode, a coding CLI."),
        ChatMessage(role="user", content=TASK[base]),
    ]
    # one assistant+tool pair per produced file (content="" — Ollama rejects None)
    for _ in RECORDS[base]:
        msgs.append(ChatMessage(role="assistant", content=""))
        msgs.append(ChatMessage(role="tool", content="Wrote file successfully."))
    return SessionContext(
        messages=msgs,
        tools=client_tools,
        state=SessionState(
            identity=SessionIdentity(value="rho", method="user_field")
        ),
    )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _post(messages: list[dict], tools: list[dict] | None) -> dict:
    body = {"model": MODEL, "messages": messages, "stream": False}
    if tools is not None:
        body["tools"] = tools
    r = httpx.post(OLLAMA, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


# ---- rho1: judge remaining-naming ----


def _run_rho1(base: str) -> dict:
    records = _action_record(base).records("rho")
    message = compose_judgment_message(TASK[base], records)
    resp = _post(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": message}],
        tools=None,
    )
    text = resp.get("content") or ""
    verdict = parse_verdict(text)
    statement = strip_verdict(text)
    return {"verdict": verdict, "statement": statement, "raw": text}


# ---- rho2 / rho2_imp / control: anchored call 2 ----


def _call2_messages(base: str, anchor: str) -> list[dict]:
    driver = _driver()
    messages = driver._seat_filler_messages(_file_done_context(base))  # noqa: SLF001
    messages = copy.deepcopy(messages)
    messages[-1]["content"] = messages[-1]["content"] + "\n\n" + anchor
    return messages


def _tools(base: str) -> list[dict]:
    driver = _driver()
    return driver._delegation_tools() + list(_file_done_context(base).tools)  # noqa: SLF001


def _classify_target(path: str, base: str) -> str:
    p = (path or "").lower()
    if UNPRODUCED[base].lower().split(".")[0] in p and "test" in p:
        return "advance"
    if any(prod.lower() in p for prod in PRODUCED[base]):
        return "stuck"
    if p:
        return "other"
    return "none"


def _run_call2(base: str, anchor: str) -> dict:
    resp = _post(_call2_messages(base, anchor), tools=_tools(base))
    calls = resp.get("tool_calls") or []
    if not calls:
        return {"target": "none", "delegated": False, "tool": None}
    fn = calls[0]["function"]
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    return {
        "target": _classify_target(args.get("filePath", ""), base),
        "delegated": fn.get("name") == "invoke_ensemble",
        "tool": fn.get("name"),
    }


def main() -> None:
    arm = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "B2"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    results = []
    if arm == "rho1":
        for i in range(n):
            try:
                r = _run_rho1(base)
            except Exception as e:  # noqa: BLE001
                r = {"verdict": None, "statement": "", "raw": f"ERR:{e}"}
            results.append(r)
            print(f"  rho1 {base} {i + 1}/{n}: verdict={r['verdict']} "
                  f"stmt={r['statement'][:80]!r}")
        (HERE / f"statements_{base}.json").write_text(json.dumps(results, indent=2))
    else:
        # anchored call-2 arms
        if arm == "control":
            anchors = [NEUTRAL] * n
        else:
            stmts = json.loads((HERE / f"statements_{base}.json").read_text())
            anchors = []
            for s in stmts[:n]:
                if s["verdict"] != "REMAINING" or not s["statement"]:
                    anchors.append(None)  # no usable anchor
                elif arm == "rho2_imp":
                    anchors.append(s["statement"] + IMPERATIVE)
                else:
                    anchors.append(s["statement"])
        for i, anchor in enumerate(anchors):
            if anchor is None:
                r = {"target": "none", "delegated": False, "tool": "NO_ANCHOR"}
            else:
                try:
                    r = _run_call2(base, anchor)
                except Exception as e:  # noqa: BLE001
                    r = {"target": "none", "delegated": False, "tool": f"ERR:{e}"}
            results.append(r)
            print(f"  {arm} {base} {i + 1}/{n}: {r}")
        runnable = sum(1 for a in anchors if a is not None)
        adv = sum(1 for r in results if r["target"] == "advance")
        stuck = sum(1 for r in results if r["target"] == "stuck")
        deleg = sum(1 for r in results if r["delegated"])
        none = sum(1 for r in results if r["target"] == "none")
        print(f"\n{arm} {base}: advance={adv}/{n} stuck={stuck}/{n} "
              f"delegated={deleg}/{n} none={none}/{n} runnable={runnable}/{n}")
        (HERE / f"results_{arm}_{base}.json").write_text(
            json.dumps({"arm": arm, "base": base, "n": n, "runnable": runnable,
                        "results": results}, indent=2)
        )


if __name__ == "__main__":
    main()
