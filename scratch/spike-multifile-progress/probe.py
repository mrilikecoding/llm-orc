"""Rung-1 probe — multi-file progress on the REMAINING branch (Finding G).

PRE-REGISTRATION (2026-06-07, BUILD progressive-ladder rung 1; informal,
crawl-before-walk — if this confirms the fix it grounds a DECIDE loop-back #6
where the formal methods review fires).

Failure observed in the WP-LB-K acceptance Run 1: a 2-file task
("string_utils.py AND test_string_utils.py") never converged — qwen3:14b
re-wrote string_utils.py on every REMAINING trailing turn and never advanced
to the test file. Diagnosis: on REMAINING, call 2 is the ADR-036 C3 composition
(conversation + generic delegation guidance) with the judgment exchange
DISCARDED (FC-66). The judge computes "what remains" and it is thrown away, so
the seat-filler's next-action selection has no remaining-deliverable anchor.

HYPOTHESIS: routing the judge's "what remains" forward to anchor call 2 moves
the next-action selection from file 1 (stuck) to the test file (advance),
without collapsing delegation.

ARMS (n=10 each, qwen3:14b via Ollama /v1, $0 local):
  A_current  — call 2 composed exactly as the Loop Driver does today
               (real _seat_filler_messages on the file-1-done context).
  B_anchored — same, plus a remaining-work anchor appended to the trailing
               guidance ("The remaining deliverable not yet produced is
               test_string_utils.py. Produce it next.").

MEASURE (per call, from the model's chosen tool call):
  - target: the filePath the model names (invoke_ensemble.filePath or a direct
    write filePath). Classify advance (test file) vs stuck (string_utils.py)
    vs other.
  - delegated: did it call invoke_ensemble (vs an inline write / other)?
Denominator is always n (a no-tool-call or unparseable response counts as
neither advance nor delegated — recorded as 'none').

PASS (informal, for grounding a loop-back): B_anchored advances markedly more
than A_current AND B_anchored keeps delegating (does not collapse to inline
writes). Numbers reported, not a hard threshold — this is characterization.

FIDELITY: call 2 is composed via the real LoopDriver._seat_filler_messages +
_delegation_tools (not hand-built), so the bytes match production. The anchor
in arm B is a probe-only string; the production form would be the judge's
actual one-sentence "what remains" output.
"""

import copy
import json
import sys
from pathlib import Path

import httpx

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.loop_driver import LoopDriver
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:14b"
CAPS = frozenset({"code-generator"})

TWO_FILE_TASK = (
    "Write a python module string_utils.py with a function that reverses the "
    "word order of a string, and a test_string_utils.py with unit tests for it."
)

ANCHOR = (
    "\n\nThe remaining deliverable not yet produced is test_string_utils.py "
    "(string_utils.py has already been written successfully). Produce "
    "test_string_utils.py next."
)


class _NullSeat:
    async def generate_with_tools(self, *, messages, tools):  # pragma: no cover
        raise NotImplementedError

    async def generate_response(self, message, role_prompt):  # pragma: no cover
        raise NotImplementedError


class _NullDispatch:
    async def dispatch(self, call, *, session_id=""):  # pragma: no cover
        raise NotImplementedError


def _driver() -> LoopDriver:
    """A driver used only for its real composition helpers."""
    return LoopDriver(
        seat_filler=_NullSeat(),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_NullDispatch(),
        action_record=SessionActionRecord(),
        judgment_seat=_NullSeat(),
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=CAPS,
    )


def _file1_done_context() -> SessionContext:
    """The 2-file task with string_utils.py written (a trailing tool tail)."""
    client_tools = [
        {"type": "function", "function": {"name": n}}
        for n in ("write", "edit", "read", "bash")
    ]
    return SessionContext(
        messages=[
            ChatMessage(role="system", content="You are opencode, a coding CLI."),
            ChatMessage(role="user", content=TWO_FILE_TASK),
            # Empty string, not None: matches the OpenCode capture and Ollama
            # rejects a null content type (a tool-call assistant turn).
            ChatMessage(role="assistant", content=""),
            ChatMessage(role="tool", content="Wrote file successfully."),
        ],
        tools=client_tools,
        state=SessionState(
            identity=SessionIdentity(value="probe", method="user_field")
        ),
    )


def _call2_messages(anchored: bool) -> list[dict]:
    driver = _driver()
    messages = driver._seat_filler_messages(_file1_done_context())  # noqa: SLF001
    if anchored:
        messages = copy.deepcopy(messages)
        messages[-1]["content"] = messages[-1]["content"] + ANCHOR
    return messages


def _tools() -> list[dict]:
    driver = _driver()
    ctx = _file1_done_context()
    return driver._delegation_tools() + list(ctx.tools)  # noqa: SLF001


def _classify_target(path: str) -> str:
    p = (path or "").lower()
    if "test_string_utils" in p or ("test" in p and "string" in p):
        return "advance"
    if "string_utils" in p:
        return "stuck"
    if p:
        return "other"
    return "none"


def _run_one(anchored: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": _call2_messages(anchored),
        "tools": _tools(),
        "stream": False,
    }
    r = httpx.post(OLLAMA, json=body, timeout=300)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {"delegated": False, "target": "none", "tool": None}
    call = calls[0]["function"]
    name = call.get("name")
    try:
        args = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    path = args.get("filePath", "")
    return {
        "delegated": name == "invoke_ensemble",
        "target": _classify_target(path),
        "tool": name,
    }


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else "A_current"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    anchored = arm == "B_anchored"
    results = []
    for i in range(n):
        try:
            res = _run_one(anchored)
        except Exception as e:  # noqa: BLE001 — record and continue
            res = {"delegated": False, "target": "none", "tool": f"ERR:{e}"}
        results.append(res)
        print(f"  {arm} {i + 1}/{n}: {res}")
    advance = sum(1 for r in results if r["target"] == "advance")
    stuck = sum(1 for r in results if r["target"] == "stuck")
    delegated = sum(1 for r in results if r["delegated"])
    out = Path(__file__).parent / f"results_{arm}.json"
    out.write_text(json.dumps({"arm": arm, "n": n, "results": results}, indent=2))
    print(
        f"\n{arm}: advance={advance}/{n} stuck={stuck}/{n} "
        f"delegated={delegated}/{n}  -> {out.name}"
    )


if __name__ == "__main__":
    main()
