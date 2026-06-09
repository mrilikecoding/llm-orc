"""Ladder axis C — repair-shaped (read-modify-write), production composition.

PRE-REGISTRATION: docs/agentic-serving/essays/research-logs/
cycle-7-progressive-ladder.md §4 axis C + the §5 P1 resolution (repair-turn
scoring). Rungs 2 (axis A) and axis B both passed. Axis C adds the
`boundary_excluded` turn shape — the repair instruction ("fix the bug …")
triggers it — and tests two things the prior rungs could not:

  1. Does the repair flow engage + advance + converge (the mechanism on a
     repair-then-write task)?
  2. Does the judge count a REPAIR delivered as a write of the target file as
     that deliverable produced (the P1 accounting resolution — under the current
     driver, generation maps to `write`, so a fix is a rewrite the existing
     accounting standard already recognizes)?
  3. Meter: the first turn is `boundary_excluded` (the third turn shape,
     completing the A/B/C meter evidence).

Task (repair-then-write; buggy module inlined so the repair is a single write):
  fix string_utils.py (count_vowels misses uppercase) + write test_string_utils.py.
Two deliverables: the repaired module (delivered as a write) + the test.

States (real production composition judge -> anchor -> seat):
  RC0  (nothing done)              -> expect the repair (write string_utils.py);
                                      turn_shape boundary_excluded
  RC1  (module repaired)           -> judge REMAINING (test remains; the repair
                                      counts as produced); advance to the test
  RCc  (module repaired + test)    -> judge COMPLETE; finish (convergence)

MEASURE (n=10/state): chosen action target (module/test/other/none), advance vs
churn, judge verdict, delegated, turn_shape (meter dogfood — the third shape).
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import httpx

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.loop_driver import ApplyWork, FinishTurn, LoopDriver, TurnDecision
from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallSuccess
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall, ToolCallingResponse

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:14b"
CAPS = frozenset({"code-generator"})

# Buggy module inlined (the repair becomes a single rewrite, isolating the
# repair-classification + judge-accounting questions from the read flow that
# axis B already covered).
BUGGY = (
    "def count_vowels(s):\n"
    "    return sum(1 for c in s if c in 'aeiou')\n"
)
TASK = (
    "The file string_utils.py contains a bug:\n\n```python\n" + BUGGY + "```\n\n"
    "Fix the bug in string_utils.py where count_vowels misses uppercase vowels, "
    "then write a test_string_utils.py with unit tests that cover uppercase vowels."
)

WRITE_MODULE = ("module", "write", "string_utils.py", "Wrote string_utils.py successfully.")
WRITE_TEST = ("test", "write", "test_string_utils.py", "Wrote test_string_utils.py successfully.")
STATES = {"RC0": [], "RC1": [WRITE_MODULE], "RCc": [WRITE_MODULE, WRITE_TEST]}
DELIVERABLES = {"module", "test"}


def _target(path: str) -> str:
    p = (path or "").lower()
    if "test_string" in p or ("test" in p and "string" in p):
        return "test"
    if "string_utils" in p:
        return "module"
    return "other" if p else "none"


class _OllamaSeat:
    async def generate_with_tools(self, *, messages, tools) -> ToolCallingResponse:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                OLLAMA,
                json={"model": MODEL, "messages": messages, "tools": tools, "stream": False},
            )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            calls.append(
                ToolCall(id=c.get("id") or f"c{i}", name=fn.get("name", ""),
                         arguments_json=fn.get("arguments") or "{}")
            )
        return ToolCallingResponse(
            content=msg.get("content") or "",
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
        )


class _OllamaJudge:
    async def generate_response(self, message: str, role_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                OLLAMA,
                json={"model": MODEL, "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": message},
                ], "stream": False},
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""


class _FakeDispatch:
    async def dispatch(self, call, *, session_id=""):
        return ToolCallSuccess(
            id=call.id, name=call.name, content="generated",
            envelope=DispatchEnvelope(status="success", primary="generated"),
        )


def _client_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": n}} for n in ("write", "edit", "read", "bash")]


def _state(prior: list) -> tuple[SessionContext, SessionActionRecord]:
    record = SessionActionRecord()
    messages = [
        ChatMessage(role="system", content="You are opencode, a coding CLI."),
        ChatMessage(role="user", content=TASK),
    ]
    for _label, action_kind, path, result in prior:
        record.record_action("probe", action_kind=action_kind, target_path=path)
        record.join_result("probe", result)
        messages.append(ChatMessage(role="assistant", content=""))
        messages.append(ChatMessage(role="tool", content=result))
    ctx = SessionContext(
        messages=messages, tools=_client_tools(),
        state=SessionState(identity=SessionIdentity(value="probe", method="user_field")),
    )
    return ctx, record


def _driver(record, events) -> LoopDriver:
    substrate = DispatchEventSubstrate()
    substrate.register_sink(type("S", (), {"consume": lambda self, e: events.append(e)})())
    return LoopDriver(
        seat_filler=_OllamaSeat(), enforcer=SingleStepEnforcer(), tool_dispatch=_FakeDispatch(),
        action_record=record, judgment_seat=_OllamaJudge(),
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=CAPS, event_substrate=substrate,
    )


async def _run_state(state: str) -> dict:
    prior = STATES[state]
    produced = {label for (label, *_ ) in prior}
    ctx, record = _state(prior)
    events: list = []
    outcome = await _driver(record, events).decide(ctx)
    td = next((e for e in events if isinstance(e, TurnDecision)), None)

    if isinstance(outcome, ApplyWork):
        tgt = _target(outcome.file_path)
        row = {"action": f"write:{tgt}", "target": tgt,
               "advanced": tgt in DELIVERABLES and tgt not in produced,
               "churn": tgt in produced, "delegated": outcome.delegated_ensemble is not None}
    elif isinstance(outcome, FinishTurn):
        row = {"action": "finish", "target": "none", "advanced": False, "churn": False, "delegated": False}
    else:
        inv = getattr(outcome, "invocation", None)
        name = getattr(inv, "name", "carry")
        row = {"action": f"{name}", "target": "none", "advanced": False, "churn": False, "delegated": False}

    row["verdict"] = td.judgment_verdict if td else None
    row["turn_shape"] = td.turn_shape if td else None
    return row


async def main() -> None:
    state = sys.argv[1] if len(sys.argv) > 1 else "RC1"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rows = []
    for i in range(n):
        try:
            row = await _run_state(state)
        except Exception as e:  # noqa: BLE001
            row = {"action": f"ERR:{e}", "target": "none", "advanced": False, "churn": False,
                   "delegated": False, "verdict": None, "turn_shape": None}
        rows.append(row)
        print(f"  {state} {i + 1}/{n}: {row}")
    out = Path(__file__).parent / f"results_{state}.json"
    out.write_text(json.dumps({"state": state, "n": n, "rows": rows}, indent=2))
    print(f"\nstate {state}: actions={dict(Counter(r['action'].split(':')[0] for r in rows))} "
          f"advance={sum(r['advanced'] for r in rows)}/{n} churn={sum(r['churn'] for r in rows)} "
          f"delegated={sum(r['delegated'] for r in rows)} "
          f"verdict={dict(Counter(r['verdict'] for r in rows))} "
          f"turn_shape={dict(Counter(r['turn_shape'] for r in rows))} -> {out.name}")


if __name__ == "__main__":
    asyncio.run(main())
