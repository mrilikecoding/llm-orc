"""Ladder axis B — mixed read-then-write (production composition, depth-faithful).

PRE-REGISTRATION: docs/agentic-serving/essays/research-logs/
cycle-7-progressive-ladder.md §4 axis B (methods-reviewed). Rung-2 (axis A)
passed: the anchor scales with deliverable count. Axis B adds a leading READ —
the `carry` turn shape — and tests three things rung 2 could not:

  1. Does the seat-filler READ FIRST when told to (mixed flow works)?
  2. Does the judge correctly NOT count the read as a deliverable (a read is
     context, not a produced file — the FC-61 carry-side assertion)?
  3. Can the meter distinguish a read turn from a write turn on a REMAINING
     tail (the rung-2 meter gap: both read the descriptive anchor -> carry)?

Task (mixed): read config.py, then write settings_loader.py + test_settings_loader.py.
Two WRITE deliverables; the read is setup, not a deliverable.

States (real production composition: judge call 1 -> anchor -> seat call 2):
  R0  (nothing done; first turn)        -> expect the first action to be READ
  R1  (read done, 0 writes)             -> judge REMAINING (read != deliverable);
                                           seat advances to write the module
  R2  (read + module done)              -> advance to the test
  RC  (read + both writes done)         -> judge COMPLETE; finish (convergence
                                           with a read in the record)

MEASURE (n=10/state): the seat-filler's chosen action (read / write-module /
write-test / finish / none), advance vs churn (re-doing a produced WRITE), the
judge verdict, delegated, and turn_shape (meter dogfood — the gap diagnosis).
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.loop_driver import (
    ApplyWork,
    CarryClientTool,
    FinishTurn,
    LoopDriver,
    TurnDecision,
)
from llm_orc.agentic.orchestrator_tool_dispatch import ToolCallSuccess
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall, ToolCallingResponse

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:14b"
CAPS = frozenset({"code-generator"})

TASK = (
    "Read config.py to see the settings, then write a python module "
    "settings_loader.py with a function that loads those settings, and a "
    "test_settings_loader.py with unit tests for it."
)
CONFIG_CONTENTS = 'timeout = 30\nretries = 3\nbase_url = "https://api.example.com"\n'

# Prior-action specs: (label, action_kind, path, result-content)
READ = ("read", "read", "config.py", CONFIG_CONTENTS)
WRITE_MODULE = ("module", "write", "settings_loader.py", "Wrote settings_loader.py successfully.")
WRITE_TEST = ("test", "write", "test_settings_loader.py", "Wrote test_settings_loader.py successfully.")

STATES = {
    "R0": [],
    "R1": [READ],
    "R2": [READ, WRITE_MODULE],
    "RC": [READ, WRITE_MODULE, WRITE_TEST],
}
WRITE_DELIVERABLES = {"module", "test"}


def _target(path: str) -> str:
    p = (path or "").lower()
    if "test_settings" in p or ("test" in p and "settings" in p):
        return "test"
    if "settings_loader" in p:
        return "module"
    if "config" in p:
        return "read"
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
    produced_writes = {label for (label, ak, *_ ) in prior if ak == "write"}
    ctx, record = _state(prior)
    events: list = []
    outcome = await _driver(record, events).decide(ctx)
    td = next((e for e in events if isinstance(e, TurnDecision)), None)

    if isinstance(outcome, ApplyWork):
        tgt = _target(outcome.file_path)
        row = {"action": f"write:{tgt}", "target": tgt,
               "advanced": tgt in WRITE_DELIVERABLES and tgt not in produced_writes,
               "churn": tgt in produced_writes, "delegated": outcome.delegated_ensemble is not None}
    elif isinstance(outcome, CarryClientTool):
        args = {}
        try:
            args = json.loads(outcome.invocation.arguments or "{}")
        except Exception:  # noqa: BLE001
            pass
        tgt = _target(args.get("filePath", ""))
        row = {"action": f"{outcome.invocation.name}:{tgt}", "target": tgt,
               "advanced": False, "churn": False, "delegated": False}
    elif isinstance(outcome, FinishTurn):
        row = {"action": "finish", "target": "none", "advanced": False, "churn": False, "delegated": False}
    else:
        row = {"action": "?", "target": "none", "advanced": False, "churn": False, "delegated": False}

    row["verdict"] = td.judgment_verdict if td else None
    row["turn_shape"] = td.turn_shape if td else None
    return row


async def main() -> None:
    state = sys.argv[1] if len(sys.argv) > 1 else "R1"
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
    from collections import Counter
    acts = Counter(r["action"].split(":")[0] for r in rows)
    print(f"\nstate {state}: actions={dict(acts)} advance={sum(r['advanced'] for r in rows)}/{n} "
          f"churn={sum(r['churn'] for r in rows)} delegated={sum(r['delegated'] for r in rows)} "
          f"verdict={dict(Counter(r['verdict'] for r in rows))} "
          f"turn_shape={dict(Counter(r['turn_shape'] for r in rows))} -> {out.name}")


if __name__ == "__main__":
    asyncio.run(main())
