"""Ladder rung 2 — axis A, depth-3 write-only progress (production composition).

PRE-REGISTRATION: see docs/agentic-serving/essays/research-logs/
cycle-7-progressive-ladder.md (methods-reviewed; rung 2 = axis A depth 3).

Unlike the rung-1 probe (which isolated one depth-2 decision with a HARDCODED
anchor), this runs the REAL production two-call composition now that WP-LB-L
shipped: real judgment seat (call 1) -> real strip_verdict -> real
_seat_filler_messages(remaining_anchor=...) (call 2). Both composing factors
are exercised at depth 3:  P(judge names the unproduced deliverable correctly)
x P(anchored call 2 advances).

Task (depth 3, write-only, deliverable count legible — the WP-LB-K Run 1 shape):
  string_utils.py  +  test_string_utils.py  +  README.md

States (the two deeper REMAINING decision points + the convergence point),
each with the SessionActionRecord and the conversation pre-populated so the
judge digest and the seat-filler conversation are production-faithful:
  A  (1 of 3 done: module)        -> expect REMAINING; advance to test|readme
  B  (2 of 3 done: module+test)   -> expect REMAINING; advance to readme  [the deep test]
  C  (3 of 3 done)                -> expect COMPLETE; finish (convergence)

MEASURE per state (n=10), from the REAL LoopDriver.decide() outcome + the
emitted TurnDecision (turn_shape stamped by the WP-LB-J meter — dogfooded):
  - judge verdict (COMPLETE/REMAINING/parse-miss) and, on REMAINING, whether
    the judge's statement named an UNPRODUCED deliverable (remaining-naming
    accuracy — rho measured ~1.0 at depth 2; does it hold at depth 3?);
  - target: the filePath the seat-filler chose -> advance (an unproduced
    deliverable) vs churn (a produced one) vs none (no tool call);
  - delegated: invoke_ensemble vs inline-write/none;
  - turn_shape (should be generation for write turns; the meter denominator).

first-churn turn (the SlopCodeBench-style metric): A churn => 2, B churn => 3.
PASS (characterization, not a hard threshold): A and B both advance markedly
(>= the rho depth-2 statement-only 8/10 reference) with delegation preserved,
and C converges (COMPLETE). Cloud-contrast trigger: rung-2 advance <= 7/10.
"""

import asyncio
import json
import sys
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

TASK = (
    "Write a python module string_utils.py with a function that reverses the "
    "word order of a string, a test_string_utils.py with unit tests for it, "
    "and a README.md describing the module."
)

DELIVERABLES = {
    "module": "string_utils.py",
    "test": "test_string_utils.py",
    "readme": "README.md",
}


def _kind(path: str) -> str:
    p = (path or "").lower()
    if "readme" in p:
        return "readme"
    if "test" in p and "string" in p:
        return "test"
    if "string_utils" in p:
        return "module"
    return "other" if p else "none"


class _OllamaSeat:
    """Real qwen3:14b seat-filler — the production SeatFiller port over Ollama."""

    async def generate_with_tools(self, *, messages, tools) -> ToolCallingResponse:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                OLLAMA,
                json={"model": MODEL, "messages": messages, "tools": tools, "stream": False},
            )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        raw = msg.get("tool_calls") or []
        calls = []
        for i, c in enumerate(raw):
            fn = c.get("function", {})
            calls.append(
                ToolCall(
                    id=c.get("id") or f"c{i}",
                    name=fn.get("name", ""),
                    arguments_json=fn.get("arguments") or "{}",
                )
            )
        return ToolCallingResponse(
            content=msg.get("content") or "",
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
        )


class _OllamaJudge:
    """Real qwen3:14b judgment seat — bare-form, no tools (call 1)."""

    async def generate_response(self, message: str, role_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                OLLAMA,
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": message},
                    ],
                    "stream": False,
                },
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""


class _FakeDispatch:
    """The callee content is not under test — selection of the next file is."""

    async def dispatch(self, call, *, session_id=""):
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content="generated",
            envelope=DispatchEnvelope(status="success", primary="generated"),
        )


def _client_tools() -> list[dict]:
    return [
        {"type": "function", "function": {"name": n}}
        for n in ("write", "edit", "read", "bash")
    ]


def _state(produced: list[str]) -> tuple[SessionContext, SessionActionRecord]:
    """Build the context + pre-joined action record for `produced` deliverables."""
    record = SessionActionRecord()
    messages = [
        ChatMessage(role="system", content="You are opencode, a coding CLI."),
        ChatMessage(role="user", content=TASK),
    ]
    for kind in produced:
        path = DELIVERABLES[kind]
        record.record_action("probe", action_kind="write", target_path=path)
        record.join_result("probe", f"Wrote {path} successfully.")
        messages.append(ChatMessage(role="assistant", content=""))
        messages.append(ChatMessage(role="tool", content=f"Wrote {path} successfully."))
    ctx = SessionContext(
        messages=messages,
        tools=_client_tools(),
        state=SessionState(identity=SessionIdentity(value="probe", method="user_field")),
    )
    return ctx, record


def _driver(record: SessionActionRecord, sink_events: list) -> LoopDriver:
    substrate = DispatchEventSubstrate()
    substrate.register_sink(type("S", (), {"consume": lambda self, e: sink_events.append(e)})())
    return LoopDriver(
        seat_filler=_OllamaSeat(),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_FakeDispatch(),
        action_record=record,
        judgment_seat=_OllamaJudge(),
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=CAPS,
        event_substrate=substrate,
    )


async def _run_state(produced: list[str]) -> dict:
    ctx, record = _state(produced)
    events: list = []
    driver = _driver(record, events)
    outcome = await driver.decide(ctx)
    td = next((e for e in events if isinstance(e, TurnDecision)), None)

    produced_kinds = set(produced)
    if isinstance(outcome, ApplyWork):
        target = _kind(outcome.file_path)
        advanced = target in DELIVERABLES and target not in produced_kinds
        result = {
            "outcome": "apply_work",
            "target": target,
            "advanced": advanced,
            "churn": target in produced_kinds,
            "delegated": outcome.delegated_ensemble is not None,
        }
    elif isinstance(outcome, FinishTurn):
        result = {"outcome": "finish", "target": "none", "advanced": False, "churn": False, "delegated": False}
    else:
        result = {"outcome": "carry", "target": "none", "advanced": False, "churn": False, "delegated": False}

    result["verdict"] = td.judgment_verdict if td else None
    result["turn_shape"] = td.turn_shape if td else None
    return result


async def main() -> None:
    state = sys.argv[1] if len(sys.argv) > 1 else "B"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    produced = {"A": ["module"], "B": ["module", "test"], "C": ["module", "test", "readme"]}[state]

    rows = []
    for i in range(n):
        try:
            row = await _run_state(produced)
        except Exception as e:  # noqa: BLE001
            row = {"outcome": f"ERR:{e}", "target": "none", "advanced": False, "churn": False,
                   "delegated": False, "verdict": None, "turn_shape": None}
        rows.append(row)
        print(f"  {state} {i + 1}/{n}: {row}")

    adv = sum(1 for r in rows if r["advanced"])
    churn = sum(1 for r in rows if r["churn"])
    deleg = sum(1 for r in rows if r["delegated"])
    complete = sum(1 for r in rows if r["verdict"] == "COMPLETE")
    remaining = sum(1 for r in rows if r["verdict"] == "REMAINING")
    out = Path(__file__).parent / f"results_state_{state}.json"
    out.write_text(json.dumps({"state": state, "produced": produced, "n": n, "rows": rows}, indent=2))
    print(
        f"\nstate {state} ({'+'.join(produced)} done): advance={adv}/{n} churn={churn}/{n} "
        f"delegated={deleg}/{n} verdict[COMPLETE={complete} REMAINING={remaining}] -> {out.name}"
    )


if __name__ == "__main__":
    asyncio.run(main())
