"""Compound ladder rung A×C — multi-file flow with a repair inside it.

PRE-REGISTRATION (predictions recorded before running; see also
docs/agentic-serving/essays/research-logs/cycle-7-progressive-ladder.md §"Compound
rungs"). Methods review open question #1: do failures only surface when axes
compound? This rung compounds axis A (deliverable count) with axis C (repair).

It is axis-C's task (fix string_utils.py + write test_string_utils.py) PLUS one
README deliverable. The ONLY difference from the axis-C single rung is the added
README, so the comparison isolates the interaction effect:

  axis-C RC1 (module fixed, 1 remaining: test)     -> churn 2/10 (re-target the
                                                       already-fixed module)
  A×C   AC1 (module fixed, 2 remaining: test+readme) -> churn = ?

HYPOTHESES at AC1 (the repaired module stays salient via the "fix" framing while
MORE deliverables remain):
  H1 (amplification — a limit): churn > 2/10. More remaining work gives more
     chances to re-pull next-action selection back to the salient fixed module.
  H0 (no interaction): churn ≈ 2/10. Repair salience is independent of the
     remaining-set size.
  H-dilute: churn < 2/10. A larger unproduced set dilutes the fixed module's pull.

Task (3 deliverables; one is a repair-write):
  string_utils.py (repair)  +  test_string_utils.py  +  README.md

States (n=10/state; real production composition judge -> strip_verdict ->
_seat_filler_messages(remaining_anchor) -> seat, through the real decide()):
  AC0 (nothing)              -> expect repair-write the module (the axis-C RC0
                                shape: boundary_excluded first turn, delegated)
  AC1 (module fixed)         -> 2 remain; advance to test|readme. THE KEY STATE —
                                churn (re-target module|test) is the interaction
                                measurement vs RC1's 2/10.
  AC2 (module fixed + test)  -> 1 remains; advance to readme. Second churn slot.
  ACc (all 3 done)           -> expect COMPLETE -> finish (convergence).

MEASURE per state: advance (an unproduced deliverable), churn (a produced one)
+ churn_target (module|test|readme — which produced file got re-pulled),
delegated, judge verdict, turn_shape.

PASS (characterization, not a hard threshold): AC0 engages the repair-write;
AC1/AC2 advance markedly (>= axis-C's 8/10 RC1 reference) with delegation
preserved; ACc converges (COMPLETE). The interaction finding is the AC1 churn
vs RC1 2/10 comparison.

Cloud-contrast trigger (the established ladder rule): any REMAINING-state advance
<= 7/10 -> escalate one cloud rung to separate seat ceiling from mechanism ceiling.

WP-LB-M acceptance (dogfooded — the harness runs the just-committed outcome-derived
stamping): write-advance turns should stamp turn_shape=generation from the action,
NOT read off the judge's "fix" phrasing (the axis-C RC1 carry<->boundary_excluded
fragility). A repair first turn (AC0) stays boundary_excluded from the instruction;
a REMAINING write whose anchor happens to carry a repair verb is the accepted
instruction-side boundary_excluded residual.
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

# Buggy module inlined — identical to the axis-C single rung, so AC vs RC differs
# only by the added README deliverable (clean interaction-effect isolation).
BUGGY = (
    "def count_vowels(s):\n"
    "    return sum(1 for c in s if c in 'aeiou')\n"
)
TASK = (
    "The file string_utils.py contains a bug:\n\n```python\n" + BUGGY + "```\n\n"
    "Fix the bug in string_utils.py where count_vowels misses uppercase vowels, "
    "then write a test_string_utils.py with unit tests that cover uppercase "
    "vowels, and a README.md describing the module."
)

DELIVERABLES = {
    "module": "string_utils.py",
    "test": "test_string_utils.py",
    "readme": "README.md",
}
STATES = {
    "AC0": [],
    "AC1": ["module"],
    "AC2": ["module", "test"],
    "ACc": ["module", "test", "readme"],
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
                json={"model": MODEL, "messages": messages, "tools": tools,
                      "stream": False},
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
    """Real qwen3:14b judgment seat — bare-form, no tools (call 1)."""

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
    substrate.register_sink(
        type("S", (), {"consume": lambda self, e: sink_events.append(e)})()
    )
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
        churn = target in produced_kinds
        result = {
            "outcome": "apply_work",
            "target": target,
            "advanced": target in DELIVERABLES and target not in produced_kinds,
            "churn": churn,
            "churn_target": target if churn else None,
            "delegated": outcome.delegated_ensemble is not None,
        }
    elif isinstance(outcome, FinishTurn):
        result = {"outcome": "finish", "target": "none", "advanced": False,
                  "churn": False, "churn_target": None, "delegated": False}
    else:
        target = _kind(getattr(outcome.invocation, "arguments", "") or "")
        churn = target in produced_kinds
        result = {"outcome": "carry", "target": target, "advanced": False,
                  "churn": churn, "churn_target": target if churn else None,
                  "delegated": False}

    result["verdict"] = td.judgment_verdict if td else None
    result["turn_shape"] = td.turn_shape if td else None
    return result


async def _run(state: str, n: int) -> None:
    produced = STATES[state]
    rows = []
    for i in range(n):
        try:
            row = await _run_state(produced)
        except Exception as e:  # noqa: BLE001
            row = {"outcome": f"ERR:{e}", "target": "none", "advanced": False,
                   "churn": False, "churn_target": None, "delegated": False,
                   "verdict": None, "turn_shape": None}
        rows.append(row)
        print(f"  {state} {i + 1}/{n}: {row}", flush=True)

    adv = sum(1 for r in rows if r["advanced"])
    churn = sum(1 for r in rows if r["churn"])
    churn_mod = sum(1 for r in rows if r.get("churn_target") == "module")
    deleg = sum(1 for r in rows if r["delegated"])
    complete = sum(1 for r in rows if r["verdict"] == "COMPLETE")
    remaining = sum(1 for r in rows if r["verdict"] == "REMAINING")
    shapes = {}
    for r in rows:
        shapes[r["turn_shape"]] = shapes.get(r["turn_shape"], 0) + 1
    out = Path(__file__).parent / f"results_state_{state}.json"
    out.write_text(json.dumps({"state": state, "produced": produced, "n": n,
                               "rows": rows}, indent=2))
    print(
        f"\nstate {state} ({'+'.join(produced) or 'nothing'} done): advance={adv}/{n} "
        f"churn={churn}/{n} (module={churn_mod}) delegated={deleg}/{n} "
        f"verdict[COMPLETE={complete} REMAINING={remaining}] shapes={shapes} -> {out.name}\n",
        flush=True,
    )


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    order = sys.argv[2].split(",") if len(sys.argv) > 2 else list(STATES)
    for state in order:
        await _run(state, n)


if __name__ == "__main__":
    asyncio.run(main())
