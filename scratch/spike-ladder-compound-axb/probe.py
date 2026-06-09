"""Compound ladder rung A×B — depth × carry-side (read-then-write under depth).

PRE-REGISTRATION (predictions + decision boundaries recorded before running; see
docs/agentic-serving/essays/research-logs/cycle-7-progressive-ladder.md §"Compound
rungs"). Methods review (`housekeeping/audits/research-methods-compound-axc.md`,
applied to this A×B redesign) directed: probe the undercharacterized carry-side
under depth, with adequate power and pre-registered boundaries.

This compounds axis A (deliverable count) with axis B (a leading read). Axis B
characterized the carry-side at DEPTH 2 (read + 2 writes): read-first 10/10,
advance 9/10, churn 0, converge 10/10. The uncharacterized question is whether
the carry-side holds as the post-read write sequence DEEPENS to 3:
  - does the seat-filler still READ FIRST?
  - does the leading read stay correctly non-counted (never re-pulled / churned)
    across THREE remaining-work states, not one?
  - does the judge keep not-counting the read as depth grows (FC-61 under depth)?
  - does advance hold at ~axis-B 9/10 with a deeper write sequence after the read?

Task (read + 3 code writes — all code, no prose deliverable, so depth is the
only manipulation vs axis B's depth-2):
  read config.py  ->  write settings_loader.py + test_settings_loader.py
                      + validator.py

States (n per state below; real production composition judge call 1 -> anchor ->
seat call 2 through the real decide()):
  R0  (nothing; first turn)          -> expect READ first                  n=10
  R1  (read done, 0 writes)          -> 3 writes remain; advance to a write n=15
  R2  (read + 1 write)               -> 2 remain; advance                  n=15
  R3  (read + 2 writes)              -> 1 remains; advance (the deep state) n=15
  RC  (read + 3 writes)              -> expect COMPLETE -> finish           n=10

MEASURE per state: advance (an unproduced WRITE); write-churn (re-target a
produced write) + churn_target; read-churn (re-read config when already read —
axis B saw 1/10, tracked separately, not a deliverable churn); no_tool (a finish
at a REMAINING state — premature-finish, a distinct failure mode); delegated;
judge verdict; the judge's remaining statement + which deliverable filenames it
names (remaining-naming accuracy — distinguishes seat churn from judge error);
turn_shape.

PRE-REGISTERED DECISION BOUNDARIES (set before running, to prevent post-hoc
reinterpretation; n=15 on R1/R2/R3 gives ~baseline resolution):
  Read-first (R0):      >=9/10 reads first = holds (axis B 10/10); <8/10 = limit.
  Advance (R1/R2/R3):   >=12/15 (0.8) = carry-side-under-depth holds;
                        8-11/15 (0.53-0.73) = degradation, the compound surfaces
                        what single-axis missed (CLOUD-CONTRAST trigger);
                        <=7/15 = clear depth limit.
  Carry-side integrity: read-churn <=1/15 per state AND judge names only
                        unproduced writes (never the read, never a produced file)
                        >=13/15. Rising read-churn or naming drift with depth = a
                        carry-side-under-depth limit (the headline finding if it
                        appears).
  Convergence (RC):     COMPLETE >=9/10; rising REMAINING-at-RC (false-continue,
                        the read+3-writes record miscounted) = a convergence limit.
  no_tool:              tracked; axis B ~10%. Rising with depth = premature-finish
                        limit.
  first-churn turn:     the SlopCodeBench depth at which churn first appears.

Cloud-contrast trigger (the established ladder rule): any REMAINING-state advance
<=0.73 (<=11/15) -> escalate one cloud rung to separate seat ceiling from
mechanism ceiling.

WP-LB-M acceptance (dogfooded — the harness runs the just-committed outcome-derived
stamping). Axis B documented the turn_shape bug in BOTH directions; this run is its
real-model acceptance:
  R0 read       -> expect turn_shape=carry      (was generation pre-WP-LB-M)
  R1/R2/R3 write-> expect turn_shape=generation (was carry pre-WP-LB-M)
  RC finish     -> carry (was already correct)
A REMAINING write whose anchor happens to carry a repair verb is the accepted
instruction-side boundary_excluded residual (not expected here — no repair).
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
    "settings_loader.py with a function that loads those settings, a "
    "test_settings_loader.py with unit tests for it, and a validator.py "
    "module that validates the loaded settings."
)
CONFIG_CONTENTS = 'timeout = 30\nretries = 3\nbase_url = "https://api.example.com"\n'

# (label, action_kind, path, joined-result-content)
READ = ("read", "read", "config.py", CONFIG_CONTENTS)
WRITE_MODULE = ("module", "write", "settings_loader.py", "Wrote settings_loader.py successfully.")
WRITE_TEST = ("test", "write", "test_settings_loader.py", "Wrote test_settings_loader.py successfully.")
WRITE_VALIDATOR = ("validator", "write", "validator.py", "Wrote validator.py successfully.")

STATES = {
    "R0": [],
    "R1": [READ],
    "R2": [READ, WRITE_MODULE],
    "R3": [READ, WRITE_MODULE, WRITE_TEST],
    "RC": [READ, WRITE_MODULE, WRITE_TEST, WRITE_VALIDATOR],
}
DEFAULT_N = {"R0": 10, "R1": 15, "R2": 15, "R3": 15, "RC": 10}
WRITE_DELIVERABLES = {"module", "test", "validator"}
# Filename fragments for remaining-naming detection in the judge's statement.
NAME_FRAGMENTS = {
    "module": "settings_loader.py",
    "test": "test_settings_loader.py",
    "validator": "validator.py",
    "read": "config.py",
}


def _kind(path: str) -> str:
    p = (path or "").lower()
    if "validator" in p:
        return "validator"
    if "test_settings" in p or ("test" in p and "settings" in p):
        return "test"
    if "settings_loader" in p:
        return "module"
    if "config" in p:
        return "read"
    return "other" if p else "none"


def _names_in(text: str) -> list[str]:
    """Which deliverable filenames the judge's statement names. ``test_settings_
    loader.py`` contains ``settings_loader.py``, so module counts as named only
    when ``settings_loader.py`` appears MORE often than the test fragment."""
    t = (text or "").lower()
    found = []
    if NAME_FRAGMENTS["test"] in t:
        found.append("test")
    if t.count(NAME_FRAGMENTS["module"]) > t.count(NAME_FRAGMENTS["test"]):
        found.append("module")
    if NAME_FRAGMENTS["validator"] in t:
        found.append("validator")
    if NAME_FRAGMENTS["read"] in t:
        found.append("read")
    return found


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
    """Real qwen3:14b judgment seat — bare-form (call 1); records its last text."""

    def __init__(self) -> None:
        self.last: str | None = None

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
        self.last = r.json()["choices"][0]["message"].get("content") or ""
        return self.last


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


def _state(produced: list) -> tuple[SessionContext, SessionActionRecord]:
    record = SessionActionRecord()
    messages = [
        ChatMessage(role="system", content="You are opencode, a coding CLI."),
        ChatMessage(role="user", content=TASK),
    ]
    for _label, kind, path, content in produced:
        record.record_action("probe", action_kind=kind, target_path=path)
        record.join_result("probe", content)
        messages.append(ChatMessage(role="assistant", content=""))
        messages.append(ChatMessage(role="tool", content=content))
    ctx = SessionContext(
        messages=messages,
        tools=_client_tools(),
        state=SessionState(identity=SessionIdentity(value="probe", method="user_field")),
    )
    return ctx, record


def _driver(record, judge, sink_events: list) -> LoopDriver:
    substrate = DispatchEventSubstrate()
    substrate.register_sink(
        type("S", (), {"consume": lambda self, e: sink_events.append(e)})()
    )
    return LoopDriver(
        seat_filler=_OllamaSeat(),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_FakeDispatch(),
        action_record=record,
        judgment_seat=judge,
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=CAPS,
        event_substrate=substrate,
    )


async def _run_state(produced: list) -> dict:
    ctx, record = _state(produced)
    judge = _OllamaJudge()
    events: list = []
    driver = _driver(record, judge, events)
    outcome = await driver.decide(ctx)
    td = next((e for e in events if isinstance(e, TurnDecision)), None)

    produced_kinds = {p[0] for p in produced}  # labels: read/module/test/validator

    if isinstance(outcome, ApplyWork):
        target = _kind(outcome.file_path)
        result = _classify_action(target, produced_kinds, delegated=True)
    elif isinstance(outcome, CarryClientTool):
        # A passthrough client tool (e.g. a read carry). filePath is in args.
        args = getattr(outcome.invocation, "arguments", None)
        path = ""
        if isinstance(args, dict):
            path = args.get("filePath") or args.get("path") or ""
        elif isinstance(args, str):
            path = args
        target = _kind(path)
        result = _classify_action(target, produced_kinds, delegated=False)
    elif isinstance(outcome, FinishTurn):
        result = {"outcome": "finish", "target": "none", "advanced": False,
                  "write_churn": False, "read_churn": False, "churn_target": None,
                  "no_tool": True, "delegated": False}
    else:
        result = {"outcome": str(type(outcome).__name__), "target": "none",
                  "advanced": False, "write_churn": False, "read_churn": False,
                  "churn_target": None, "no_tool": False, "delegated": False}

    result["verdict"] = td.judgment_verdict if td else None
    result["turn_shape"] = td.turn_shape if td else None
    result["judge_names"] = _names_in(judge.last) if judge.last else []
    result["judge_text"] = (judge.last or "")[:240]
    return result


def _classify_action(target: str, produced_kinds: set, *, delegated: bool) -> dict:
    is_write_deliverable = target in WRITE_DELIVERABLES
    advanced = is_write_deliverable and target not in produced_kinds
    write_churn = is_write_deliverable and target in produced_kinds
    read_churn = target == "read" and "read" in produced_kinds
    return {
        "outcome": "apply_work" if delegated else "carry",
        "target": target,
        "advanced": advanced,
        "write_churn": write_churn,
        "read_churn": read_churn,
        "churn_target": target if write_churn else None,
        "no_tool": False,
        "delegated": delegated,
    }


async def _run(state: str, n: int) -> None:
    produced = STATES[state]
    rows = []
    for i in range(n):
        try:
            row = await _run_state(produced)
        except Exception as e:  # noqa: BLE001
            row = {"outcome": f"ERR:{e}", "target": "none", "advanced": False,
                   "write_churn": False, "read_churn": False, "churn_target": None,
                   "no_tool": False, "delegated": False, "verdict": None,
                   "turn_shape": None, "judge_names": [], "judge_text": ""}
        rows.append(row)
        print(f"  {state} {i + 1}/{n}: adv={row['advanced']} target={row['target']} "
              f"wchurn={row['write_churn']} rchurn={row['read_churn']} "
              f"deleg={row['delegated']} verdict={row['verdict']} "
              f"shape={row['turn_shape']} names={row['judge_names']}", flush=True)

    adv = sum(1 for r in rows if r["advanced"])
    wchurn = sum(1 for r in rows if r["write_churn"])
    rchurn = sum(1 for r in rows if r["read_churn"])
    no_tool = sum(1 for r in rows if r["no_tool"])
    deleg = sum(1 for r in rows if r["delegated"])
    complete = sum(1 for r in rows if r["verdict"] == "COMPLETE")
    remaining = sum(1 for r in rows if r["verdict"] == "REMAINING")
    reads = sum(1 for r in rows if r["target"] == "read")
    shapes: dict = {}
    for r in rows:
        shapes[r["turn_shape"]] = shapes.get(r["turn_shape"], 0) + 1
    # naming: did the judge name ONLY unproduced writes?
    produced_kinds = {p[0] for p in produced}
    unproduced = WRITE_DELIVERABLES - produced_kinds
    clean_naming = sum(
        1 for r in rows
        if r["judge_names"] and set(r["judge_names"]) <= unproduced
    )
    named_read = sum(1 for r in rows if "read" in r["judge_names"])
    out = Path(__file__).parent / f"results_state_{state}.json"
    out.write_text(json.dumps({"state": state, "produced": [p[0] for p in produced],
                               "n": n, "rows": rows}, indent=2))
    print(
        f"\nstate {state} ({'+'.join(p[0] for p in produced) or 'nothing'} done): "
        f"advance={adv}/{n} write_churn={wchurn}/{n} read_churn={rchurn}/{n} "
        f"reads={reads}/{n} no_tool={no_tool}/{n} delegated={deleg}/{n} "
        f"verdict[COMPLETE={complete} REMAINING={remaining}] shapes={shapes} "
        f"naming[clean={clean_naming} named_read={named_read}] -> {out.name}\n",
        flush=True,
    )


async def main() -> None:
    order = sys.argv[1].split(",") if len(sys.argv) > 1 else list(STATES)
    override_n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for state in order:
        await _run(state, override_n or DEFAULT_N[state])


if __name__ == "__main__":
    asyncio.run(main())
