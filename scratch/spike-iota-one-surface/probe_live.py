"""Spike ι Arm B — live no-tools graceful-finish confirm (PRIMARY).

A real qwen3:14b seat (Ollama /v1, $0 local — the `agentic-orchestrator-
offline-tools` fully-local swap seat) handed plain and capability-matched
no-tools questions through the LANDED composition (`_seat_filler_messages`,
`_delegation_tools`) — not hand-built messages. Measures, per cell, whether the
seat finishes with text vs delegates.

Cells:
  plain_nocaps — plain question, no capabilities. Expect finish (no tool offered).
  plain_caps   — plain question, capabilities present. Expect finish-with-text;
                 the delegation-carrying guidance must NOT drive over-delegation
                 (H-ι.3).
  match_caps   — capability-matched (coding) question, capabilities present.
                 Expect delegate — confirms no-tools delegation still works.

Usage: uv run python scratch/spike-iota-one-surface/probe_live.py <cell> [n]
"""

from __future__ import annotations

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

HERE = Path(__file__).parent
OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:14b"
CAPS = frozenset({"code-generator"})

# Plain questions — domain-disjoint from a `code-generator` capability, so a
# finish-with-text is the correct outcome (delegation would be over-reach).
PLAIN = [
    "What is the capital of France?",
    "What is 17 times 4?",
    "Who wrote the play Hamlet?",
    "What is the boiling point of water in Celsius at sea level?",
    "Name the largest planet in our solar system.",
    "What year did the first human land on the Moon?",
    "What is the chemical symbol for gold?",
    "How many continents are there on Earth?",
    "What is the square root of 144?",
    "What language is primarily spoken in Brazil?",
]
# Capability-matched questions — a code-generator should be delegated to.
MATCH = [
    "Write a Python function that reverses a string.",
    "Write a Python function that checks whether a number is prime.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function that sorts a list of integers ascending.",
    "Write a Python function that counts vowels in a string.",
    "Write a Python function that flattens a nested list.",
    "Write a Python function that computes the factorial of n.",
    "Write a Python function that removes duplicates from a list.",
    "Write a Python function that converts Celsius to Fahrenheit.",
    "Write a Python function that finds the max of three numbers.",
]


class _Null:
    async def generate_with_tools(self, *, messages, tools):  # pragma: no cover
        raise NotImplementedError

    async def generate_response(self, message, role_prompt):  # pragma: no cover
        raise NotImplementedError


def _driver(caps: frozenset[str]) -> LoopDriver:
    return LoopDriver(
        seat_filler=_Null(),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_Null(),
        action_record=SessionActionRecord(),
        judgment_seat=_Null(),
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=caps,
    )


def _no_tools_context(question: str) -> SessionContext:
    """A plain no-tools request — just the user question, no client tools."""
    return SessionContext(
        messages=[ChatMessage(role="user", content=question)],
        tools=[],
        state=SessionState(
            identity=SessionIdentity(value="iota", method="user_field")
        ),
    )


def _composed(question: str, caps: frozenset[str]) -> tuple[list[dict], list[dict] | None]:
    """Messages + tools as the real loop would compose them for the seat."""
    driver = _driver(caps)
    context = _no_tools_context(question)
    messages = driver._seat_filler_messages(context)  # noqa: SLF001
    seat_tools = driver._delegation_tools() + list(context.tools)  # noqa: SLF001
    return messages, (seat_tools or None)


def _post(messages: list[dict], tools: list[dict] | None) -> dict:
    body: dict = {"model": MODEL, "messages": messages, "stream": False}
    if tools is not None:
        body["tools"] = tools
    r = httpx.post(OLLAMA, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _classify(resp: dict) -> str:
    calls = resp.get("tool_calls") or []
    if calls and (calls[0].get("function") or {}).get("name") == "invoke_ensemble":
        return "delegate"
    if calls:
        return "other_tool"
    return "finish"


def main() -> None:
    cell = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    caps = frozenset() if cell == "plain_nocaps" else CAPS
    questions = MATCH if cell == "match_caps" else PLAIN
    results = []
    for i in range(n):
        q = questions[i % len(questions)]
        messages, tools = _composed(q, caps)
        try:
            resp = _post(messages, tools)
            outcome = _classify(resp)
            content = (resp.get("content") or "")
        except Exception as e:  # noqa: BLE001
            outcome = f"ERR:{e}"
            content = ""
        results.append({"q": q, "outcome": outcome, "content": content[:160]})
        print(f"  {cell} {i + 1}/{n}: {outcome:10s} | {q[:48]}")
    finish = sum(1 for r in results if r["outcome"] == "finish")
    deleg = sum(1 for r in results if r["outcome"] == "delegate")
    other = sum(1 for r in results if r["outcome"] not in ("finish", "delegate"))
    print(f"\n{cell}: finish={finish}/{n} delegate={deleg}/{n} other={other}/{n}")
    (HERE / f"results_{cell}.json").write_text(
        json.dumps(
            {"cell": cell, "n": n, "finish": finish, "delegate": deleg,
             "other": other, "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
