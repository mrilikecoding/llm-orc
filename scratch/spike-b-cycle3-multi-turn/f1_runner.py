"""F1 turn-by-turn dispatch runner for Spike B tau-shape frontier arm.

Each scenario × trial has its own context file. The runner maintains state
across multiple invocations:
  - init           : create 6 context files (3 scenarios × 2 trials)
  - prompt <ctx>   : print the dispatch prompt for one context
  - process <ctx> --response-file <path> : parse subagent JSON response,
                     simulate tools, update context. Print TERMINATED/ACTIVE.
  - active         : list active (not yet terminated) context IDs
  - grade          : produce final grading JSON across all contexts

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import library_catalog as lc

# Match the cheap-arm tau-shape harness system prompt verbatim
LIBRARY_AGENT_SYSTEM_PROMPT = """\
You are a library checkout assistant. Patrons message you with check-out \
requests. Your job is to take the right action on their behalf and then \
reply to confirm what you did. Do not ask the patron for permission to take \
routine actions like placing holds — patrons expect you to handle these \
proactively.

Library policy: patrons with outstanding fines may not check out additional \
materials until fines are paid in full. Always check patron status when \
patrons attempt a check-out, and surface any fine balance to the patron in \
your response.

Available actions:
- search_catalog: find books by title or author
- check_patron_status: get patron's holds, overdue books, and fine balance
- check_out: check out an available book to a patron
- place_hold: reserve a checked-out book (proactively, do not ask first)
- pay_fine: apply a fine payment
- send_response: send the final reply to the patron

Action protocol:
- For an available book + clean patron status → check_out, then send_response with due date.
- For a checked-out book → place_hold proactively (no need to ask first), then send_response with hold position and the book's expected return date.
- For a patron with outstanding fines → DO NOT check out. Surface the fine balance in send_response and either invite payment or refuse with explanation.

When you have all the information you need and have taken the appropriate \
action, call send_response with your reply to the patron. You MUST call \
send_response to conclude the conversation — do not stop generating without \
calling it. Do not call other tools after send_response."""

CONTEXTS_DIR = HERE / "trials" / "f1-contexts"


def context_path(scenario_name: str, trial_index: int) -> Path:
    return CONTEXTS_DIR / f"frontier-{scenario_name}-trial{trial_index:02d}.json"


def _normalize_state_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Convert string-int keys back to int after JSON round-trip.

    JSON only supports string keys, so books/patrons dicts that originally
    had int keys (e.g., 7142) come back as strings ('7142'). Tool functions
    look them up with int keys; restore the integer typing here.
    """
    if "books" in state and isinstance(state["books"], dict):
        state["books"] = {int(k) if isinstance(k, str) and k.isdigit() else k: v
                          for k, v in state["books"].items()}
    if "patrons" in state and isinstance(state["patrons"], dict):
        state["patrons"] = {int(k) if isinstance(k, str) and k.isdigit() else k: v
                            for k, v in state["patrons"].items()}
    return state


def load_ctx(ctx_id: str) -> dict[str, Any]:
    path = CONTEXTS_DIR / f"{ctx_id}.json"
    ctx = json.loads(path.read_text())
    if "state" in ctx:
        ctx["state"] = _normalize_state_keys(ctx["state"])
    if "scenario_data" in ctx:
        ctx["scenario_data"] = _normalize_state_keys(ctx["scenario_data"])
    return ctx


def save_ctx(ctx_id: str, ctx: dict[str, Any]) -> None:
    path = CONTEXTS_DIR / f"{ctx_id}.json"
    path.write_text(json.dumps(ctx, indent=2))


def init_contexts() -> list[str]:
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    ctx_ids = []
    for scenario in lc.ALL_SCENARIOS:
        for trial_idx in [1, 2]:
            ctx = {
                "scenario_name": scenario["name"],
                "scenario_data": scenario,
                "trial_index": trial_idx,
                "state": lc.fresh_state(scenario),
                "conversation": [
                    {"role": "system", "content": LIBRARY_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": scenario["request"]},
                ],
                "tool_call_log": [],
                "turns": [],
                "current_turn": 0,
                "terminated": False,
                "terminated_via_send_response": False,
                "max_turns": 8,
            }
            ctx_id = context_path(scenario["name"], trial_idx).stem
            save_ctx(ctx_id, ctx)
            ctx_ids.append(ctx_id)
    return ctx_ids


def render_conversation(convo: list[dict[str, Any]]) -> str:
    """Render conversation history as a readable transcript for the dispatch prompt."""
    parts = []
    for msg in convo:
        role = msg["role"]
        if role == "system":
            continue  # included separately in prompt
        elif role == "user":
            parts.append(f"[USER]\n{msg['content']}\n")
        elif role == "assistant":
            text = msg.get("content", "") or ""
            block = f"[ASSISTANT]\n{text}".rstrip()
            tcs = msg.get("tool_calls") or []
            if tcs:
                tc_summary = []
                for tc in tcs:
                    fn = tc["function"]
                    args = json.loads(fn["arguments"]) if fn["arguments"] else {}
                    tc_summary.append(f"  {fn['name']}({json.dumps(args)})")
                block += "\nTool calls:\n" + "\n".join(tc_summary)
            parts.append(block + "\n")
        elif role == "tool":
            tcid = msg.get("tool_call_id", "?")
            content = msg.get("content", "")
            parts.append(f"[TOOL RESULT for {tcid}]\n{content}\n")
    return "\n".join(parts)


def build_dispatch_prompt(ctx: dict[str, Any]) -> str:
    convo = render_conversation(ctx["conversation"])
    turn = ctx["current_turn"]
    prompt = f"""You are simulating ONE turn of a frontier-tier library-checkout agent for a Spike B research probe (Cycle 3 RQ-3 turn-by-turn multi-turn reliability test). The harness will execute your tool calls deterministically and feed results back as the next dispatch's input. You do NOT need to imagine tool results — only output ONE turn's response.

CRITICAL CONSTRAINTS:
- Do NOT use any tools. The library tools below are simulated by the harness; your role is to plan ONE turn.
- Output ONE assistant turn's response: text content + tool calls.
- Format your response as a JSON code block at the end of your message. The harness extracts the LAST JSON block.

ROLE (verbatim from the harness's system prompt):
{LIBRARY_AGENT_SYSTEM_PROMPT}

Available tools (the harness executes these; do not invoke them yourself):
- search_catalog(query: str)
- check_patron_status(patron_id: int)
- check_out(book_id: int, patron_id: int)
- place_hold(book_id: int, patron_id: int)
- pay_fine(patron_id: int, amount: float)
- send_response(message: str) — terminates the conversation when called successfully

CONVERSATION SO FAR (turn {turn}):

{convo}
[ASSISTANT — your turn now]

Output ONE assistant turn as a JSON code block:

```json
{{
  "content": "your assistant text for this turn (can be empty)",
  "tool_calls": [{{"name": "tool_name", "arguments": {{...}}}}]
}}
```

If you have already called send_response in a prior turn this conversation has terminated and you should not have been dispatched. Otherwise: produce ONE turn. Respond with the JSON code block. Output NO TEXT after the closing ``` of the JSON block."""
    return prompt


def parse_response(response_text: str) -> dict[str, Any]:
    """Extract the LAST JSON code block from the subagent's response."""
    matches = re.findall(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
    if not matches:
        # Fallback: any code block
        matches = re.findall(r"```\s*\n?(.*?)\n```", response_text, re.DOTALL)
    if not matches:
        raise ValueError(
            f"No JSON code block found. First 800 chars of response:\n{response_text[:800]}"
        )
    return json.loads(matches[-1])


def process_response(ctx: dict[str, Any], response_text: str) -> dict[str, Any]:
    try:
        parsed = parse_response(response_text)
    except (ValueError, json.JSONDecodeError) as e:
        ctx["terminated"] = True
        ctx["error"] = f"Parse error: {e}"
        return ctx

    turn = ctx["current_turn"]
    content = parsed.get("content", "") or ""
    tool_calls = parsed.get("tool_calls", []) or []

    # Append assistant message in OpenAI format
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        oai_tool_calls = []
        for i, tc in enumerate(tool_calls):
            oai_tool_calls.append({
                "id": f"call_t{turn}_{i}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            })
        assistant_msg["tool_calls"] = oai_tool_calls
    ctx["conversation"].append(assistant_msg)

    ctx["turns"].append({
        "turn": turn,
        "assistant_content": content,
        "tool_calls": tool_calls,
    })

    if not tool_calls:
        ctx["terminated"] = True
        return ctx

    state = ctx["state"]
    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        args = tc.get("arguments", {}) or {}
        if not isinstance(args, dict):
            args = {}
        result = lc.dispatch_tool(state, name, args)
        ctx["tool_call_log"].append({
            "name": name,
            "arguments": args,
            "result": result,
            "turn": turn,
        })
        ctx["conversation"].append({
            "role": "tool",
            "tool_call_id": f"call_t{turn}_{i}",
            "content": json.dumps(result),
        })
        if name == "send_response" and isinstance(result, dict) and result.get("success"):
            ctx["terminated"] = True
            ctx["terminated_via_send_response"] = True

    ctx["current_turn"] = turn + 1
    if ctx["current_turn"] >= ctx["max_turns"] and not ctx["terminated"]:
        ctx["terminated"] = True
        ctx["error"] = "max_turns_reached"

    return ctx


def list_active() -> list[str]:
    if not CONTEXTS_DIR.exists():
        return []
    active = []
    for path in sorted(CONTEXTS_DIR.glob("frontier-*.json")):
        ctx = json.loads(path.read_text())
        if not ctx["terminated"]:
            active.append(path.stem)
    return active


def grade_all() -> list[dict[str, Any]]:
    grades = []
    for path in sorted(CONTEXTS_DIR.glob("frontier-*.json")):
        ctx = json.loads(path.read_text())
        scenario_name = ctx["scenario_name"]
        tcl = [
            {"name": r["name"], "arguments": r["arguments"], "result": r["result"]}
            for r in ctx["tool_call_log"]
        ]
        grade = lc.grade_scenario(scenario_name, ctx["state"], tcl)
        grades.append({
            "context": path.stem,
            "scenario_name": scenario_name,
            "trial_index": ctx["trial_index"],
            "success": grade.success,
            "failure_modes": grade.failure_modes,
            "notes": grade.notes,
            "tool_call_count": len(ctx["tool_call_log"]),
            "turns": len(ctx["turns"]),
            "terminated_via_send_response": ctx["terminated_via_send_response"],
            "error": ctx.get("error"),
            "responses_sent": ctx["state"].get("_responses_sent", []),
        })
    return grades


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p_prompt = sub.add_parser("prompt")
    p_prompt.add_argument("ctx_id")
    p_proc = sub.add_parser("process")
    p_proc.add_argument("ctx_id")
    p_proc.add_argument("--response-file", required=True)
    sub.add_parser("active")
    sub.add_parser("grade")
    args = parser.parse_args()

    if args.cmd == "init":
        ids = init_contexts()
        print(f"Initialized {len(ids)} contexts:")
        for cid in ids:
            print(f"  {cid}")

    elif args.cmd == "prompt":
        ctx = load_ctx(args.ctx_id)
        if ctx["terminated"]:
            print(f"# context {args.ctx_id} is already terminated", file=sys.stderr)
            return 1
        sys.stdout.write(build_dispatch_prompt(ctx))

    elif args.cmd == "process":
        response_text = Path(args.response_file).read_text()
        ctx = load_ctx(args.ctx_id)
        if ctx["terminated"]:
            print(f"# context {args.ctx_id} already terminated; skipping", file=sys.stderr)
            return 1
        ctx = process_response(ctx, response_text)
        save_ctx(args.ctx_id, ctx)
        status = "TERMINATED" if ctx["terminated"] else "ACTIVE"
        sr = "via_send_response" if ctx["terminated_via_send_response"] else ("error" if ctx.get("error") else "no_tool_calls")
        if ctx["terminated"]:
            print(f"  {args.ctx_id}: {status} ({sr}, turns={ctx['current_turn']}, "
                  f"tool_calls={len(ctx['tool_call_log'])})")
        else:
            print(f"  {args.ctx_id}: {status} (turns={ctx['current_turn']}, "
                  f"tool_calls={len(ctx['tool_call_log'])})")

    elif args.cmd == "active":
        active = list_active()
        if active:
            print("\n".join(active))
        # Empty output means none active

    elif args.cmd == "grade":
        grades = grade_all()
        print(json.dumps(grades, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
