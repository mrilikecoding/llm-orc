"""Spike B tau-shape harness — RQ-3 multi-turn reliability probe.

Three arms (Decision G3) on the library-checkout fixture:
  arm-cheap-bare           : MiniMax M2.5 Free orchestrator, multi-turn loop with
                             generate_with_tools; deterministic library tools.
  arm-cheap-with-script    : Same as cheap-bare, plus a deterministic preprocessor
                             that runs check_patron_status before the LLM enters
                             the loop (parallels Spike A's arm2 — script context
                             prepended). Tests whether the script-as-grounding
                             pattern from Spike A extends to multi-turn.
  arm-frontier             : Sonnet 4.6 facsimile via Claude Code Agent dispatch.
                             Frontier-arm design defers to a separate dispatch
                             pattern after cheap arms run (see TODO in run_frontier).

Each trial runs 3 scenarios (available / checked-out / patron-fines) per the
library_catalog module. N=2 trials per arm.

Failure-mode instrumentation per Decision A:
  - meltdown onset (repeated tool calls)
  - premature stop (no send_response)
  - error self-conditioning (repeated tool call after error)
  - memory drift (re-asks for resolved info — manual review)
  - early stall (no search_catalog within 2 calls)

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add llm-orc src to path for production-client access (Path A from Spike A)
SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Add scratch dir for library_catalog import
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm_orc.cli_library.template_provider import LibraryTemplateProvider
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import ModelFactory

import library_catalog as lc


# -------------------- system prompts --------------------

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


# -------------------- multi-turn loop --------------------

@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    turn: int


@dataclass
class TurnRecord:
    turn: int
    assistant_content: str
    tool_calls: list[dict[str, Any]]  # raw OpenAI-format
    finish_reason: str
    duration_s: float


@dataclass
class ScenarioResult:
    scenario_name: str
    arm: str
    trial_index: int
    turns: list[TurnRecord]
    tool_call_log: list[ToolCallRecord]
    final_state_summary: dict[str, Any]
    grade: lc.ScenarioGrade
    error: str | None = None
    total_duration_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def to_dict(obj: Any) -> Any:
    """Serialize dataclasses recursively for JSON output."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


async def run_scenario_cheap(
    model: Any,
    scenario: dict[str, Any],
    arm_name: str,
    trial_index: int,
    *,
    use_script_preprocess: bool,
    max_turns: int = 15,
) -> ScenarioResult:
    """Run a single scenario through the multi-turn cheap-arm loop."""
    state = lc.fresh_state(scenario)
    tool_call_log: list[ToolCallRecord] = []
    turns: list[TurnRecord] = []
    error: str | None = None
    total_input = 0
    total_output = 0

    # Build initial messages
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": LIBRARY_AGENT_SYSTEM_PROMPT},
    ]

    # Cheap-with-script arm: preprocess by deterministically calling
    # check_patron_status before the LLM enters the loop. Pass result as
    # a system-side context block. Parallels Spike A arm2's script-as-input.
    if use_script_preprocess:
        patron_id = scenario["patron_id"]
        preprocess_result = lc.check_patron_status(state, patron_id)
        tool_call_log.append(ToolCallRecord(
            name="check_patron_status",
            arguments={"patron_id": patron_id},
            result=preprocess_result,
            turn=-1,  # preprocessing turn
        ))
        context_block = (
            f"DETERMINISTIC PREPROCESSING (already completed by script):\n"
            f"check_patron_status({patron_id}) returned: "
            f"{json.dumps(preprocess_result, indent=2)}\n"
        )
        messages.append({"role": "system", "content": context_block})

    # Initial user message
    messages.append({"role": "user", "content": scenario["request"]})

    start_time = time.time()
    terminated_via_send_response = False

    for turn in range(max_turns):
        turn_start = time.time()
        try:
            response = await model.generate_with_tools(
                messages=messages,
                tools=lc.TOOL_DEFINITIONS,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break

        turn_duration = time.time() - turn_start
        usage = response.usage
        total_input += getattr(usage, "prompt_tokens", 0)
        total_output += getattr(usage, "completion_tokens", 0)

        # Capture turn record
        raw_tool_calls = []
        for tc in response.tool_calls:
            raw_tool_calls.append({
                "id": tc.id,
                "name": tc.name,
                "arguments_json": tc.arguments_json,
            })
        turns.append(TurnRecord(
            turn=turn,
            assistant_content=response.content,
            tool_calls=raw_tool_calls,
            finish_reason=response.finish_reason,
            duration_s=turn_duration,
        ))

        # Append assistant message to messages (OpenAI format)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # If no tool calls, agent has stopped — possible premature stop
        if not response.tool_calls:
            break

        # Dispatch each tool call
        for tc in response.tool_calls:
            try:
                args = json.loads(tc.arguments_json) if tc.arguments_json else {}
            except json.JSONDecodeError as e:
                args = {}
                result: dict[str, Any] = {
                    "error": f"Invalid arguments JSON: {e}; raw: {tc.arguments_json!r}"
                }
            else:
                result = lc.dispatch_tool(state, tc.name, args)

            tool_call_log.append(ToolCallRecord(
                name=tc.name,
                arguments=args,
                result=result,
                turn=turn,
            ))

            # Append tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

            # Termination: send_response succeeded
            if tc.name == "send_response" and result.get("success"):
                terminated_via_send_response = True

        if terminated_via_send_response:
            break

    total_duration = time.time() - start_time

    # Grade scenario
    grade = lc.grade_scenario(
        scenario_name=scenario["name"],
        final_state=state,
        tool_call_log=[
            {"name": r.name, "arguments": r.arguments, "result": r.result}
            for r in tool_call_log
            if r.turn >= 0  # exclude preprocessing entries from grading
        ],
    )

    # Add max-turn reached flag if applicable
    if not terminated_via_send_response and not error and len(turns) == max_turns:
        grade.failure_modes.append("max_turns_reached_without_send_response")

    final_summary = {
        "books": {bid: b["status"] for bid, b in state["books"].items()},
        "patron_fine_balance": state["patrons"][scenario["patron_id"]]["fine_balance"],
        "responses_sent": state.get("_responses_sent", []),
    }

    return ScenarioResult(
        scenario_name=scenario["name"],
        arm=arm_name,
        trial_index=trial_index,
        turns=turns,
        tool_call_log=tool_call_log,
        final_state_summary=final_summary,
        grade=grade,
        error=error,
        total_duration_s=total_duration,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


# -------------------- arm runners --------------------

async def run_cheap_arm(
    arm_name: str,
    profile_name: str,
    trial_index: int,
    use_script_preprocess: bool,
    output_dir: Path,
) -> list[ScenarioResult]:
    """Run all 3 scenarios for one cheap-arm trial."""
    config_manager = ConfigurationManager(template_provider=LibraryTemplateProvider())
    credential_storage = CredentialStorage(config_manager)
    model_factory = ModelFactory(config_manager, credential_storage)
    model = await model_factory.load_model_from_agent_config(
        {"model_profile": profile_name}
    )

    if not model.supports_tool_calling:
        raise RuntimeError(
            f"Profile {profile_name!r} resolves to a model that does not support "
            f"tool calling. Cannot run multi-turn tau-shape fixture."
        )

    results: list[ScenarioResult] = []
    for scenario in lc.ALL_SCENARIOS:
        print(f"    [{arm_name} trial{trial_index} scenario={scenario['name']}] ", end="", flush=True)
        result = await run_scenario_cheap(
            model=model,
            scenario=scenario,
            arm_name=arm_name,
            trial_index=trial_index,
            use_script_preprocess=use_script_preprocess,
        )
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            grade = result.grade
            grade_str = "OK" if grade.success else "FAILED_GRADE"
            failure_str = (
                f" failures={','.join(grade.failure_modes)}"
                if grade.failure_modes else ""
            )
            print(
                f"{grade_str} turns={len(result.turns)} "
                f"tool_calls={len(result.tool_call_log)} "
                f"duration={result.total_duration_s:.1f}s "
                f"in/out={result.total_input_tokens}/{result.total_output_tokens}"
                f"{failure_str}"
            )

        # Save per-scenario JSON
        ts = int(time.time())
        out_path = output_dir / f"{arm_name}-trial{trial_index:02d}-{scenario['name']}-{ts}.json"
        out_path.write_text(json.dumps(to_dict(result), indent=2))
        results.append(result)

    return results


# -------------------- main --------------------

async def execute(args: argparse.Namespace) -> int:
    profile = args.profile
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arms_to_run = args.arms.split(",") if args.arms else ["cheap-bare", "cheap-with-script"]
    valid_arms = {"cheap-bare", "cheap-with-script"}  # frontier handled separately
    for a in arms_to_run:
        if a not in valid_arms:
            print(f"FATAL: arm '{a}' not in {valid_arms} for tau_shape_harness "
                  f"(frontier arm uses dispatch pattern, run separately).", file=sys.stderr)
            return 2

    if args.dry_run:
        print("=" * 80)
        print("DRY-RUN — no LLM calls will be made")
        print(f"Profile: {profile}")
        print(f"Arms: {arms_to_run}")
        print(f"Trials per arm: {args.trials}")
        print(f"Scenarios per trial: {len(lc.ALL_SCENARIOS)} ({[s['name'] for s in lc.ALL_SCENARIOS]})")
        print(f"Output dir: {output_dir}")
        print("=" * 80)

        # Bootstrap + profile resolution check (no network)
        try:
            cm = ConfigurationManager(template_provider=LibraryTemplateProvider())
            CredentialStorage(cm)
            resolved_model, resolved_provider = cm.resolve_model_profile(profile)
            print(f"\nProfile resolves: model={resolved_model}, provider={resolved_provider}")
            p = cm.get_model_profile(profile) or {}
            print(f"  base_url: {p.get('base_url', '(default)')}")
        except Exception as e:
            print(f"\nFAILED to resolve profile: {type(e).__name__}: {e}", file=sys.stderr)
            return 2

        # Print scenario summary
        print("\nScenarios in fixture:")
        for s in lc.ALL_SCENARIOS:
            print(f"\n  [{s['name']}] expected_action: {s['expected_action']}")
            print(f"    request: {s['request']}")

        print("\nTool definitions count:", len(lc.TOOL_DEFINITIONS))
        print("Tool names:", [t["function"]["name"] for t in lc.TOOL_DEFINITIONS])
        print()
        print("System prompt preview (first 200 chars):")
        print(LIBRARY_AGENT_SYSTEM_PROMPT[:200] + "...")
        print()
        print("=" * 80)
        print("Dry-run complete. To execute trials, re-run without --dry-run.")
        return 0

    # Live mode
    print(f"Bootstrapping llm-orc client for profile '{profile}'...")
    cm_check = ConfigurationManager(template_provider=LibraryTemplateProvider())
    CredentialStorage(cm_check)
    try:
        resolved_model, resolved_provider = cm_check.resolve_model_profile(profile)
        print(f"  Resolved: model={resolved_model}, provider={resolved_provider}")
    except Exception as e:
        print(f"FATAL: profile {profile!r} did not resolve: {e}", file=sys.stderr)
        return 2

    print(f"Running {args.trials} trial(s) per arm across {len(arms_to_run)} arm(s); "
          f"{len(lc.ALL_SCENARIOS)} scenario(s) per trial...\n")

    all_results: list[ScenarioResult] = []
    for arm_name in arms_to_run:
        use_script = (arm_name == "cheap-with-script")
        for trial_idx in range(1, args.trials + 1):
            print(f"--- {arm_name} trial {trial_idx}/{args.trials} ---")
            arm_results = await run_cheap_arm(
                arm_name=arm_name,
                profile_name=profile,
                trial_index=trial_idx,
                use_script_preprocess=use_script,
                output_dir=output_dir,
            )
            all_results.extend(arm_results)
            print()

    # Aggregate summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for arm_name in arms_to_run:
        for scenario_name in [s["name"] for s in lc.ALL_SCENARIOS]:
            sub = [r for r in all_results if r.arm == arm_name and r.scenario_name == scenario_name]
            if not sub:
                continue
            ok_count = sum(1 for r in sub if r.grade.success)
            print(f"  {arm_name} / {scenario_name}: {ok_count}/{len(sub)} graded success", end="")
            failures: dict[str, int] = {}
            for r in sub:
                for fm in r.grade.failure_modes:
                    failures[fm] = failures.get(fm, 0) + 1
            if failures:
                fm_str = ", ".join(f"{name}×{count}" for name, count in failures.items())
                print(f" | failure modes: {fm_str}")
            else:
                print()
    print()
    print(f"All trial outputs saved to: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike B tau-shape harness (Cycle 3 RQ-3).")
    parser.add_argument(
        "--profile",
        default="orchestrator-minimax-m25-free",
        help="llm-orc model_profile name",
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/spike-b-cycle3-multi-turn/trials",
        help="Directory for trial output JSON files",
    )
    parser.add_argument(
        "--arms",
        default="cheap-bare,cheap-with-script",
        help="Comma-separated arms (cheap-bare, cheap-with-script). Frontier dispatched separately.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="Trials per arm (default: 2 per Decision G3 trial count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Bootstrap, resolve profile, print fixture summary; no LLM calls",
    )
    args = parser.parse_args()
    return asyncio.run(execute(args))


if __name__ == "__main__":
    sys.exit(main())
