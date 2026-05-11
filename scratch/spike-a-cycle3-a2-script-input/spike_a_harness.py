"""Spike A harness — RQ-1 isolation test (A2 + script input vs script-as-orchestrator).

Cycle 3 RQ-1: Does A2 + script input produce equivalent factual grounding to A3
on the cycle-2 README-review task class? Plus Cycle 3 Decision C (hybrid):
adds Arm 3 (script-as-orchestrator) as an in-cycle probe contingent on free-tier
quota holding for three arms × N=3 trials.

Uses llm-orc's production model factory (Path A per practitioner decision):
constructs the model via the configured `orchestrator-minimax-m25-free` profile
through `ConfigurationManager` + `CredentialStorage` + `ModelFactory`. Auth is
handled by llm-orc's existing flow; harness never touches credentials.yaml.

Three arms:
- arm1 (A2 baseline): single LLM doing review on README alone — replicates Cycle 2's A2.
- arm2 (A2 + script input): single LLM doing review with script's deterministic
  report prepended as input context — RQ-1's primary isolation test.
- arm3 (script-as-orchestrator): script ran first; LLM is a bounded subordinate
  step asked to synthesize script findings + perform additional README review —
  Decision C / lit-review's Routine + Compiled AI shape.

Scope conditions on this reconstruction:
1. Cycle 2's A2 ran via `opencode run` (OpenCode CLI agent mode); this harness uses
   llm-orc's production OpenAI-compatible client. The LLM endpoint and model are
   identical (MiniMax M2.5 Free via OpenCode Zen); the path-to-API differs.
2. The system prompt is reconstructed from `code-review.yaml`'s `default_task`
   (the production code-review prompt — same project register as Cycle 2's
   A1 cascade arm). Cycle 2's exact A2 prompt is not recoverable.

NOTE: Spike code. Will be deleted after findings are recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make the llm-orc package importable when running from repo root
SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_orc.cli_library.template_provider import LibraryTemplateProvider
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import ModelFactory


# --- prompts (reconstructed from code-review.yaml's default_task) ---

DEFAULT_TASK_SYSTEM_PROMPT = """\
Review this code submission for a production system. Provide a thorough analysis covering:

**SCOPE:** Full production readiness assessment
**FOCUS:** Security, performance, maintainability, and team standards
**OUTPUT:** Actionable feedback with specific recommendations

Please analyze the code and provide detailed recommendations for improvement."""

# Arm 3's framing (script-as-orchestrator): LLM is a bounded subordinate step
# synthesizing already-completed deterministic analysis. Distinct from Arm 2
# where the LLM is positioned as the reviewer doing analysis with script as
# context.
#
# CONTAMINATED VERSION (preserved for record): the parenthetical example below
# explicitly directs the LLM toward "do YAML examples reference profiles or
# keys that are actually defined in the document?" — the exact failure mode
# RQ-1 is testing for. arm3 results from this prompt cannot cleanly attribute
# success to the script-as-orchestrator shape vs. the explicit direction.
ARM3_SYNTHESIS_SYSTEM_PROMPT = """\
A deterministic analysis script has already run on the README under review. \
Its findings are provided to you as completed evidence (verified link checks, \
canonical-section presence checks, code-block parseability checks). Your role \
is to synthesize the script's findings together with your own additional \
review of the README content into a structured code-review report.

**YOUR ADDITIONAL REVIEW** should cover what deterministic analysis cannot: \
semantic correctness (e.g., do YAML examples reference profiles or keys that \
are actually defined in the document?), consistency across sections, \
documentation gaps, and other content-level issues.

**OUTPUT:** A structured code-review report. Cite both the script's verified \
findings and your own additional findings. Provide actionable recommendations."""

# Arm 3 debiased: same script-as-orchestrator framing, but no specific
# failure-mode hints. Tests whether arm3's bug detection survives without
# the prompt-level direction toward the bug category.
ARM3_DEBIASED_SYSTEM_PROMPT = """\
A deterministic analysis script has already run on the README under review. \
Its findings are provided to you as completed evidence. Your role is to \
synthesize the script's findings together with your own additional review \
of the README content into a structured code-review report.

**OUTPUT:** A structured code-review report. Cite both the script's findings \
and your own findings. Provide actionable recommendations."""

# Arm 4: same debiased system prompt, but user message has ONLY the script
# report (no README content). Tests whether arm3's bug detection requires
# README content for semantic reasoning, or whether the LLM is hallucinating
# from the script's structural output alone. Bug detection here would be a
# strong signal of LLM-prior hallucination, not grounded analysis.


# --- arm definitions ---

@dataclass
class Arm:
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str  # Uses {script_report}, {readme} placeholders


def build_arms(script_report: str, readme: str) -> dict[str, Arm]:
    return {
        "arm1": Arm(
            name="arm1",
            description="A2 baseline (single LLM, README only — no script context)",
            system_prompt=DEFAULT_TASK_SYSTEM_PROMPT,
            user_prompt_template="README UNDER REVIEW:\n\n{readme}",
        ),
        "arm2": Arm(
            name="arm2",
            description="A2 + script input (single LLM with script report prepended as context)",
            system_prompt=DEFAULT_TASK_SYSTEM_PROMPT,
            user_prompt_template=(
                "DETERMINISTIC ANALYSIS REPORT (verified facts from a script analyzer):\n\n"
                "{script_report}\n\n"
                "---\n\n"
                "README UNDER REVIEW:\n\n{readme}"
            ),
        ),
        "arm3": Arm(
            name="arm3",
            description="Script-as-orchestrator CONTAMINATED (system prompt directs LLM to check undefined profiles — preserved for record)",
            system_prompt=ARM3_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt_template=(
                "DETERMINISTIC ANALYSIS REPORT (already completed by script):\n\n"
                "{script_report}\n\n"
                "---\n\n"
                "README CONTENT (for your additional semantic review):\n\n{readme}"
            ),
        ),
        "arm3_debiased": Arm(
            name="arm3_debiased",
            description="Script-as-orchestrator DEBIASED (same script + README, no specific failure-mode hints in system prompt)",
            system_prompt=ARM3_DEBIASED_SYSTEM_PROMPT,
            user_prompt_template=(
                "DETERMINISTIC ANALYSIS REPORT (already completed by script):\n\n"
                "{script_report}\n\n"
                "---\n\n"
                "README CONTENT (for your additional review):\n\n{readme}"
            ),
        ),
        "arm4": Arm(
            name="arm4",
            description="Script-only (debiased synthesis prompt, NO README content) — tests whether bug-detection requires README for grounded analysis",
            system_prompt=ARM3_DEBIASED_SYSTEM_PROMPT,
            user_prompt_template=(
                "DETERMINISTIC ANALYSIS REPORT (already completed by script):\n\n"
                "{script_report}"
            ),
        ),
    }


def render_user_prompt(arm: Arm, script_report: str, readme: str) -> str:
    return arm.user_prompt_template.format(script_report=script_report, readme=readme)


# --- trial execution ---

@dataclass
class TrialResult:
    arm: str
    trial_index: int
    started_at: float
    completed_at: float
    duration_s: float
    response_text: str
    input_tokens: int | None
    output_tokens: int | None
    response_chars: int
    error: str | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "trial_index": self.trial_index,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "response_chars": self.response_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "raw_usage": self.raw_usage,
            "response_text": self.response_text,
            "error": self.error,
        }


async def run_trial(
    model_factory: ModelFactory,
    profile_name: str,
    arm: Arm,
    user_prompt: str,
    trial_index: int,
) -> TrialResult:
    started = time.time()
    try:
        model = await model_factory.load_model_from_agent_config(
            {"model_profile": profile_name}
        )
        response = await model.generate_response(
            message=user_prompt,
            role_prompt=arm.system_prompt,
        )
        usage = model.get_last_usage() or {}
        completed = time.time()
        return TrialResult(
            arm=arm.name,
            trial_index=trial_index,
            started_at=started,
            completed_at=completed,
            duration_s=completed - started,
            response_text=response,
            response_chars=len(response),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            raw_usage=dict(usage),
        )
    except Exception as e:  # network / API errors
        completed = time.time()
        return TrialResult(
            arm=arm.name,
            trial_index=trial_index,
            started_at=started,
            completed_at=completed,
            duration_s=completed - started,
            response_text="",
            response_chars=0,
            input_tokens=None,
            output_tokens=None,
            error=f"{type(e).__name__}: {e}",
        )


def save_trial(result: TrialResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(result.started_at)
    path = output_dir / f"{result.arm}-trial{result.trial_index:02d}-{timestamp}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


# --- main ---

async def execute(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    readme_path = repo_root / args.readme
    script_report_path = Path(args.script_report)
    output_dir = Path(args.output_dir)

    if not readme_path.exists():
        print(f"FATAL: README not found at {readme_path}", file=sys.stderr)
        return 2
    if not script_report_path.exists():
        print(f"FATAL: script report not found at {script_report_path}", file=sys.stderr)
        return 2

    readme = readme_path.read_text()
    script_report = script_report_path.read_text()
    arms = build_arms(script_report=script_report, readme=readme)

    selected_arms = args.arms.split(",") if args.arms else list(arms.keys())
    for a in selected_arms:
        if a not in arms:
            print(f"FATAL: unknown arm '{a}'. Known: {list(arms.keys())}", file=sys.stderr)
            return 2

    # --- dry-run mode: bootstrap llm-orc, resolve profile, print prompts; NO LLM calls ---
    if args.dry_run:
        print("=" * 80)
        print(f"DRY-RUN MODE — no LLM calls will be made")
        print(f"Profile: {args.profile}")
        print(f"Arms selected: {selected_arms}")
        print(f"Trials per arm: {args.trials}")
        print(f"README: {readme_path} ({len(readme)} chars)")
        print(f"Script report: {script_report_path} ({len(script_report)} chars)")
        print(f"Output dir: {output_dir}")
        print("=" * 80)

        # Validate bootstrap path + profile resolution (config-file reads only; no network)
        print("\nBootstrapping llm-orc client for profile resolution check...")
        try:
            config_manager = ConfigurationManager(template_provider=LibraryTemplateProvider())
            CredentialStorage(config_manager)  # constructs but does not consume credentials
            resolved_model, resolved_provider = config_manager.resolve_model_profile(args.profile)
            print(f"  Profile resolves: model={resolved_model}, provider={resolved_provider}")
            profile = config_manager.get_model_profile(args.profile) or {}
            base_url = profile.get("base_url", "(not set)")
            print(f"  base_url: {base_url}")
            print(f"  timeout_seconds: {profile.get('timeout_seconds', '(default)')}")
        except Exception as e:
            print(f"  FAILED to resolve profile: {type(e).__name__}: {e}")
            print("  Live mode will not work without resolving this. Inspect llm-orc config.")
            return 2

        for arm_name in selected_arms:
            arm = arms[arm_name]
            user_prompt = render_user_prompt(arm, script_report=script_report, readme=readme)
            print(f"\n### Arm: {arm.name}")
            print(f"### Description: {arm.description}")
            print(f"### System prompt ({len(arm.system_prompt)} chars):")
            print(arm.system_prompt[:300] + ("..." if len(arm.system_prompt) > 300 else ""))
            print(f"### User prompt ({len(user_prompt)} chars):")
            print(user_prompt[:500] + ("..." if len(user_prompt) > 500 else ""))
            print()
        print("=" * 80)
        print("Dry-run validation complete. To execute trials, re-run without --dry-run.")
        return 0

    # --- live mode: bootstrap llm-orc, run trials ---
    print(f"Bootstrapping llm-orc client for profile '{args.profile}'...")
    config_manager = ConfigurationManager(template_provider=LibraryTemplateProvider())
    credential_storage = CredentialStorage(config_manager)
    model_factory = ModelFactory(config_manager, credential_storage)

    # Verify profile resolves before running any trials
    try:
        resolved_model, resolved_provider = config_manager.resolve_model_profile(args.profile)
        print(f"  Resolved: model={resolved_model}, provider={resolved_provider}")
    except Exception as e:
        print(f"FATAL: profile '{args.profile}' did not resolve: {e}", file=sys.stderr)
        return 2

    print(f"Running {args.trials} trial(s) per arm across {len(selected_arms)} arm(s)...")
    print()

    all_results: list[TrialResult] = []
    for arm_name in selected_arms:
        arm = arms[arm_name]
        user_prompt = render_user_prompt(arm, script_report=script_report, readme=readme)
        print(f"--- {arm.name} ({arm.description}) ---")
        for i in range(1, args.trials + 1):
            print(f"  Trial {i}/{args.trials}... ", end="", flush=True)
            result = await run_trial(
                model_factory=model_factory,
                profile_name=args.profile,
                arm=arm,
                user_prompt=user_prompt,
                trial_index=i,
            )
            saved = save_trial(result, output_dir)
            if result.error:
                print(f"ERROR after {result.duration_s:.2f}s — {result.error}")
            else:
                print(
                    f"OK — {result.duration_s:.2f}s, "
                    f"{result.input_tokens} in / {result.output_tokens} out, "
                    f"saved to {saved.name}"
                )
            all_results.append(result)
        print()

    # --- summary ---
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for arm_name in selected_arms:
        arm_results = [r for r in all_results if r.arm == arm_name and not r.error]
        if not arm_results:
            print(f"  {arm_name}: no successful trials")
            continue
        durations = [r.duration_s for r in arm_results]
        durations.sort()
        median = durations[len(durations) // 2]
        in_toks = [r.input_tokens for r in arm_results if r.input_tokens]
        out_toks = [r.output_tokens for r in arm_results if r.output_tokens]
        print(
            f"  {arm_name}: n={len(arm_results)}/{args.trials} ok, "
            f"median={median:.2f}s, range=[{min(durations):.2f}s, {max(durations):.2f}s], "
            f"avg in/out tokens={sum(in_toks) // len(in_toks) if in_toks else '?'}"
            f"/{sum(out_toks) // len(out_toks) if out_toks else '?'}"
        )
    print()
    print(f"All trial outputs saved to: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike A harness (Cycle 3 RQ-1).")
    parser.add_argument(
        "--profile",
        default="orchestrator-minimax-m25-free",
        help="llm-orc model_profile name (default: orchestrator-minimax-m25-free)",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="README fixture path (relative to repo root; default: README.md)",
    )
    parser.add_argument(
        "--script-report",
        default="scratch/spike-a-cycle3-a2-script-input/script-output.txt",
        help="Path to deterministic_analyzer.py output",
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/spike-a-cycle3-a2-script-input/trials",
        help="Directory for trial output JSON files",
    )
    parser.add_argument(
        "--arms",
        default="arm1,arm2,arm3",
        help="Comma-separated arms to run (subset of arm1,arm2,arm3)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Trials per arm (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt structure for each arm without making LLM calls",
    )
    args = parser.parse_args()
    return asyncio.run(execute(args))


if __name__ == "__main__":
    sys.exit(main())
