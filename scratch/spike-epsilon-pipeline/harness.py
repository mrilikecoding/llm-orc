"""
Spike ε — plan → dispatch → synthesize pipeline harness.

Cycle 7 DISCOVER spike (2026-05-21+). Tests the framework-driven pipeline
shape end-to-end against (a) the PLAY note 22 composition prompt that
confabulated under orchestrator-LLM dispatch, and (b) Spike δ's web-searcher
→ claim-extractor deterministic chain with the new synthesizer at the end
(positive-control comparison).

Free-tier only: routing-planner = qwen3:8b (Spike ζ ensemble);
synthesizer = qwen3:8b (Spike ε ensemble); web-searcher = script-agent
(DDG via ddgs, no LLM cost); claim-extractor = qwen3:8b.

Pipelines:

    pipeline_planner_driven(request)
        1. Plan: invoke routing-planner with request content.
        2. Dispatch (single): invoke the named ensemble, OR fall through
           to direct completion via the synthesizer.
        3. Synthesize: invoke response-synthesizer with structured input
           (ORIGINAL REQUEST / PLAN / DISPATCH RESULTS).
        Returns: full audit trail.

    pipeline_deterministic_chain(request, chain)
        1. Dispatch each ensemble in `chain` in order. Output of step N
           is input to step N+1.
        2. Synthesize: invoke response-synthesizer with full chain
           context.
        Returns: full audit trail.

Outputs:
    results.json — structured audit of all runs (planner output, dispatch
                   results, synthesizer output, latency, errors).
    summary.md   — human-readable summary of findings.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parents[1]
RESULTS_FILE = SPIKE_DIR / "results.json"

# Ensure the project venv's python is on PATH so the web-searcher script-agent
# can resolve `ddgs`. The script uses `#!/usr/bin/env python3`; without the
# venv on PATH it picks system python which lacks ddgs.
VENV_BIN = REPO_ROOT / ".venv" / "bin"
SUBPROC_ENV = {**os.environ, "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}"}

PLANNER_ENSEMBLE = "spike-cycle7-zeta-routing-planner"
SYNTHESIZER_ENSEMBLE = "spike-cycle7-epsilon-response-synthesizer"
CAPABILITY_ENSEMBLES = {
    "web-searcher": "agentic-serving/web-searcher",
    "text-summarizer": "agentic-serving/text-summarizer",
    "code-generator": "agentic-serving/code-generator",
    "claim-extractor": "agentic-serving/claim-extractor",
    "argument-mapper": "agentic-serving/argument-mapper",
    "prose-improver": "agentic-serving/prose-improver",
}


def invoke_ensemble(name: str, input_data: str, timeout: int = 300) -> dict[str, Any]:
    """Run `llm-orc invoke <name> --input-data <input>` and return parsed JSON.

    Returns a dict with: ok (bool), response (str | None), raw (dict | None),
    error (str | None), latency_seconds (float).
    """
    start = time.time()
    try:
        result = subprocess.run(
            [
                "llm-orc",
                "invoke",
                name,
                "--input-data",
                input_data,
                "--output-format",
                "json",
                "--no-streaming",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=SUBPROC_ENV,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "response": None,
            "raw": None,
            "error": f"timeout after {timeout}s",
            "latency_seconds": time.time() - start,
        }
    latency = time.time() - start
    if result.returncode != 0:
        return {
            "ok": False,
            "response": None,
            "raw": None,
            "error": f"exit {result.returncode}: {result.stderr.strip()[:500]}",
            "latency_seconds": latency,
        }
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "response": None,
            "raw": None,
            "error": f"json parse: {e}",
            "latency_seconds": latency,
        }
    # The response shape: raw["results"][<agent-name>]["response"]
    results = raw.get("results", {})
    if not results:
        return {
            "ok": False,
            "response": None,
            "raw": raw,
            "error": "no results in raw output",
            "latency_seconds": latency,
        }
    # Take first agent's response
    agent_name = next(iter(results))
    response = results[agent_name].get("response", "")
    return {
        "ok": True,
        "response": response,
        "agent": agent_name,
        "raw": raw,
        "error": None,
        "latency_seconds": latency,
    }


def parse_plan(planner_response: str) -> dict[str, Any]:
    """Strip <think>...</think> blocks and parse JSON plan."""
    if not planner_response:
        return {"ok": False, "error": "empty planner response"}
    stripped = re.sub(r"<think>.*?</think>", "", planner_response, flags=re.DOTALL).strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {"ok": False, "error": "no JSON object found", "raw": planner_response}
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"json parse: {e}", "raw": match.group(0)}
    return {"ok": True, "plan": plan}


def format_synthesizer_input(
    original_request: str,
    dispatched: list[str],
    planned_but_not_run: list[str],
    dispatch_results: list[tuple[str, str]],
) -> str:
    dispatched_str = ", ".join(dispatched) if dispatched else "none"
    planned_str = ", ".join(planned_but_not_run) if planned_but_not_run else "none"
    parts = [
        "ORIGINAL REQUEST",
        original_request,
        "",
        "PLAN",
        f"Dispatched: {dispatched_str}",
        f"Planned-but-not-run: {planned_str}",
        "",
        "DISPATCH RESULTS",
    ]
    if not dispatch_results:
        parts.append("(none — direct completion)")
    else:
        for ensemble_name, result in dispatch_results:
            parts.append(f"[{ensemble_name}]")
            parts.append(result)
            parts.append("")
    return "\n".join(parts)


def pipeline_planner_driven(request: str) -> dict[str, Any]:
    """Planner-driven single-step pipeline.

    1. Plan: routing-planner.
    2. Dispatch (single): named ensemble OR fall through.
    3. Synthesize: response-synthesizer with structured context.
    """
    trace: dict[str, Any] = {"request": request, "stages": []}

    # Stage 1: Plan
    plan_invoke = invoke_ensemble(PLANNER_ENSEMBLE, request)
    trace["stages"].append({"stage": "plan", "invoke": plan_invoke})
    if not plan_invoke["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "plan"
        return trace
    plan_parse = parse_plan(plan_invoke["response"])
    trace["plan_parse"] = plan_parse
    if not plan_parse["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "plan_parse"
        return trace
    plan = plan_parse["plan"]
    action = plan.get("action")
    ensemble = plan.get("ensemble")

    dispatched: list[str] = []
    planned_but_not_run: list[str] = []
    dispatch_results: list[tuple[str, str]] = []

    # Detect explicit chain language in request (for "planned-but-not-run" honesty)
    chain_pattern = re.compile(
        r"(?:then|after that|and then)\s+use\s+(?:the\s+)?(\w+(?:-\w+)*)",
        flags=re.IGNORECASE,
    )
    chained_ensemble_names = chain_pattern.findall(request)
    # Also detect "use the X capability ensemble" patterns to enumerate all named
    explicit_pattern = re.compile(
        r"use\s+(?:the\s+)?(\w+(?:-\w+)*)\s+capability\s+ensemble",
        flags=re.IGNORECASE,
    )
    explicit_names = explicit_pattern.findall(request)
    all_explicit = list({*chained_ensemble_names, *explicit_names})

    # Stage 2: Dispatch (single, per planner output)
    if action == "dispatch" and ensemble and ensemble in CAPABILITY_ENSEMBLES:
        target = CAPABILITY_ENSEMBLES[ensemble]
        dispatch_invoke = invoke_ensemble(target, request)
        trace["stages"].append(
            {"stage": "dispatch", "ensemble": ensemble, "invoke": dispatch_invoke}
        )
        if dispatch_invoke["ok"]:
            dispatched.append(ensemble)
            dispatch_results.append((ensemble, dispatch_invoke["response"]))
        else:
            planned_but_not_run.append(ensemble)
        # Any other ensemble named explicitly in the request but not dispatched
        for name in all_explicit:
            if name in CAPABILITY_ENSEMBLES and name not in dispatched:
                planned_but_not_run.append(name)
    elif action == "direct":
        trace["stages"].append({"stage": "dispatch", "ensemble": None, "skipped": "direct"})
    else:
        trace["stages"].append(
            {
                "stage": "dispatch",
                "ensemble": ensemble,
                "skipped": "unknown_ensemble_or_bad_action",
            }
        )

    # Stage 3: Synthesize
    synth_input = format_synthesizer_input(
        request, dispatched, planned_but_not_run, dispatch_results
    )
    synth_invoke = invoke_ensemble(SYNTHESIZER_ENSEMBLE, synth_input)
    trace["synth_input"] = synth_input
    trace["stages"].append({"stage": "synthesize", "invoke": synth_invoke})
    if not synth_invoke["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "synthesize"
        return trace

    trace["final_response"] = synth_invoke["response"]
    trace["dispatched"] = dispatched
    trace["planned_but_not_run"] = planned_but_not_run
    trace["verdict"] = "OK"
    return trace


def pipeline_deterministic_chain(
    request: str, chain: list[tuple[str, str]]
) -> dict[str, Any]:
    """Deterministic-chain pipeline.

    chain: list of (ensemble_name, input_strategy). input_strategy is either
        "request" (use original request) or "prior" (use prior step's output).
    """
    trace: dict[str, Any] = {
        "request": request,
        "chain": [(name, strat) for name, strat in chain],
        "stages": [],
    }

    dispatched: list[str] = []
    dispatch_results: list[tuple[str, str]] = []
    prior_output: str = request

    for ensemble_name, strategy in chain:
        if ensemble_name not in CAPABILITY_ENSEMBLES:
            trace["verdict"] = "FAIL"
            trace["failure_stage"] = f"unknown ensemble {ensemble_name}"
            return trace
        target = CAPABILITY_ENSEMBLES[ensemble_name]
        step_input = request if strategy == "request" else prior_output
        invoke = invoke_ensemble(target, step_input)
        trace["stages"].append(
            {
                "stage": "dispatch",
                "ensemble": ensemble_name,
                "strategy": strategy,
                "invoke": invoke,
            }
        )
        if not invoke["ok"]:
            trace["verdict"] = "FAIL"
            trace["failure_stage"] = f"dispatch {ensemble_name}"
            return trace
        dispatched.append(ensemble_name)
        dispatch_results.append((ensemble_name, invoke["response"]))
        prior_output = invoke["response"]

    # Synthesize
    synth_input = format_synthesizer_input(request, dispatched, [], dispatch_results)
    synth_invoke = invoke_ensemble(SYNTHESIZER_ENSEMBLE, synth_input)
    trace["synth_input"] = synth_input
    trace["stages"].append({"stage": "synthesize", "invoke": synth_invoke})
    if not synth_invoke["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "synthesize"
        return trace

    trace["final_response"] = synth_invoke["response"]
    trace["dispatched"] = dispatched
    trace["dispatch_results"] = dispatch_results
    trace["verdict"] = "OK"
    return trace


def numbers_in(text: str) -> set[str]:
    """Extract distinct multi-digit numbers (Spike δ's analysis pattern)."""
    return set(re.findall(r"\b\d[\d,]{2,}\b", text or ""))


def analyze_fidelity(trace: dict[str, Any]) -> dict[str, Any]:
    """Analyze whether the final synthesizer output matches dispatch result content.

    Number-overlap analysis per Spike δ. Returns set comparisons.
    """
    final = trace.get("final_response", "")
    final_numbers = numbers_in(final)
    source_numbers: set[str] = set()
    for stage in trace.get("stages", []):
        if stage.get("stage") == "dispatch" and stage.get("invoke", {}).get("ok"):
            source_numbers |= numbers_in(stage["invoke"].get("response", ""))
    return {
        "final_numbers": sorted(final_numbers),
        "source_numbers": sorted(source_numbers),
        "overlap": sorted(final_numbers & source_numbers),
        "fabricated_in_final": sorted(final_numbers - source_numbers),
        "fidelity_pass": (final_numbers - source_numbers) == set() or not final_numbers,
    }


def run_all() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []

    # Test 1: PLAY note 22 composition prompt under planner-driven pipeline
    test1_request = (
        "Use the web-searcher capability ensemble to find information about "
        "the current population of Iceland, then use the claim-extractor "
        "capability ensemble on the results."
    )
    print("Test 1: PLAY note 22 prompt under planner-driven pipeline", flush=True)
    test1 = pipeline_planner_driven(test1_request)
    test1["test_id"] = "T1-play-note-22"
    test1["fidelity"] = analyze_fidelity(test1)
    runs.append(test1)

    # Test 2: Spike δ deterministic chain with synthesizer at end
    test2_request = "current population of Iceland"
    print("Test 2: Spike δ chain (web-searcher → claim-extractor) with synthesizer", flush=True)
    test2 = pipeline_deterministic_chain(
        test2_request,
        [("web-searcher", "request"), ("claim-extractor", "prior")],
    )
    test2["test_id"] = "T2-spike-delta-chain-with-synth"
    test2["fidelity"] = analyze_fidelity(test2)
    runs.append(test2)

    # Test 3 (bonus): Simple single-step planner-driven on Spike δ prompt
    # This tests the basic plan→dispatch→synthesize cleanly.
    test3_request = "What is the current population of Iceland?"
    print("Test 3: Simple single-capability question under planner-driven pipeline", flush=True)
    test3 = pipeline_planner_driven(test3_request)
    test3["test_id"] = "T3-simple-single-capability"
    test3["fidelity"] = analyze_fidelity(test3)
    runs.append(test3)

    return {
        "spike": "cycle-7-epsilon-pipeline",
        "run_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": runs,
    }


def main() -> int:
    results = run_all()
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_FILE}")
    for run in results["runs"]:
        tid = run["test_id"]
        verdict = run.get("verdict", "?")
        fid = run.get("fidelity", {})
        fab = fid.get("fabricated_in_final", [])
        print(
            f"  {tid}: verdict={verdict}, "
            f"final_numbers={len(fid.get('final_numbers', []))}, "
            f"source_numbers={len(fid.get('source_numbers', []))}, "
            f"fabricated={fab}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
