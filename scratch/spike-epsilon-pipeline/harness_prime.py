"""
Spike ε' — bounds-closing tests for the response-synthesizer ensemble.

Extends Spike ε (`harness.py`) to address three plausible-but-untested bounds
named in the Cycle 7 DISCOVER → MODEL gate scope-of-claim partition:

  A. Synthesizer's Rule 5 direct-completion path under varied request shapes.
  B. Rounding-drift base rate under numerical-dense dispatch results.
  C. Multi-turn conversational continuity under the synthesizer-only
     architecture (whether prior-turn context carries into subsequent turns).

Free-tier only (qwen3:8b throughout). Results to results_prime.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Reuse infrastructure from Spike ε harness
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (
    SPIKE_DIR,
    SYNTHESIZER_ENSEMBLE,
    analyze_fidelity,
    format_synthesizer_input,
    invoke_ensemble,
    numbers_in,
    pipeline_deterministic_chain,
    pipeline_planner_driven,
)

RESULTS_FILE = SPIKE_DIR / "results_prime.json"


# ---------------------------------------------------------------------------
# Test family A — Direct-completion path (Rule 5)
# ---------------------------------------------------------------------------

def test_a_direct_completion_simple_conversational() -> dict[str, Any]:
    """A1: Simple conversational request that should fall to direct completion."""
    request = "Hi! What's a fun fact about octopuses?"
    return _annotate(
        pipeline_planner_driven(request),
        test_id="A1-direct-simple-conversational",
        purpose="Does the planner route to direct + does the synthesizer answer honestly with Rule 5 framing?",
    )


def test_a_direct_completion_factual_question() -> dict[str, Any]:
    """A2: Factual question with no capability match."""
    request = "What's the capital of Mongolia, and roughly when was it founded?"
    return _annotate(
        pipeline_planner_driven(request),
        test_id="A2-direct-factual-question",
        purpose="Direct-completion path on a factual question; check for fidelity to known answers.",
    )


def test_a_direct_completion_ambiguous() -> dict[str, Any]:
    """A3: Ambiguous prompt where capability fit is weak."""
    request = "I'm not sure how to approach this — any thoughts on writing a good commit message?"
    return _annotate(
        pipeline_planner_driven(request),
        test_id="A3-direct-ambiguous-no-capability",
        purpose="Ambiguous NL request; planner likely routes direct (no clear capability slot).",
    )


# ---------------------------------------------------------------------------
# Test family B — Numerical-density fidelity
# ---------------------------------------------------------------------------

_DENSE_NUMERICAL_FIXTURE = """
COUNTRY STATISTICS REPORT (synthetic fixture for spike ε' B1)

Population figures across selected European nations (2024 estimates):
- France: 68,374,591 (2024-01-01); 67,656,682 (2023-01-01); 65,273,511 (2020-01-01)
- Germany: 84,358,845 (2024-01-01); 84,358,845 (2023-12-31); 83,166,711 (2020-12-31)
- Italy: 58,761,146 (2024-01-01); 58,853,482 (2023-01-01); 60,317,116 (2019-12-31)
- Spain: 48,946,035 (2024-01-01); 48,345,223 (2023-01-01); 47,332,614 (2020-01-01)
- Poland: 37,636,508 (2024-01-01); 37,766,327 (2023-01-01); 38,265,013 (2019-12-31)

Population density (per km²):
- France: 124.7; Germany: 235.2; Italy: 194.8; Spain: 96.7; Poland: 120.3

GDP per capita (USD, 2023):
- France: $44,460; Germany: $52,746; Italy: $38,373; Spain: $32,677; Poland: $22,113

Note: figures are synthetic for spike testing and may not match real-world data.
"""


def test_b_numerical_density() -> dict[str, Any]:
    """B1: Synthesizer over a dense-numerical dispatch result; count drift violations."""
    # Synthesize directly (skip planner+dispatch — we want a controlled fixture)
    request = "Summarize the European country statistics for me, focusing on the most populous and the wealthiest."
    synth_input = format_synthesizer_input(
        original_request=request,
        dispatched=["text-summarizer"],
        planned_but_not_run=[],
        dispatch_results=[("text-summarizer", _DENSE_NUMERICAL_FIXTURE)],
    )
    invoke = invoke_ensemble(SYNTHESIZER_ENSEMBLE, synth_input)
    trace: dict[str, Any] = {
        "test_id": "B1-numerical-density",
        "purpose": "Rounding/restatement-drift base rate over a many-figure dispatch result.",
        "request": request,
        "synth_input": synth_input,
        "fixture": _DENSE_NUMERICAL_FIXTURE,
        "stages": [{"stage": "synthesize", "invoke": invoke}],
    }
    if not invoke["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "synthesize"
        return trace
    trace["final_response"] = invoke["response"]
    source_nums = numbers_in(_DENSE_NUMERICAL_FIXTURE)
    final_nums = numbers_in(invoke["response"])
    fab = sorted(final_nums - source_nums)
    overlap = sorted(final_nums & source_nums)
    trace["fidelity"] = {
        "source_numbers_count": len(source_nums),
        "final_numbers_count": len(final_nums),
        "overlap": overlap,
        "fabricated_in_final": fab,
        "fidelity_pass": fab == [],
        "drift_rate": (len(fab) / len(final_nums)) if final_nums else 0.0,
    }
    trace["verdict"] = "OK"
    return trace


def test_b_precise_roundable() -> dict[str, Any]:
    """B2: Dispatch result with precise round-able figures; does synthesizer round?"""
    fixture = (
        "Country populations (precise): Iceland 388,790; "
        "Luxembourg 660,809; Malta 542,051; San Marino 33,581; "
        "Liechtenstein 39,580."
    )
    request = "What are the populations of the smaller European countries?"
    synth_input = format_synthesizer_input(
        original_request=request,
        dispatched=["web-searcher"],
        planned_but_not_run=[],
        dispatch_results=[("web-searcher", fixture)],
    )
    invoke = invoke_ensemble(SYNTHESIZER_ENSEMBLE, synth_input)
    trace: dict[str, Any] = {
        "test_id": "B2-precise-roundable",
        "purpose": "Does the synthesizer round precise figures (e.g., 388,790 → 388,000)?",
        "request": request,
        "synth_input": synth_input,
        "fixture": fixture,
        "stages": [{"stage": "synthesize", "invoke": invoke}],
    }
    if not invoke["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "synthesize"
        return trace
    trace["final_response"] = invoke["response"]
    source_nums = numbers_in(fixture)
    final_nums = numbers_in(invoke["response"])
    fab = sorted(final_nums - source_nums)
    trace["fidelity"] = {
        "source_numbers": sorted(source_nums),
        "final_numbers": sorted(final_nums),
        "fabricated_in_final": fab,
        "fidelity_pass": fab == [],
    }
    trace["verdict"] = "OK"
    return trace


# ---------------------------------------------------------------------------
# Test family C — Multi-turn continuity
# ---------------------------------------------------------------------------

def test_c_multi_turn_context_dependent() -> dict[str, Any]:
    """C1: Two-turn session. First turn establishes context; second turn depends on it."""
    # Turn 1
    turn1_request = "I'm planning a trip to Reykjavik next month."
    turn1 = pipeline_planner_driven(turn1_request)

    # Turn 2 — depends on turn 1's context. The synthesizer's input must
    # somehow incorporate prior conversation. The current pipeline has no
    # conversation-state — this is the architectural question being probed.
    # We test the simplest mitigation: include turn 1 in the synthesizer's
    # ORIGINAL REQUEST section as conversation history.
    turn2_user_request = "What's the weather like there?"
    turn2_request_with_history = (
        f"[Prior conversation: user said: \"{turn1_request}\". "
        f"Assistant responded: \"{turn1.get('final_response', '(empty)')}\"]\n\n"
        f"Current user message: {turn2_user_request}"
    )
    turn2 = pipeline_planner_driven(turn2_request_with_history)

    trace: dict[str, Any] = {
        "test_id": "C1-multi-turn-context-dependent",
        "purpose": "Does the synthesizer-only architecture handle multi-turn context when history is passed in the request?",
        "turn1_request": turn1_request,
        "turn1_response": turn1.get("final_response"),
        "turn2_user_message": turn2_user_request,
        "turn2_request_with_history": turn2_request_with_history,
        "turn2_response": turn2.get("final_response"),
        "turn2_plan": turn2.get("plan_parse", {}).get("plan"),
        "verdict": "OK" if turn2.get("verdict") == "OK" else "FAIL",
    }
    # Subjective check: does turn2 response reference "Reykjavik" or location-specific?
    response = (turn2.get("final_response") or "").lower()
    trace["fidelity"] = {
        "references_reykjavik": "reykjavik" in response,
        "references_iceland": "iceland" in response,
        "references_weather": "weather" in response or "temperature" in response or "climate" in response,
        "context_preserved": (
            ("reykjavik" in response or "iceland" in response)
            and ("weather" in response or "temperature" in response or "climate" in response)
        ),
    }
    return trace


def test_c_multi_turn_dispatch_referenced() -> dict[str, Any]:
    """C2: Two-turn session where turn 2 references turn 1's dispatch result."""
    # Turn 1 — dispatches web-searcher
    turn1_request = "What's the current population of Iceland?"
    turn1 = pipeline_planner_driven(turn1_request)

    # Turn 2 — references turn 1's response (the synthesized answer, which
    # carries dispatch result content). Tests whether the synthesizer can
    # reason over its own prior response when given as history.
    turn2_user_request = "How does that compare to Luxembourg's population?"
    turn2_request_with_history = (
        f"[Prior conversation: user said: \"{turn1_request}\". "
        f"Assistant responded: \"{turn1.get('final_response', '(empty)')}\"]\n\n"
        f"Current user message: {turn2_user_request}"
    )
    turn2 = pipeline_planner_driven(turn2_request_with_history)

    trace: dict[str, Any] = {
        "test_id": "C2-multi-turn-dispatch-referenced",
        "purpose": "Does the synthesizer correctly use prior dispatch-result content when answering a follow-up?",
        "turn1_request": turn1_request,
        "turn1_response": turn1.get("final_response"),
        "turn2_user_message": turn2_user_request,
        "turn2_request_with_history": turn2_request_with_history,
        "turn2_response": turn2.get("final_response"),
        "turn2_plan": turn2.get("plan_parse", {}).get("plan"),
        "verdict": "OK" if turn2.get("verdict") == "OK" else "FAIL",
    }
    response = (turn2.get("final_response") or "").lower()
    trace["fidelity"] = {
        "references_iceland": "iceland" in response,
        "references_luxembourg": "luxembourg" in response,
        "comparison_attempted": "compar" in response or "vs" in response or "than" in response,
    }
    return trace


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _annotate(trace: dict[str, Any], test_id: str, purpose: str) -> dict[str, Any]:
    trace["test_id"] = test_id
    trace["purpose"] = purpose
    if "fidelity" not in trace and trace.get("verdict") == "OK":
        trace["fidelity"] = analyze_fidelity(trace)
    return trace


def run_all() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    print("Spike ε' — bounds-closing tests for the response-synthesizer", flush=True)

    print("\nA1: simple conversational direct-completion", flush=True)
    runs.append(test_a_direct_completion_simple_conversational())

    print("A2: factual-question direct-completion", flush=True)
    runs.append(test_a_direct_completion_factual_question())

    print("A3: ambiguous-no-capability direct-completion", flush=True)
    runs.append(test_a_direct_completion_ambiguous())

    print("\nB1: numerical-density fidelity (dense fixture)", flush=True)
    runs.append(test_b_numerical_density())

    print("B2: precise-roundable figures fidelity", flush=True)
    runs.append(test_b_precise_roundable())

    print("\nC1: multi-turn context-dependent follow-up", flush=True)
    runs.append(test_c_multi_turn_context_dependent())

    print("C2: multi-turn dispatch-referenced follow-up", flush=True)
    runs.append(test_c_multi_turn_dispatch_referenced())

    return {
        "spike": "cycle-7-epsilon-prime-bounds-closing",
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
        extras = []
        if "drift_rate" in fid:
            extras.append(f"drift_rate={fid['drift_rate']:.2%}")
        if "context_preserved" in fid:
            extras.append(f"context_preserved={fid['context_preserved']}")
        if "references_luxembourg" in fid:
            extras.append(f"references_luxembourg={fid['references_luxembourg']}")
        extras_str = ", " + ", ".join(extras) if extras else ""
        print(f"  {tid}: verdict={verdict}, fabricated={fab}{extras_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
