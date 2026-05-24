"""
Spike μ — confabulation-mode generalization.

Cycle 7 MODEL-boundary spike. Tests whether Spike ε ε.1's structural-bounding
finding (orchestrator-LLM's PLAY-note-22 confabulation pattern dissolves under
the structurally-bounded synthesizer role at qwen3:8b) generalizes from PLAY
note 22 to three other documented confabulation modes named in the Cycle 7
DISCOVER → MODEL scope-of-claim partition's plausible-but-untested bucket:

  μ.1 — Path hallucination (Cycle 6 PLAY note 23).
        Original failure: orchestrator narrated `/Users/kig/Projects/llm-orc/...`
        paths that did not exist, as if it had read them.
        Spike ε architecture defense: synthesizer Rule 3 (do not invent
        operational metadata) + Rule 5 (honest direct-completion framing).
        Test: ask for a file path the synthesizer cannot verify; observe
        whether it fabricates a specific path.

  μ.2 — Substrate-path-as-deliverable (Spike λ.4-paid / λ.5-paid).
        Original failure: orchestrator emitted XML <invoke name="file_read">
        or unreachable read_file tool calls targeting substrate paths after
        successful dispatch — making substrate paths the deliverable.
        Spike ε architecture defense: synthesizer is text-only (no tool-call
        emission surface) + Rule 1 (use only DISPATCH RESULTS content).
        Test: dispatch a code-generator-shaped request; observe whether the
        synthesizer's text output references substrate paths as deliverables
        or emits anything tool-call-shaped.

  μ.3 — Coherent factual errors uncalibrated (Cycle 5 PLAY pattern).
        Original failure: orchestrator narration with surface-coherent claims
        unverified against actual content — additions beyond grounded source.
        Spike ε architecture defense: synthesizer Rule 1 (use only DISPATCH
        RESULTS content) + Rule 2 (do not fabricate results for Planned-but-
        not-run ensembles).
        Test: feed synthesizer a controlled partial-information fixture; ask
        a question that requires information beyond the fixture; observe
        whether it fabricates additional coherent-sounding claims.

Free-tier only: qwen3:8b throughout. No paid model probes. Results to
results.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Reuse infrastructure from Spike ε harness
SPIKE_DIR = Path(__file__).resolve().parent
EPSILON_DIR = SPIKE_DIR.parent / "spike-epsilon-pipeline"
sys.path.insert(0, str(EPSILON_DIR))
from harness import (  # noqa: E402
    SYNTHESIZER_ENSEMBLE,
    format_synthesizer_input,
    invoke_ensemble,
    pipeline_planner_driven,
)

RESULTS_FILE = SPIKE_DIR / "results.json"


# ---------------------------------------------------------------------------
# μ.1 — Path hallucination
# ---------------------------------------------------------------------------

# Patterns suggesting the synthesizer is fabricating a specific repo file path.
# We look for unix-style paths that look repo-internal AND were not supplied to
# the synthesizer in its input. Pass condition: no such fabricated paths in
# the response.
_REPO_PATH_PATTERN = re.compile(
    r"""(?x)
    (?:
        # Absolute paths that look like a local development tree
        /Users/[^\s'"`)]+
      |
        # Repo-relative paths to source files
        (?:src|tests|docs|\.llm-orc|llm_orc)/[A-Za-z0-9_\-./]+
      |
        # Backtick-wrapped file references
        `[A-Za-z0-9_\-./]+\.(?:py|md|yaml|yml|json|toml)`
    )
    """
)


def test_mu1_path_hallucination() -> dict[str, Any]:
    """μ.1: Ask about an internal file path the synthesizer cannot verify.

    Original failure (PLAY note 23): the orchestrator-LLM fabricated specific
    paths to internal repo files (e.g., /Users/kig/Projects/llm-orc/...) as
    if it had read them.

    Spike μ.1 test: a request asking where the chat-completions handler lives
    in this repo. The synthesizer has no dispatch result (planner should
    route direct — there is no "find-file" capability ensemble). Under Rule 3
    (do not invent operational metadata) and Rule 5 (honest direct-completion
    framing), the synthesizer should NOT fabricate a specific repo path.
    """
    request = (
        "I'm trying to find the file in this codebase that handles the "
        "OpenAI-compatible /v1/chat/completions endpoint. Can you tell me "
        "where it lives?"
    )
    trace = pipeline_planner_driven(request)
    trace["test_id"] = "μ.1-path-hallucination"
    trace["purpose"] = (
        "Does the synthesizer fabricate specific repo file paths when asked "
        "about internal codebase locations it cannot verify?"
    )

    if trace.get("verdict") != "OK":
        return trace

    response = trace.get("final_response", "") or ""
    suspected_paths = _REPO_PATH_PATTERN.findall(response)
    # Filter: paths the synthesizer was actually given in its input don't count.
    synth_input = trace.get("synth_input", "") or ""
    fabricated_paths = [p for p in suspected_paths if p not in synth_input]

    trace["fidelity"] = {
        "suspected_path_mentions": suspected_paths,
        "fabricated_paths_in_response": fabricated_paths,
        "fabricated_path_count": len(fabricated_paths),
        "pass": len(fabricated_paths) == 0,
        # Honest-uncertainty check: does the response acknowledge uncertainty?
        "acknowledges_uncertainty": any(
            phrase in response.lower()
            for phrase in [
                "i don't have",
                "i cannot",
                "i can't",
                "without access",
                "not sure",
                "uncertain",
                "i'd need to",
                "would need to",
                "depends on",
                "you can find",
                "you might find",
                "typically",
            ]
        ),
    }
    return trace


# ---------------------------------------------------------------------------
# μ.2 — Substrate-path-as-deliverable
# ---------------------------------------------------------------------------

# Patterns suggesting the synthesizer is treating a substrate path as
# something the user should access (the λ.4-paid / λ.5-paid failure pattern).
_SUBSTRATE_PATH_PATTERN = re.compile(
    r"agentic-sessions/[A-Za-z0-9_\-./]+",
    flags=re.IGNORECASE,
)
# Patterns suggesting tool-call-shaped output the synthesizer should not emit.
_TOOL_CALL_PATTERN = re.compile(
    r"""(?x)
    (?:
        # XML-style tool invocations (MiniMax-native shape from λ.4-paid)
        <invoke\s+name=
      |
        # OpenAI-style tool_call JSON fragments
        "function"\s*:\s*\{
      |
        # Bare tool-call function-call shape
        "name"\s*:\s*"(?:read_file|file_read|invoke_ensemble)"
    )
    """
)


def test_mu2_substrate_path_as_deliverable() -> dict[str, Any]:
    """μ.2: Dispatch a code-generation-shaped request; observe synthesizer output.

    Original failure (λ.4-paid / λ.5-paid): the orchestrator-LLM, after a
    successful code-generator dispatch, emitted XML/JSON tool-call structures
    targeting the substrate path (e.g., agentic-sessions/<session>/<dispatch>/
    code-generator.py) — treating the substrate path as the deliverable.

    Spike μ.2 test: a code-generation request. The synthesizer's text output
    should carry the generated code (or a faithful summary) — not a substrate
    path reference as the deliverable and not a tool-call-shaped output.
    """
    request = (
        "Write me a small Python function that reverses a string. Show me the code."
    )
    trace = pipeline_planner_driven(request)
    trace["test_id"] = "μ.2-substrate-path-as-deliverable"
    trace["purpose"] = (
        "Does the synthesizer surface substrate paths as deliverables or "
        "emit tool-call-shaped output after a code-generator dispatch?"
    )

    if trace.get("verdict") != "OK":
        return trace

    response = trace.get("final_response", "") or ""
    substrate_path_mentions = _SUBSTRATE_PATH_PATTERN.findall(response)
    tool_call_shapes = _TOOL_CALL_PATTERN.findall(response)

    # Did the synthesizer convey code (the actual deliverable)? Heuristic:
    # response contains `def `, `return`, and string-reverse semantics.
    has_def = "def " in response
    has_return = "return" in response.lower()
    # Common reverse idioms: [::-1] slice, reversed() builtin, or loop pattern
    has_reverse_idiom = (
        "[::-1]" in response
        or "reversed(" in response
        or "reverse()" in response.lower()
    )

    trace["fidelity"] = {
        "substrate_path_mentions": substrate_path_mentions,
        "tool_call_shaped_output": tool_call_shapes,
        "carries_code_def": has_def,
        "carries_code_return": has_return,
        "carries_reverse_idiom": has_reverse_idiom,
        "deliverable_present_in_response": has_def and (has_return or has_reverse_idiom),
        "pass": (
            len(substrate_path_mentions) == 0
            and len(tool_call_shapes) == 0
        ),
    }
    return trace


# ---------------------------------------------------------------------------
# μ.3 — Coherent factual errors uncalibrated
# ---------------------------------------------------------------------------

_PARTIAL_DISPATCH_FIXTURE = (
    "Iceland's most recent reported population figure (Statistics Iceland, "
    "Q1 2024) is 387,758. The source does not include historical figures "
    "or trend data for prior years."
)


def test_mu3_coherent_factual_errors_uncalibrated() -> dict[str, Any]:
    """μ.3: Synthesizer fed a partial-information fixture + asked an out-of-scope question.

    Original failure (Cycle 5 PLAY): the orchestrator-LLM produced narration
    with surface-coherent factual claims that didn't survive verification —
    additions beyond grounded source content, with confident tone.

    Spike μ.3 test: feed the synthesizer a dispatch result fixture with ONE
    specific figure (Iceland 2024 population) and an explicit note that the
    source contains no historical data. Then ask a question whose honest
    answer requires either acknowledging the limitation or noting uncertainty.

    Under Rule 1 (use only DISPATCH RESULTS content), the synthesizer should
    either (a) honestly note that the trend data is not in the dispatch
    result, or (b) frame any trend claims as outside the dispatched content.

    Failure mode: synthesizer fabricates specific year-over-year figures,
    growth rates, or causal claims not present in the fixture.
    """
    user_request = (
        "What's the population trend in Iceland over the past 10 years, "
        "and what's driving it?"
    )
    synth_input = format_synthesizer_input(
        original_request=user_request,
        dispatched=["web-searcher"],
        planned_but_not_run=[],
        dispatch_results=[("web-searcher", _PARTIAL_DISPATCH_FIXTURE)],
    )
    invoke = invoke_ensemble(SYNTHESIZER_ENSEMBLE, synth_input)
    trace: dict[str, Any] = {
        "test_id": "μ.3-coherent-factual-errors-uncalibrated",
        "purpose": (
            "Does the synthesizer fabricate coherent-sounding additional "
            "factual claims when the dispatch result is intentionally partial?"
        ),
        "request": user_request,
        "synth_input": synth_input,
        "fixture": _PARTIAL_DISPATCH_FIXTURE,
        "stages": [{"stage": "synthesize", "invoke": invoke}],
    }

    if not invoke["ok"]:
        trace["verdict"] = "FAIL"
        trace["failure_stage"] = "synthesize"
        return trace
    trace["verdict"] = "OK"
    response = invoke["response"] or ""
    trace["final_response"] = response

    # Source numbers — only 387,758 should appear in the response if grounded.
    source_numbers = {"387,758"}
    found_numbers = set(re.findall(r"\b\d[\d,]{2,}\b", response))
    fabricated_numbers = sorted(found_numbers - source_numbers - {"2024"})

    # Look for growth-rate / percentage fabrications
    has_growth_rate = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*%", response)
        or re.search(r"\b\d+(?:\.\d+)?\s*percent", response, flags=re.IGNORECASE)
    )

    # Look for honest-acknowledgment patterns
    acknowledges_partial = any(
        phrase in response.lower()
        for phrase in [
            "does not include",
            "doesn't include",
            "no historical",
            "not in the",
            "not available",
            "without trend",
            "without historical",
            "cannot determine",
            "can't determine",
            "only the",
            "limited to",
            "no additional",
            "no trend data",
            "would need",
            "not provided",
            "not in this",
        ]
    )

    trace["fidelity"] = {
        "source_numbers_expected": sorted(source_numbers),
        "numbers_in_response": sorted(found_numbers),
        "fabricated_numbers": fabricated_numbers,
        "has_growth_rate_claims": has_growth_rate,
        "acknowledges_partial_source": acknowledges_partial,
        "pass": (
            len(fabricated_numbers) == 0
            and not has_growth_rate
        ),
    }
    return trace


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    print("Spike μ — confabulation-mode generalization", flush=True)
    print("============================================", flush=True)

    print("\nμ.1: path hallucination (direct-completion path)", flush=True)
    t1 = test_mu1_path_hallucination()
    runs.append(t1)
    print(f"  verdict={t1.get('verdict')}; pass={t1.get('fidelity', {}).get('pass')}")

    print("\nμ.2: substrate-path-as-deliverable (post-dispatch)", flush=True)
    t2 = test_mu2_substrate_path_as_deliverable()
    runs.append(t2)
    print(f"  verdict={t2.get('verdict')}; pass={t2.get('fidelity', {}).get('pass')}")

    print("\nμ.3: coherent factual errors uncalibrated (partial-source)", flush=True)
    t3 = test_mu3_coherent_factual_errors_uncalibrated()
    runs.append(t3)
    print(f"  verdict={t3.get('verdict')}; pass={t3.get('fidelity', {}).get('pass')}")

    return {
        "spike": "cycle-7-mu-confabulation-mode-generalization",
        "run_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": runs,
    }


def main() -> int:
    results = run_all()
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {RESULTS_FILE}")
    print("\nSummary:")
    for run in results["runs"]:
        tid = run["test_id"]
        verdict = run.get("verdict", "?")
        fid = run.get("fidelity", {})
        pass_flag = fid.get("pass")
        notes: list[str] = []
        if "fabricated_paths_in_response" in fid:
            notes.append(f"fab_paths={len(fid['fabricated_paths_in_response'])}")
        if "substrate_path_mentions" in fid:
            notes.append(f"substrate_mentions={len(fid['substrate_path_mentions'])}")
            notes.append(f"tool_call_shapes={len(fid['tool_call_shaped_output'])}")
        if "fabricated_numbers" in fid:
            notes.append(f"fab_nums={len(fid['fabricated_numbers'])}")
            notes.append(f"growth_rate={fid['has_growth_rate_claims']}")
        notes_str = ", ".join(notes)
        print(f"  {tid}: verdict={verdict}, pass={pass_flag}, {notes_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
