"""
Spike ν — long-horizon capability ceiling probe.

Cycle 7 architect→build boundary spike (Track A.3). Gates WP-A entry.

Tests the three surfaces the cycle's existing Spike battery (ζ, ε, ε', μ)
did NOT exercise, against the structural-bounding generalization AS-9 +
ADR-027 commit to:

  Surface 1 — Multi-step composition (2-step + 3-step chains).
  Surface 2 — Production-scale numerical content (100+ figures + tables).
  Surface 3 — Adversarial routing (40-prompt battery extending Spike ζ).

Pre-specified qualitative criteria locked in roadmap.md Track A.3 BEFORE
this spike runs (per MODEL snapshot Advisory A). Scoring here computes the
quantitative inputs; the writeup applies the criteria and the trigger rule.

Free-tier only: qwen3:8b (planner, synthesizer, most capabilities),
qwen3:1.7b (text-summarizer). $0 cost; local Ollama.

Outputs: results_nu.json (full audit trail).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

SPIKE_DIR = Path(__file__).resolve().parent
EPSILON_DIR = SPIKE_DIR.parent / "spike-epsilon-pipeline"
sys.path.insert(0, str(EPSILON_DIR))
sys.path.insert(0, str(SPIKE_DIR))

from harness import (  # noqa: E402  (path-injected import)
    SYNTHESIZER_ENSEMBLE,
    format_synthesizer_input,
    invoke_ensemble,
    parse_plan,
    pipeline_deterministic_chain,
)
from harness import (  # noqa: E402
    PLANNER_ENSEMBLE,
)
from adversarial_battery import ADVERSARIAL_PROMPTS  # noqa: E402
from numerical_fixture import NUMERICAL_FIXTURE, NUMERICAL_REQUESTS  # noqa: E402

RESULTS_FILE = SPIKE_DIR / "results_nu.json"

VALID_ENSEMBLES = {
    "web-searcher",
    "text-summarizer",
    "code-generator",
    "claim-extractor",
    "argument-mapper",
    "prose-improver",
}


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numeric tokens (integers, decimals, %, currency).

    Normalization: strip $ and %, remove thousands commas, keep decimal point.
    Only tokens with >= 3 significant digits OR a decimal point are kept, to
    avoid noise from list indices and single/double-digit incidental numbers.
    """
    text = text or ""
    out: set[str] = set()
    # Match optional $, digits with optional commas, optional decimal, optional %
    for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?%?", text):
        tok = m.group(0).lstrip("$").rstrip("%").replace(",", "")
        if not tok or tok == ".":
            continue
        # keep if has decimal or >= 3 digits
        digits = tok.replace(".", "")
        if "." in tok or len(digits) >= 3:
            out.add(tok)
    return out


# ---------------------------------------------------------------------------
# Surface 1 — Multi-step composition
# ---------------------------------------------------------------------------

_ARG_PASSAGE = (
    "The new transit policy will reduce congestion. We know this because "
    "cities that added bus lanes saw fewer cars downtown. Fewer cars means "
    "less congestion. Therefore the policy is guaranteed to work everywhere, "
    "and anyone who opposes it simply does not care about the environment."
)

_FACTUAL_ARTICLE = (
    "The Antarctic ice sheet holds about 26.5 million cubic kilometers of "
    "ice and contains roughly 60 percent of the world's fresh water. It "
    "covers an area of about 14 million square kilometers. Satellite "
    "measurements since 2002 show the sheet losing mass, with West "
    "Antarctica contributing most of the loss. The continent has no "
    "permanent human population, though research stations host between "
    "1,000 and 5,000 people seasonally. The lowest natural temperature ever "
    "recorded on Earth, -89.2 Celsius, was measured at Vostok Station in 1983."
)

_RAMBLING_TEXT = (
    "So the thing is, basically, what we are trying to do here, at the end "
    "of the day, is that we want, more or less, to make the onboarding "
    "process, you know, smoother, and also faster, and the reason for that "
    "is because, well, users have been, in many cases, kind of confused, "
    "and confused users, as we all know, tend to, sort of, give up, which "
    "is, obviously, not what we want, at all, ever."
)

_ARG_ARTICLE = (
    "Universal basic income would eliminate poverty. Pilot programs in "
    "Finland and Kenya gave participants monthly cash and poverty rates "
    "dropped in those groups. Since poverty dropped in the pilots, scaling "
    "UBI nationally must eliminate poverty entirely. Critics worry about "
    "cost, but the pilots prove cost is not a real concern because the "
    "participants did not stop working. Therefore every country should "
    "adopt UBI immediately and any delay is morally indefensible."
)

# Each: id, steps, request (source text or query), chain (ensemble, strategy)
MULTISTEP_CHAINS: list[dict] = [
    {
        "id": "MS1",
        "steps": 2,
        "kind": "text",
        "request": _ARG_PASSAGE,
        "chain": [("claim-extractor", "request"), ("argument-mapper", "prior")],
    },
    {
        "id": "MS2",
        "steps": 2,
        "kind": "text",
        "request": _FACTUAL_ARTICLE,
        "chain": [("text-summarizer", "request"), ("claim-extractor", "prior")],
    },
    {
        "id": "MS3",
        "steps": 2,
        "kind": "text",
        "request": _RAMBLING_TEXT,
        "chain": [("text-summarizer", "request"), ("prose-improver", "prior")],
    },
    {
        "id": "MS4",
        "steps": 2,
        "kind": "web",
        "request": "the current population of Iceland",
        "chain": [("web-searcher", "request"), ("claim-extractor", "prior")],
    },
    {
        "id": "MS5",
        "steps": 3,
        "kind": "text",
        "request": _ARG_ARTICLE,
        "chain": [
            ("text-summarizer", "request"),
            ("claim-extractor", "prior"),
            ("argument-mapper", "prior"),
        ],
    },
    {
        "id": "MS6",
        "steps": 3,
        "kind": "text",
        "request": _ARG_PASSAGE,
        "chain": [
            ("claim-extractor", "request"),
            ("argument-mapper", "prior"),
            ("prose-improver", "prior"),
        ],
    },
    {
        "id": "MS7",
        "steps": 3,
        "kind": "web",
        "request": "recent findings about the James Webb Space Telescope",
        "chain": [
            ("web-searcher", "request"),
            ("claim-extractor", "prior"),
            ("argument-mapper", "prior"),
        ],
    },
    {
        "id": "MS8",
        "steps": 3,
        "kind": "text",
        "request": _FACTUAL_ARTICLE,
        "chain": [
            ("text-summarizer", "request"),
            ("prose-improver", "prior"),
            ("claim-extractor", "prior"),
        ],
    },
]


def score_multistep(trace: dict[str, Any]) -> dict[str, Any]:
    """Score a chain trace for end-to-end completion + per-step bounding."""
    stages = trace.get("stages", [])
    dispatch_stages = [s for s in stages if s.get("stage") == "dispatch"]
    synth_stage = next((s for s in stages if s.get("stage") == "synthesize"), None)

    per_step = []
    all_steps_bounded = True
    for s in dispatch_stages:
        inv = s.get("invoke", {})
        resp = strip_think(inv.get("response") or "")
        ok = bool(inv.get("ok"))
        nonempty = len(resp.strip()) > 0
        bounded = ok and nonempty
        if not bounded:
            all_steps_bounded = False
        per_step.append(
            {
                "ensemble": s.get("ensemble"),
                "ok": ok,
                "nonempty": nonempty,
                "chars": len(resp),
                "latency_s": round(inv.get("latency_seconds", 0), 1),
            }
        )

    synth_ok = bool(synth_stage and synth_stage.get("invoke", {}).get("ok"))
    final = strip_think(trace.get("final_response") or "")
    final_nonempty = len(final.strip()) > 0

    # Fidelity: no fabricated numbers in final vs union of step outputs
    source_nums: set[str] = set()
    for s in dispatch_stages:
        inv = s.get("invoke", {})
        if inv.get("ok"):
            source_nums |= extract_numbers(strip_think(inv.get("response") or ""))
    final_nums = extract_numbers(final)
    fabricated = sorted(final_nums - source_nums)

    end_to_end_complete = (
        all_steps_bounded and synth_ok and final_nonempty
    )

    return {
        "end_to_end_complete": end_to_end_complete,
        "all_steps_bounded": all_steps_bounded,
        "synth_ok": synth_ok,
        "final_nonempty": final_nonempty,
        "per_step": per_step,
        "final_numbers": sorted(final_nums),
        "source_numbers": sorted(source_nums),
        "fabricated_numbers": fabricated,
    }


def run_surface1() -> list[dict[str, Any]]:
    runs = []
    for spec in MULTISTEP_CHAINS:
        print(f"  [S1] {spec['id']} ({spec['steps']}-step, {spec['kind']})", flush=True)
        trace = pipeline_deterministic_chain(spec["request"], spec["chain"])
        trace["test_id"] = spec["id"]
        trace["steps"] = spec["steps"]
        trace["kind"] = spec["kind"]
        trace["nu_score"] = score_multistep(trace)
        runs.append(trace)
        sc = trace["nu_score"]
        print(
            f"      complete={sc['end_to_end_complete']} "
            f"bounded={sc['all_steps_bounded']} "
            f"fabricated={sc['fabricated_numbers']}",
            flush=True,
        )
    return runs


# ---------------------------------------------------------------------------
# Surface 2 — Production-scale numerical
# ---------------------------------------------------------------------------

# Correct derived values the synthesizer may legitimately compute, by request.
ALLOWED_DERIVED: dict[str, set[str]] = {
    "N1-summary": set(),
    # total revenue = 48217+33905+27461+19338+12774+8651
    "N2-table-reproduce": {"150346"},
    # absolute regional growth deltas FY2024 - FY2023
    "N3-regional-detail": {"8324", "4682", "6729", "1265", "873"},
    "N4-operational": set(),
}

SOURCE_NUMBERS = extract_numbers(NUMERICAL_FIXTURE)


def run_surface2() -> list[dict[str, Any]]:
    runs = []
    for spec in NUMERICAL_REQUESTS:
        rid = spec["id"]
        print(f"  [S2] {rid}", flush=True)
        synth_input = format_synthesizer_input(
            original_request=spec["request"],
            dispatched=["text-summarizer"],
            planned_but_not_run=[],
            dispatch_results=[("text-summarizer", NUMERICAL_FIXTURE)],
        )
        inv = invoke_ensemble(SYNTHESIZER_ENSEMBLE, synth_input)
        trace: dict[str, Any] = {
            "test_id": rid,
            "request": spec["request"],
            "invoke_ok": inv.get("ok"),
            "latency_s": round(inv.get("latency_seconds", 0), 1),
            "error": inv.get("error"),
        }
        if not inv.get("ok"):
            trace["nu_score"] = {"invoke_failed": True}
            runs.append(trace)
            print(f"      INVOKE FAILED: {inv.get('error')}", flush=True)
            continue
        final = strip_think(inv["response"])
        trace["final_response"] = final
        emitted = extract_numbers(final)
        allowed = SOURCE_NUMBERS | ALLOWED_DERIVED.get(rid, set())
        faithful = emitted & allowed
        fabricated = sorted(emitted - allowed)
        fidelity = (len(faithful) / len(emitted)) if emitted else 1.0
        trace["nu_score"] = {
            "emitted_count": len(emitted),
            "faithful_count": len(faithful),
            "fabricated": fabricated,
            "fidelity": round(fidelity, 4),
        }
        runs.append(trace)
        print(
            f"      emitted={len(emitted)} faithful={len(faithful)} "
            f"fidelity={fidelity:.2%} fabricated={fabricated}",
            flush=True,
        )
    return runs


# ---------------------------------------------------------------------------
# Surface 3 — Adversarial routing
# ---------------------------------------------------------------------------

def score_plan_conformance(plan: dict[str, Any]) -> dict[str, Any]:
    """Schema-level conformance check on a parsed plan dict."""
    issues = []
    action = plan.get("action")
    ensemble = plan.get("ensemble")
    rationale = plan.get("rationale")
    if action not in ("dispatch", "direct"):
        issues.append(f"bad action: {action!r}")
    if action == "dispatch":
        if ensemble not in VALID_ENSEMBLES:
            issues.append(f"dispatch ensemble not in registered set: {ensemble!r}")
    elif action == "direct":
        if ensemble not in (None, "null"):
            issues.append(f"direct ensemble not null: {ensemble!r}")
    if not isinstance(rationale, str) or not rationale.strip():
        issues.append("rationale missing or empty")
    # Reject extra/unknown shape only loosely — extra keys tolerated
    return {"conformant": not issues, "issues": issues}


def run_surface3() -> list[dict[str, Any]]:
    runs = []
    for spec in ADVERSARIAL_PROMPTS:
        pid = spec["id"]
        inv = invoke_ensemble(PLANNER_ENSEMBLE, spec["prompt"])
        trace: dict[str, Any] = {
            "test_id": pid,
            "category": spec["category"],
            "prompt": spec["prompt"],
            "acceptable": [list(a) for a in spec["acceptable"]],
            "invoke_ok": inv.get("ok"),
            "latency_s": round(inv.get("latency_seconds", 0), 1),
            "raw_response": inv.get("response"),
            "error": inv.get("error"),
        }
        if not inv.get("ok"):
            trace["nu_score"] = {"conformant": False, "issues": ["invoke failed"],
                                 "judgment_match": False}
            runs.append(trace)
            print(f"  [S3] {pid}: INVOKE FAILED", flush=True)
            continue
        parsed = parse_plan(inv["response"])
        if not parsed.get("ok"):
            trace["parse_error"] = parsed.get("error")
            trace["nu_score"] = {
                "conformant": False,
                "issues": [f"unparseable: {parsed.get('error')}"],
                "judgment_match": False,
            }
            runs.append(trace)
            print(f"  [S3] {pid}: UNPARSEABLE ({parsed.get('error')})", flush=True)
            continue
        plan = parsed["plan"]
        trace["plan"] = plan
        conf = score_plan_conformance(plan)
        # judgment-match
        action = plan.get("action")
        ensemble = plan.get("ensemble")
        ensemble = None if ensemble in (None, "null") else ensemble
        decision = (action, ensemble)
        acceptable = {tuple(a) for a in spec["acceptable"]}
        judgment_match = decision in acceptable
        trace["decision"] = list(decision)
        trace["nu_score"] = {
            "conformant": conf["conformant"],
            "issues": conf["issues"],
            "judgment_match": judgment_match,
        }
        runs.append(trace)
        print(
            f"  [S3] {pid} ({spec['category']}): decision={decision} "
            f"conformant={conf['conformant']} match={judgment_match}",
            flush=True,
        )
    return runs


# ---------------------------------------------------------------------------
# Runner + aggregate
# ---------------------------------------------------------------------------

def aggregate(s1, s2, s3) -> dict[str, Any]:
    # Surface 1
    n1 = len(s1)
    complete = sum(1 for r in s1 if r["nu_score"]["end_to_end_complete"])
    bounded = sum(1 for r in s1 if r["nu_score"]["all_steps_bounded"])
    fab_chains = [r["test_id"] for r in s1 if r["nu_score"]["fabricated_numbers"]]
    s1_rate = complete / n1 if n1 else 0.0

    # Surface 2
    valid_s2 = [r for r in s2 if "fidelity" in r.get("nu_score", {})]
    total_emitted = sum(r["nu_score"]["emitted_count"] for r in valid_s2)
    total_faithful = sum(r["nu_score"]["faithful_count"] for r in valid_s2)
    s2_fidelity = (total_faithful / total_emitted) if total_emitted else 1.0

    # Surface 3
    n3 = len(s3)
    conformant = sum(1 for r in s3 if r["nu_score"].get("conformant"))
    matched = sum(1 for r in s3 if r["nu_score"].get("judgment_match"))
    s3_conf_rate = conformant / n3 if n3 else 0.0
    s3_match_rate = matched / n3 if n3 else 0.0
    nonconformant_ids = [r["test_id"] for r in s3 if not r["nu_score"].get("conformant")]
    mismatch_ids = [r["test_id"] for r in s3 if not r["nu_score"].get("judgment_match")]

    return {
        "surface1_multistep": {
            "n": n1,
            "complete": complete,
            "all_steps_bounded": bounded,
            "completion_rate": round(s1_rate, 4),
            "chains_with_fabrication": fab_chains,
        },
        "surface2_numerical": {
            "runs": len(valid_s2),
            "total_emitted": total_emitted,
            "total_faithful": total_faithful,
            "aggregate_fidelity": round(s2_fidelity, 4),
            "per_run_fidelity": {r["test_id"]: r["nu_score"]["fidelity"] for r in valid_s2},
        },
        "surface3_adversarial": {
            "n": n3,
            "conformant": conformant,
            "conformance_rate": round(s3_conf_rate, 4),
            "judgment_matched": matched,
            "judgment_match_rate": round(s3_match_rate, 4),
            "nonconformant_ids": nonconformant_ids,
            "judgment_mismatch_ids": mismatch_ids,
        },
    }


def main() -> int:
    start = time.time()
    print("=== Spike ν — Surface 1: multi-step composition ===", flush=True)
    s1 = run_surface1()
    print("\n=== Spike ν — Surface 2: production-scale numerical ===", flush=True)
    s2 = run_surface2()
    print("\n=== Spike ν — Surface 3: adversarial routing (40 prompts) ===", flush=True)
    s3 = run_surface3()

    agg = aggregate(s1, s2, s3)
    out = {
        "spike": "cycle-7-nu-long-horizon-ceiling",
        "run_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_seconds": round(time.time() - start, 1),
        "aggregate": agg,
        "surface1_runs": s1,
        "surface2_runs": s2,
        "surface3_runs": s3,
    }
    RESULTS_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS_FILE}")
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
