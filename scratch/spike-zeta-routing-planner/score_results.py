#!/usr/bin/env python3
"""Score Spike ζ results: JSON conformance + capability-match correctness."""

import json
import re
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
RESULTS = SPIKE_DIR / "results.json"
SCORED = SPIKE_DIR / "scored.json"

VALID_ENSEMBLES = {
    "web-searcher",
    "text-summarizer",
    "code-generator",
    "claim-extractor",
    "argument-mapper",
    "prose-improver",
}


def extract_planner_text(run: dict) -> str:
    output = run.get("output", {})
    if isinstance(output, dict):
        return (
            output.get("results", {}).get("planner", {}).get("response", "") or ""
        )
    return ""


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_object(text: str) -> tuple[dict | None, str]:
    text = strip_think_blocks(text)
    json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = json_blocks + re.findall(r"\{[^{}]*?\"action\"[^{}]*?\}", text, flags=re.DOTALL)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed, "extracted"
        except json.JSONDecodeError:
            continue

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed, "raw"
        except json.JSONDecodeError:
            pass

    return None, "none"


def score_run(run: dict) -> dict:
    expected_action = run["expected_action"]
    expected_ensemble = run["expected_ensemble"]
    if expected_ensemble == "null":
        expected_ensemble = None

    planner_text = extract_planner_text(run)
    parsed, parse_status = extract_json_object(planner_text)

    json_conformant = parsed is not None
    actual_action = parsed.get("action") if parsed else None
    actual_ensemble = parsed.get("ensemble") if parsed else None
    actual_rationale = parsed.get("rationale") if parsed else None

    schema_valid = (
        json_conformant
        and actual_action in {"dispatch", "direct"}
        and (
            (actual_action == "dispatch" and actual_ensemble in VALID_ENSEMBLES)
            or (actual_action == "direct" and actual_ensemble is None)
        )
        and isinstance(actual_rationale, str)
        and len(actual_rationale) > 0
    )

    action_correct = json_conformant and actual_action == expected_action
    if expected_ensemble is None and run["expected_action"] == "dispatch":
        ensemble_correct = json_conformant and actual_ensemble in VALID_ENSEMBLES
        ensemble_correct_note = "ambiguous-accept-any-valid"
    else:
        ensemble_correct = json_conformant and actual_ensemble == expected_ensemble
        ensemble_correct_note = None

    return {
        "id": run["id"],
        "shape": run["shape"],
        "input_preview": run["input"][:80],
        "expected_action": expected_action,
        "expected_ensemble": expected_ensemble,
        "parse_status": parse_status,
        "json_conformant": json_conformant,
        "schema_valid": schema_valid,
        "actual_action": actual_action,
        "actual_ensemble": actual_ensemble,
        "actual_rationale": actual_rationale,
        "action_correct": action_correct,
        "ensemble_correct": ensemble_correct,
        "ensemble_correct_note": ensemble_correct_note,
        "latency_seconds": run["latency_seconds"],
        "raw_planner_text_preview": planner_text[:300],
    }


def main() -> int:
    with RESULTS.open() as f:
        data = json.load(f)

    scored = [score_run(run) for run in data["runs"]]

    n = len(scored)
    json_conformant_count = sum(1 for s in scored if s["json_conformant"])
    schema_valid_count = sum(1 for s in scored if s["schema_valid"])
    action_correct_count = sum(1 for s in scored if s["action_correct"])
    ensemble_correct_count = sum(1 for s in scored if s["ensemble_correct"])
    latencies = [s["latency_seconds"] for s in scored]
    latencies.sort()

    summary = {
        "total": n,
        "json_conformant": json_conformant_count,
        "json_conformant_pct": round(100 * json_conformant_count / n, 1),
        "schema_valid": schema_valid_count,
        "schema_valid_pct": round(100 * schema_valid_count / n, 1),
        "action_correct": action_correct_count,
        "action_correct_pct": round(100 * action_correct_count / n, 1),
        "ensemble_correct": ensemble_correct_count,
        "ensemble_correct_pct": round(100 * ensemble_correct_count / n, 1),
        "latency_min_s": round(latencies[0], 2),
        "latency_p50_s": round(latencies[n // 2], 2),
        "latency_p90_s": round(latencies[int(n * 0.9)], 2),
        "latency_max_s": round(latencies[-1], 2),
        "latency_mean_s": round(sum(latencies) / n, 2),
    }

    out = {
        "spike": data["spike"],
        "run_date": data["run_date"],
        "ensemble": data["ensemble"],
        "summary": summary,
        "scored": scored,
    }

    with SCORED.open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print("Per-prompt:")
    for s in scored:
        flags = []
        if not s["json_conformant"]:
            flags.append("JSON-FAIL")
        if not s["schema_valid"]:
            flags.append("SCHEMA-FAIL")
        if not s["action_correct"]:
            flags.append(f"ACTION-MISMATCH({s['actual_action']}vs{s['expected_action']})")
        if not s["ensemble_correct"]:
            flags.append(
                f"ENSEMBLE-MISMATCH({s['actual_ensemble']}vs{s['expected_ensemble']})"
            )
        flag_str = " ".join(flags) if flags else "OK"
        print(f"  {s['id']:42s} | {flag_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
