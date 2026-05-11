"""Score each Spike C trial against the 5 ground-truth issues.

For each trial output, check whether the review surfaced each of the 5 known
issues. Detection is based on keyword/phrase patterns derived from the
ground-truth descriptions. Multiple per-issue heuristics give defense-in-depth
against false negatives.

Output: per-trial detection matrix + per-arm aggregation.

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TRIALS_DIR = Path(__file__).parent / "trials"


def load_trial_text(path: Path) -> str:
    """Extract the review text from a trial output. JSON outputs need parsing."""
    raw = path.read_text()
    if path.suffix == ".json":
        try:
            data = json.loads(raw)
            # Arm B JSON: results dict with code-reviewer agent's response
            if "results" in data:
                reviewer = data["results"].get("code-reviewer", {})
                analyzer = data["results"].get("diff-analyzer", {})
                # Combine: include the analyzer output (script-agent) + reviewer's prose
                return (analyzer.get("response", "") + "\n\n" + reviewer.get("response", ""))
            return raw
        except json.JSONDecodeError:
            return raw
    return raw


def detect_issue1_off_by_one(text: str) -> tuple[bool, list[str]]:
    """ISSUE-1: off-by-one in check_limit() (uses > instead of >=)."""
    text_l = text.lower()
    matches = []
    # Strong signals
    patterns = [
        (r"off[- ]by[- ]one", "off-by-one phrase"),
        (r"check_limit.{0,80}>=", "check_limit + >="),
        (r"`>=`|use\s+>=|should be\s+>=", ">= correction recommended"),
        (r"strict(ly)?\s+greater\s+than", "strict-greater-than identified"),
        (r"exactly\s+(at\s+)?the\s+limit|at\s+the\s+limit", "at-the-limit boundary"),
        (r"==\s*self\.limit|==\s*limit|equals?\s+the\s+limit", "equality-with-limit boundary"),
        (r"boundary.{0,40}(check_limit|limit)", "boundary semantics flagged"),
    ]
    for pat, label in patterns:
        if re.search(pat, text_l):
            matches.append(label)
    return (len(matches) >= 1, matches)


def detect_issue2_apikey_in_logs(text: str) -> tuple[bool, list[str]]:
    """ISSUE-2: api_key included in logger.warning output."""
    text_l = text.lower()
    matches = []
    patterns = [
        (r"api[- ]?key.{0,80}log(?!ic)", "api_key + log"),
        (r"log.{0,80}api[- ]?key", "log + api_key"),
        (r"credential.{0,40}(log|leak|expos)", "credential leak language"),
        (r"sensitive.{0,40}(log|leak|expos)", "sensitive-data leak language"),
        (r"plaintext.{0,40}log", "plaintext + log"),
        (r"truncate|mask|hash.{0,40}(api|key|credential)", "mask/truncate recommendation"),
    ]
    for pat, label in patterns:
        if re.search(pat, text_l):
            matches.append(label)
    return (len(matches) >= 1, matches)


def detect_issue3_type_annotation(text: str) -> tuple[bool, list[str]]:
    """ISSUE-3: limit: int = None should be limit: int | None = None."""
    text_l = text.lower()
    matches = []
    patterns = [
        (r"int\s*\|\s*none", "int | None correction"),
        (r"optional\[int\]", "Optional[int] mention"),
        (r"limit.{0,30}int.{0,30}None", "limit/int/None proximity"),
        (r"none\s+is\s+not\s+(an?\s+)?int", "None-is-not-int reasoning"),
        (r"annotation.{0,40}(violat|mismatch|incorrect|wrong)", "annotation mismatch language"),
        (r"mypy.{0,40}(strict|reject|fail|error)", "mypy strict reference"),
        (r"type\s+(error|mismatch|violat).{0,40}limit", "type error on limit"),
    ]
    for pat, label in patterns:
        if re.search(pat, text_l):
            matches.append(label)
    return (len(matches) >= 1, matches)


def detect_issue4_test_gap(text: str) -> tuple[bool, list[str]]:
    """ISSUE-4: tests deferred to follow-up PR."""
    text_l = text.lower()
    matches = []
    patterns = [
        (r"test\s+coverage", "test coverage phrase"),
        (r"tests?\s+(are\s+|will\s+be\s+|to\s+be\s+|need\s+|should\s+be\s+)?(added|written|present|deferred)", "test addition language"),
        (r"defer.{0,40}test", "defer + test"),
        (r"test.{0,40}defer", "test + defer"),
        (r"(no|zero|missing|absent)\s+test", "no/zero/missing test"),
        (r"follow.?up.{0,40}test|test.{0,40}follow.?up", "follow-up + test"),
        (r"tdd|test.driven", "TDD reference"),
        (r"claude\.md.{0,40}test|test.{0,40}claude\.md", "CLAUDE.md test reference"),
    ]
    for pat, label in patterns:
        if re.search(pat, text_l):
            matches.append(label)
    return (len(matches) >= 1, matches)


def detect_issue5_cross_file(text: str) -> tuple[bool, list[str]]:
    """ISSUE-5: DEFAULT_BUDGET_LIMIT vs DEFAULT_MAX_TOKEN_LIMIT drift."""
    text_l = text.lower()
    matches = []
    patterns = [
        (r"default_max_token_limit", "DEFAULT_MAX_TOKEN_LIMIT mention"),
        (r"orchestrator_config\.py", "orchestrator_config.py reference"),
        (r"cross[- ]?file", "cross-file phrase"),
        (r"drift", "drift language"),
        (r"two\s+(constants?|sources?|values?).{0,40}(diverge|mismatch|same)", "two-sources diverge"),
        (r"50_?000_?000|50\s*million", "50M actual value reference"),
        (r"100_?000(?!_)", "100K diff value reference"),
        (r"import.{0,40}(constant|value|threshold|limit).{0,40}(orchestr|config)", "import constant from config"),
        (r"single\s+source\s+of\s+truth", "single-source-of-truth language"),
        (r"redeclar|re-declar|duplicat", "duplicate declaration language"),
    ]
    for pat, label in patterns:
        if re.search(pat, text_l):
            matches.append(label)
    return (len(matches) >= 1, matches)


ISSUE_DETECTORS = [
    ("ISSUE-1 (off-by-one)", detect_issue1_off_by_one),
    ("ISSUE-2 (api_key in logs)", detect_issue2_apikey_in_logs),
    ("ISSUE-3 (type annotation)", detect_issue3_type_annotation),
    ("ISSUE-4 (test gap)", detect_issue4_test_gap),
    ("ISSUE-5 (cross-file)", detect_issue5_cross_file),
]


def score_trial(path: Path) -> dict:
    text = load_trial_text(path)
    results: dict[str, dict] = {}
    for label, detector in ISSUE_DETECTORS:
        caught, signals = detector(text)
        results[label] = {"caught": caught, "signals": signals}
    return {
        "trial": path.name,
        "char_count": len(text),
        "results": results,
        "total_caught": sum(1 for r in results.values() if r["caught"]),
    }


def main() -> None:
    trial_files = sorted(TRIALS_DIR.glob("arm-*"))
    if not trial_files:
        print("No trial files found")
        return

    all_scores = []
    for path in trial_files:
        score = score_trial(path)
        all_scores.append(score)

    # Per-trial table
    print("=" * 100)
    print(f"{'Trial':<55} {'I-1':>4} {'I-2':>4} {'I-3':>4} {'I-4':>4} {'I-5':>4} {'Total':>6}")
    print("=" * 100)
    for s in all_scores:
        row = f"{s['trial']:<55}"
        for label, _ in ISSUE_DETECTORS:
            mark = "✓" if s["results"][label]["caught"] else "·"
            row += f" {mark:>4}"
        row += f" {s['total_caught']:>5}/5"
        print(row)
    print("=" * 100)

    # Per-arm aggregation
    print("\nPER-ARM AGGREGATION")
    print("=" * 60)
    arm_groups: dict[str, list[dict]] = {}
    for s in all_scores:
        if "arm-a-cheap-bare" in s["trial"]:
            arm = "Arm A (cheap-bare)"
        elif "arm-b-cheap-with-ensemble" in s["trial"]:
            arm = "Arm B (cheap-with-ensemble)"
        elif "arm-c-frontier" in s["trial"]:
            arm = "Arm C (frontier-bare)"
        else:
            arm = "unknown"
        arm_groups.setdefault(arm, []).append(s)

    for arm, scores in arm_groups.items():
        n = len(scores)
        print(f"\n{arm} (n={n})")
        for label, _ in ISSUE_DETECTORS:
            caught_count = sum(1 for s in scores if s["results"][label]["caught"])
            print(f"  {label}: {caught_count}/{n} trials")
        avg = sum(s["total_caught"] for s in scores) / n
        print(f"  Mean issues caught per trial: {avg:.2f}/5")

    # Save full detail
    out_path = TRIALS_DIR.parent / "scoring-results.json"
    out_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nFull detail saved to: {out_path}")


if __name__ == "__main__":
    main()
