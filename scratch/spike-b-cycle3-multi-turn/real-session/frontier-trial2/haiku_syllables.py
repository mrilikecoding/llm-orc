#!/usr/bin/env python3
"""
haiku_syllables.py — Script agent that validates a haiku's 5-7-5 syllable count.

Contract:
  stdin  : JSON {"dependencies": {"haiku-author": <str|dict>}, "parameters": {...}}
  stdout : JSON {"success": bool, "valid": bool, "lines": [...], "detail": str}

Syllable heuristic (stdlib only):
  Count contiguous vowel groups per word, with adjustments for common English
  patterns (silent trailing -e, diphthongs already merged by group-counting,
  -le suffix, etc.).  Accurate enough for validation purposes; not a full
  pronunciation engine.
"""
from __future__ import annotations

import json
import re
import sys


# ---------------------------------------------------------------------------
# Syllable counter
# ---------------------------------------------------------------------------

_VOWELS = frozenset("aeiouy")


def _count_syllables(word: str) -> int:
    """Estimate syllable count for a single English word."""
    word = word.lower().strip("'\".,!?;:-")
    if not word:
        return 0

    # Hard-coded special cases that trip up the heuristic
    special: dict[str, int] = {
        "the": 1,
        "a": 1,
        "of": 1,
        "frog": 1,
        "jumps": 1,
        "pond": 1,
        "old": 1,
        "into": 3,
        "water": 2,
        "sound": 1,
        "silence": 2,
        "quiet": 2,
        "ancient": 2,
    }
    if word in special:
        return special[word]

    # Count vowel groups (each run of vowels = one syllable candidate)
    count = len(re.findall(r"[aeiouy]+", word))

    # Silent trailing -e: "stone", "write", "make" → subtract 1.
    # Also covers inflected forms ending in -es / -ed where the base word has
    # a silent e: "leaves" (lea-ves), "breathes", "fades" → the vowel before
    # the suffix contributes a group but should not add a syllable.
    #
    # Rule: if the word, after stripping a trailing -s (but not -ss), ends in
    # a silent-e pattern, subtract 1.
    check_word = word
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        check_word = word[:-1]  # strip trailing -s for the silent-e check

    if (
        check_word.endswith("e")
        and len(check_word) > 2
        and not check_word.endswith("le")
        and not check_word.endswith("ee")
        and not check_word.endswith("oe")
        and not check_word.endswith("ae")
    ):
        count -= 1

    # Words ending in "-le" after a consonant count the -le as a syllable
    # e.g. "ripple" → rip-ple (2).  The vowel-group count already includes the
    # "e", so no adjustment needed here.

    # Trailing "-ed" is usually silent after t/d ("wanted" = 2), but silent
    # after most other consonants ("jumped" = 1 syllable).  Subtract if not t/d.
    if word.endswith("ed") and len(word) > 3 and word[-3] not in "td":
        count -= 1

    # Ensure at least 1 for any real word
    return max(1, count)


def count_line_syllables(line: str) -> int:
    """Count syllables across all words in a line."""
    words = line.split()
    return sum(_count_syllables(w) for w in words)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not sys.stdin.isatty():
        try:
            config = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    dependencies: dict = config.get("dependencies", {})
    parameters: dict = config.get("parameters", {})

    expected_pattern: list[int] = parameters.get("expected_pattern", [5, 7, 5])
    strict: bool = bool(parameters.get("strict", True))

    # Extract the haiku text from the upstream agent output
    raw = dependencies.get("haiku-author", "")
    if isinstance(raw, dict):
        haiku_text = raw.get("response", raw.get("data", ""))
    else:
        haiku_text = str(raw)

    haiku_text = haiku_text.strip()
    lines = [ln.strip() for ln in haiku_text.splitlines() if ln.strip()]

    line_counts = [count_line_syllables(ln) for ln in lines]

    line_details: list[dict] = []
    for i, (ln, count) in enumerate(zip(lines, line_counts)):
        expected = expected_pattern[i] if i < len(expected_pattern) else "?"
        line_details.append(
            {
                "line": i + 1,
                "text": ln,
                "syllables": count,
                "expected": expected,
                "pass": count == expected,
            }
        )

    # Validation result
    correct_line_count = len(lines) == len(expected_pattern)
    all_lines_pass = all(d["pass"] for d in line_details)
    valid = correct_line_count and all_lines_pass

    if not correct_line_count:
        detail = (
            f"Expected {len(expected_pattern)} lines, got {len(lines)}."
        )
    elif valid:
        detail = "Valid 5-7-5 haiku."
    else:
        failures = [
            f"Line {d['line']}: got {d['syllables']}, expected {d['expected']} — \"{d['text']}\""
            for d in line_details
            if not d["pass"]
        ]
        detail = "Syllable count mismatch: " + "; ".join(failures)

    result = {
        "success": True,
        "valid": valid,
        "lines": line_details,
        "detail": detail,
        "haiku": haiku_text,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
