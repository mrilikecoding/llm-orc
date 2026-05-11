#!/usr/bin/env python3
"""Validate haiku 5-7-5 syllable structure."""

import json
import sys
import re


def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    word = re.sub(r"[^a-z']", "", word)
    if not word:
        return 0
    count = 0
    vowels = "aeiouy"
    prev_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e"):
        count = max(1, count - 1)
    if count == 0:
        count = 1
    return count


def validate_line(line: str, expected: int) -> tuple[bool, int]:
    words = line.split()
    count = sum(count_syllables(w) for w in words)
    return count == expected, count


def main() -> None:
    if not sys.stdin.isatty():
        try:
            data = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    parameters = data.get("parameters", {})
    dependencies = data.get("dependencies", {})
    haiku_text = parameters.get("haiku", "")

    if not haiku_text and dependencies:
        for dep_output in dependencies.values():
            if isinstance(dep_output, dict):
                haiku_text = dep_output.get("response", "") or dep_output.get("data", "")
            else:
                haiku_text = str(dep_output)
            break

    lines = [l.strip() for l in haiku_text.strip().split("\n") if l.strip()]

    if len(lines) != 3:
        result = {
            "success": False,
            "error": f"Expected 3 lines, got {len(lines)}",
            "syllable_counts": [],
        }
        print(json.dumps(result, indent=2))
        return

    expected = [5, 7, 5]
    syllable_counts = []
    all_valid = True
    details = []

    for i, (line, exp) in enumerate(zip(lines, expected)):
        valid, count = validate_line(line, exp)
        syllable_counts.append(count)
        all_valid = all_valid and valid
        details.append(f"Line {i+1}: {count}/{exp} syllables - {'OK' if valid else 'FAIL'}")

    result = {
        "success": all_valid,
        "syllable_counts": syllable_counts,
        "valid": all_valid,
        "expected": expected,
        "haiku": haiku_text,
        "details": details,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()