#!/usr/bin/env python3
"""Validate haiku syllable structure (5-7-5 pattern)."""

import json
import sys

def count_syllables(word: str) -> int:
    word = word.lower().strip()
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

    if word.endswith("e") and count > 1:
        count -= 1

    return max(1, count)

def count_line_syllables(line: str) -> int:
    words = line.split()
    return sum(count_syllables(w) for w in words)

def validate_haiku(text: str) -> dict:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    if len(lines) != 3:
        return {
            "valid": False,
            "error": f"Expected 3 lines, got {len(lines)}",
            "syllables": None,
        }

    syllables = [count_line_syllables(line) for line in lines]
    expected = [5, 7, 5]

    all_match = all(s == e for s, e in zip(syllables, expected))

    return {
        "valid": all_match,
        "syllables": syllables,
        "expected": expected,
        "lines": lines,
        "error": None if all_match else f"Syllables {syllables} do not match expected {expected}",
    }

def main() -> None:
    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        input_data = {}

    context = input_data.get("context", {})
    dependencies = input_data.get("dependencies", {})

    haiku_text = None
    for agent_name, agent_output in dependencies.items():
        if isinstance(agent_output, dict):
            response = agent_output.get("response", agent_output.get("data", ""))
        else:
            response = str(agent_output)
        if response:
            haiku_text = response
            break

    if not haiku_text:
        haiku_text = context.get("haiku", "")

    if not haiku_text:
        result = {"valid": False, "error": "No haiku text provided", "syllables": None}
    else:
        result = validate_haiku(haiku_text)

    result["success"] = result.get("valid", False)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()