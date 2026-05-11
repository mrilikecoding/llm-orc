#!/usr/bin/env python3
"""
haiku_syllables.py - Validate haiku 5-7-5 syllable structure.

Input (JSON from stdin):
  {"dependencies": {"haiku_writer": {"response": "..."}}}

Output (JSON to stdout):
  {"success": bool, "valid": bool, "counts": [int, int, int], "lines": [str, str, str]}
"""

import json
import sys
import re


def count_syllables(word: str) -> int:
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def count_line_syllables(line: str) -> int:
    words = re.findall(r"[a-z]+", line.lower())
    return sum(count_syllables(w) for w in words)


def main() -> None:
    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}

    haiku_text = ""
    deps = input_data.get("dependencies", {})
    if "haiku_writer" in deps:
        haiku_writer = deps["haiku_writer"]
        if isinstance(haiku_writer, dict):
            haiku_text = haiku_writer.get("response", "")
        else:
            haiku_text = str(haiku_writer)

    lines = [l.strip() for l in haiku_text.strip().split("\n") if l.strip()]
    expected = [5, 7, 5]

    if len(lines) < 3:
        result = {"success": True, "valid": False, "counts": [], "lines": lines,
                  "error": f"Expected 3 lines, got {len(lines)}"}
    else:
        counts = [count_line_syllables(lines[i]) for i in range(3)]
        valid = counts == expected
        result = {"success": True, "valid": valid, "counts": counts, "lines": lines}
        if not valid:
            result["error"] = f"Line syllables: {counts}, expected: {expected}"

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
