#!/usr/bin/env python3
"""
haiku_syllables.py - Validate haiku 5-7-5 syllable structure.

Usage:
  As script agent in ensemble, receives JSON with 'dependencies' containing
  the haiku agent output. Validates and outputs results to stdout.
"""

import json
import re
import sys


def count_syllables(text: str) -> int:
    text = text.lower()
    text = re.sub(r"[^a-z]", "", text)
    if not text:
        return 0
    count = len(re.findall(r"[aeiouy]+", text))
    if text.endswith("e") and count > 1:
        count -= 1
    for suffix in ["le", "les", "ion"]:
        if text.endswith(suffix):
            count += 1
            break
    return max(count, 1)


def main() -> None:
    if not sys.stdin.isatty():
        try:
            config = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    haiku_text = ""
    dependencies = config.get("dependencies", {})
    for agent_output in dependencies.values():
        if isinstance(agent_output, dict):
            haiku_text = agent_output.get("response", agent_output.get("data", ""))
        elif isinstance(agent_output, str):
            haiku_text = agent_output
        if haiku_text:
            break

    if not haiku_text:
        haiku_text = config.get("input_data", "")

    lines = [l.strip() for l in haiku_text.strip().split("\n") if l.strip()]
    results = []
    expected = [5, 7, 5]
    valid = True

    for i, line in enumerate(lines[:3]):
        syllables = count_syllables(line)
        expected_count = expected[i] if i < 3 else 0
        line_valid = syllables == expected_count
        if not line_valid:
            valid = False
        results.append({
            "line": i + 1,
            "text": line,
            "syllables": syllables,
            "expected": expected_count,
            "valid": line_valid
        })

    output = {
        "success": valid,
        "haiku": "\n".join([r["text"] for r in results]),
        "validation": results,
        "all_valid": valid
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
