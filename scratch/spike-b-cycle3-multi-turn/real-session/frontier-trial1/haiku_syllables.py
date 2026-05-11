#!/usr/bin/env python3
"""
haiku_syllables.py — Script-agent that validates haiku 5-7-5 syllable structure.

Contract (llm-orc script agent):
  stdin:  JSON {"input": <str>, "parameters": {...}, "context": {"dependencies": {...}}}
  stdout: JSON {"success": bool, "data": {...}, "validation": {...}}

The upstream haiku-writer agent output is expected in:
  context["dependencies"]["haiku-writer"]["response"]   (primary)
  OR input (fallback)
"""

import json
import re
import sys


# ---------------------------------------------------------------------------
# Syllable counting — heuristic vowel-group approach (no external deps)
# ---------------------------------------------------------------------------

VOWELS = "aeiouyAEIOUY"


def count_syllables_word(word: str) -> int:
    """Return estimated syllable count for a single word.

    Heuristic strategy (no external deps):
    1. Strip punctuation.
    2. Count vowel groups (consecutive vowels = 1 nucleus).
    3. Subtract 1 for silent trailing 'e' (consonant+e at end: "lake", "shine").
    4. Subtract 1 for consonant+e+s ending with only 2 vowel nuclei
       (i.e. "shines", "drives" but NOT "ripples" which has 2 real syllables).
    5. Minimum of 1 per non-empty word.

    Known limitations: heuristic accuracy ~85-90% for common English words.
    Multi-syllable words with unusual vowel patterns may miscount.
    """
    word = re.sub(r"[^a-zA-Z]", "", word)
    if not word:
        return 0

    lower = word.lower()

    # Count vowel groups (each contiguous run of vowels = 1 syllable nucleus)
    vowel_groups = re.findall(r"[aeiouy]+", lower)
    count = len(vowel_groups)

    # Rule: silent trailing 'e' preceded by a consonant
    # "lake", "shine", "drive" → consonant before 'e' at end
    if (
        len(lower) > 2
        and lower[-1] == "e"
        and lower[-2] not in VOWELS
        and count > 1
    ):
        count -= 1
    # Rule: consonant+e+s ending where 'e' is silent (verb/noun inflection)
    # "shines" (sh-i-n-e-s → 1), "drives", "leaves"
    # Guard: skip if the consonant before '-es' is itself preceded by another
    # consonant (cluster like "-pples", "-ttles") — those form real syllables.
    # e.g. "ripples": lower[-4]='p' (consonant) → do NOT reduce
    #      "shines":  lower[-4]='i' (vowel)     → reduce
    elif (
        len(lower) > 3
        and lower[-1] == "s"
        and lower[-2] == "e"
        and lower[-3] not in VOWELS
        and (len(lower) < 5 or lower[-4] in VOWELS)
        and count == 2
    ):
        count -= 1

    return max(1, count)


def count_syllables_line(line: str) -> int:
    """Return total syllable count for a line of text."""
    words = line.strip().split()
    return sum(count_syllables_word(w) for w in words)


# ---------------------------------------------------------------------------
# Haiku extraction helpers
# ---------------------------------------------------------------------------

def extract_haiku_lines(text: str) -> list[str]:
    """Extract up to 3 non-empty lines from the haiku text.

    Handles both newline-separated and slash-separated conventions.
    """
    # Try slash-separated first (common in single-line haiku citations)
    if "/" in text and "\n" not in text.strip():
        lines = [l.strip() for l in text.split("/") if l.strip()]
        if len(lines) == 3:
            return lines

    # Fall back to newlines, skipping blanks
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines


def extract_haiku_text(input_data: str, dependencies: dict) -> str:
    """Pull haiku text from dependencies or raw input_data.

    Looks in upstream agent outputs first (any agent that produced a
    non-empty 'response' string), then falls back to raw input_data.
    """
    for agent_name, agent_output in dependencies.items():
        if isinstance(agent_output, dict):
            candidate = agent_output.get("response", agent_output.get("data", ""))
            if candidate and isinstance(candidate, str):
                return candidate
        elif isinstance(agent_output, str) and agent_output.strip():
            return agent_output

    # Fallback: use raw input_data
    return input_data or ""


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

HAIKU_PATTERN = [5, 7, 5]


def validate_haiku(text: str) -> dict:
    """Validate syllable counts for a haiku. Returns structured result."""
    lines = extract_haiku_lines(text)

    if len(lines) != 3:
        return {
            "valid": False,
            "error": f"Expected 3 lines, found {len(lines)}",
            "lines_found": len(lines),
            "raw_text": text,
        }

    counts = [count_syllables_line(line) for line in lines]
    line_results = []
    all_pass = True

    for i, (line, count, expected) in enumerate(zip(lines, counts, HAIKU_PATTERN)):
        passed = count == expected
        all_pass = all_pass and passed
        line_results.append(
            {
                "line_number": i + 1,
                "text": line,
                "syllables_counted": count,
                "syllables_expected": expected,
                "pass": passed,
            }
        )

    return {
        "valid": all_pass,
        "lines": line_results,
        "summary": (
            "Valid 5-7-5 haiku"
            if all_pass
            else f"Invalid haiku: counts are {counts}, expected {HAIKU_PATTERN}"
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point (script-agent contract)
# ---------------------------------------------------------------------------

def main() -> None:
    if not sys.stdin.isatty():
        try:
            config = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    # Support both the llm-orc ScriptAgentInput schema format and
    # the legacy {"input": ..., "context": {"dependencies": ...}} format.
    #
    # ScriptAgentInput (from orchestrator):
    #   {"agent_name": ..., "input_data": ..., "context": {}, "dependencies": {...}}
    #
    # Legacy direct-test format:
    #   {"input": ..., "parameters": ..., "context": {"dependencies": {...}}}
    if "agent_name" in config and "input_data" in config:
        # ScriptAgentInput schema (llm-orc orchestrator path)
        input_data: str = config.get("input_data", "")
        parameters: dict = config.get("parameters", {})
        dependencies: dict = config.get("dependencies", {})
    else:
        # Legacy / direct-test format
        input_data = config.get("input", "")
        parameters = config.get("parameters", {})
        # Dependencies may be at top-level or nested under context
        legacy_context: dict = config.get("context", {})
        dependencies = config.get("dependencies", legacy_context.get("dependencies", {}))

    haiku_text = extract_haiku_text(input_data, dependencies)
    validation = validate_haiku(haiku_text)

    result = {
        "success": True,
        "data": {
            "haiku_text": haiku_text,
            "validation": validation,
        },
        "parameters_received": parameters,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
