#!/usr/bin/env python3
"""Classify input text into topics (statements) and questions.

Produces keyed JSON output for input_key routing:
  {"topics": ["..."], "questions": ["..."]}
"""

import json
import re
import sys


def classify(text: str) -> dict[str, list[str]]:
    """Split text into topics (statements) and questions."""
    topics: list[str] = []
    questions: list[str] = []

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence.endswith("?"):
            questions.append(sentence)
        else:
            topics.append(sentence)

    return {"topics": topics, "questions": questions}


def main() -> None:
    """Read input and output classified results."""
    try:
        raw = sys.stdin.read().strip()

        # Try JSON first (executor sends ScriptAgentInput)
        text = ""
        try:
            data = json.loads(raw)
            text = data.get("input_data", data.get("input", ""))
        except (json.JSONDecodeError, AttributeError):
            # Fall back to raw text
            text = raw

        if not text:
            print(json.dumps({"success": False, "error": "No input text"}))
            return

        result = classify(text)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "topics": [], "questions": []}))


if __name__ == "__main__":
    main()
