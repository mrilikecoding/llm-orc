#!/usr/bin/env python3
"""Chunker script that splits input text into sentences for fan-out processing."""

import json
import re
import sys


def chunk_text(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting on . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def main() -> None:
    """Read input and output chunks as JSON array."""
    try:
        # Read input from stdin (JSON format)
        input_data = json.load(sys.stdin)
        text = input_data.get("input", "")

        if not text:
            result = {
                "success": False,
                "error": "No input text provided",
                "data": []
            }
        else:
            chunks = chunk_text(text)
            result = {
                "success": True,
                "data": chunks  # This array will trigger fan-out
            }
    except json.JSONDecodeError:
        result = {
            "success": False,
            "error": "Invalid JSON input",
            "data": []
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "data": []
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
