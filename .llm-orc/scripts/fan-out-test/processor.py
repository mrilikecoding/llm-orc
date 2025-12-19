#!/usr/bin/env python3
"""Processor script that analyzes a single chunk (fan-out instance)."""

import json
import sys


def analyze_chunk(chunk: str, index: int, total: int) -> dict:
    """Analyze a single chunk of text."""
    return {
        "chunk_index": index,
        "total_chunks": total,
        "original_text": chunk,
        "word_count": len(chunk.split()),
        "char_count": len(chunk),
        "has_question": "?" in chunk,
        "has_exclamation": "!" in chunk,
    }


def main() -> None:
    """Read chunk input and output analysis."""
    try:
        # Read input from stdin (JSON format with chunk metadata)
        input_data = json.load(sys.stdin)

        # Fan-out provides: input (the chunk), chunk_index, total_chunks, base_input
        chunk = input_data.get("input", "")
        chunk_index = input_data.get("chunk_index", 0)
        total_chunks = input_data.get("total_chunks", 1)

        if not chunk:
            result = {
                "success": False,
                "error": "No chunk provided",
                "data": None
            }
        else:
            analysis = analyze_chunk(chunk, chunk_index, total_chunks)
            result = {
                "success": True,
                "data": analysis
            }
    except json.JSONDecodeError:
        result = {
            "success": False,
            "error": "Invalid JSON input",
            "data": None
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "data": None
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
