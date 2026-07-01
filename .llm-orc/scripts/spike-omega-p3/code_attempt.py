#!/usr/bin/env python3
"""Ω-P3 code-attempt: one build attempt inside the per-file build loop.

Body root of the code-build loop. A loop body receives the loop node's input,
which for build-code is the engine's *enhanced input*: the fan-out chunk
("...Chunk content:\n{deliverable, contract}") wrapped again in dependency prose.
So recovering the deliverable means locating the chunk marker and JSON-decoding
just the object that follows (a naive parse over the whole blob fails — two JSON
blobs plus prose). Stand-in for a real cheap code generator; carry-driven retry
is proven by spike Ω-loop, so this emits valid content on the first attempt.
"""

from __future__ import annotations

import json
import os
import sys

CHUNK_MARKER = "Chunk content:\n"


def _input_text() -> str:
    text = os.environ.get("INPUT_TEXT")
    if text:
        return text
    try:
        return str(json.loads(sys.stdin.read()).get("input_data", ""))
    except (json.JSONDecodeError, ValueError, AttributeError):
        return ""


def _deliverable_file() -> str:
    text = _input_text()
    idx = text.find(CHUNK_MARKER)
    if idx == -1:
        return ""
    start = idx + len(CHUNK_MARKER)
    try:
        chunk, _ = json.JSONDecoder().raw_decode(text[start:].lstrip())
    except (json.JSONDecodeError, ValueError):
        return ""
    return chunk.get("deliverable", {}).get("file", "") if isinstance(chunk, dict) else ""


def main() -> None:
    file = _deliverable_file()
    print(
        json.dumps(
            {
                "file": file,
                "tier": "code",
                "content": f"# {file}\n# (built by the CODE tier via the build loop)\n",
            }
        )
    )


if __name__ == "__main__":
    main()
