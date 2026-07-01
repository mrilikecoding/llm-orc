#!/usr/bin/env python3
"""Ω-P3 classify: recover the build chunk and emit the routing booleans.

Root of the build-one sub-ensemble. A fan-out instance delivers its chunk to the
sub-ensemble as a formatted prompt in INPUT_TEXT ("...Chunk content:\n<json>"),
so this node parses the composite {deliverable, contract} back out and emits the
guard booleans (is_code / is_doc) plus the deliverable + contract passed through
for the downstream build nodes.
"""

from __future__ import annotations

import json
import os
import sys

CHUNK_MARKER = "Chunk content:\n"


def _chunk() -> dict:
    text = os.environ.get("INPUT_TEXT", "")
    if not text:
        try:
            text = json.loads(sys.stdin.read()).get("input", "")
        except (json.JSONDecodeError, ValueError, AttributeError):
            text = ""
    body = text.split(CHUNK_MARKER, 1)[-1] if CHUNK_MARKER in text else text
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return {}


def main() -> None:
    chunk = _chunk()
    deliverable = chunk.get("deliverable", {})
    kind = deliverable.get("kind", "")
    print(
        json.dumps(
            {
                "is_code": kind in ("python_module", "python_cli"),
                "is_doc": kind == "markdown_doc",
                "file": deliverable.get("file", ""),
                "kind": kind,
                "deliverable": deliverable,
                "contract": chunk.get("contract", []),
            }
        )
    )


if __name__ == "__main__":
    main()
