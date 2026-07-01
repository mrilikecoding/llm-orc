#!/usr/bin/env python3
"""Ω-P3 classify (live): recover the chunk and build the real dispatch prompt.

Root of the live build-one sub-ensemble. Recovers the {deliverable, contract}
chunk a fan-out instance delivers in INPUT_TEXT, then emits the routing booleans
plus the full per-file build prompt (dispatch_input) for the cheap-local
generator — built via the Ω-E build_dispatch_input so the real builders get the
same file spec + sibling APIs the Ω-E driver gave them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scratch" / "spike-omega-p3"))
from buildlib import build_dispatch_input  # noqa: E402

CHUNK_MARKER = "Chunk content:\n"


def _chunk() -> dict:
    text = os.environ.get("INPUT_TEXT", "")
    if not text:
        try:
            text = str(json.loads(sys.stdin.read()).get("input_data", ""))
        except (json.JSONDecodeError, ValueError, AttributeError):
            text = ""
    body = text.split(CHUNK_MARKER, 1)[-1] if CHUNK_MARKER in text else text
    try:
        chunk, _ = json.JSONDecoder().raw_decode(body.lstrip())
    except (json.JSONDecodeError, ValueError):
        return {}
    return chunk if isinstance(chunk, dict) else {}


def main() -> None:
    chunk = _chunk()
    deliverable = chunk.get("deliverable", {})
    contract = chunk.get("contract", [])
    kind = deliverable.get("kind", "")
    print(
        json.dumps(
            {
                "is_code": kind in ("python_module", "python_cli"),
                "is_doc": kind == "markdown_doc",
                "file": deliverable.get("file", ""),
                "kind": kind,
                "dispatch_input": build_dispatch_input(deliverable, contract),
            }
        )
    )


if __name__ == "__main__":
    main()
