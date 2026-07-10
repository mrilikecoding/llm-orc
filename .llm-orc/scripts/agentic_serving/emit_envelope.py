#!/usr/bin/env python3
"""code-seat terminal — emit an ADR-024 ``DispatchEnvelope`` for the deliverable.

The code-generation flow (``code-generator``) ends in a chatty synthesizer
response wrapping the code in prose. This deterministic node extracts the code
deliverable and emits it as an ADR-024 ``DispatchEnvelope`` so the serving
marshal (the shape node) consumes a faithful structured artifact rather than
guessing at prose (scenarios.md "the marshal node consumes the seat's real
common I/O envelope"; ADR-024; ADR-046 §2). The seat owns the deliverable
CONTENT; the serving classify owns the DESTINATION path — shape combines them.
"""

from __future__ import annotations

import json
import sys

from _helpers import extract_code as _extract_code
from _helpers import payload as __payload
from _helpers import response as _response
from _helpers import terminal as _terminal


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}








def main() -> None:
    deps_map = __payload(sys.stdin.read().strip()).get("dependencies", {})
    deps = deps_map if isinstance(deps_map, dict) else {}
    generated = _terminal(_response(deps.get("generate", {})))
    code = _extract_code(generated, drop_test_blocks=True)
    summary = code.splitlines()[0][:80] if code.strip() else "code deliverable"

    envelope = {
        "status": "success",
        "primary": code,
        "structured": {"content": code},
        "artifacts": [
            {
                "content_type": "text/x-python",
                "content": code,
                "summary": summary,
            }
        ],
        "diagnostics": {"ensemble": "code-seat"},
    }
    print(json.dumps(envelope))


if __name__ == "__main__":
    main()
