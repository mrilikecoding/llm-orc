#!/usr/bin/env python3
"""build-round router — pick the round shape from the carried input (issue #100).

Head node of the retry loop's body. Round 1's input is the turn itself; a
rejected round whose tests collected and were judged adequate carries them
under the HELD TESTS sentinel (composed by build_gated_envelope). The route
is deterministic: sentinel present -> the code-only held round (tests are
the spec), else the full fresh TDD round. A sentinel in user-authored turn
text worst-cases into a held round the gate rejects — degraded, never a
wrong accept.

Emits JSON: {target, round_input}
"""

from __future__ import annotations

import json
import sys

from _helpers import HELD_TESTS_MARKER as _HELD_MARKER
from _helpers import payload as _payload


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    round_input = ""
    for key in ("input_data", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            round_input = value
            break
    target = (
        "build-code-round" if _HELD_MARKER in round_input else "build-gated-round"
    )
    print(json.dumps({"target": target, "round_input": round_input}))


if __name__ == "__main__":
    main()
