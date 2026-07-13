#!/usr/bin/env python3
"""not-grounded shape — deterministic echo node (grounded-explain design).

The honest not-grounded message rides the ROUTING decision (classify ->
resolve -> shape -> form_gate -> emit), not the seat envelope; this node
only satisfies the skeleton's dispatch step with a minimal ADR-024
envelope, at zero model cost.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    sys.stdin.read()
    print(json.dumps({"status": "ok", "primary": "Not grounded in this session."}))


if __name__ == "__main__":
    main()
