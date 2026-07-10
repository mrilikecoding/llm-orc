#!/usr/bin/env python3
"""need-run shape — deterministic echo node (issue #83, run half).

The run request rides the ROUTING decision (classify -> resolve -> shape ->
form_gate -> emit), not the seat envelope; this node only satisfies the
skeleton's dispatch step with a minimal ADR-024 envelope, at zero model cost.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    sys.stdin.read()
    print(json.dumps({"status": "ok", "primary": "Requesting a client test run."}))


if __name__ == "__main__":
    main()
