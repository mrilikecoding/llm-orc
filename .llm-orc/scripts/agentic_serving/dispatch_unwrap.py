#!/usr/bin/env python3
"""Peel a dispatch node's wrapper back to the child's terminal envelope.

A ``dispatch:`` node responds with the child ensemble's full result dict
(``{"ensemble": ..., "deliverable": <terminal response>, "results": ...}``).
The retry loop's ``until``/``carry`` predicates and the serving marshal
expect the bare ADR-024 envelope the round's terminal emits — this node
restores that contract deterministically (the dispatch sibling of
``loop_unwrap``).
"""

from __future__ import annotations

import json
import sys

from _helpers import deps as _deps
from _helpers import payload as _payload
from _helpers import response as _response


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    envelope: dict[str, object] = {}
    for dep in _deps(payload).values():
        try:
            wrapper = json.loads(_response(dep))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(wrapper, dict):
            continue
        deliverable = wrapper.get("deliverable")
        if isinstance(deliverable, str):
            try:
                parsed = json.loads(deliverable)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                envelope = parsed
                break
    print(json.dumps(envelope))


if __name__ == "__main__":
    main()
