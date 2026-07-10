#!/usr/bin/env python3
"""Serving marshal — emit node (client permission seam).

Terminal node of the serving ensemble: emits the serve outcome the caller maps
onto the client permission seam — a file-write for a valid build deliverable, a
prose finish otherwise (scenarios.md "Per-Turn Serving Handler"; ADR-046 §1,
ADR-034 re-homes the Client-Tool-Action Terminal). A build deliverable the
form-gate refused degrades to a prose finish carrying the refusal reason: the
serve never writes a deliverable that failed destination-validity.

    read failed:     {"finish": true, "content": "Refused: <read_failed reason>"}
    needs files:     {"finish": false, "reads": ["<path>", ...]}
    build + valid:   {"finish": false, "file": "<path>", "content": "<source>"}
    build + refused: {"finish": true, "content": "Refused: <reason>"}
    non-build:       {"finish": true, "content": "<prose>"}
"""

from __future__ import annotations

import json
import sys


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        gated = json.loads(_response(deps.get("form_gate", {})))
    except json.JSONDecodeError:
        gated = {}
    if not isinstance(gated, dict):
        gated = {}

    build = bool(gated.get("build", False))
    content = str(gated.get("content", ""))
    accept = gated.get("accept")
    seat_admitted = gated.get("seat_admitted")
    needs_files = gated.get("needs_files") or []
    read_failed = str(gated.get("read_failed", ""))

    if read_failed:
        # issue #83: one read round per turn — a failed request refuses
        # honestly, never re-requests.
        outcome = {"finish": True, "content": f"Refused: {read_failed}"}
    elif needs_files:
        # issue #83: delegate the file reads to the client permission seam.
        outcome = {"finish": False, "reads": list(needs_files)}
    elif seat_admitted is False:
        # The seat's output did not meet its own seat-owned contract (WP-E8;
        # ADR-046 §2). Refuse before shipping — a distinct, higher-priority gate
        # than the loop-level accept below. Only an explicit False refuses; an
        # ungated seat (None) or an admitted one falls through.
        reason = gated.get("seat_contract_reason") or "seat contract not met"
        outcome = {"finish": True, "content": f"Seat contract not met: {reason}"}
    elif build and accept is False:
        # The accept gate rejected the deliverable: route another round rather
        # than ship it, even though it parses (ODP-2, the client owns the loop;
        # ADR-048 §1). Only an explicit False rejects — an ungated turn (accept
        # None) or an accepted one falls through to the normal path.
        reason = gated.get("accept_reason") or "accept gate rejected"
        outcome = {"finish": True, "content": f"Another round needed: {reason}"}
    elif build and gated.get("valid", False):
        outcome = {
            "finish": False,
            "file": gated.get("file", "solution.py"),
            "content": content,
        }
    elif build:
        outcome = {
            "finish": True,
            "content": f"Refused: {gated.get('reason', 'invalid deliverable')}",
        }
    else:
        outcome = {"finish": True, "content": content}

    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
