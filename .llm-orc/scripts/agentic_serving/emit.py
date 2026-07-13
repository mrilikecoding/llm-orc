#!/usr/bin/env python3
"""Serving marshal — emit node (client permission seam).

Terminal node of the serving ensemble: emits the serve outcome the caller maps
onto the client permission seam — a file-write for a valid build deliverable, a
prose finish otherwise (scenarios.md "Per-Turn Serving Handler"; ADR-046 §1,
ADR-034 re-homes the Client-Tool-Action Terminal). A build deliverable the
form-gate refused degrades to a prose finish carrying the refusal reason: the
serve never writes a deliverable that failed destination-validity.

    read failed:     {"finish": true, "content": "Refused: <read_failed reason>"}
    glob failed:     {"finish": true, "content": "Refused: <glob_failed reason>"}
    needs files:     {"finish": false, "reads": ["<path>", ...]}
    needs glob:      {"finish": false, "glob": "<stem>"}
    needs run:       {"finish": false, "run": "<command>"}
    not grounded:    {"finish": true, "content": "No `<target>` in this session..."}
    build + valid:   {"finish": false, "file": "<path>", "content": "<source>"}
    build + refused: {"finish": true, "content": "Refused: <reason>"}
    non-build:       {"finish": true, "content": "<prose>"}

The read/glob/run branches are mutually exclusive by construction — classify
routes each turn to exactly one seam — so their order below only mirrors the
failure-before-request style, never resolves a real conflict.
"""

from __future__ import annotations

import json
import sys

# grounded-explain design (docs/plans/2026-07-12-grounded-explain-design.md):
# a deterministic, non-speculative honest message — never "Refused:", since
# nothing was requested and refused; the turn is answered honestly instead of
# guessed. classify supplies the target basename via ``not_grounded``.
_NOT_GROUNDED_MESSAGE = (
    "No `{target}` in this session (no successful build or read of it), so "
    "I can't explain its internals without guessing. If it's in your "
    "workspace, ask me to read it."
)


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _seam_outcome(gated: dict) -> dict | None:
    """The issue-#83 delegation-seam outcome, or ``None`` when the turn
    rides no seam (a build/prose turn). Failures refuse honestly before any
    request fires — one round per seam per turn, never a re-request."""
    read_failed = str(gated.get("read_failed", ""))
    if read_failed:
        return {"finish": True, "content": f"Refused: {read_failed}"}
    glob_failed = str(gated.get("glob_failed", ""))
    if glob_failed:
        # issue #83 discovery: zero or ambiguous candidates refuse honestly.
        return {"finish": True, "content": f"Refused: {glob_failed}"}
    needs_files = gated.get("needs_files") or []
    if needs_files:
        # delegate the file reads to the client permission seam.
        return {"finish": False, "reads": list(needs_files)}
    needs_glob = str(gated.get("needs_glob", ""))
    if needs_glob:
        # issue #83 discovery: delegate one workspace listing.
        return {"finish": False, "glob": needs_glob}
    needs_run = str(gated.get("needs_run", ""))
    if needs_run:
        # issue #83 run half: delegate one closed-template test run.
        return {"finish": False, "run": needs_run}
    not_grounded = str(gated.get("not_grounded", ""))
    if not_grounded:
        # grounded-explain design: the target named in an explain turn has
        # no visible build or read on the wire — the explainer seat was
        # never called, so there is no speculation path to guard here.
        message = _NOT_GROUNDED_MESSAGE.format(target=not_grounded)
        return {"finish": True, "content": message}
    return None


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

    seam = _seam_outcome(gated)
    if seam is not None:
        outcome = seam
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
