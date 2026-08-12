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
from typing import NamedTuple

# grounded-explain design (docs/plans/2026-07-12-grounded-explain-design.md):
# a deterministic, non-speculative honest message — never "Refused:", since
# nothing was requested and refused; the turn is answered honestly instead of
# guessed. classify supplies the target basename via ``not_grounded``.
_NOT_GROUNDED_MESSAGE = (
    "No `{target}` in this session (no successful build or read of it), so "
    "I can't explain its internals without guessing. If it's in your "
    "workspace, ask me to read it."
)


class Terminal(NamedTuple):
    """One emit reject/refuse terminal: its wire PREFIX and its ask-outcome-
    ledger MINTING classification (review round 2 major 3 — the project's
    behavior-migrates-downward doctrine applied to emit). Every reject/
    refuse terminal emit can produce declares both here, in ONE place, so
    the caller-side invariant test iterating ``TERMINALS`` catches a newly
    added terminal that doesn't declare (or mis-declares) how the ledger
    should read it, instead of drifting from a hand-maintained parallel
    list. ``mints`` is ``""`` for a terminal that never contributes a
    build-outcome ledger entry (a refusal on a turn that carried no build
    signal — never attributed to a gate the record doesn't support).
    """

    prefix: str
    mints: str


# Recap grounding (docs/plans/2026-07-17-recap-grounding-design.md, #133/#134):
# the prefix-stable templates a build ask degrades to when nothing shipped
# for it — exported so the caller-side ask-outcome ledger can recognize a
# REJECTED/REFUSED entry from the serve's own wire messages, never a
# duplicated regex guessing at this wording. Every prefix is anchored at
# message start only (a startswith check, never equality — the reason text
# after it varies).
#
# Known bound (wrong-accept-hunt target 3): the caller loads these dynamically
# from THIS project's current emit.py at request time (never a version pinned
# into a session), so a session that merely spans a long-running process with
# no restart always matches the live prefix text. The one real skew case is
# changing this literal text itself in a way old wire messages (already
# rendered into a client's history before the edit) won't match — the ledger
# then silently under-reports that one old rejection as "no outcome" rather
# than misreporting it, which is the safe direction to fail. Never rename
# these constants without also considering that old-session cost.
SEAT_CONTRACT_REJECT_PREFIX = "Seat contract not met: "
ACCEPT_GATE_REJECT_PREFIX = "Another round needed: "
# Review round 2 new blocker 2: the invariant is "a ledger entry may claim a
# build outcome only when the turn carried a build ask" — the wire-only
# ledger means the PREFIX itself must encode build-ness, since a read/glob
# refusal renders identically (build=False) whether it answers a build ask's
# discovery round or a bare-symbol explain's. BUILD_REFUSED_PREFIX is used
# EXACTLY on refusal paths where the turn carried classify's build signal
# (threaded as ``is_build_ask``); REFUSED_PREFIX (below) is everything else
# and never mints a ledger entry.
#
# Known historical bound: wire text from a session predating this split (the
# plain "Refused: " prefix) will not mint a refused entry on replay — the
# same safe-direction-to-fail bound already recorded for the other prefixes
# above (a session spanning a template-wording change under-reports, never
# misreports).
BUILD_REFUSED_PREFIX = "Build refused: "
REFUSED_PREFIX = "Refused: "

TERMINALS: dict[str, Terminal] = {
    "seat_contract": Terminal(SEAT_CONTRACT_REJECT_PREFIX, "rejected_contract"),
    "accept_gate": Terminal(ACCEPT_GATE_REJECT_PREFIX, "rejected_gate"),
    "build_refused": Terminal(BUILD_REFUSED_PREFIX, "refused"),
    "refused": Terminal(REFUSED_PREFIX, ""),
}


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
    request fires — one round per seam per turn, never a re-request.

    Review round 2 new blocker 2: a read/glob refusal mints a BUILD-scoped
    ledger entry only when the turn carried classify's build signal
    (``is_build_ask``, threaded from classify through resolve/shape/
    form_gate) — the SAME refusal shape (build=False) otherwise answers a
    bare-symbol explain's discovery round, which is never a build ask.
    """
    is_build_ask = bool(gated.get("is_build_ask", False))
    refused = TERMINALS["build_refused"] if is_build_ask else TERMINALS["refused"]
    read_failed = str(gated.get("read_failed", ""))
    if read_failed:
        return {"finish": True, "content": f"{refused.prefix}{read_failed}"}
    glob_failed = str(gated.get("glob_failed", ""))
    if glob_failed:
        # issue #83 discovery: zero or ambiguous candidates refuse honestly.
        return {"finish": True, "content": f"{refused.prefix}{glob_failed}"}
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
    recall_answer = str(gated.get("recall_answer", ""))
    if recall_answer:
        # #82 deep recall: the deterministic ordinal-selection answer, composed
        # by classify from the chronological ledger. No seat, no guessing.
        return {"finish": True, "content": recall_answer}
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
        outcome = {
            "finish": True,
            "content": f"{TERMINALS['seat_contract'].prefix}{reason}",
        }
    elif build and accept is False:
        # The accept gate rejected the deliverable: route another round rather
        # than ship it, even though it parses (ODP-2, the client owns the loop;
        # ADR-048 §1). Only an explicit False rejects — an ungated turn (accept
        # None) or an accepted one falls through to the normal path.
        reason = gated.get("accept_reason") or "accept gate rejected"
        outcome = {
            "finish": True,
            "content": f"{TERMINALS['accept_gate'].prefix}{reason}",
        }
    elif build and gated.get("valid", False):
        outcome = {
            "finish": False,
            "file": gated.get("file", "solution.py"),
            "content": content,
        }
    elif build:
        # build=True already proves this turn carried a build ask (this
        # branch is a form-gate parse failure, only ever reachable on the
        # build path) — no is_build_ask threading needed, always the
        # build-scoped prefix (review round 2 new blocker 2).
        reason = gated.get("reason", "invalid deliverable")
        outcome = {
            "finish": True,
            "content": f"{TERMINALS['build_refused'].prefix}{reason}",
        }
    else:
        outcome = {"finish": True, "content": content}

    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
