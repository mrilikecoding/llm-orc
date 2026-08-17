#!/usr/bin/env python3
"""Serving marshal — emit node (client permission seam).

Terminal node of the serving ensemble: emits the serve outcome the caller maps
onto the client permission seam — a file-write for a valid build deliverable, a
prose finish otherwise (scenarios.md "Per-Turn Serving Handler"; ADR-046 §1,
ADR-034 re-homes the Client-Tool-Action Terminal). A build deliverable the
form-gate refused degrades to a prose finish carrying the refusal reason: the
serve never writes a deliverable that failed destination-validity.

Recap grounding (#133/#134): every reject/refuse terminal below carries one of
the module-level prefixes in ``TERMINALS`` (never a literal duplicated at the
call site), so the caller-side ask-outcome ledger can recognize it on the wire.
A read/glob refusal picks its prefix from whether THIS turn carried a build ask
(``is_build_ask``, threaded from classify) — the same failure renders
identically (``build=False``) whether it answers a build ask's discovery round
or a bare-symbol explain's, so the prefix itself is what tells the ledger
whether a build outcome may be attributed at all.

    routing failed (#152):       {"finish": true, "content": "Refused: serving pipeline error: <reason>"}
    read failed (build ask):     {"finish": true, "content": "Build refused: <read_failed reason>"}
    read failed (non-build ask): {"finish": true, "content": "Refused: <read_failed reason>"}
    glob failed (build ask):     {"finish": true, "content": "Build refused: <glob_failed reason>"}
    glob failed (non-build ask): {"finish": true, "content": "Refused: <glob_failed reason>"}
    needs files:      {"finish": false, "reads": ["<path>", ...]}
    needs glob:       {"finish": false, "glob": "<stem>"}
    needs run:        {"finish": false, "run": "<command>"}
    not grounded:     {"finish": true, "content": "No `<target>` in this session..."}
    recall answer:    {"finish": true, "content": "<deterministic ledger answer>"}
    seat contract:    {"finish": true, "content": "Seat contract not met: <reason>"}
    accept gate:      {"finish": true, "content": "Another round needed: <reason>"}
    build + valid:    {"finish": false, "file": "<path>", "content": "<source>"}
    build + invalid:  {"finish": true, "content": "Build refused: <reason>"}
    non-build:        {"finish": true, "content": "<prose>"}

The read/glob/run branches are mutually exclusive by construction — classify
routes each turn to exactly one seam — so their order below only mirrors the
failure-before-request style, never resolves a real conflict.

#155 adds two refuse terminals that reuse existing prefixes rather than
adding table entries: a pipeline-read failure (``REFUSED_PREFIX``, never
mints — an unreadable shape or form_gate makes ``is_build_ask``
unknowable) and a dead seat-side gate on a build turn
(``BUILD_REFUSED_PREFIX``, mints ``refused`` — routing succeeded by
construction, so ``is_build_ask`` is known).

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
# minor 3 (review round 1, #145): when classify recorded an ATTEMPT reason
# for this exact target (``not_grounded_reason``, threaded from
# _visibility's ``attempted`` dict — a prior turn's build read that
# refused as oversize/failed/over-budget), the honest message must not
# suggest the very action that just failed. States the recorded reason
# instead of the generic "ask me to read it" invitation.
_NOT_GROUNDED_WITH_REASON_MESSAGE = (
    "No `{target}` in this session: {reason}, so I can't explain its "
    "internals without guessing."
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
    node_failed = str(gated.get("node_failed", ""))
    if node_failed:
        # #155: the SHAPE node could not be read, so the routing decision,
        # every delegation request and the deliverable all came from an
        # unreadable source — nothing this turn is trustworthy. Refuses
        # before every other outcome, with the same non-minting prefix and
        # the same reason as routing_failed below: a broken pipeline makes
        # is_build_ask unknowable.
        #
        # Deliberately NOT where the seat-gate failure goes. Review found
        # that treating them alike refused eight routes the seat contract
        # has no bearing on: it is a vacuous echo on every non-build route,
        # so its death cannot change those outcomes but was killing the
        # turn anyway.
        prefix = TERMINALS["refused"].prefix
        return {
            "finish": True,
            "content": (
                f"{prefix}serving pipeline error: {node_failed}; "
                f"nothing was built or written"
            ),
        }
    routing_failed = str(gated.get("routing_failed", ""))
    if routing_failed:
        # #152 fail-closed routing: no readable routing decision — the seat
        # dispatched on a failed decision, so no content-bearing route is
        # trustworthy; refuse before every other outcome. Always the plain
        # non-minting prefix: an unreadable decision makes ``is_build_ask``
        # unknowable, and the ledger doctrine is under-report, never
        # misreport.
        prefix = TERMINALS["refused"].prefix
        return {"finish": True, "content": f"{prefix}{routing_failed}"}
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
    needs_self_files = gated.get("needs_self_files") or []
    if needs_self_files:
        # #144 serve-native self-reference: the caller reads the serve's
        # own script server-side — never a client tool call.
        return {"finish": False, "self_reads": list(needs_self_files)}
    needs_grep = str(gated.get("needs_grep", ""))
    if needs_grep:
        # #121 content-grep: delegate ONE def-anchored search round.
        return {"finish": False, "grep": needs_grep}
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
        not_grounded_reason = str(gated.get("not_grounded_reason", ""))
        if not_grounded_reason:
            message = _NOT_GROUNDED_WITH_REASON_MESSAGE.format(
                target=not_grounded, reason=not_grounded_reason
            )
        else:
            message = _NOT_GROUNDED_MESSAGE.format(target=not_grounded)
        return {"finish": True, "content": message}
    recall_answer = str(gated.get("recall_answer", ""))
    if recall_answer:
        # #82 deep recall: the deterministic ordinal-selection answer, composed
        # by classify from the chronological ledger. No seat, no guessing.
        return {"finish": True, "content": recall_answer}
    return None


def _readable_gate(dep: object) -> dict | None:
    """The form_gate node's output, or ``None`` when it cannot be read (#155).

    Positive recognition on ``valid``, which ``form_gate.main`` emits on
    every path — build and non-build alike, from a single ``print`` with
    no early return. The engine's wrap for a dead serving node carries
    ``success``/``data``/``error``/``agent_requests``, disjoint from it.

    This has to run BEFORE any field is read off the result, because the
    old ``except: gated = {}`` is not merely unhandled — ``{}`` answers
    every subsequent question plausibly (``build=False``, ``content=""``,
    no refusal reason), so the non-build branch printed
    ``{"finish": true, "content": ""}``. A failure was converted into a
    well-formed success the client cannot distinguish from "the model had
    nothing to say".
    """
    try:
        parsed = json.loads(_response(dep))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or "valid" not in parsed:
        return None
    return parsed


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    readable = _readable_gate(deps.get("form_gate", {}))
    if readable is None:
        # Non-minting prefix, for #152's reason: with the pipeline itself
        # unreadable, ``is_build_ask`` is unknowable, and the ledger
        # doctrine is under-report rather than misreport.
        prefix = TERMINALS["refused"].prefix
        print(
            json.dumps(
                {
                    "finish": True,
                    "content": (
                        f"{prefix}serving pipeline error: the form gate node "
                        f"returned unreadable output; nothing was built or "
                        f"written"
                    ),
                }
            )
        )
        return
    gated = readable

    build = bool(gated.get("build", False))
    content = str(gated.get("content", ""))
    accept = gated.get("accept")
    seat_admitted = gated.get("seat_admitted")

    seat_gate_failed = str(gated.get("seat_gate_failed", ""))

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
    elif build and seat_gate_failed:
        # #155: the seat-side gate died on a turn that actually depends on
        # it.
        #
        # PLACEMENT, which the design doc demanded a decision on and an
        # earlier round answered silently. Two constraints:
        #
        # - After the delegation seams, because the seat contract is a
        #   vacuous echo on every non-build route (only four ensembles
        #   declare a `seat_contract:` block), so refusing there kills
        #   turns its verdict cannot affect.
        # - After the accept gate, because that gate holds a REAL verdict
        #   the system computed ("tests do not pass") carrying a retry
        #   invitation, while this one only says the admission verdict is
        #   unknown. Refusing ahead of it discarded the better answer and
        #   converted a `rejected_gate` ledger entry into a `refused` one.
        #
        # So this fires only where it must: a turn that would otherwise
        # SHIP, on an unknown admission verdict. That is the wrong-accept
        # it exists to prevent, and nothing else.
        #
        # MINTING prefix, unlike the pipeline failure above, because
        # routing succeeded by construction to reach the build branch, so
        # is_build_ask is KNOWN rather than unknowable. (An earlier draft
        # justified it as preserving a `rejected_contract` entry the system
        # already earned. That was wrong, and was lifted from the design
        # doc's Arc B bullet about a dead `seat` DISPATCH node — a
        # different fault. Measured: before this arc a dead seat_contract
        # SHIPPED, minting `shipped`. The change converts a wrong-accept
        # into a refusal, which is why the entry must still mint.)
        outcome = {
            "finish": True,
            "content": (
                f"{TERMINALS['build_refused'].prefix}serving pipeline error: "
                f"{seat_gate_failed}; nothing was built or written"
            ),
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
