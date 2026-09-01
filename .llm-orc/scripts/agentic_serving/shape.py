#!/usr/bin/env python3
"""Serving marshal — shape node (fidelity marshalling).

Reads the seat's ADR-024 ``DispatchEnvelope`` and the resolved routing decision
and produces the faithful deliverable: the deliverable CONTENT comes from the
envelope (``artifacts[0].content``, else ``primary``), the DESTINATION path and
build flag come from the routing decision (``resolve`` when the guarded decider
ran, else ``classify`` directly; scenarios.md "Per-Turn Serving Handler";
ADR-046 §1, ADR-034 re-homes the Artifact Bridge). Consumers read ``artifacts``
/ ``structured``, never parse ``primary`` structurally (ADR-024).

When the seat did not emit an envelope (e.g. a non-build explain seat that
returns raw prose), the raw terminal text is the deliverable — shape degrades
gracefully rather than requiring every seat to envelope first.
"""

from __future__ import annotations

import json
import re
import sys

from _helpers import terminal as _terminal

# #168: the engine's wrap is f"Schema JSON execution failed: {str(e)}" over a
# blanket except, and subprocess exceptions stringify with the whole argv —
# so the interpreter path, the script path, and hence the operator's home
# directory and username used to ride to the client on every wrapped node
# failure. Ten captures in .llm-orc/.serve-trace/turns.jsonl.
#
# Positive extraction rather than stripping the `Command '[...]'` clause the
# captures happen to show. Both recognised residues are NUMERIC, so neither
# can carry a path by construction, and an unrecognised shape contributes no
# verbatim text at all. The clause-strip the issue proposed fails open on
# FileNotFoundError, which carries an absolute path and no Command clause —
# the same denylist failure #152 and #155 Arc A were both paid for.
#
# One pattern per FACT, not per producer wording. script_agent.py states the
# same two facts three ways — the blanket except's stringified subprocess
# exception, its own "Script failed with exit code N" envelope, and its
# "Script timed out after N seconds" envelope — and the client should see one
# vocabulary regardless. A producer that words it a fourth way degrades to
# "failed" rather than leaking, which is the right direction.
_EXIT_STATUS_RE = re.compile(
    r"(?:returned non-zero exit status|failed with exit code) (\d+)"
)
_TIMEOUT_RE = re.compile(r"timed out after ([\d.]+) seconds?")


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _readable_decision(dep: object) -> dict | None:
    """The parsed routing decision when it carries the producers' contract
    (#152 fail-closed): a dict with a NON-EMPTY ``target`` and at least
    one of ``build``/``kind`` present. Presence of the dep alone proves
    nothing — a crashed node RETURNS its failure envelope as a normal
    response (script_agent.py's exit-code wrap), and a crashed classify
    laundered through a healthy resolve arrives with ``target: ""`` (as
    does an out-of-set decider vote). Positive readability: an unknown
    future failure shape fails closed instead of sailing past an
    error-shape denylist."""
    try:
        parsed = json.loads(_response(dep))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    target = parsed.get("target")
    # non-empty STRING, not merely truthy (review finding 8): a drifted
    # producer emitting a number/list/whitespace target must not route.
    if not isinstance(target, str) or not target.strip():
        return None
    if "build" not in parsed and "kind" not in parsed:
        return None
    return parsed


def _engine_failure_summary(error: str) -> str:
    """The client-safe residue of an engine wrap error (#168).

    The operator keeps the full text: ``turn_trace.py`` records raw node
    responses server-side, so nothing debuggable is lost by refusing to
    quote it here.
    """
    status = _EXIT_STATUS_RE.search(error)
    if status:
        return f"exited non-zero, status {status.group(1)}"
    timeout = _TIMEOUT_RE.search(error)
    if timeout:
        return f"timed out after {timeout.group(1)} seconds"
    return "failed"


def _routing_failure_reason(deps: dict) -> str:
    """The deterministic refusal reason for a turn with no readable
    routing decision, naming the failing node and what happened to it —
    never the engine wrap's text, which embeds the subprocess argv
    (#168), and never raw stderr."""
    for name in ("resolve", "classify"):
        try:
            parsed = json.loads(_response(deps.get(name)))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, str) and error:
                summary = _engine_failure_summary(error)
                return (
                    "serving pipeline error: no readable routing decision "
                    f"this turn ({name}: {summary}); nothing was built or "
                    "written"
                )
    return (
        "serving pipeline error: no readable routing decision this turn; "
        "nothing was built or written"
    )


def _envelope_deliverable(seat_terminal: str) -> str | None:
    """The deliverable content from an ADR-024 envelope, or ``None`` when the
    seat terminal is not an envelope."""
    try:
        env = json.loads(seat_terminal)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(env, dict) or "status" not in env:
        return None
    artifacts = env.get("artifacts")
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[0], dict):
        content = artifacts[0].get("content")
        if isinstance(content, str):
            return content
    primary = env.get("primary")
    return primary if isinstance(primary, str) else None


def _dead_seat_reason(seat_terminal: str) -> str:
    """Why the seat is dead, or ``""`` when it isn't (#174).

    Positive recognition, not a denylist: a seat terminal that parses as a
    JSON dict WITHOUT ``status`` is not a healthy seat output whatever else
    it may be — a raw-prose seat (the explainer) never parses as a dict at
    all, and a healthy ADR-024 envelope always carries ``status``. Both
    engine failure families are dicts of this disjoint shape by
    construction: ``execute_with_schema_json``'s wrap
    (``success``/``data``/``error``/``agent_requests``) and the sub-ensemble
    ``ScriptAgent.execute`` ``{success, error, stderr}`` shape.

    Named bound (accepted the way #155 accepted its seat_contract
    trip-wire): an explainer answer that happens to BE a parseable JSON
    dict without ``status`` is indistinguishable from a dead seat here and
    wrong-refuses. The system prompt forbids that shape.

    The reason text is built ONLY from ``_engine_failure_summary`` over the
    dict's string ``error`` field — never ``stderr``, argv, or any other
    dict value; ``turn_trace.py`` keeps the whole thing server-side.
    """
    try:
        env = json.loads(seat_terminal)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(env, dict) or "status" in env:
        return ""
    error = env.get("error")
    if isinstance(error, str) and error:
        return _engine_failure_summary(error)
    return "failed"


def _seat_verdict(dep: object) -> tuple[bool | None, str]:
    """The per-seat admission verdict from the ``seat_contract`` node, or
    ``(None, "")`` when no seat contract ran. ``None`` means "no per-seat gate";
    emit treats only an explicit ``False`` as a refusal (WP-E8; ADR-046 §2). This
    is a different granularity from the accept-gate verdict below and rides
    alongside it."""
    try:
        verdict = json.loads(_response(dep))
    except (json.JSONDecodeError, TypeError):
        return None, ""
    if not isinstance(verdict, dict) or "seat_admitted" not in verdict:
        return None, ""
    return bool(verdict["seat_admitted"]), str(verdict.get("seat_contract_reason", ""))


def _unreadable_seat_contract(deps: dict) -> str:
    """Why the ``seat_contract`` node could not be read, or ``""`` (#155).

    ``_seat_verdict`` answers ``(None, "")`` for anything without a
    ``seat_admitted`` key, and emit reads ``None`` as "no per-seat gate
    ran" — which is indistinguishable from a gate that ran and died.

    Fails closed on an ABSENT dep too. Two drafts got the reason wrong,
    so it is stated carefully. The first said "the explain path has no
    seat_contract block", conflating the ensemble's optional
    ``seat_contract:`` YAML block with the skeleton's ``seat_contract``
    NODE — ``serving.yaml`` declares that node unconditionally, so it runs
    on every route and emits ``seat_admitted: true`` vacuously when the
    route's ensemble declares no contract. The second said an absent dep
    "means the node was filtered out for not succeeding", which is false
    in general: ``decide`` and ``pick`` are absent on most turns because
    ``when:`` skipped them, and every ``ScriptAgent`` failure is caught
    inside the agent and returned as a ``status="success"`` response
    carrying an error envelope, so a crashed seat_contract is always
    PRESENT.

    Measured: zero absences across 650 recorded live turns. So this branch
    is unreachable in the current skeleton, and it is kept as a deliberate
    trip-wire rather than as handling for a live failure mode — if a
    future skeleton guards or removes this node, build turns refuse
    loudly instead of silently losing the gate. The bound that comes with
    that: such a skeleton refuses EVERY build turn until this check is
    updated with it.

    Positive recognition, not a denylist: a healthy ``seat_contract``
    always emits ``seat_admitted``, so anything lacking it is not a
    seat_contract output whatever else it may be. The engine's wrap for a
    dead serving node carries ``success``/``data``/``error``/
    ``agent_requests`` — disjoint from that — so the check discriminates
    without enumerating failure shapes.

    Reported separately from a pipeline read failure, because it is a
    different KIND of thing: see the placement note in emit.
    """
    try:
        parsed = json.loads(_response(deps.get("seat_contract", {})))
    except (json.JSONDecodeError, TypeError):
        return "the seat contract node returned unreadable output"
    if not isinstance(parsed, dict) or "seat_admitted" not in parsed:
        return "the seat contract node returned unreadable output"
    return ""


def _envelope_verdict(seat_terminal: str) -> tuple[bool | None, str]:
    """The accept-gate verdict from a build-gated envelope's diagnostics, or
    ``(None, "")`` when the seat carries no verdict (an ungated code-seat or a
    non-build explainer). ``None`` means "no gate ran here"; the emit node treats
    only an explicit ``False`` as a rejection (WP-D8; ADR-048 §1)."""
    try:
        env = json.loads(seat_terminal)
    except (json.JSONDecodeError, TypeError):
        return None, ""
    if not isinstance(env, dict):
        return None, ""
    diagnostics = env.get("diagnostics")
    if not isinstance(diagnostics, dict) or "accept" not in diagnostics:
        return None, ""
    return bool(diagnostics["accept"]), str(diagnostics.get("accept_reason", ""))


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    # The routing decision is ``resolve`` when the guarded decider ran, else
    # the structural ``classify`` decision directly (the unit-harness /
    # pre-resolve back-compat source; live wiring carries only resolve).
    # #152: readability-gated, never truthiness — with no readable decision
    # the turn fails CLOSED (build=False plus a ``routing_failed`` reason
    # emit renders as an honest refusal; the seat dispatched on the failed
    # decision, so no content-bearing route is trustworthy).
    decision = _readable_decision(deps.get("resolve")) or _readable_decision(
        deps.get("classify")
    )
    routing_failed = ""
    if decision is None:
        decision = {}
        routing_failed = _routing_failure_reason(deps)

    seat_terminal = _terminal(_response(deps.get("seat", {})))
    deliverable = _envelope_deliverable(seat_terminal)
    seat_failed = ""
    if deliverable is None:
        # #174: a dead seat (a JSON dict without `status`) is not a healthy
        # seat output — zero the deliverable as defense in depth rather than
        # ship the engine's failure envelope. Everything that does not parse
        # to a dict stays raw prose, unaffected.
        seat_failed = _dead_seat_reason(seat_terminal)
        deliverable = "" if seat_failed else seat_terminal.strip()

    accept, accept_reason = _envelope_verdict(seat_terminal)
    seat_admitted, seat_contract_reason = _seat_verdict(deps.get("seat_contract"))
    seat_gate_failed = _unreadable_seat_contract(deps)

    print(
        json.dumps(
            {
                # fail CLOSED: an unreadable routing decision must not
                # default a turn onto the build path
                "build": bool(
                    decision.get("build", decision.get("kind") != "explanation")
                )
                if decision
                else False,
                "file": decision.get("file", "solution.py"),
                "content": deliverable,
                "accept": accept,
                "accept_reason": accept_reason,
                "seat_admitted": seat_admitted,
                "seat_contract_reason": seat_contract_reason,
                # issue #83: read, run, and glob requests ride the routing
                # decision
                "needs_files": decision.get("needs_files", []),
                "read_failed": str(decision.get("read_failed", "")),
                "needs_run": str(decision.get("needs_run", "")),
                "needs_glob": str(decision.get("needs_glob", "")),
                "glob_failed": str(decision.get("glob_failed", "")),
                # #144 serve-native self-reference: rides the routing decision.
                "needs_self_files": decision.get("needs_self_files", []),
                # #121 content-grep: rides the routing decision.
                "needs_grep": str(decision.get("needs_grep", "")),
                "picked": str(decision.get("picked", "")),
                "not_grounded": str(decision.get("not_grounded", "")),
                "not_grounded_reason": str(decision.get("not_grounded_reason", "")),
                "recall_answer": str(decision.get("recall_answer", "")),
                # Review round 2 new blocker 2: pass through unchanged.
                "is_build_ask": bool(decision.get("is_build_ask", False)),
                # #152: non-empty exactly when no readable routing decision
                # arrived — emit refuses on it before every other outcome.
                "routing_failed": routing_failed,
                # #155: non-empty exactly when the seat-side gate could not
                # be read. NOT a pipeline-read failure — the seat contract has
                # no bearing on a delegation or prose route, so emit consumes
                # this on the BUILD branch only.
                "seat_gate_failed": seat_gate_failed,
                # #174: non-empty exactly when the seat itself is dead (a
                # JSON dict without `status`) — emit consumes this exactly
                # where content ships, never as a turn-wide precondition, so
                # a dead placeholder seat cannot refuse a delegation/prose
                # route whose answer never touches the seat terminal.
                "seat_failed": seat_failed,
            }
        )
    )


if __name__ == "__main__":
    main()
