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
import sys

from _helpers import terminal as _terminal


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
    if not isinstance(parsed, dict) or not parsed.get("target"):
        return None
    if "build" not in parsed and "kind" not in parsed:
        return None
    return parsed


def _routing_failure_reason(deps: dict) -> str:
    """The deterministic refusal reason for a turn with no readable
    routing decision, carrying the engine failure envelope's one-line
    ``error`` when a decision dep has one (never raw stderr)."""
    for name in ("resolve", "classify"):
        try:
            parsed = json.loads(_response(deps.get(name)))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, str) and error:
                return (
                    "serving pipeline error: no readable routing decision "
                    f"this turn ({name}: {error}); nothing was built or "
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
    if deliverable is None:
        deliverable = seat_terminal.strip()

    accept, accept_reason = _envelope_verdict(seat_terminal)
    seat_admitted, seat_contract_reason = _seat_verdict(deps.get("seat_contract"))

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
            }
        )
    )


if __name__ == "__main__":
    main()
