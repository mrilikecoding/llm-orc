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


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _terminal(text: str) -> str:
    """The seat's deliverable, unwrapping the layers the engine adds
    (``deliverable`` / script ``{"success","output"}`` / nested ``results``).
    For a build seat this yields the ADR-024 envelope JSON; for a raw seat it
    yields the plain terminal text."""
    current = text
    for _ in range(6):
        try:
            obj = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return current
        if not isinstance(obj, dict):
            return current
        if isinstance(obj.get("deliverable"), str):
            current = obj["deliverable"]
            continue
        if isinstance(obj.get("output"), str):
            current = obj["output"]
            continue
        results = obj.get("results")
        if isinstance(results, dict) and results:
            node = results[list(results.keys())[-1]]
            current = node.get("response", "") if isinstance(node, dict) else str(node)
            continue
        return current
    return current


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
    # The routing decision is ``resolve`` when the guarded decider ran, else the
    # structural ``classify`` decision directly (backward-compatible).
    decision_dep = deps.get("resolve") or deps.get("classify", {})
    try:
        decision = json.loads(_response(decision_dep))
    except json.JSONDecodeError:
        decision = {}
    if not isinstance(decision, dict):
        decision = {}

    seat_terminal = _terminal(_response(deps.get("seat", {})))
    deliverable = _envelope_deliverable(seat_terminal)
    if deliverable is None:
        deliverable = seat_terminal.strip()

    accept, accept_reason = _envelope_verdict(seat_terminal)
    seat_admitted, seat_contract_reason = _seat_verdict(deps.get("seat_contract"))

    print(
        json.dumps(
            {
                "build": bool(
                    decision.get("build", decision.get("kind") != "explanation")
                ),
                "file": decision.get("file", "solution.py"),
                "content": deliverable,
                "accept": accept,
                "accept_reason": accept_reason,
                "seat_admitted": seat_admitted,
                "seat_contract_reason": seat_contract_reason,
            }
        )
    )


if __name__ == "__main__":
    main()
