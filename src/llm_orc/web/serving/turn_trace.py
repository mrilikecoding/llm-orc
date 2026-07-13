"""Serving-turn introspection — a light, vendor-neutral trace (WP-A8).

Built from the L0 engine's own execution result. The engine already returns
every node's output (and runs a performance-event queue + a usage collector);
this reads that surviving surface into a readable per-turn record so an operator
can see how each seat — and the model inside a dispatched seat — actually
behaved. It iterates to build understanding of the ensemble, which is the whole
point of the standing "don't build in a vacuum" grounding directive.

The trace sits above the model layer, so it is agnostic to the inference
backend (Ollama vs llama.cpp) and to any future observability backend: the same
per-node shape maps onto OpenTelemetry spans if a backend (Phoenix, Langfuse)
is later adopted. No new dependency, no infra.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_SNIPPET = 280


def _snippet_cap() -> int:
    """The response clip length: readable-short by default,
    ``LLM_ORC_SERVE_TRACE_SNIPPET`` raises it for live diagnosis."""
    raw = os.environ.get("LLM_ORC_SERVE_TRACE_SNIPPET", "")
    try:
        return int(raw) if raw else _SNIPPET
    except ValueError:
        return _SNIPPET


def _snippet(value: Any) -> str:
    cap = _snippet_cap()
    text = value if isinstance(value, str) else json.dumps(value)
    text = " ".join(text.split())
    return text if len(text) <= cap else text[:cap] + "…"


def _child_results(response: Any) -> dict[str, Any] | None:
    """The child ensemble's node results when ``response`` is a dispatched
    seat's serialized child result — so the trace can show the model's real
    output inside the seat, not just the seat's opaque envelope string."""
    if not isinstance(response, str):
        return None
    try:
        child = json.loads(response)
    except json.JSONDecodeError:
        return None
    results = child.get("results") if isinstance(child, dict) else None
    return results if isinstance(results, dict) else None


def _diagnostics(response: Any) -> dict[str, Any] | None:
    """The envelope's structured ``diagnostics`` dict when ``response`` is a
    (possibly output-wrapped) envelope JSON. The small typed verdict fields —
    accept, held_round, tests_pass/adequate — survive verbatim so a battery
    post-mortem can answer gate questions the snippet cap otherwise eats
    (issue #114); prose-sized string values still clip."""
    if not isinstance(response, str):
        return None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for candidate in (data, data.get("output")):
        if isinstance(candidate, dict):
            diagnostics = candidate.get("diagnostics")
            if isinstance(diagnostics, dict):
                return {
                    key: _snippet(value) if isinstance(value, str) else value
                    for key, value in diagnostics.items()
                }
    return None


_CHAIN_PLAN_KEYS = ("chain", "step_index", "target")


def _chain_plan(response: Any) -> dict[str, Any] | None:
    """The classify node's ``{chain, step_index, target}`` routing decision,
    read from its FULL response (before snippeting) so the values survive
    un-clipped — mirrors ``_diagnostics``. ``None`` when the response is
    absent, unparseable, or missing a routing key (e.g. a toolless/short-
    circuit turn that never runs the serving ensemble)."""
    if not isinstance(response, str):
        return None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not all(key in data for key in _CHAIN_PLAN_KEYS):
        return None
    return {key: data[key] for key in _CHAIN_PLAN_KEYS}


def _seat_entry(name: str, node: Any) -> dict[str, Any]:
    """One child-node trace entry: snippeted response plus the structured
    envelope diagnostics when the node carries them."""
    response = node.get("response") if isinstance(node, dict) else node
    entry: dict[str, Any] = {"node": name, "response": _snippet(response)}
    diagnostics = _diagnostics(response)
    if diagnostics is not None:
        entry["diagnostics"] = diagnostics
    return entry


def build_turn_trace(ensemble_name: str, result_dict: dict[str, Any]) -> dict[str, Any]:
    """Per-node introspection from the engine's execution result."""
    results = result_dict.get("results", {})
    nodes: list[dict[str, Any]] = []
    classify_response: Any = None
    if isinstance(results, dict):
        classify_node = results.get("classify")
        classify_response = (
            classify_node.get("response") if isinstance(classify_node, dict) else None
        )
        for name, node in results.items():
            response = node.get("response") if isinstance(node, dict) else None
            entry: dict[str, Any] = {
                "node": name,
                "status": node.get("status", "ok") if isinstance(node, dict) else "?",
                "response": _snippet(response),
            }
            child = _child_results(response)
            if child is not None:
                entry["seat"] = [
                    _seat_entry(child_name, child_node)
                    for child_name, child_node in child.items()
                ]
            nodes.append(entry)
    trace: dict[str, Any] = {
        "ensemble": ensemble_name,
        "execution_order": result_dict.get("execution_order", []),
        "nodes": nodes,
    }
    chain_plan = _chain_plan(classify_response)
    if chain_plan is not None:
        trace["chain_plan"] = chain_plan
    return trace


def summarize_turn_trace(trace: dict[str, Any]) -> str:
    order = trace.get("execution_order") or [n["node"] for n in trace["nodes"]]
    return f"[serve-trace] {trace['ensemble']}: {' -> '.join(order)}"


def emit_turn_trace(
    ensemble_name: str, result_dict: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Build the turn trace, append it to ``<root>/turns.jsonl``, and write a
    one-line summary to stderr. Returns the trace so callers/tests can inspect
    it. Tracing must never break the serve, so IO failures are swallowed."""
    try:
        trace = build_turn_trace(ensemble_name, result_dict)
    except Exception:  # noqa: BLE001 — tracing must never break the serve
        trace = {"ensemble": ensemble_name, "execution_order": [], "nodes": []}
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "turns.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace) + "\n")
        sys.stderr.write(summarize_turn_trace(trace) + "\n")
    except OSError:
        pass
    return trace
