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
import sys
from pathlib import Path
from typing import Any

_SNIPPET = 280


def _snippet(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET else text[:_SNIPPET] + "…"


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


def build_turn_trace(ensemble_name: str, result_dict: dict[str, Any]) -> dict[str, Any]:
    """Per-node introspection from the engine's execution result."""
    results = result_dict.get("results", {})
    nodes: list[dict[str, Any]] = []
    if isinstance(results, dict):
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
                    {
                        "node": child_name,
                        "response": _snippet(
                            child_node.get("response")
                            if isinstance(child_node, dict)
                            else child_node
                        ),
                    }
                    for child_name, child_node in child.items()
                ]
            nodes.append(entry)
    return {
        "ensemble": ensemble_name,
        "execution_order": result_dict.get("execution_order", []),
        "nodes": nodes,
    }


def summarize_turn_trace(trace: dict[str, Any]) -> str:
    order = trace.get("execution_order") or [n["node"] for n in trace["nodes"]]
    return f"[serve-trace] {trace['ensemble']}: {' -> '.join(order)}"


def emit_turn_trace(
    ensemble_name: str, result_dict: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Build the turn trace, append it to ``<root>/turns.jsonl``, and write a
    one-line summary to stderr. Returns the trace so callers/tests can inspect
    it. Tracing must never break the serve, so IO failures are swallowed."""
    trace = build_turn_trace(ensemble_name, result_dict)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "turns.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace) + "\n")
        sys.stderr.write(summarize_turn_trace(trace) + "\n")
    except OSError:
        pass
    return trace
