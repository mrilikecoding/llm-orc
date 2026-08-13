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

from llm_orc.web.serving.token_estimate import projected_tokens_v2

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


def _child_usage(response: Any) -> dict[str, Any]:
    """Per-agent usage dicts (``metadata.usage.agents``) from a dispatched
    seat's serialized child result — the same parse ``_child_results`` does,
    read again for the usage side (C2, #145): the raw ``prompt_eval_count``
    Ollama returns is the only direct signal a truncated prompt was
    actually sent (no seat sets ``num_ctx``, so runtime truncation is
    otherwise unobservable). ``{}`` when the response is absent,
    unparseable, or carries no usage."""
    if not isinstance(response, str):
        return {}
    try:
        child = json.loads(response)
    except json.JSONDecodeError:
        return {}
    metadata = child.get("metadata") if isinstance(child, dict) else None
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    agents = usage.get("agents") if isinstance(usage, dict) else None
    return agents if isinstance(agents, dict) else {}


def _usage_counts(usage: Any) -> dict[str, int]:
    """``{"prompt_eval_count": N, "eval_count": N}`` from one agent's raw
    usage dict (ollama.py's ``_record_usage`` idiom) — only the keys Ollama
    actually returned survive, so a call that fell back to the text-length
    estimate contributes nothing (the honest absence of a truncation
    signal, not a fabricated one)."""
    if not isinstance(usage, dict):
        return {}
    counts: dict[str, int] = {}
    for key in ("prompt_eval_count", "eval_count"):
        value = usage.get(key)
        if isinstance(value, int):
            counts[key] = value
    return counts


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


def _seat_entry(name: str, node: Any, usage: Any = None) -> dict[str, Any]:
    """One child-node trace entry: snippeted response plus the structured
    envelope diagnostics when the node carries them, plus (C2, #145) the
    raw prompt_eval_count/eval_count when the caller found usage for this
    agent in the child ensemble's own metadata."""
    response = node.get("response") if isinstance(node, dict) else node
    entry: dict[str, Any] = {"node": name, "response": _snippet(response)}
    diagnostics = _diagnostics(response)
    if diagnostics is not None:
        entry["diagnostics"] = diagnostics
    entry.update(_usage_counts(usage))
    return entry


def _classify_response(results: dict[str, Any]) -> Any:
    """The classify node's raw (un-snippeted) response, for ``_chain_plan``
    to read its routing decision from — ``None`` when the results carry no
    classify node."""
    classify_node = results.get("classify")
    return classify_node.get("response") if isinstance(classify_node, dict) else None


def _top_level_usage(result_dict: dict[str, Any]) -> dict[str, Any]:
    """``metadata.usage.agents`` from the execution result, or ``{}`` when
    absent (C2, #145) — the parent ensemble's OWN agents' usage, distinct
    from a dispatched seat's nested child usage (``_child_usage``)."""
    metadata = result_dict.get("metadata")
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    agents = usage.get("agents") if isinstance(usage, dict) else None
    return agents if isinstance(agents, dict) else {}


def _node_entry(name: str, node: Any, top_usage: dict[str, Any]) -> dict[str, Any]:
    """One top-level node's trace entry, plus (C2, #145) its own
    prompt_eval_count/eval_count from ``top_usage`` and — when the node is
    a dispatched seat — its nested seat entries with the child ensemble's
    own per-agent usage."""
    response = node.get("response") if isinstance(node, dict) else None
    entry: dict[str, Any] = {
        "node": name,
        "status": node.get("status", "ok") if isinstance(node, dict) else "?",
        "response": _snippet(response),
    }
    entry.update(_usage_counts(top_usage.get(name)))
    child = _child_results(response)
    if child is not None:
        child_usage = _child_usage(response)
        entry["seat"] = [
            _seat_entry(child_name, child_node, child_usage.get(child_name))
            for child_name, child_node in child.items()
        ]
    return entry


# Runtime truncation backstop (#151's core; review round 2 Part 2, #145).
# token_estimate.projected_tokens_v2 is a PRE-FLIGHT guard on the read
# accumulator — a held read's projected size can still admit while the
# ACTUAL dispatched prompt (system/role wrapping, conversation history the
# accumulator doesn't project over) crosses the real model window. This is
# the general backstop no pre-flight estimator can evade: compare Ollama's
# own recorded prompt_eval_count (C2) against the projected size of what
# classify actually dispatched, and flag the turn when either trigger
# fires.
#
# WINDOW = 40,960: the model's context window. Hardcoded rather than
# server-queried — no seat sets num_ctx today, so 40,960 is qwen3:8b's own
# default (measured, not configured). #151's remainder: this stays open
# for a server-queried window and threshold re-measurement whenever the
# model or window changes.
WINDOW = 40960

# DIVISION OF LABOR (review round 3 blocker C): trigger 1 catches deep
# overflow on LARGE (near-window) projections; trigger 2 catches the
# measured discard SIGNATURE, also gated to large projections. Both
# triggers now share the SAME near-window gate — round 2's ungated
# trigger 1 false-positived on small, legitimate calls (a deferred
# decide-child ~40 tokens, a run-verdict child ~260 tokens, other tiny
# deferred calls) whose ratio dips low purely from small-N overhead
# (header/template tokens dominating a tiny real prompt), never from
# truncation. Gating both on "the dispatched prompt was plausibly
# near-window sized" (projected > WINDOW * 0.8) means NEITHER trigger can
# ever fire for a small call, regardless of its ratio.
_NEAR_WINDOW_FRACTION = 0.8

# Trigger 1 (deep overflow, caught directly on ratio, near-window only):
# review round 3 minor 1 — the measured in-window floor for estimator
# v2's most over-projected density class (CJK+code) came in at 0.438,
# leaving only ~4% headroom at the round-2 threshold of 0.42. Lowered to
# 0.35 (~25% headroom below the measured floor). Coverage of the
# borderline ratio band [0.35, 0.438) that trigger 1 alone no longer
# reaches is NOT lost: the measured discard signature (prompt_eval_count
# landing at almost exactly half the window) still lands there via
# trigger 2, independent of ratio — a projected prompt >= 48K already has
# a discard-signature ratio <= 0.43, and trigger 2 fires the SAME instant
# regardless of which side of 0.35 it lands on.
_DEEP_TRUNCATION_RATIO = 0.35

# Trigger 2 (the measured discard SIGNATURE, caught directly): both real
# captured over-window prompts returned EXACTLY 20,482 = 40,960 // 2 + 2,
# regardless of how far over the window the real prompt was — Ollama's
# truncation halves the window, not a proportional cut, so ratio alone
# (trigger 1) undersells how confidently identifiable this shape is. The
# +/-64 band absorbs template/rounding variation across model versions.
_DISCARD_SIGNATURE_BAND = 64

# RESIDUAL (documented, not eliminated): a false positive requires BOTH
# prompt_eval_count landing within the signature band of half-window AND
# the estimator over-projecting the real prompt by more than 1/0.35 = 2.86x
# simultaneously — on the single most over-projected measured density
# class (CJK+code, ~2.15x), the real token count would ALSO have to land
# within the 128-wide band purely by chance, on the order of ~0.3% of
# plausible real-token sizes for that class. This fails SAFE: a false
# positive is a loud, retryable refusal — never a silently corrupted
# answer.


def _dispatch_input(classify_response: Any) -> str:
    """classify's own composed ``dispatch_input`` — the closest available
    proxy for what the seat actually dispatched. The seat's own
    role_prompt/system wrapping is a roughly-fixed per-agent overhead this
    doesn't capture, but dispatch_input carries the UNBOUNDED, variable
    component (conversation history, held read bodies) that is the actual
    truncation risk."""
    if not isinstance(classify_response, str):
        return ""
    try:
        data = json.loads(classify_response)
    except json.JSONDecodeError:
        return ""
    return str(data.get("dispatch_input", "")) if isinstance(data, dict) else ""


def _truncation_detected(prompt_eval_count: int, projected_prompt_tokens: int) -> bool:
    """Dual trigger (review rounds 2-3 Part 2/blocker C, #151's core) —
    refuse when EITHER fires. Both share the near-window gate (review
    round 3 blocker C: an ungated trigger 1 false-positived on small,
    legitimate calls). See the module-level constants above for the
    derivation of each threshold and the documented residual."""
    if projected_prompt_tokens <= WINDOW * _NEAR_WINDOW_FRACTION:
        return False
    if prompt_eval_count < projected_prompt_tokens * _DEEP_TRUNCATION_RATIO:
        return True
    return abs(prompt_eval_count - WINDOW // 2) <= _DISCARD_SIGNATURE_BAND


def _prompt_eval_counts(nodes: list[dict[str, Any]]) -> list[int]:
    """Every recorded prompt_eval_count in this turn's trace — top-level
    agent calls and nested seat-child calls alike (C2's recording seam)."""
    counts: list[int] = []
    for node in nodes:
        value = node.get("prompt_eval_count")
        if isinstance(value, int):
            counts.append(value)
        for seat_entry in node.get("seat", ()) or ():
            seat_value = (
                seat_entry.get("prompt_eval_count")
                if isinstance(seat_entry, dict)
                else None
            )
            if isinstance(seat_value, int):
                counts.append(seat_value)
    return counts


def _truncation_check(
    nodes: list[dict[str, Any]], classify_response: Any
) -> dict[str, Any] | None:
    """The turn's truncation verdict, or ``None`` when nothing indicates
    truncation — including when there is no dispatch_input or no recorded
    usage to check at all (never a false positive on absent data)."""
    dispatch_input = _dispatch_input(classify_response)
    if not dispatch_input:
        return None
    projected = projected_tokens_v2(dispatch_input)
    counts = _prompt_eval_counts(nodes)
    if not any(_truncation_detected(count, projected) for count in counts):
        return None
    return {"projected_prompt_tokens": projected, "prompt_eval_counts": counts}


def build_turn_trace(ensemble_name: str, result_dict: dict[str, Any]) -> dict[str, Any]:
    """Per-node introspection from the engine's execution result."""
    results = result_dict.get("results", {})
    top_usage = _top_level_usage(result_dict)
    nodes: list[dict[str, Any]] = []
    classify_response: Any = None
    if isinstance(results, dict):
        classify_response = _classify_response(results)
        nodes = [_node_entry(name, node, top_usage) for name, node in results.items()]
    trace: dict[str, Any] = {
        "ensemble": ensemble_name,
        "execution_order": result_dict.get("execution_order", []),
        "nodes": nodes,
    }
    chain_plan = _chain_plan(classify_response)
    if chain_plan is not None:
        trace["chain_plan"] = chain_plan
    truncation = _truncation_check(nodes, classify_response)
    if truncation is not None:
        trace["truncation_detected"] = True
        trace["truncation_detail"] = truncation
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
