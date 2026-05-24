"""Ensemble-backed Routing Planner / Response Synthesizer (Cycle 7 WP-A).

Interim adapters implementing the :class:`RoutingPlanner` and
:class:`ResponseSynthesizer` ports defined in ``dispatch_pipeline``. Each
invokes a named ensemble through the in-process ``OrchestraService.invoke``
surface (the same facade Orchestrator Tool Dispatch uses for capability
dispatch) and translates the result into the port's return type.

Per the ports + spike-backed-adapters decomposition (BUILD entry,
2026-05-23): WP-A wires these against the ζ/ε spike ensembles so the
Dispatch Pipeline runs end-to-end. WP-B (production Routing Planner, ADR-028
— adds the `input` field per Track A.1, bonus-path layering) and WP-C
(production Response Synthesizer, ADR-029 — Rule 6 per Track A.2, token
streaming via the EnsembleExecutor's per-ensemble streaming surface) replace
these adapters behind the unchanged ports.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol

from llm_orc.agentic.dispatch_pipeline import DispatchPlan

__all__ = [
    "EnsembleInvoker",
    "EnsembleResponseSynthesizer",
    "EnsembleRoutingPlanner",
]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class EnsembleInvoker(Protocol):
    """The in-process ensemble-execution surface (``OrchestraService``)."""

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _first_response_text(result: dict[str, Any]) -> str:
    """Extract the first agent's response text from an execution result.

    Single-agent ensembles (the planner and synthesizer) carry their
    output under ``results[<agent>]["response"]`` — the same shape the
    Spike ε / ζ harnesses read.
    """
    results = result.get("results") or {}
    if not results:
        return ""
    first = next(iter(results.values()))
    if isinstance(first, dict):
        return str(first.get("response") or "")
    return str(first)


def _strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


class EnsembleRoutingPlanner:
    """Stage-1 Routing Planner backed by a routing-planner ensemble.

    Parses the ensemble's JSON output into a :class:`DispatchPlan`,
    reporting the model's decision faithfully (an injected action or a
    fabricated ensemble name passes through unchanged — the Dispatch
    Pipeline's validation stage rejects it). Returns ``None`` when the
    output is empty or contains no parseable JSON object (the Spike ν A6
    reliability mode).
    """

    def __init__(self, *, invoker: EnsembleInvoker, ensemble_name: str) -> None:
        self._invoker = invoker
        self._ensemble_name = ensemble_name

    async def plan(self, *, request: str) -> DispatchPlan | None:
        result = await self._invoker.invoke(
            {"ensemble_name": self._ensemble_name, "input": request}
        )
        text = _strip_think(_first_response_text(result))
        if not text:
            return None
        match = _JSON_OBJECT.search(text)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or "action" not in parsed:
            return None
        action = str(parsed.get("action"))
        ensemble = parsed.get("ensemble")
        ensemble = None if ensemble in (None, "null") else str(ensemble)
        # The spike planner emits no `input` field (Track A.1 adds it before
        # WP-B); default the dispatch input to the original request.
        plan_input = parsed.get("input")
        plan_input = str(plan_input) if plan_input is not None else request
        rationale = str(parsed.get("rationale") or "")
        return DispatchPlan(
            action=action, ensemble=ensemble, input=plan_input, rationale=rationale
        )


class EnsembleResponseSynthesizer:
    """Stage-3 Response Synthesizer backed by a response-synthesizer ensemble.

    Serializes the (request, plan, dispatch-results) structured input the
    synthesizer ensemble expects, invokes it, and yields the response text.
    WP-A yields the full response as a single chunk; WP-C makes this
    token-streaming via the EnsembleExecutor's per-ensemble streaming API.
    """

    def __init__(self, *, invoker: EnsembleInvoker, ensemble_name: str) -> None:
        self._invoker = invoker
        self._ensemble_name = ensemble_name

    async def synthesize(
        self,
        *,
        original_request: str,
        dispatched: list[str],
        planned_but_not_run: list[str],
        dispatch_results: list[tuple[str, str]],
    ) -> AsyncIterator[str]:
        synth_input = _format_synthesizer_input(
            original_request=original_request,
            dispatched=dispatched,
            planned_but_not_run=planned_but_not_run,
            dispatch_results=dispatch_results,
        )
        result = await self._invoker.invoke(
            {"ensemble_name": self._ensemble_name, "input": synth_input}
        )
        yield _strip_think(_first_response_text(result))


def _format_synthesizer_input(
    *,
    original_request: str,
    dispatched: list[str],
    planned_but_not_run: list[str],
    dispatch_results: list[tuple[str, str]],
) -> str:
    """Compose the three-section structured input the synthesizer reads.

    Mirrors the ALL-CAPS section contract the spike synthesizer ensemble's
    system prompt enforces (ORIGINAL REQUEST / PLAN / DISPATCH RESULTS).
    """
    dispatched_str = ", ".join(dispatched) if dispatched else "none"
    planned_str = ", ".join(planned_but_not_run) if planned_but_not_run else "none"
    parts = [
        "ORIGINAL REQUEST",
        original_request,
        "",
        "PLAN",
        f"Dispatched: {dispatched_str}",
        f"Planned-but-not-run: {planned_str}",
        "",
        "DISPATCH RESULTS",
    ]
    if not dispatch_results:
        parts.append("(none — direct completion)")
    else:
        for ensemble_name, output in dispatch_results:
            parts.append(f"[{ensemble_name}]")
            parts.append(output)
            parts.append("")
    return "\n".join(parts)
