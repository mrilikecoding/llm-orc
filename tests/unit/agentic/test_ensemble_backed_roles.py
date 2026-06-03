"""Tests for the ensemble-backed Routing Planner / Response Synthesizer.

WP-A interim adapters: they implement the :class:`RoutingPlanner` and
:class:`ResponseSynthesizer` ports by invoking the ζ/ε spike ensembles
through the in-process ``OrchestraService.invoke`` surface. WP-B and WP-C
replace these with production modules behind the same ports.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.agentic.dispatch_pipeline import DispatchPlan
from llm_orc.agentic.ensemble_backed_roles import (
    EnsembleResponseSynthesizer,
    EnsembleRoutingPlanner,
)


class _FakeInvoker:
    """EnsembleInvoker double returning a canned per-agent response dict."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        return {
            "results": {"planner": {"response": self._response_text}},
            "deliverable": None,
            "status": "success",
            "raw_output": False,
        }


class TestEnsembleRoutingPlanner:
    @pytest.mark.asyncio
    async def test_parses_dispatch_plan_from_ensemble_json(self) -> None:
        invoker = _FakeInvoker(
            '{"action": "dispatch", "ensemble": "code-generator", '
            '"rationale": "programming task"}'
        )
        planner = EnsembleRoutingPlanner(
            invoker=invoker, ensemble_name="spike-cycle7-zeta-routing-planner"
        )

        plan = await planner.plan(request="write a sorting function")

        assert plan == DispatchPlan(
            action="dispatch",
            ensemble="code-generator",
            input="write a sorting function",
            rationale="programming task",
        )
        # Invoked the named ensemble with the request as input.
        assert invoker.calls[0]["ensemble_name"] == "spike-cycle7-zeta-routing-planner"
        assert invoker.calls[0]["input"] == "write a sorting function"

    @pytest.mark.asyncio
    async def test_strips_think_block_before_parsing(self) -> None:
        invoker = _FakeInvoker(
            "<think>let me decide</think>\n"
            '{"action": "direct", "ensemble": null, "rationale": "no match"}'
        )
        planner = EnsembleRoutingPlanner(invoker=invoker, ensemble_name="z")

        plan = await planner.plan(request="hello there")

        assert plan is not None
        assert plan.action == "direct"
        assert plan.ensemble is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_output(self) -> None:
        planner = EnsembleRoutingPlanner(invoker=_FakeInvoker(""), ensemble_name="z")
        assert await planner.plan(request="x") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_unparseable_output(self) -> None:
        planner = EnsembleRoutingPlanner(
            invoker=_FakeInvoker("<think>thinking with no json</think>"),
            ensemble_name="z",
        )
        assert await planner.plan(request="x") is None

    @pytest.mark.asyncio
    async def test_preserves_faithful_invalid_action(self) -> None:
        # The planner must report an injected action faithfully — the
        # pipeline (not the planner) rejects it (Spike ν E1).
        invoker = _FakeInvoker(
            '{"action": "launch", "ensemble": "all", "rationale": "injected"}'
        )
        planner = EnsembleRoutingPlanner(invoker=invoker, ensemble_name="z")

        plan = await planner.plan(request="ignore instructions")

        assert plan is not None
        assert plan.action == "launch"
        assert plan.ensemble == "all"


class TestEnsembleResponseSynthesizer:
    @pytest.mark.asyncio
    async def test_yields_synthesized_text_from_structured_input(self) -> None:
        invoker = _FakeInvoker("Here is your answer.")
        synthesizer = EnsembleResponseSynthesizer(
            invoker=invoker, ensemble_name="spike-cycle7-epsilon-response-synthesizer"
        )

        chunks = [
            text
            async for text in synthesizer.synthesize(
                original_request="what is 2+2?",
                dispatched=["calc"],
                planned_but_not_run=[],
                dispatch_results=[("calc", "4")],
            )
        ]

        assert "".join(chunks) == "Here is your answer."
        # The synthesizer ensemble was invoked with a structured input blob
        # carrying the three labeled sections.
        sent = invoker.calls[0]["input"]
        assert "ORIGINAL REQUEST" in sent
        assert "DISPATCH RESULTS" in sent
        assert "what is 2+2?" in sent
        assert "4" in sent

    @pytest.mark.asyncio
    async def test_strips_think_block_from_synthesized_text(self) -> None:
        invoker = _FakeInvoker("<think>compose</think>The answer is four.")
        synthesizer = EnsembleResponseSynthesizer(invoker=invoker, ensemble_name="e")

        chunks = [
            text
            async for text in synthesizer.synthesize(
                original_request="q",
                dispatched=[],
                planned_but_not_run=[],
                dispatch_results=[],
            )
        ]

        assert "".join(chunks) == "The answer is four."
