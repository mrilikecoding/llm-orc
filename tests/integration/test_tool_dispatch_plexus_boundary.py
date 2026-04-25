"""Boundary integration test: Orchestrator Tool Dispatch → Plexus Adapter.

Per ``docs/agentic-serving/system-design.md`` §Test Architecture:

    Orchestrator Tool Dispatch → Plexus Adapter
    test_query_knowledge_and_record_outcome_round_trip — Real
    Plexus-active path; also run in stateless mode to verify no-op
    returns.

WP-I ships only the no-op (stateless) branch. WP-K replaces the
Adapter's method bodies with real plexus MCP client calls and adds
a Plexus-active variant of this test. The stateless variant stays
identical — its only requirement is that the no-op fallback returns
well-formed values that flow through Tool Dispatch into
``ToolCallSuccess.content``.

This test exercises the production call chain:

``OrchestratorToolDispatch → PlexusAdapter`` (real instances on both
sides). The other dispatch dependencies (Harness, Autonomy Policy,
Composition Validator, ensemble operations) are real but the only
boundary under test here is Tool Dispatch ↔ Plexus Adapter — the
configuration matches the production wiring in
``v1_chat_completions.get_orchestrator_tool_dispatch``.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.composition_validator import CompositionRequest
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.plexus_adapter import PlexusAdapter
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness


class _StubOperations:
    """Minimal :class:`EnsembleOperations` double — Plexus tests never invoke."""

    async def invoke(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("Plexus boundary tests must not invoke ensembles")

    async def read_ensembles(self) -> list[dict[str, Any]]:  # pragma: no cover
        raise AssertionError("Plexus boundary tests must not list ensembles")


class _UnusedValidator:
    def validate(self, request: CompositionRequest) -> Any:  # pragma: no cover
        raise AssertionError(
            f"composition_validator should not be called: {request.name!r}"
        )


class _UnusedWriter:
    def write(self, config: Any) -> str:  # pragma: no cover
        raise AssertionError(f"local_ensemble_writer should not be called: {config!r}")


def _make_dispatch_with_real_adapter() -> OrchestratorToolDispatch:
    """Build the full dispatch stack with a real :class:`PlexusAdapter`.

    Mirrors the production wiring in
    ``v1_chat_completions.get_orchestrator_tool_dispatch``.
    """
    operations = _StubOperations()
    harness = ResultSummarizerHarness(
        invoker=operations, summarizer_name="agentic-result-summarizer"
    )
    policy = AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL)
    return OrchestratorToolDispatch(
        operations=operations,
        harness=harness,
        autonomy_policy=policy,
        composition_validator=_UnusedValidator(),
        local_ensemble_writer=_UnusedWriter(),
        plexus_adapter=PlexusAdapter(),
    )


class TestQueryKnowledgeAndRecordOutcomeRoundTrip:
    """FC-7 stateless coverage: Plexus-absent paths return well-formed values.

    Production wiring constructs a :class:`PlexusAdapter` whose method
    bodies are no-op fallbacks. These tests verify the no-op values
    flow through Tool Dispatch's match-case routing and arrive at the
    orchestrator surface as ``ToolCallSuccess`` rather than as
    ``not_yet_wired`` errors (which would be the WP-G state).
    """

    @pytest.mark.asyncio
    async def test_query_knowledge_returns_empty_well_formed(self) -> None:
        dispatch = _make_dispatch_with_real_adapter()

        result = await dispatch.dispatch(
            InternalToolCall(
                id="qk-1",
                name="query_knowledge",
                arguments={"topic": "ensembles for refactoring"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "query_knowledge"
        assert result.content == {"results": [], "context": ""}

    @pytest.mark.asyncio
    async def test_record_outcome_returns_acknowledgement_promptly(self) -> None:
        dispatch = _make_dispatch_with_real_adapter()

        result = await dispatch.dispatch(
            InternalToolCall(
                id="ro-1",
                name="record_outcome",
                arguments={
                    "ensemble_name": "composed-x",
                    "quality_signal": "positive",
                    "context": "succeeded on a refactor task",
                },
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "record_outcome"
        assert result.content == {"acknowledged": True}

    @pytest.mark.asyncio
    async def test_react_loop_remains_responsive_while_enrichment_lags(
        self,
    ) -> None:
        """No-op fallback satisfies the responsiveness contract trivially.

        ADR-009: a Plexus-active deployment with backlog must still let
        the ReAct loop continue without blocking on enrichment. With
        Plexus absent there is no enrichment to lag on, so the contract
        is satisfied by construction. WP-K extends this test with a
        Plexus-active variant that exercises real enrichment-lag
        semantics.
        """
        dispatch = _make_dispatch_with_real_adapter()

        first = await dispatch.dispatch(
            InternalToolCall(
                id="ro-2",
                name="record_outcome",
                arguments={"ensemble_name": "x", "quality_signal": "positive"},
            )
        )
        second = await dispatch.dispatch(
            InternalToolCall(
                id="qk-2",
                name="query_knowledge",
                arguments={"topic": "anything"},
            )
        )

        assert isinstance(first, ToolCallSuccess)
        assert isinstance(second, ToolCallSuccess)
        # Query reflects the currently-visible state of the graph; the
        # recently-recorded outcome is not yet enriched (in WP-I, no
        # enrichment runs at all — the empty result is correct).
        assert second.content == {"results": [], "context": ""}
