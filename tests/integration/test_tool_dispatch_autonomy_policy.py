"""Boundary integration test: Orchestrator Tool Dispatch → Autonomy Policy.

Per ``docs/agentic-serving/system-design.md`` §Test Architecture:

    Orchestrator Tool Dispatch → Autonomy Policy
    test_autonomy_gate_fires_before_every_dispatch — Every tool path
    passes through the gate; baseline level allows invoke, composes,
    denies promotion.

The Group 2 unit tests use a ``_RecordingPolicy`` stub to assert that
the gate was called. This integration test uses the production
:class:`AutonomyPolicy` across the Tool Dispatch boundary so the
real decision logic (level lookup + per-tool routing of events)
observes the full lifecycle from
``OrchestratorToolDispatch.dispatch`` through the match-case routing.

Promotion is not an orchestrator tool (``TOOL_NAMES`` excludes it)
— scenario-level "denies promotion" holds structurally via the
unknown-tool short-circuit, covered end-to-end in the Serving Layer
acceptance tests. The focus here is the per-tool gate consultation
and event propagation on the valid-tool paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.agentic.autonomy_policy import (
    BASELINE_LEVEL,
    PURE_TOOL_USER_VISIBLE_LEVEL,
    AutonomyDecision,
    AutonomyPolicy,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness


class _SpyAutonomyPolicy:
    """Delegates to the real :class:`AutonomyPolicy` and records each call.

    The delegating wrapper keeps the production decision logic on the
    boundary while recording arguments so the test can assert the gate
    fires for every dispatched tool. Satisfies ``AutonomyGate``
    structurally.
    """

    def __init__(self, *, level: str) -> None:
        self._inner = AutonomyPolicy(level_provider=lambda: level)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def decide(self, *, tool_name: str, arguments: dict[str, Any]) -> AutonomyDecision:
        self.calls.append((tool_name, dict(arguments)))
        return self._inner.decide(tool_name=tool_name, arguments=arguments)


class _StubEnsembleOperations:
    """Minimal ``EnsembleOperations`` for the Autonomy boundary test.

    The gate fires before tool routing, so the downstream ensemble
    operations are only exercised on the ``invoke_ensemble`` and
    ``list_ensembles`` paths. ``invoke`` returns a ``raw_output: True``
    result so the Harness's pass-through branch is taken — this test
    focuses on the gate boundary, not summarization.
    """

    def __init__(self) -> None:
        self.invoke_calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.invoke_calls.append(dict(arguments))
        return {
            "results": {"a": {"response": "ok"}},
            "synthesis": None,
            "status": "success",
            "raw_output": True,
        }

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return []


def _build_dispatch(
    *, policy: _SpyAutonomyPolicy, operations: _StubEnsembleOperations
) -> OrchestratorToolDispatch:
    harness = ResultSummarizerHarness(
        invoker=operations, summarizer_name="unused-in-this-test"
    )
    return OrchestratorToolDispatch(
        operations=operations, harness=harness, autonomy_policy=policy
    )


class TestAutonomyGateFiresBeforeEveryDispatch:
    """Real AutonomyPolicy across the Tool Dispatch boundary."""

    @pytest.mark.asyncio
    async def test_gate_consulted_exactly_once_per_dispatch_at_baseline(
        self,
    ) -> None:
        policy = _SpyAutonomyPolicy(level=BASELINE_LEVEL)
        operations = _StubEnsembleOperations()
        dispatch = _build_dispatch(policy=policy, operations=operations)

        for tool_name in sorted(TOOL_NAMES):
            await dispatch.dispatch(
                InternalToolCall(
                    id=f"call_{tool_name}",
                    name=tool_name,
                    arguments={"name": "analysis", "input": "x"},
                )
            )

        # Every tool in the closed set consulted the gate, in order, once each.
        seen_names = [call[0] for call in policy.calls]
        assert sorted(seen_names) == sorted(TOOL_NAMES)
        assert len(policy.calls) == len(TOOL_NAMES)

    @pytest.mark.asyncio
    async def test_unknown_tool_short_circuits_before_gate(self) -> None:
        policy = _SpyAutonomyPolicy(level=BASELINE_LEVEL)
        operations = _StubEnsembleOperations()
        dispatch = _build_dispatch(policy=policy, operations=operations)

        await dispatch.dispatch(
            InternalToolCall(
                id="call_unknown",
                name="author_script",
                arguments={"name": "evil.py"},
            )
        )

        # TOOL_NAMES closure is AS-6's enforcement, not the gate's.
        assert policy.calls == []

    @pytest.mark.asyncio
    async def test_tightened_level_surfaces_composition_event_end_to_end(
        self,
    ) -> None:
        """At ``pure-tool-user-visible``, the real policy emits a
        composition event and Tool Dispatch attaches it to the result."""
        policy = _SpyAutonomyPolicy(level=PURE_TOOL_USER_VISIBLE_LEVEL)
        operations = _StubEnsembleOperations()
        dispatch = _build_dispatch(policy=policy, operations=operations)

        compose_result = await dispatch.dispatch(
            InternalToolCall(
                id="call_compose",
                name="compose_ensemble",
                arguments={"name": "new"},
            )
        )

        invoke_result = await dispatch.dispatch(
            InternalToolCall(
                id="call_invoke",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "x"},
            )
        )

        # Composition surfaces one event; invoke stays silent — only
        # compose_ensemble is flagged for visibility at this level.
        assert len(compose_result.events) == 1
        assert compose_result.events[0].kind == "composition"
        assert compose_result.events[0].payload["arguments"] == {"name": "new"}
        assert invoke_result.events == ()
        # Invoke still reached the downstream ensemble operations.
        assert isinstance(invoke_result, ToolCallSuccess)
        assert operations.invoke_calls == [{"ensemble_name": "analysis", "input": "x"}]
