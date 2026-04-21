"""Tests for the Orchestrator Tool Dispatch module.

Per `docs/agentic-serving/system-design.md` §Orchestrator Tool Dispatch
(L2) and §Integration Contracts (Orchestrator Runtime → Orchestrator
Tool Dispatch).

Covers scenarios:

* §Orchestrator tool surface is exactly the committed set (FC-5)
* §Invocation outside the tool set is rejected
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallError,
    ToolCallSuccess,
)


class _RaisingOperations:
    """Defaults to raising on any call — vacuous-mock hazard prevention.

    Individual tests subclass or replace the specific method they
    exercise.
    """

    async def invoke(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError(
            f"operations.invoke should not be called in this test: {arguments!r}"
        )

    async def read_ensembles(self) -> list[dict[str, Any]]:  # pragma: no cover
        raise AssertionError(
            "operations.read_ensembles should not be called in this test"
        )


class _ScriptedOperations(_RaisingOperations):
    """Programmable ``EnsembleOperations`` double.

    Feeds canned ``invoke`` results and ensemble listings, records
    arguments for assertions, and optionally raises ``ValueError`` to
    simulate the "ensemble not found" path that the real
    ``ExecutionHandler.invoke`` surfaces.
    """

    def __init__(
        self,
        *,
        invoke_result: dict[str, Any] | None = None,
        invoke_raises: BaseException | None = None,
        ensembles: list[dict[str, Any]] | None = None,
    ) -> None:
        self._invoke_result = invoke_result
        self._invoke_raises = invoke_raises
        self._ensembles = list(ensembles or [])
        self.invoke_calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.invoke_calls.append(dict(arguments))
        if self._invoke_raises is not None:
            raise self._invoke_raises
        return self._invoke_result or {}

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return list(self._ensembles)


class TestDispatchRejectsUnknownTool:
    """Scenario: Invocation outside the tool set is rejected.

    Tool Dispatch is the structural enforcement point for ADR-003 —
    the closed tool set. A name outside the five committed tools
    returns a typed tool error the Runtime can surface to the
    orchestrator LLM as an observation; the ReAct loop continues.
    """

    @pytest.mark.asyncio
    async def test_dispatch_rejects_unknown_tool_name(self) -> None:
        dispatch = OrchestratorToolDispatch(operations=_RaisingOperations())

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_abc",
                name="hallucinated_tool",
                arguments={},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.id == "call_abc"
        assert result.name == "hallucinated_tool"
        assert result.kind == "unknown_tool"


class TestClosedToolSet:
    """Scenario: Orchestrator tool surface is exactly the committed set (FC-5).

    The closed-set property (ADR-003) is enforced structurally: the
    five tool names in ``TOOL_NAMES`` correspond to exactly five
    async methods on the dispatch class. A sixth public async tool
    method would mean the closed set is no longer closed.
    """

    def test_tool_dispatch_exposes_exactly_five_tool_methods(self) -> None:
        tool_methods = {
            name
            for name, member in inspect.getmembers(OrchestratorToolDispatch)
            if inspect.iscoroutinefunction(member) and name in TOOL_NAMES
        }

        assert tool_methods == {
            "invoke_ensemble",
            "compose_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        }
        assert len(tool_methods) == 5

    def test_tool_names_set_matches_committed_five(self) -> None:
        """ADR-003's tool set is what the module advertises as ``TOOL_NAMES``."""
        assert TOOL_NAMES == frozenset(
            {
                "invoke_ensemble",
                "compose_ensemble",
                "list_ensembles",
                "query_knowledge",
                "record_outcome",
            }
        )


class TestNotYetWiredTools:
    """WP-C wires only invoke_ensemble and list_ensembles.

    The other three tools exist (to honor the closed-set property) but
    return a typed ``not_yet_wired`` tool error. The Runtime surfaces
    these to the orchestrator LLM as an observation — the LLM is free
    to plan around them.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "landing_wp"),
        [
            ("compose_ensemble", "WP-G"),
            ("query_knowledge", "WP-I"),
            ("record_outcome", "WP-I"),
        ],
    )
    async def test_not_yet_wired_tool_returns_typed_error(
        self, tool_name: str, landing_wp: str
    ) -> None:
        dispatch = OrchestratorToolDispatch(operations=_RaisingOperations())

        result = await dispatch.dispatch(
            InternalToolCall(id="call_1", name=tool_name, arguments={})
        )

        assert isinstance(result, ToolCallError)
        assert result.name == tool_name
        assert result.kind == "not_yet_wired"
        assert landing_wp in result.reason


class TestListEnsembles:
    """list_ensembles delegates to ``EnsembleOperations.read_ensembles``."""

    @pytest.mark.asyncio
    async def test_list_ensembles_returns_library_entries(self) -> None:
        operations = _ScriptedOperations(
            ensembles=[
                {
                    "name": "analysis",
                    "description": "Analyzes code",
                    "source": "local",
                    "relative_path": "analysis.yaml",
                    "agent_count": 1,
                }
            ]
        )
        dispatch = OrchestratorToolDispatch(operations=operations)

        result = await dispatch.dispatch(
            InternalToolCall(id="call_1", name="list_ensembles", arguments={})
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "list_ensembles"
        entries = result.content
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert entries[0]["name"] == "analysis"
        assert entries[0]["description"] == "Analyzes code"


class TestInvokeEnsemble:
    """invoke_ensemble delegates to ``EnsembleOperations.invoke``.

    ``OrchestraService.invoke`` returns a normalized
    ``{results, synthesis, status}`` shape; that dict flows through as
    the ``ToolCallSuccess.content``. Missing-ensemble errors surface
    as ``ValueError`` from the handler and translate to a
    ``ToolCallError(kind="invocation_failed")``.
    """

    @pytest.mark.asyncio
    async def test_invoke_ensemble_delegates_to_operations(self) -> None:
        normalized_result = {
            "results": {"analyst": {"response": "ok"}},
            "synthesis": "analysis complete",
            "status": "success",
        }
        operations = _ScriptedOperations(invoke_result=normalized_result)
        dispatch = OrchestratorToolDispatch(operations=operations)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_7",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "refactor the parser"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.id == "call_7"
        assert result.name == "invoke_ensemble"
        assert result.content == normalized_result
        # Delegation uses the handler's field name (``ensemble_name``).
        assert operations.invoke_calls == [
            {"ensemble_name": "analysis", "input": "refactor the parser"}
        ]

    @pytest.mark.asyncio
    async def test_invoke_ensemble_returns_error_when_name_not_in_library(
        self,
    ) -> None:
        """A hallucinated ensemble name becomes an observation, not a crash.

        The real handler raises ``ValueError("Ensemble does not exist: ...")``
        — Tool Dispatch converts that to ``ToolCallError``.
        """
        operations = _ScriptedOperations(
            invoke_raises=ValueError("Ensemble does not exist: does-not-exist")
        )
        dispatch = OrchestratorToolDispatch(operations=operations)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_8",
                name="invoke_ensemble",
                arguments={"name": "does-not-exist", "input": "anything"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.name == "invoke_ensemble"
        assert result.kind == "invocation_failed"
        assert "does-not-exist" in result.reason

    @pytest.mark.asyncio
    async def test_invoke_ensemble_rejects_missing_name_argument(self) -> None:
        """Input validation — missing or empty ``name`` is a typed error
        surfaced without calling the handler."""
        operations = _ScriptedOperations()  # invoke would return {} if called
        dispatch = OrchestratorToolDispatch(operations=operations)

        result = await dispatch.dispatch(
            InternalToolCall(id="call_9", name="invoke_ensemble", arguments={})
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invalid_arguments"
        # Handler must not have been called — validation is local.
        assert operations.invoke_calls == []
