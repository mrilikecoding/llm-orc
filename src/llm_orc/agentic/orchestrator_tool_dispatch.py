"""Orchestrator Tool Dispatch — closed five-tool surface (ADR-003).

Per ``docs/agentic-serving/system-design.md`` §Orchestrator Tool
Dispatch (L2) and §Integration Contracts (Orchestrator Runtime →
Orchestrator Tool Dispatch, Orchestrator Tool Dispatch → Result
Summarizer Harness). The Runtime calls ``dispatch(call)`` with an
``InternalToolCall``; this module routes by name through the closed
set or returns a typed tool error for names outside the set.

The closed-set property is structurally enforced: the five tool names
live in ``TOOL_NAMES`` and correspond to five async methods on this
class. FC-5 checks the count of public async methods whose names are
in ``TOOL_NAMES`` — exactly five.

Per system design Amendment #3, Tool Dispatch also interposes the
Result Summarizer Harness on ``invoke_ensemble``'s return path.
Unsummarized ensemble results never reach the Orchestrator Runtime
unless the invoked ensemble's ``raw_output`` flag is set (ADR-004
escape hatch). The Runtime is unaware of the Harness — summarization
is a Tool-Dispatch-side concern (FC-4, FC-8).

WP-C wires ``invoke_ensemble`` and ``list_ensembles`` by delegating to
the project's existing ``OrchestraService`` (invoke + read_ensembles).
This avoids a parallel find-and-execute code path — the orchestrator
tool surface is a thin adapter on the existing ensemble-operations
facade, not a re-implementation.

``compose_ensemble`` (WP-G), ``query_knowledge`` (WP-I), and
``record_outcome`` (WP-I) return typed not-yet-wired errors so the
closed-set property holds from WP-C onward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from llm_orc.agentic.result_summarizer_harness import (
    RawOutputPassthrough,
    ResultSummarizerHarness,
    SummarizationFailure,
    SummarizationSuccess,
)

TOOL_NAMES: frozenset[str] = frozenset(
    {
        "invoke_ensemble",
        "compose_ensemble",
        "list_ensembles",
        "query_knowledge",
        "record_outcome",
    }
)
"""The closed tool set committed by ADR-003."""


ToolErrorKind = Literal[
    "unknown_tool",
    "not_yet_wired",
    "invocation_failed",
    "invalid_arguments",
    "summarization_failed",
]


@dataclass(frozen=True)
class InternalToolCall:
    """A tool call emitted by the orchestrator LLM.

    ``arguments`` is pre-parsed by the Runtime — the orchestrator LLM
    emits JSON-string arguments per OpenAI convention; the Runtime
    parses before handing off so JSON handling is centralized.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallSuccess:
    """A successful tool result surfaced to the orchestrator's context."""

    id: str
    name: str
    content: Any


@dataclass(frozen=True)
class ToolCallError:
    """A failed tool call surfaced as an observation, not an exception.

    The ReAct loop continues with this result — the LLM sees the
    error, may adjust its plan, and emits the next tool call.
    """

    id: str
    name: str
    kind: ToolErrorKind
    reason: str


ToolCallResult = ToolCallSuccess | ToolCallError


class EnsembleOperations(Protocol):
    """Narrow facade over ensemble invocation and listing.

    ``OrchestraService`` satisfies this structurally; tests pass a
    handwritten double. The Protocol names only the two operations
    the orchestrator tool surface delegates to, so Tool Dispatch is
    decoupled from the full service construction surface.
    """

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def read_ensembles(self) -> list[dict[str, Any]]: ...


class OrchestratorToolDispatch:
    """Closed five-tool dispatch surface (ADR-003, FC-5)."""

    def __init__(
        self,
        *,
        operations: EnsembleOperations,
        harness: ResultSummarizerHarness,
    ) -> None:
        self._operations = operations
        self._harness = harness

    async def dispatch(self, call: InternalToolCall) -> ToolCallResult:
        """Route a tool call by name through the closed set.

        Match-case makes the five committed tools visible at the
        dispatch site. A name outside the set falls through to the
        ``_`` arm and becomes a typed ``unknown_tool`` error — the
        ReAct loop continues with the error as an observation.
        """
        match call.name:
            case "invoke_ensemble":
                return await self.invoke_ensemble(call.id, call.arguments)
            case "compose_ensemble":
                return await self.compose_ensemble(call.id, call.arguments)
            case "list_ensembles":
                return await self.list_ensembles(call.id, call.arguments)
            case "query_knowledge":
                return await self.query_knowledge(call.id, call.arguments)
            case "record_outcome":
                return await self.record_outcome(call.id, call.arguments)
            case _:
                return ToolCallError(
                    id=call.id,
                    name=call.name,
                    kind="unknown_tool",
                    reason=(
                        f"tool '{call.name}' is not in the orchestrator's "
                        "committed tool set"
                    ),
                )

    async def invoke_ensemble(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Resolve an ensemble by name, execute, then interpose the Harness.

        Per Amendment #3: on every successful invocation, Tool Dispatch
        calls :class:`ResultSummarizerHarness` with the raw result and
        the invoked ensemble's ``raw_output`` flag. The Runtime never
        sees unsummarized output unless the ensemble explicitly opts
        into the ADR-004 escape hatch.
        """
        name = arguments.get("name")
        input_data = arguments.get("input", "")
        if not isinstance(name, str) or not name:
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invalid_arguments",
                reason="invoke_ensemble requires 'name' (non-empty string)",
            )
        if not isinstance(input_data, str):
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invalid_arguments",
                reason="invoke_ensemble 'input' must be a string",
            )

        try:
            result = await self._operations.invoke(
                {"ensemble_name": name, "input": input_data}
            )
        except ValueError as exc:
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invocation_failed",
                reason=str(exc),
            )

        raw_output = bool(result.get("raw_output", False))
        summarization = await self._harness.summarize(result, raw_output=raw_output)
        match summarization:
            case SummarizationSuccess(summary=summary):
                return ToolCallSuccess(
                    id=id_,
                    name="invoke_ensemble",
                    content={"summary": summary},
                )
            case RawOutputPassthrough(content=passthrough):
                return ToolCallSuccess(
                    id=id_, name="invoke_ensemble", content=passthrough
                )
            case SummarizationFailure(reason=reason):
                return ToolCallError(
                    id=id_,
                    name="invoke_ensemble",
                    kind="summarization_failed",
                    reason=reason,
                )

    async def compose_ensemble(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Compose a new ensemble (WP-G)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="compose_ensemble",
            kind="not_yet_wired",
            reason="compose_ensemble lands in WP-G (Composition Validator)",
        )

    async def list_ensembles(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Enumerate library ensembles via ``OrchestraService.read_ensembles``."""
        del arguments
        entries = await self._operations.read_ensembles()
        return ToolCallSuccess(id=id_, name="list_ensembles", content=entries)

    async def query_knowledge(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Query the knowledge graph (WP-I Plexus Adapter)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="query_knowledge",
            kind="not_yet_wired",
            reason="query_knowledge lands in WP-I (Plexus Adapter)",
        )

    async def record_outcome(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Record a routing decision or outcome (WP-I Plexus Adapter)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="record_outcome",
            kind="not_yet_wired",
            reason="record_outcome lands in WP-I (Plexus Adapter)",
        )
