"""Orchestrator Runtime — the ReAct loop behind the serving layer.

Per ``docs/agentic-serving/system-design.md`` §Orchestrator Runtime
(L2) and §Integration Contracts (Serving Layer → Orchestrator Runtime,
Orchestrator Runtime → Budget Controller, Orchestrator Runtime →
Orchestrator Tool Dispatch).

The Runtime drives the orchestrator LLM through a tool-calling loop:

1. Check Budget before each iteration (FC-10; AS-3).
2. Call the LLM with the session's messages and the five-tool schema.
3. If the LLM emits tool calls, dispatch each through Orchestrator
   Tool Dispatch, feed the results back as observations, and loop.
4. If the LLM emits a stop response with no tool calls, terminate
   with ``Completion(finish_reason="stop")``.
5. On Budget exhaustion, terminate with an explicit ContentDelta
   explaining the exhaustion followed by ``Completion("length")`` —
   the termination is not silent.

FC-4: the Runtime imports only BudgetController and a dispatcher
Protocol — no Plexus, no config, no Autonomy, no Calibration. Result
summarization is interposed by Orchestrator Tool Dispatch (WP-D adds
the dedicated Harness). The ``OrchestratorLLM`` Protocol keeps the
Runtime decoupled from any particular model client — WP-C tests pass
a scripted double; production wiring passes a tool-calling adapter.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

from llm_orc.agentic.budget_controller import (
    BudgetCheckExhausted,
    BudgetController,
)
from llm_orc.agentic.orchestrator_chunk import (
    Completion,
    ContentDelta,
    InternalToolCallInFlight,
    InternalToolCallResult,
    OrchestratorChunk,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_start import SessionContext
from llm_orc.models.base import ToolCallingResponse


class OrchestratorLLM(Protocol):
    """Tool-calling LLM interface the Runtime drives.

    ``ModelInterface`` satisfies this Protocol structurally — any
    provider that implements ``generate_with_tools`` (opt-in per the
    ``supports_tool_calling`` flag) works here. The Protocol keeps
    the Runtime decoupled from ``ModelInterface``'s abstract
    ``name`` and ``generate_response`` surface so tests can pass
    minimal doubles.
    """

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse: ...


class ToolDispatcher(Protocol):
    """Minimum Tool Dispatch surface the Runtime calls."""

    async def dispatch(self, call: InternalToolCall) -> ToolCallResult: ...


def _build_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-compatible tool schemas for the closed five-tool set.

    Argument schemas are intentionally minimal for WP-C — the LLM sees
    enough to emit valid calls; richer schemas can land with each tool's
    wiring WP (G for compose_ensemble, I for query_knowledge /
    record_outcome).
    """
    specs = {
        "invoke_ensemble": {
            "description": "Execute an ensemble by name with input text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the library ensemble to invoke.",
                    },
                    "input": {
                        "type": "string",
                        "description": "Task input text passed to the ensemble.",
                    },
                },
                "required": ["name", "input"],
            },
        },
        "compose_ensemble": {
            "description": (
                "Compose a new ensemble from library primitives. Not yet wired."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        "list_ensembles": {
            "description": "List ensembles available in the library.",
            "parameters": {"type": "object", "properties": {}},
        },
        "query_knowledge": {
            "description": (
                "Query the knowledge graph for prior outcomes. Not yet wired."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        "record_outcome": {
            "description": ("Record a routing decision or outcome. Not yet wired."),
            "parameters": {"type": "object", "properties": {}},
        },
    }
    # Ensure the closed set is exactly the keys we advertise.
    assert set(specs) == TOOL_NAMES
    return [
        {
            "type": "function",
            "function": {"name": name, **spec},
        }
        for name, spec in specs.items()
    ]


class OrchestratorRuntime:
    """Drives the orchestrator's ReAct loop, yielding streaming chunks."""

    def __init__(
        self,
        *,
        llm: OrchestratorLLM,
        budget: BudgetController,
        tool_dispatch: ToolDispatcher,
    ) -> None:
        self._llm = llm
        self._budget = budget
        self._tool_dispatch = tool_dispatch

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        """Run the ReAct loop for this session turn."""
        state = context.state
        messages: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in context.messages
        ]
        tools = _build_tool_schemas()

        while True:
            check = self._budget.check(
                turn_count=state.turn_count,
                token_spend=state.token_spend,
            )
            if isinstance(check, BudgetCheckExhausted):
                yield ContentDelta(content=_format_exhaustion_message(check))
                yield Completion(finish_reason="length")
                return

            response = await self._llm.generate_with_tools(
                messages=messages, tools=tools
            )
            state.record_iteration(response.usage.total_tokens)

            if response.content:
                yield ContentDelta(content=response.content)

            if not response.tool_calls:
                yield Completion(finish_reason="stop")
                return

            messages.append(_assistant_message(response))
            for tool_call in response.tool_calls:
                yield InternalToolCallInFlight(id=tool_call.id, name=tool_call.name)
                parsed = InternalToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=_safe_parse_arguments(tool_call.arguments_json),
                )
                result = await self._tool_dispatch.dispatch(parsed)
                for event in result.events:
                    yield event
                yield InternalToolCallResult(
                    id=result.id, summary=_tool_result_summary(result)
                )
                messages.append(_tool_result_message(result))


def _safe_parse_arguments(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON-string arguments, defaulting to empty on bad JSON.

    Invalid JSON becomes an empty dict; the tool handler surfaces any
    argument-validation error as a tool observation (see Tool Dispatch
    ``invalid_arguments`` handling). The Runtime does not distinguish
    malformed JSON from missing fields — both become validation errors
    at the tool boundary.
    """
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_message(response: ToolCallingResponse) -> dict[str, Any]:
    """Assistant-turn message containing tool calls for the next iteration."""
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments_json},
            }
            for tc in response.tool_calls
        ],
    }


def _tool_result_summary(result: ToolCallResult) -> str:
    """Produce a short summary string for the ``InternalToolCallResult`` chunk.

    WP-C uses a trivial format (success/error + tool name). WP-D's
    Result Summarizer Harness produces the properly-summarized content
    for AS-7 compliance; this placeholder will be superseded.
    """
    if isinstance(result, ToolCallSuccess):
        return f"ok: {result.name}"
    return f"error: {result.name} ({result.kind})"


def _tool_result_message(result: ToolCallResult) -> dict[str, Any]:
    """``role: tool`` observation fed back to the LLM for the next iteration."""
    content = (
        json.dumps(result.content, default=str)
        if isinstance(result, ToolCallSuccess)
        else json.dumps({"error": result.kind, "reason": result.reason})
    )
    return {
        "role": "tool",
        "tool_call_id": result.id,
        "content": content,
    }


def _format_exhaustion_message(check: BudgetCheckExhausted) -> str:
    """Human-readable termination message surfaced to the client."""
    if check.reason == "turn_limit":
        return (
            f"[Session budget exhausted: turn limit reached "
            f"({check.turn_count}/{check.turn_limit})]"
        )
    return (
        f"[Session budget exhausted: token limit reached "
        f"({check.token_spend}/{check.token_limit})]"
    )
