"""Loop Driver (Cycle 7 loop-back WP-LB-B, ADR-033) — L2.

The layer-A control structure for the tool-driven multi-turn chat-
completions surface — the role no ADR-027 component held (the routing
planner decides *which capability*, the synthesizer composes a response;
neither *drives*). The Serving Layer's surface-mode discriminator engages
the Loop Driver when a request carries client ``tools[]``.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Loop Driver.
The driver occupies the client's "model" seat (ADR-033 §Decision 5): the
model the client talks to *is* the loop-driver, with the framework
interposing single-step enforcement and per-turn ensemble delegation. Each
``run`` is **one turn** — the multi-turn loop is realized across HTTP
requests (the client executes the emitted tool call locally and returns
its result in a follow-up request; ADR-034's terminal participates).

**Seat-filler contract (system-design Amendment, BUILD-resolved 2026-06-02).**
The seat-filler decides one of two things per turn, and the framework
truncates any batch to that one action (Single-Step Enforcer, ADR-033 §3):

* a **client tool call** (``write``/``edit``/``bash``/``read``) with literal
  arguments — passed through to the client **verbatim**. This is the
  grounded-carry path: a value observed in a prior tool result, which the
  seat-filler read from the conversation, reaches the client tool call
  argument unchanged (no ``${...}`` template, no fabrication; FC-45);
* an internal **``invoke_ensemble``** call — the per-turn **callee**
  generation delegation. The driver dispatches a *single* capability
  ensemble through the existing Tool Dispatch chokepoint (no routing-planner,
  no response-synthesizer; FC-44), then **maps** the deliverable to a client
  tool call (the tool-mapping decision the Loop Driver owns).

This keeps generated content delegated to ensembles (the cost-distribution
value proposition) while keeping literal/observed values exact — the
distinction the grounded-carry path needs. The seat-filler emits
``invoke_ensemble`` for content it wants generated and a client tool call
for content it already determines.

The seat-filler is injected (a swappable Model Profile per ADR-011) so a
driver-model swap (cheap-tier ↔ frontier-tier — the named axis-2 fallback)
touches only configuration, never this control structure (FC-46). The
driver depends on the ``SeatFiller`` *protocol*, not on Orchestrator
Configuration — it does not import the L3 config module.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_start import SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall, ToolCallingResponse

__all__ = ["LoopDriver", "SeatFiller", "ToolDispatcher"]

_GENERATION_TOOL = "invoke_ensemble"
"""The internal tool name the seat-filler emits to delegate per-turn
content generation to a capability ensemble (the callee). Any other tool
name is a client tool call passed through verbatim."""


class SeatFiller(Protocol):
    """The tool-calling LLM that fills the client's "model" seat.

    Structurally identical to the Orchestrator Runtime's ``OrchestratorLLM``
    port — any provider implementing ``generate_with_tools`` (opt-in per the
    ``supports_tool_calling`` flag) fills the seat. Defined here as the
    narrow port the Loop Driver needs, following the per-module-port pattern
    the Dispatch Pipeline uses, so the driver stays decoupled from the
    Runtime and from ``ModelInterface``'s wider surface.
    """

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse: ...


class ToolDispatcher(Protocol):
    """The Orchestrator Tool Dispatch surface the driver delegates through.

    The same ``dispatch`` chokepoint the single-turn Dispatch Pipeline uses,
    so the per-turn callee generation rides the existing calibration-gate +
    tier-router + autonomy interpositions.
    """

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult: ...


class LoopDriver:
    """Layer-A multi-turn control structure (callee delegation)."""

    def __init__(
        self,
        *,
        seat_filler: SeatFiller,
        enforcer: SingleStepEnforcer,
        tool_dispatch: ToolDispatcher,
    ) -> None:
        self._seat_filler = seat_filler
        self._enforcer = enforcer
        self._tool_dispatch = tool_dispatch

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        """Drive one turn of the tool-driven multi-turn loop."""
        response = await self._seat_filler.generate_with_tools(
            messages=_to_openai_messages(context),
            tools=context.tools,
        )
        enforced = self._enforcer.enforce(response.tool_calls)

        if enforced.action is None:
            if response.content:
                yield ContentDelta(content=response.content)
            yield Completion(finish_reason="stop")
            return

        action = enforced.action
        if action.name == _GENERATION_TOOL:
            invocation = await self._delegate_generation(action, context)
        else:
            invocation = _passthrough_client_tool(action)
        yield ClientToolCall(tool_calls=(invocation,))

    async def _delegate_generation(
        self, action: ToolCall, context: SessionContext
    ) -> ToolCallInvocation:
        """Dispatch the per-turn callee ensemble and map its deliverable.

        The seat-filler's ``invoke_ensemble`` call names the capability and
        the generation task (selected by task content — AS-10); the driver
        dispatches that single ensemble (no routing-planner / synthesizer
        stage — FC-44) and marshals the deliverable into a client ``write``.
        Richer tool-mapping (``edit``/``bash``) and capability-list
        validation are deferred to the WP-D Capability List Builder
        integration.
        """
        args = _parse_arguments(action.arguments_json)
        result = await self._tool_dispatch.dispatch(
            InternalToolCall(
                id=action.id,
                name=_GENERATION_TOOL,
                arguments={
                    "name": _string_field(args, "name"),
                    "input": _string_field(args, "input"),
                },
            ),
            session_id=context.state.identity.value,
        )
        client_arguments = json.dumps(
            {
                "filePath": _string_field(args, "filePath"),
                "content": _deliverable_text(result),
            }
        )
        return ToolCallInvocation(
            id=action.id, name="write", arguments=client_arguments
        )


def _passthrough_client_tool(action: ToolCall) -> ToolCallInvocation:
    """Carry a literal client tool call through to the client verbatim.

    The seat-filler's exact ``arguments_json`` is preserved unchanged — the
    grounded-carry guarantee (FC-45): an observed value the seat-filler placed
    in the argument is not regenerated, summarized, or templated.
    """
    return ToolCallInvocation(
        id=action.id, name=action.name, arguments=action.arguments_json
    )


def _to_openai_messages(context: SessionContext) -> list[dict[str, Any]]:
    """Project the session's chat messages into the seat-filler's payload.

    Carries ``role`` and ``content`` for every message, so a prior observed
    tool result (a ``role: tool`` message) is surfaced to the seat-filler —
    the precondition for grounded carry. Fuller tool-round-trip fidelity
    (``tool_call_id``, the assistant turn's ``tool_calls``) is WP-LB-C
    loop-participation work.
    """
    return [
        {"role": message.role, "content": message.content}
        for message in context.messages
    ]


def _parse_arguments(arguments_json: str) -> dict[str, Any]:
    """Parse a tool call's JSON arguments, tolerating malformed input."""
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string_field(arguments: dict[str, Any], key: str) -> str:
    """Read a string-typed argument field, defaulting to empty."""
    value = arguments.get(key, "")
    return value if isinstance(value, str) else ""


def _deliverable_text(result: ToolCallResult) -> str:
    """Extract the deliverable content from a per-turn dispatch result.

    A successful inline ensemble carries the deliverable on
    ``envelope.primary`` (ADR-024). Substrate-routed deliverables — where
    ``envelope.primary`` is a summary plus an ``ArtifactReference`` — are
    resolved to full fidelity by the Artifact Bridge (WP-LB-D); WP-LB-B
    marshals the inline ``primary``. A typed error is rendered as text.
    """
    if isinstance(result, ToolCallSuccess):
        if result.envelope is not None:
            return result.envelope.primary
        return str(result.content)
    return f"[dispatch error: {result.kind}] {result.reason}"
