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

Per turn the driver:

* invokes the injected **seat-filler** LLM to decide the next agentic step
  (which client tool to call, or finish), conditioned on the conversation
  including any prior observed tool result;
* enforces **single-action-per-turn** via the Single-Step Enforcer
  (batch-truncation — the framework's grounding guarantee, ADR-033 §3);
* delegates per-turn content generation to a **single capability ensemble**
  (the callee — not the ``plan → dispatch → synthesize`` pipeline; WP-LB-B
  callee delegation lands the generation path);
* emits a per-turn ``TurnDecision`` diagnostic for axis-2 split-vs-callee
  diagnosis (FC-51).

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

__all__ = ["CapabilitySelector", "LoopDriver", "SeatFiller", "ToolDispatcher"]


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


class CapabilitySelector(Protocol):
    """Selects the capability ensemble for a generation turn (AS-10).

    The selection is a function of the turn's task content, never the
    client-declared tools. WP-LB-B injects a selector backed by a stub list;
    the single-turn WP-D Capability List Builder is the shared production
    source.
    """

    def select(self, *, task: str) -> str | None: ...


class LoopDriver:
    """Layer-A multi-turn control structure (callee delegation)."""

    def __init__(
        self,
        *,
        seat_filler: SeatFiller,
        enforcer: SingleStepEnforcer,
        tool_dispatch: ToolDispatcher,
        capability_selector: CapabilitySelector,
    ) -> None:
        self._seat_filler = seat_filler
        self._enforcer = enforcer
        self._tool_dispatch = tool_dispatch
        self._capability_selector = capability_selector

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

        invocation = await self._generate_action(enforced.action, context)
        yield ClientToolCall(tool_calls=(invocation,))

    async def _generate_action(
        self, action: ToolCall, context: SessionContext
    ) -> ToolCallInvocation:
        """Resolve a client tool call's content via the per-turn callee.

        A generation action (``write``/``edit``) delegates content to a
        single capability ensemble selected by task content; the ensemble's
        deliverable becomes the tool call's ``content`` (callee — not the
        plan -> dispatch -> synthesize pipeline). The path and tool name come
        from the seat-filler's decision.
        """
        task = _generation_task(context)
        capability = self._capability_selector.select(task=task)
        result = await self._tool_dispatch.dispatch(
            InternalToolCall(
                id=action.id,
                name="invoke_ensemble",
                arguments={"name": capability, "input": task},
            ),
            session_id=context.state.identity.value,
        )
        arguments = json.dumps(
            {"filePath": _action_path(action), "content": _deliverable_text(result)}
        )
        return ToolCallInvocation(id=action.id, name=action.name, arguments=arguments)


def _to_openai_messages(context: SessionContext) -> list[dict[str, Any]]:
    """Project the session's chat messages into the seat-filler's payload.

    Carries ``role`` and ``content`` for every message. Tool-round-trip
    fields (``tool_call_id``, ``tool_calls``) are surfaced by WP-LB-B's
    grounded-carry path / WP-LB-C's loop participation so the seat-filler
    observes prior tool results.
    """
    return [
        {"role": message.role, "content": message.content}
        for message in context.messages
    ]


def _generation_task(context: SessionContext) -> str:
    """The task string the per-turn callee ensemble generates content for.

    WP-LB-B derives it from the latest user message (matching the single-turn
    pipeline's request extraction). Richer per-step task derivation — drawing
    on the seat-filler's stated intent for the current step — is a refinement
    tracked for the multi-turn build-out.
    """
    for message in reversed(context.messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


def _action_path(action: ToolCall) -> str:
    """Extract the target file path from the seat-filler's tool-call args."""
    try:
        arguments = json.loads(action.arguments_json)
    except json.JSONDecodeError:
        return ""
    path = arguments.get("filePath", "")
    return path if isinstance(path, str) else ""


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
