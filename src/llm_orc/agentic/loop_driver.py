"""Loop Driver (Cycle 7 loop-back WP-LB-B, ADR-033) — L2.

The layer-A control structure for the tool-driven multi-turn chat-
completions surface — the role no ADR-027 component held (the routing
planner decides *which capability*, the synthesizer composes a response;
neither *drives*). The Serving Layer's surface-mode discriminator engages
the tool-driven surface when a request carries client ``tools[]``; the
Client-Tool-Action Terminal (ADR-034) composes the Loop Driver and emits
the wire response.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Loop Driver.
The driver occupies the client's "model" seat (ADR-033 §Decision 5): the
model the client talks to *is* the loop-driver, with the framework
interposing single-step enforcement and per-turn ensemble delegation. Each
``decide`` is **one turn** — it returns a :class:`TurnOutcome` the Terminal
marshals into the wire response. The multi-turn loop is realized across HTTP
requests (the client executes the emitted tool call locally and returns its
result in a follow-up request; ADR-034's terminal participates).

The driver owns the per-turn *decision* — the tool *choice* and the callee
generation dispatch — but **not** the deliverable-content marshalling or the
wire emission, which the Client-Tool-Action Terminal owns (ADR-034). So
``decide`` carries the deliverable *envelope* (ADR-024) in an
:class:`ApplyWork` outcome rather than baking content into a tool call; the
Terminal resolves full-fidelity content (substrate-routed deliverables via
the Artifact Bridge, ADR-025/ADR-034 §Decision 3) before emitting.

**Seat-filler contract (system-design Amendment, BUILD-resolved 2026-06-02).**
The seat-filler decides one of two things per turn, and the framework
truncates any batch to that one action (Single-Step Enforcer, ADR-033 §3):

* a **client tool call** (``write``/``edit``/``bash``/``read``) with literal
  arguments — passed through to the client **verbatim**. This is the
  grounded-carry path: a value observed in a prior tool result, which the
  seat-filler read from the conversation, reaches the client tool call
  argument unchanged (no ``${...}`` template, no fabrication; FC-45). The
  driver returns it as a :class:`CarryClientTool` the Terminal emits as-is;
* an internal **``invoke_ensemble``** call — the per-turn **callee**
  generation delegation. The driver dispatches a *single* capability
  ensemble through the existing Tool Dispatch chokepoint (no routing-planner,
  no response-synthesizer; FC-44), then **maps** the deliverable to a client
  tool call (the tool-mapping decision the Loop Driver owns) and returns it
  as an :class:`ApplyWork` carrying the deliverable envelope.

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
from dataclasses import dataclass
from typing import Any, Protocol

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.orchestrator_chunk import ToolCallInvocation
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_start import SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall, ToolCallingResponse

__all__ = [
    "ApplyWork",
    "CarryClientTool",
    "FinishTurn",
    "LoopDriver",
    "SeatFiller",
    "ToolDispatcher",
    "TurnDecision",
    "TurnOutcome",
]

_GENERATION_TOOL = "invoke_ensemble"
"""The internal tool name the seat-filler emits to delegate per-turn
content generation to a capability ensemble (the callee). Any other tool
name is a client tool call passed through verbatim."""


@dataclass(frozen=True)
class ApplyWork:
    """A generation turn — apply ensemble-generated work to the client.

    The Loop Driver dispatched the per-turn callee ensemble (FC-44) and
    decided the tool-mapping (``write`` for new files — ADR-034 §Decision 4).
    It carries the deliverable ``envelope`` (ADR-024) — *not* the content —
    so the Client-Tool-Action Terminal resolves full-fidelity content
    (substrate-routed deliverables via the Artifact Bridge; FC-49) before
    emitting the tool call.
    """

    invocation_id: str
    tool_name: str
    file_path: str
    envelope: DispatchEnvelope
    delegated_ensemble: str


@dataclass(frozen=True)
class CarryClientTool:
    """A grounded-carry turn — a literal client tool call passed through verbatim.

    The seat-filler's exact arguments are preserved (FC-45): a value observed
    in a prior tool result reaches the client tool-call argument unchanged.
    No ensemble dispatch, no bridge marshalling; the Terminal emits the
    invocation as-is.
    """

    invocation: ToolCallInvocation


@dataclass(frozen=True)
class FinishTurn:
    """A finish turn — optional assistant text, then a stop completion.

    The safe terminal (ADR-033 §Decision 1) that makes engaging the driver on
    tools-presence safe: a tool-capable client asking a plain question is
    served correctly. ``content`` is ``None`` when the seat-filler proposed no
    text, so the Terminal emits a bare ``Completion``.
    """

    content: str | None


TurnOutcome = ApplyWork | CarryClientTool | FinishTurn
"""The per-turn decision the Loop Driver returns and the Terminal emits.

A flat union (not a base class) so exhaustiveness over the three outcomes is
visible in the Terminal's ``isinstance`` dispatch.
"""


@dataclass(frozen=True)
class TurnDecision:
    """Per-turn diagnostic event for axis-2 split-vs-callee diagnosis (FC-51).

    Emitted once per turn through the Dispatch Event Substrate. The fields
    let a failing long-horizon run be reconstructed and classified:

    * ``action`` — the client tool the driver decided (``write``/``edit``/
      ``bash``/``read``) or ``"finish"``. A wrong ``action`` points at the
      driver (split-incorrect).
    * ``delegated_ensemble`` — the capability ensemble a generation turn
      delegated to, else ``None``. Wrong content from a named ensemble points
      at the callee (callee-incorrect).
    * ``grounded_carry_held`` — the turn was a literal client-tool passthrough
      (an observed value carried verbatim), not a generation.
    * ``replanned_after_truncation`` — the Single-Step Enforcer truncated a
      proposed batch this turn, so the driver re-plans next turn.

    ``dispatch_id`` is the ADR-023 correlation identifier (``None`` when no
    substrate is configured), satisfying the substrate's ``DispatchEvent``
    protocol.
    """

    dispatch_id: str | None
    turn_index: int
    action: str
    delegated_ensemble: str | None
    grounded_carry_held: bool
    replanned_after_truncation: bool


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
        event_substrate: DispatchEventSubstrate | None = None,
    ) -> None:
        self._seat_filler = seat_filler
        self._enforcer = enforcer
        self._tool_dispatch = tool_dispatch
        self._event_substrate = event_substrate

    async def decide(self, context: SessionContext) -> TurnOutcome:
        """Decide one turn of the tool-driven multi-turn loop.

        Returns the per-turn :class:`TurnOutcome`; the Client-Tool-Action
        Terminal (ADR-034) marshals it into the wire response. Each ``decide``
        is one turn — the multi-turn loop is realized across HTTP requests.
        """
        response = await self._seat_filler.generate_with_tools(
            messages=_to_openai_messages(context),
            tools=context.tools,
        )
        enforced = self._enforcer.enforce(response.tool_calls)
        dispatch_id = self._new_dispatch_id(context)
        turn_index = _turn_index(context)

        if enforced.action is None:
            self._emit_turn_decision(
                dispatch_id, turn_index, "finish", None, False, enforced.truncated
            )
            return FinishTurn(content=response.content or None)

        action = enforced.action
        if action.name == _GENERATION_TOOL:
            envelope, ensemble, file_path = await self._delegate_generation(
                action, context
            )
            self._emit_turn_decision(
                dispatch_id, turn_index, "write", ensemble, False, enforced.truncated
            )
            return ApplyWork(
                invocation_id=action.id,
                tool_name="write",
                file_path=file_path,
                envelope=envelope,
                delegated_ensemble=ensemble,
            )

        invocation = _passthrough_client_tool(action)
        self._emit_turn_decision(
            dispatch_id, turn_index, invocation.name, None, True, enforced.truncated
        )
        return CarryClientTool(invocation=invocation)

    async def _delegate_generation(
        self, action: ToolCall, context: SessionContext
    ) -> tuple[DispatchEnvelope, str, str]:
        """Dispatch the per-turn callee ensemble and return its deliverable envelope.

        The seat-filler's ``invoke_ensemble`` call names the capability and
        the generation task (selected by task content — AS-10); the driver
        dispatches that single ensemble (no routing-planner / synthesizer
        stage — FC-44) and returns the deliverable *envelope* (ADR-024) for
        the Terminal to marshal. Returns ``(envelope, capability, file_path)``;
        the capability name feeds the ``TurnDecision`` diagnostic. Richer
        tool-mapping (``edit``/``bash``) and capability-list validation are
        deferred to the WP-D Capability List Builder integration.
        """
        args = _parse_arguments(action.arguments_json)
        capability = _string_field(args, "name")
        file_path = _string_field(args, "filePath")
        result = await self._tool_dispatch.dispatch(
            InternalToolCall(
                id=action.id,
                name=_GENERATION_TOOL,
                arguments={"name": capability, "input": _string_field(args, "input")},
            ),
            session_id=context.state.identity.value,
        )
        return _result_to_envelope(result), capability, file_path

    def _new_dispatch_id(self, context: SessionContext) -> str | None:
        if self._event_substrate is None:
            return None
        return self._event_substrate.new_dispatch_id(context.state.identity.value)

    def _emit_turn_decision(
        self,
        dispatch_id: str | None,
        turn_index: int,
        action: str,
        delegated_ensemble: str | None,
        grounded_carry_held: bool,
        replanned_after_truncation: bool,
    ) -> None:
        if self._event_substrate is None:
            return
        self._event_substrate.emit(
            TurnDecision(
                dispatch_id=dispatch_id,
                turn_index=turn_index,
                action=action,
                delegated_ensemble=delegated_ensemble,
                grounded_carry_held=grounded_carry_held,
                replanned_after_truncation=replanned_after_truncation,
            )
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


def _turn_index(context: SessionContext) -> int:
    """The 1-based position of this turn in the trajectory.

    Each ``decide`` is one assistant turn; the multi-turn loop spans requests,
    so the index is recovered from the conversation — one past the count of
    prior assistant turns. Lets the ``TurnDecision`` stream order a
    long-horizon run even though the driver is stateless across requests.
    """
    prior_assistant_turns = sum(
        1 for message in context.messages if message.role == "assistant"
    )
    return prior_assistant_turns + 1


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


def _result_to_envelope(result: ToolCallResult) -> DispatchEnvelope:
    """Extract the deliverable envelope from a per-turn dispatch result.

    A successful dispatch carries the typed :class:`DispatchEnvelope`
    (ADR-024); substrate-routed deliverables (``envelope.artifacts`` populated
    per ADR-025) are resolved to full fidelity by the Artifact Bridge in the
    Terminal (FC-49). A success without an envelope wraps ``content`` inline;
    a typed error becomes an error envelope the Terminal renders into the
    response.
    """
    if isinstance(result, ToolCallSuccess):
        if result.envelope is not None:
            return result.envelope
        return DispatchEnvelope(status="success", primary=str(result.content))
    return DispatchEnvelope(
        status="error",
        primary=f"[dispatch error: {result.kind}] {result.reason}",
    )
