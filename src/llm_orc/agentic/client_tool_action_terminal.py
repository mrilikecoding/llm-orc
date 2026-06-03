"""Client-Tool-Action Terminal (Cycle 7 loop-back WP-LB-C, ADR-034) — L3.

Emits an ensemble deliverable as a ``tool_calls`` response the client
executes locally, and participates in the client's multi-turn tool loop.
The Serving Layer's surface-mode discriminator engages the Terminal for
tool-driven requests (client ``tools[]`` present); the Terminal composes
the Loop Driver (ADR-033, which decides the per-turn action) and emits the
wire response.

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Client-Tool-Action Terminal. The split is load-bearing: the **Loop Driver
owns the tool *choice*** (which tool, which deliverable, the callee
dispatch); the **Terminal owns the tool *emission*** — turning the
per-turn :class:`~llm_orc.agentic.loop_driver.TurnOutcome` into the shared
``OrchestratorChunk`` vocabulary the SSE formatter and non-streaming
collector consume. The terminal emits ``tool_calls`` rather than writing
server-side because the client's *execution model* is load-bearing (Spike π
Phase A established a co-located server-side write fails parity even when the
bytes land — FC-48).

The Terminal is the ``_ChatCompletionsCaller`` for the tool-driven surface:
its ``run`` yields ``OrchestratorChunk`` values, so the heartbeat +
context-sink lifecycle in the Serving Layer wraps it unchanged.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

from llm_orc.agentic.loop_driver import (
    ApplyWork,
    CarryClientTool,
    FinishTurn,
    TurnOutcome,
)
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.agentic.session_start import SessionContext

__all__ = ["ClientToolActionTerminal", "TurnDecider"]


class TurnDecider(Protocol):
    """The per-turn decision surface the Terminal emits.

    The narrow port the Terminal needs from the Loop Driver (ADR-033) —
    ``decide`` returns one :class:`~llm_orc.agentic.loop_driver.TurnOutcome`
    per turn. Defined here following the per-module-port pattern (``SeatFiller``,
    ``ToolDispatcher``) so the Terminal stays decoupled from the driver's wider
    construction surface.
    """

    async def decide(self, context: SessionContext) -> TurnOutcome: ...


class ClientToolActionTerminal:
    """Tool-call emission + multi-turn loop participation (the tool-driven caller).

    Composes the Loop Driver: each ``run`` asks the driver to decide one turn
    and emits the resulting :class:`~llm_orc.agentic.loop_driver.TurnOutcome`
    as wire chunks. Loop participation is structural — the Serving Layer builds
    the ``SessionContext`` from every request message (including the trailing
    ``role: "tool"`` result), and the Terminal hands that context straight to
    the driver, so the tool result reaches the per-turn decision (FC-50). It
    does not route through the single-turn pipeline's last-user-message
    extraction, which would drop the tool result.
    """

    def __init__(self, *, loop_driver: TurnDecider) -> None:
        self._loop_driver = loop_driver

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        """Decide one turn and emit it as wire chunks."""
        outcome = await self._loop_driver.decide(context)

        if isinstance(outcome, FinishTurn):
            if outcome.content:
                yield ContentDelta(content=outcome.content)
            yield Completion(finish_reason="stop")
            return

        if isinstance(outcome, CarryClientTool):
            yield ClientToolCall(tool_calls=(outcome.invocation,))
            return

        yield ClientToolCall(tool_calls=(self._invocation_for(outcome),))

    def _invocation_for(self, outcome: ApplyWork) -> ToolCallInvocation:
        """Marshal a generation outcome's deliverable into a client tool call.

        Reads the deliverable content from the envelope and places it into the
        client tool-call ``content`` argument (ADR-034 §Decision 3). WP-LB-C's
        first increment marshals the inline ``primary`` directly; wiring the
        Artifact Bridge for substrate-routed full-fidelity content (FC-49) is
        the next increment.
        """
        content = outcome.envelope.primary
        return ToolCallInvocation(
            id=outcome.invocation_id,
            name=outcome.tool_name,
            arguments=json.dumps({"filePath": outcome.file_path, "content": content}),
        )
