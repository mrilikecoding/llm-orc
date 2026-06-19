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
from typing import Any, Protocol

from llm_orc.agentic.artifact_bridge import ArtifactBridge, FormRefusedError
from llm_orc.agentic.loop_driver import (
    ApplyWork,
    CarryClientTool,
    TurnOutcome,
)
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_artifact_store import ArtifactNotFoundError
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

    def __init__(
        self,
        *,
        loop_driver: TurnDecider,
        bridge: ArtifactBridge,
        action_record: SessionActionRecord | None = None,
    ) -> None:
        self._loop_driver = loop_driver
        self._bridge = bridge
        self._action_record = action_record

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        """Decide one turn and emit it as wire chunks."""
        outcome = await self._loop_driver.decide(context)
        client_tools = _offered_tool_names(context.tools)
        for chunk in self._emit(outcome, context.state.identity.value, client_tools):
            yield chunk

    def _emit(
        self,
        outcome: TurnOutcome,
        session_id: str,
        client_tools: frozenset[str],
    ) -> list[OrchestratorChunk]:
        """Map a per-turn outcome to the wire chunks for it."""
        if isinstance(outcome, CarryClientTool):
            return [ClientToolCall(tool_calls=(outcome.invocation,))]
        if isinstance(outcome, ApplyWork):
            return self._emit_apply_work(outcome, session_id, client_tools)
        return _finish_chunks(outcome.content)

    def _emit_apply_work(
        self, outcome: ApplyWork, session_id: str, client_tools: frozenset[str]
    ) -> list[OrchestratorChunk]:
        """Marshal a generation outcome's deliverable for the client.

        When the client offered the deliverable's destination tool, the content
        is placed into a client tool call it executes locally. When it did not
        (a toolless request — F-ι.1, ADR-043), the content is returned as a text
        completion instead of an un-executable tool call.

        The Artifact Bridge (ADR-034 §Decision 3) resolves the deliverable
        content from the envelope — full fidelity for substrate-routed
        deliverables (ADR-025; FC-49), ``envelope.primary`` for inline ones —
        and the content is placed into the client tool-call ``content``
        argument. A bridge error (unresolvable reference) or a binary
        deliverable — not yet established, ADR-034 §Negative — degrades to a
        dispatch-failure completion, not a malformed tool call (system-design
        Terminal error handling; FC-48 forbids fabricated content).

        On a resolved string deliverable the content is captured onto the
        turn's action record (ADR-039 V-04) — the same bytes that land in the
        client ``write`` become the content anchor's source for a later turn's
        callee. A failed or binary deliverable captures nothing (no usable
        content to anchor on).
        """
        try:
            content = self._bridge.marshal(
                outcome.envelope,
                destination_tool=outcome.tool_name,
                destination_path=outcome.file_path,  # ADR-041 §2 — gate's extension
            )
        except (ArtifactNotFoundError, FormRefusedError) as error:
            return _finish_chunks(f"[dispatch failed: {error}]")
        if not isinstance(content, str):
            return _finish_chunks(
                f"[dispatch failed: binary deliverable for "
                f"{outcome.file_path} is not yet supported]"
            )
        if self._action_record is not None:
            self._action_record.record_content(session_id, content)
        # F-ι.1 (ADR-043): a client that did not offer the destination tool
        # cannot execute it (a toolless request — e.g. OpenCode's title-gen aux
        # call), so the marshalled deliverable is returned as a text completion
        # rather than an un-executable tool call. Delegation stays uniform in the
        # Loop Driver; only this output shape adapts.
        if outcome.tool_name not in client_tools:
            return _finish_chunks(content)
        invocation = ToolCallInvocation(
            id=outcome.invocation_id,
            name=outcome.tool_name,
            arguments=json.dumps({"filePath": outcome.file_path, "content": content}),
        )
        return [ClientToolCall(tool_calls=(invocation,))]


def _finish_chunks(content: str | None) -> list[OrchestratorChunk]:
    """Build the chunks for a finish turn — optional text, then a stop."""
    chunks: list[OrchestratorChunk] = []
    if content:
        chunks.append(ContentDelta(content=content))
    chunks.append(Completion(finish_reason="stop"))
    return chunks


def _offered_tool_names(tools: list[dict[str, Any]]) -> frozenset[str]:
    """The client tool names a request offered (the OpenAI tool shape).

    Used by the F-ι.1 adaptive-marshalling branch to decide whether a delegated
    deliverable's destination tool can be executed by the client.
    """
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return frozenset(names)
