"""Orchestrator Runtime → Serving Layer chunk type contract.

Per ``docs/serving.md`` §Serving Layer →
Orchestrator Runtime (Integration Contracts). The Runtime yields an
async iterator of ``OrchestratorChunk`` values; the Serving Layer
translates each variant to an OpenAI-compatible SSE chunk via the
``sse_format`` module. The Runtime must not import SSE-specific code —
these types are the shared contract that keeps FC-4 intact.

Variants enumerated by the system design:

* ``ContentDelta`` — assistant content streaming in (maps to
  ``delta.content``).
* ``InternalToolCallInFlight`` / ``InternalToolCallResult`` — the
  orchestrator's own actions against llm-orc's five-tool surface
  (ADR-003). Not surfaced to clients in Phase 1; form decided by OQ #2.
* ``ClientToolCall`` — final-turn delegation per the Client Tool
  Surface Commitment (Option C). Maps to ``delta.tool_calls`` with
  ``finish_reason: tool_calls``.
* ``Completion`` — clean end of turn (maps to empty delta plus
  ``finish_reason: stop``).
* ``ErrorChunk`` — Runtime exception translated to a typed error per
  the integration contract's error-handling clause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ContentDelta:
    """Assistant content being streamed to the client."""

    content: str


@dataclass(frozen=True)
class Completion:
    """Clean end of an assistant turn.

    ``finish_reason`` is constrained to the two reasons a stop-style
    completion can carry. Tool-call delegation uses the separate
    ``ClientToolCall`` variant, which encodes ``finish_reason: tool_calls``
    alongside the tool-call payload.
    """

    finish_reason: Literal["stop", "length"]


@dataclass(frozen=True)
class InternalToolCallInFlight:
    """Observation that the orchestrator has begun an internal tool call.

    Internal tools are the five ADR-003 operations against llm-orc. The
    Runtime yields these so the Serving Layer can choose what to
    surface. Phase 1 surfaces nothing (the formatter silent-drops) —
    visibility form resolves in WP-E per OQ #2.
    """

    id: str
    name: str


@dataclass(frozen=True)
class InternalToolCallResult:
    """Summarized result of an internal tool call flowing back to the orchestrator.

    Carries the post-summarization payload that AS-7 and ADR-004
    require before the observation reaches the Runtime's context.
    Phase 1 surfaces nothing externally (see ``InternalToolCallInFlight``).
    """

    id: str
    summary: str


@dataclass(frozen=True)
class ToolCallInvocation:
    """One tool call the orchestrator is delegating to the client.

    ``arguments`` is the OpenAI-compatible JSON-encoded string. Storing
    it pre-encoded keeps the Runtime's emission logic close to the wire
    form and avoids re-encoding during formatting.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ClientToolCall:
    """Final-turn delegation to client-declared tools (Option C).

    Per ``system-design.md`` §Client Tool Surface Commitment: when a
    task step requires a client-side action (bash, file edit, etc.),
    the orchestrator closes the turn with this chunk. Maps to
    ``finish_reason: tool_calls`` and a ``delta.tool_calls`` array
    enumerating the delegations.
    """

    tool_calls: tuple[ToolCallInvocation, ...]


@dataclass(frozen=True)
class VisibilityEvent:
    """Operator / tool-user visibility surface per ADR-008 and OQ #2.

    Emitted by Autonomy Policy when the configured level calls for surfacing
    an event (Phase 1: composition events at ``pure-tool-user-visible``). The
    SSE formatter translates the event into inline narration on
    ``delta.content`` so a vanilla OpenAI-compat client surfaces it in the
    assistant message text — the tool user observes what llm-orc is doing
    under the hood in the same stream they converse through.

    Shape kept neutral (``kind`` + ``payload``) so later levels can surface
    additional event types without changing the chunk contract.
    """

    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ErrorChunk:
    """Runtime exception surfaced as an SSE error payload.

    The Serving Layer → Orchestrator Runtime integration contract
    requires that Runtime exceptions become an SSE error chunk rather
    than propagating as HTTP errors once streaming has begun. ``type``
    mirrors OpenAI's error-type vocabulary (``server_error``,
    ``rate_limit``, etc.) so clients handle it using their existing
    error-parsing paths.
    """

    message: str
    type: str


OrchestratorChunk = (
    ContentDelta
    | Completion
    | ClientToolCall
    | InternalToolCallInFlight
    | InternalToolCallResult
    | VisibilityEvent
    | ErrorChunk
)
"""Union of all chunk variants the Runtime can yield.

Kept as a type alias rather than a base class so each variant stays a
flat dataclass and exhaustiveness checking over the union is visible in
``match`` statements.
"""
