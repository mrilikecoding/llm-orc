"""Serving Layer session-start function.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3) and
§Integration Contracts (Serving Layer → ``resolve_session_start_context``).
ADR-009's Phase 2 hook is structurally reserved by a typed function —
signature and call site are the commitment. Phase 1 returns ``[]``
unconditionally; Phase 2 populates the body by reading from the Plexus
Adapter.

``SessionContext`` is the shared type on two integration contracts
(Serving Layer → ``resolve_session_start_context`` and Serving Layer →
Orchestrator Runtime). It is defined here because Serving Layer owns
both edges.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_orc.agentic.session_registry import (
    SessionIdentity,
    SessionState,
)


@dataclass(frozen=True)
class ChatMessage:
    """An OpenAI-compatible chat message flowing through the Serving Layer.

    Defined here because ChatMessage is a *contract type* on the Serving
    Layer → Orchestrator Runtime edge (it rides inside ``SessionContext``
    below) — it is not Session Registry's internal concern. Locating it
    in this module keeps FC-4 intact: downstream consumers like the
    Runtime can import ChatMessage from ``session_start`` (allow-listed)
    rather than reaching into ``session_registry`` (forbidden).
    """

    role: str
    content: str


@dataclass(frozen=True)
class PromptFragment:
    """A fragment of system-prompt material loaded at session start.

    Phase 2 (ADR-009) populates these from Plexus-backed queries; Phase 1
    never emits any. ``source`` records provenance so observability
    downstream (OQ #2) can attribute injected context to its origin.
    """

    content: str
    source: str


@dataclass(frozen=True)
class SessionContext:
    """Snapshot passed from the Serving Layer across its two L3 edges.

    The Orchestrator Runtime (WP-C) reads ``messages``, ``tools``, and
    ``state`` each iteration. ``resolve_session_start_context`` reads
    the same fields once at session start.
    """

    messages: list[ChatMessage]
    tools: list[dict[str, Any]]
    state: SessionState


def resolve_session_start_context(context: SessionContext) -> list[PromptFragment]:
    """Phase 1 body: returns ``[]``.

    The call site and typed return are the structural reservation
    (ADR-009, FC-9). When Phase 2 lands, this body will call the Plexus
    Adapter and return the resulting fragments; callers already accept
    ``list[PromptFragment]`` so no structural change is needed.
    """
    del context
    return []


class SessionStartCache:
    """Enforces the once-per-session invariant (FC-9) outside Session Registry.

    The Serving Layer reaches this cache on every HTTP request; the
    resolver runs only on the first request per ``SessionIdentity``.
    Keeping the cache here (next to the resolver it guards) means
    Session Registry stays focused on turn and token accounting and
    does not need to know about ``PromptFragment``.

    Phase 2 (ADR-009) populates the resolver body; the cache shape is
    unaffected. If a later work package requires refreshing fragments
    mid-session, the refresh logic lives here rather than rippling
    through ``SessionState``.
    """

    def __init__(
        self,
        resolver: (Callable[[SessionContext], list[PromptFragment]] | None) = None,
    ) -> None:
        self._resolver = resolver or resolve_session_start_context
        self._cache: dict[SessionIdentity, list[PromptFragment]] = {}

    def resolve(self, context: SessionContext) -> list[PromptFragment]:
        """Return fragments for this session, resolving on first encounter."""
        identity = context.state.identity
        cached = self._cache.get(identity)
        if cached is not None:
            return cached
        fragments = self._resolver(context)
        self._cache[identity] = fragments
        return fragments
