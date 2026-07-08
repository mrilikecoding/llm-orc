"""Serving Layer session-start function.

Per ``docs/serving.md`` §Serving Layer (L3) and
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

# ChatMessage relocated to the session substrate at Cycle-8 WP-B8;
# re-exported here so this module remains the serving-layer contract
# surface its consumers import from.
from llm_orc.core.session.messages import ChatMessage as ChatMessage
from llm_orc.core.session.registry import (
    SessionIdentity,
    SessionState,
)


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
