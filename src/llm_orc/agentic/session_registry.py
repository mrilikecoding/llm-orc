"""Session Registry — identifies and continues multi-request Sessions.

Per `docs/agentic-serving/system-design.md` §Session Registry (L3).
Reconstructs orchestrator state from the OpenAI-compatible chat
conversation and tracks cumulative turn and token accounting. Phase 1
is in-memory; persistence is added later only when required by
Autonomy Level or Calibration state (ADR-007, ADR-008).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Literal

IdentityMethod = Literal["user_field", "message_prefix", "cold_start"]


@dataclass(frozen=True)
class ChatMessage:
    """A single OpenAI-compatible chat message.

    Kept minimal for Phase 1. The Serving Layer translates request
    bodies into these; downstream modules read them.
    """

    role: str
    content: str


@dataclass(frozen=True)
class SessionIdentity:
    """Identifies a Session across HTTP requests.

    Derivation-method-agnostic per the Serving Layer → Session Registry
    integration contract: the identity value may come from the
    OpenAI `user` field, a message-prefix hash, or (future) an
    explicit session-id header. The method is retained so consumers
    can reason about identity stability.
    """

    value: str
    method: IdentityMethod


@dataclass
class SessionState:
    """Mutable per-Session accounting.

    Tracks cumulative turn count (ReAct iterations) and cumulative
    token spend, summed across requests that share a SessionIdentity.
    Budget, Autonomy, and Calibration state live in their own modules
    and read this state through Session Registry contracts.
    """

    identity: SessionIdentity
    turn_count: int = 0
    token_spend: int = 0

    def record_iteration(self, tokens: int) -> None:
        """Record one ReAct iteration's contribution to the Session."""
        self.turn_count += 1
        self.token_spend += tokens


class SessionRegistry:
    """Per-process registry of active Sessions.

    Resolves identity from request features and returns the canonical
    mutable state object for a given identity. Cold-start requests
    (no user field, no user message) produce a fresh identity so that
    each such request is treated as its own Session.
    """

    def __init__(self) -> None:
        self._states: dict[SessionIdentity, SessionState] = {}

    def resolve_identity(
        self,
        *,
        messages: list[ChatMessage],
        user_field: str | None,
    ) -> SessionIdentity:
        if user_field is not None:
            return SessionIdentity(value=user_field, method="user_field")

        first_user = next((m for m in messages if m.role == "user"), None)
        if first_user is None:
            return SessionIdentity(value=uuid.uuid4().hex, method="cold_start")

        digest = hashlib.sha256(first_user.content.encode("utf-8")).hexdigest()
        return SessionIdentity(value=digest, method="message_prefix")

    def get_or_create_state(self, identity: SessionIdentity) -> SessionState:
        existing = self._states.get(identity)
        if existing is not None:
            return existing
        created = SessionState(identity=identity)
        self._states[identity] = created
        return created
