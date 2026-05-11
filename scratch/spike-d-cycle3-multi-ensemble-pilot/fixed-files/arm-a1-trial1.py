"""Session-token budget tracking for the agentic-serving layer.

Tracks input + output tokens per session against a configured budget. The
Orchestrator Runtime calls these methods at each ReAct iteration to enforce
budget limits before continuing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_orc.agentic.orchestrator_config import DEFAULT_MAX_TOKEN_LIMIT
else:
    DEFAULT_MAX_TOKEN_LIMIT = 50_000_000

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_LIMIT = DEFAULT_MAX_TOKEN_LIMIT


class SessionNotFoundError(LookupError):
    """Raised when a session_id is not found in the registry."""


@dataclass
class SessionBudget:
    """Per-session token budget tracking.

    Attributes:
        session_id: Unique session identifier.
        api_key: The API key associated with this session, used for billing.
        input_tokens_used: Cumulative input tokens consumed.
        output_tokens_used: Cumulative output tokens consumed.
        limit: Maximum total tokens allowed for this session.
    """

    session_id: str
    api_key: str
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    limit: int = DEFAULT_BUDGET_LIMIT
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tokens(self, input_count: int, output_count: int) -> None:
        """Record token consumption from a single LLM call."""
        if input_count < 0 or output_count < 0:
            raise ValueError("Token counts must be non-negative")
        self.input_tokens_used += input_count
        self.output_tokens_used += output_count

    def total_used(self) -> int:
        return self.input_tokens_used + self.output_tokens_used

    def check_limit(self) -> bool:
        """Return True if the session is within budget, False if exceeded.

        A session at exactly the limit is considered within budget.
        Called at each ReAct iteration by the Orchestrator Runtime.
        """
        if self.total_used() >= self.limit:
            api_key_hash = hashlib.sha256(self.api_key.encode()).hexdigest()[:8]
            logger.warning(
                f"Session {self.session_id} exceeded budget: "
                f"{self.total_used()}/{self.limit} tokens. "
                f"API key hash: ***{api_key_hash}"
            )
            return False
        return True

    def remaining(self) -> int:
        remaining = self.limit - self.total_used()
        return max(remaining, 0)

    def get_metadata(self) -> dict[str, Any]:
        """Return session metadata for logging or persistence."""
        return dict(self.metadata)


class SessionBudgetRegistry:
    """In-memory registry of active session budgets."""

    def __init__(self) -> None:
        self._budgets: dict[str, SessionBudget] = {}

    def register(
        self, session_id: str, api_key: str, limit: int | None = None
    ) -> SessionBudget:
        """Create a new session budget and register it."""
        if session_id in self._budgets:
            raise ValueError(f"Session {session_id!r} is already registered")
        budget = SessionBudget(
            session_id=session_id,
            api_key=api_key,
            limit=limit if limit is not None else DEFAULT_BUDGET_LIMIT,
        )
        self._budgets[session_id] = budget
        return budget

    def get(self, session_id: str) -> SessionBudget:
        """Look up the budget for a session.

        Raises:
            SessionNotFoundError: If the session_id is not registered.
        """
        budget = self._budgets.get(session_id)
        if budget is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found in registry")
        return budget

    def add_usage(self, session_id: str, input_count: int, output_count: int) -> bool:
        """Record token usage and return whether the session is still within budget."""
        budget = self.get(session_id)
        budget.add_tokens(input_count, output_count)
        return budget.check_limit()

    def all_sessions(self) -> dict[str, SessionBudget]:
        return dict(self._budgets)