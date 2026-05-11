"""Session-token budget tracking for the agentic-serving layer.

Tracks input + output tokens per session against a configured budget. The
Orchestrator Runtime calls these methods at each ReAct iteration to enforce
budget limits before continuing.

NOTE: Tests for this module are required before the Orchestrator Runtime
integrates this path. See follow-up issue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_orc.agentic.orchestrator_config import DEFAULT_MAX_TOKEN_LIMIT

logger = logging.getLogger(__name__)

# Re-export the canonical default so callers that import from this module
# still get a single source of truth.
DEFAULT_BUDGET_LIMIT: int = DEFAULT_MAX_TOKEN_LIMIT


@dataclass
class SessionBudget:
    """Per-session token budget tracking.

    Attributes:
        session_id: Unique session identifier.
        input_tokens_used: Cumulative input tokens consumed.
        output_tokens_used: Cumulative output tokens consumed.
        limit: Maximum total tokens allowed for this session.
        metadata: Arbitrary caller-supplied key/value pairs.
    """

    session_id: str
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    limit: int = DEFAULT_BUDGET_LIMIT
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tokens(self, input_count: int, output_count: int) -> None:
        """Record token consumption from a single LLM call."""
        self.input_tokens_used += input_count
        self.output_tokens_used += output_count

    def total_used(self) -> int:
        return self.input_tokens_used + self.output_tokens_used

    def check_limit(self) -> bool:
        """Return True if the session is within budget, False if at or over limit.

        Called at each ReAct iteration by the Orchestrator Runtime.
        """
        if self.total_used() >= self.limit:
            logger.warning(
                "Session %s exceeded budget: %d/%d tokens used.",
                self.session_id,
                self.total_used(),
                self.limit,
            )
            return False
        return True

    def remaining(self) -> int:
        return self.limit - self.total_used()

    def get_metadata(self) -> dict[str, Any]:
        """Return a shallow copy of session metadata for logging or persistence.

        Returns a copy so callers cannot mutate the internal dict.
        """
        return dict(self.metadata)


class SessionBudgetRegistry:
    """In-memory registry of active session budgets."""

    def __init__(self) -> None:
        self._budgets: dict[str, SessionBudget] = {}

    def register(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> SessionBudget:
        """Create a new session budget and register it.

        Raises:
            ValueError: If a budget for *session_id* is already registered.
                Call ``unregister`` first if re-registration is intentional.
        """
        if session_id in self._budgets:
            raise ValueError(
                f"Session '{session_id}' is already registered. "
                "Call unregister() before re-registering."
            )
        budget = SessionBudget(
            session_id=session_id,
            limit=limit if limit is not None else DEFAULT_BUDGET_LIMIT,
        )
        self._budgets[session_id] = budget
        return budget

    def get(self, session_id: str) -> SessionBudget:
        """Look up the budget for a session.

        Raises:
            KeyError: If *session_id* has not been registered.
        """
        try:
            return self._budgets[session_id]
        except KeyError:
            raise KeyError(
                f"No budget registered for session '{session_id}'. "
                "Ensure register() is called before get() or add_usage()."
            ) from None

    def unregister(self, session_id: str) -> None:
        """Remove a session budget from the registry.

        Raises:
            KeyError: If *session_id* is not registered.
        """
        try:
            del self._budgets[session_id]
        except KeyError:
            raise KeyError(
                f"Cannot unregister unknown session '{session_id}'."
            ) from None

    def add_usage(
        self, session_id: str, input_count: int, output_count: int
    ) -> bool:
        """Record token usage and return whether the session is still within budget."""
        budget = self.get(session_id)
        budget.add_tokens(input_count, output_count)
        return budget.check_limit()

    def all_sessions(self) -> dict[str, SessionBudget]:
        return self._budgets
