"""Budget Controller — per-iteration turn and token limit check.

Per ``docs/agentic-serving/system-design.md`` §Budget Controller (L1)
and §Integration Contracts (Orchestrator Runtime → Budget Controller).
Enforces AS-3: Budget is a control-plane concern, checked at each
ReAct iteration boundary regardless of what the orchestrator LLM
decides. The orchestrator LLM does not observe Budget state.

The check is purely functional — the controller holds the Session's
Budget limits at construction and answers a single question per call:
is the next iteration permitted given the Session's current cumulative
spend? The Runtime passes in ``turn_count`` and ``token_spend`` read
from Session state.

Return semantics (not raise): a failed check returns
``BudgetCheckExhausted`` carrying the exhaustion reason and enough
context for the Runtime to shape a graceful final response. This keeps
the per-iteration control flow a value-comparison rather than
exception-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExhaustionReason = Literal["turn_limit", "token_limit"]


@dataclass(frozen=True)
class BudgetCheckPass:
    """Budget permits the next iteration."""


@dataclass(frozen=True)
class BudgetCheckExhausted:
    """Budget would be exhausted by the next iteration.

    Carries the cumulative state and the configured limits so the
    Runtime can shape the graceful-termination response without
    re-reading Session state.
    """

    reason: ExhaustionReason
    turn_count: int
    token_spend: int
    turn_limit: int
    token_limit: int


BudgetCheck = BudgetCheckPass | BudgetCheckExhausted
"""Union of possible check outcomes."""


class BudgetController:
    """Enforces turn and token limits at each ReAct iteration boundary."""

    def __init__(self, *, turn_limit: int, token_limit: int) -> None:
        self._turn_limit = turn_limit
        self._token_limit = token_limit

    def check(self, *, turn_count: int, token_spend: int) -> BudgetCheck:
        """Return pass or exhausted based on Session state vs. limits."""
        if turn_count >= self._turn_limit:
            return BudgetCheckExhausted(
                reason="turn_limit",
                turn_count=turn_count,
                token_spend=token_spend,
                turn_limit=self._turn_limit,
                token_limit=self._token_limit,
            )
        if token_spend >= self._token_limit:
            return BudgetCheckExhausted(
                reason="token_limit",
                turn_count=turn_count,
                token_spend=token_spend,
                turn_limit=self._turn_limit,
                token_limit=self._token_limit,
            )
        return BudgetCheckPass()
