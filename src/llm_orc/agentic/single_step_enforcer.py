"""Single-Step Enforcer (Cycle 7 loop-back WP-LB-B, ADR-033) — L2.

The framework's grounding guarantee for the tool-driven multi-turn
surface. The Loop Driver invokes the seat-filler LLM to decide a turn's
action; an unconstrained cheap-tier driver batches multiple tool calls
and commits later steps to values it never observed (Spike τ, n=4 — the
``${bash_output}`` failure signature). This module truncates that batch
to a single action so the next action cannot presuppose an unobserved
result; the driver re-plans only after the first call's result returns.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Single-Step
Enforcer. **D2 resolution (ARCHITECT): batch-truncation** — the only
enforcement technique with direct spike evidence (Spike τ′, n=3, used
it), model-independent (so it does not change when the seat-filler swaps
tiers — the precondition for FC-46), and placeable as a stateless policy.
The ``tool_choice`` family is empirically closed (three negatives: Spike
κ — the framework does not forward it and MiniMax did not honor it;
Spike ψ.3 — Ollama+qwen3 silently ignores forcing). A re-planning prompt
remains the only untested candidate behind this module's boundary.

The policy is stateless — it has no module-level dependencies and holds
no per-session state. The Loop Driver owns the loop; this module owns the
single structural invariant FC-43 names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llm_orc.models.base import ToolCall

__all__ = ["EnforcedTurn", "SingleStepEnforcer"]


@dataclass(frozen=True)
class EnforcedTurn:
    """The single action permitted this turn after batch truncation.

    ``action`` is the one client tool call the framework dispatches this
    turn, or ``None`` when the seat-filler proposed no action (a
    finish-with-text turn). ``truncated`` records whether additional
    proposed calls were dropped, and ``dropped`` how many — both feed the
    Loop Driver's ``TurnDecision`` diagnostic (FC-51) so an axis-2 failure
    is reconstructable from the event stream.
    """

    action: ToolCall | None
    truncated: bool
    dropped: int


class SingleStepEnforcer:
    """Stateless batch-truncation policy (ADR-033 §Decision 3; D2)."""

    def enforce(self, proposed: Sequence[ToolCall]) -> EnforcedTurn:
        """Truncate a proposed batch to its first action.

        The first proposed call is the permitted action; any others are
        dropped, not queued (FC-43: a turn that dispatches a second call
        before the first's result returns violates single-action-per-turn).
        An empty batch is a finish turn — no action, nothing dropped.
        """
        if not proposed:
            return EnforcedTurn(action=None, truncated=False, dropped=0)
        dropped = len(proposed) - 1
        return EnforcedTurn(
            action=proposed[0],
            truncated=dropped > 0,
            dropped=dropped,
        )
