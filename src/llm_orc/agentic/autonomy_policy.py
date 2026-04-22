"""Autonomy Policy — gates orchestrator actions against the Session's Autonomy Level.

Per ``docs/agentic-serving/system-design.md`` §Autonomy Policy (L1 Domain
Policy) and §Integration Contracts (Orchestrator Tool Dispatch → Autonomy
Policy). ADR-008 defines the per-Session Autonomy Level and its baseline:

* Invoke existing ensembles — unrestricted (within Budget, ADR-005)
* Compose new ensembles from library primitives — allowed, subject to
  validation (ADR-006) and calibration (ADR-007)
* Authorship of new scripts or profiles — forbidden at any level (AS-6)
* Promotion — requires explicit operator approval outside the orchestrator
  tool surface

AS-6 is load-bearing and structurally enforced by the closed tool set
(FC-5 in ``orchestrator_tool_dispatch.py``). This module's scope is the
*decision* for in-surface tool calls — whether to allow, whether to
emit visibility, and (future WPs) whether to deny pending approval.
The module does not enumerate or re-check ``TOOL_NAMES``; Tool Dispatch
filters unknown names before the gate is consulted.

The ``level_provider`` callable is the layering-clean shape borrowed from
:class:`~llm_orc.agentic.budget_controller.BudgetController`: L1 modules
take plain values injected by their L3 callers, so the policy never
imports ``ConfigurationManager`` or ``SessionRegistry`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_orc.agentic.orchestrator_chunk import VisibilityEvent

BASELINE_LEVEL = "operator-as-tool-user"
"""Default level — silent invocation and composition (ADR-008)."""

PURE_TOOL_USER_VISIBLE_LEVEL = "pure-tool-user-visible"
"""Tighter level for deployments where the tool user is not the operator.

Surfaces composition events on the response stream so a vanilla client
(OpenCode, Roo Code, Cline) shows the tool user what llm-orc composed.
"""


@dataclass(frozen=True)
class Allow:
    """Permit the dispatch; optionally surface visibility events.

    Events propagate out of Tool Dispatch as :class:`VisibilityEvent`
    chunks; the Serving Layer's formatter renders each as a narration
    line on ``delta.content``.
    """

    events: tuple[VisibilityEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Deny:
    """Block the dispatch; surfaces as a tool error to the orchestrator.

    Reserved for tighter levels (approve-before-uncalibrated etc.) that
    land with WP-H. Phase 1 never returns this; the type is part of the
    contract so WP-H extends policy logic without widening the union.
    """

    reason: str


AutonomyDecision = Allow | Deny


class AutonomyPolicy:
    """Per-dispatch gate honoring the Session's Autonomy Level."""

    def __init__(self, *, level_provider: Callable[[], str]) -> None:
        self._level_provider = level_provider

    def decide(self, *, tool_name: str, arguments: dict[str, Any]) -> AutonomyDecision:
        """Resolve the current level and compute the decision for this call."""
        level = self._level_provider()
        if level == PURE_TOOL_USER_VISIBLE_LEVEL and tool_name == "compose_ensemble":
            return Allow(events=(_composition_event(arguments),))
        return Allow()


def _composition_event(arguments: dict[str, Any]) -> VisibilityEvent:
    """Build a composition visibility event from ``compose_ensemble`` arguments."""
    return VisibilityEvent(
        kind="composition",
        payload={"tool": "compose_ensemble", "arguments": arguments},
    )
