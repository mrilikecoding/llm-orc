"""Tests for the Autonomy Policy module.

Per ``docs/agentic-serving/system-design.md`` §Autonomy Policy (L1 Domain
Policy) and §Integration Contracts (Orchestrator Tool Dispatch →
Autonomy Policy). ADR-008 defines the per-Session Autonomy Level and
baseline (default) semantics. AS-6 is load-bearing: no configuration can
enable authorship of new scripts or profiles — that closure is enforced
by ``TOOL_NAMES`` (FC-5) and Tool Dispatch's name check, not by
AutonomyPolicy. This module's scope is *decision* about in-surface tool
calls, not *closure* of the surface.

Covers scenarios (``docs/agentic-serving/scenarios.md``):

* §Default Autonomy Level permits invocation, permits composition, gates
  promotion — here at unit scope (gate decision per tool)
* §Tool user without operator role observes composition events when
  configured — here at unit scope (Allow with VisibilityEvent)
* §Pure tool-user session at default Autonomy Level experiences silent
  composition — here at unit scope (Allow with empty events)
"""

from __future__ import annotations

import pytest

from llm_orc.agentic.autonomy_policy import (
    BASELINE_LEVEL,
    PURE_TOOL_USER_VISIBLE_LEVEL,
    Allow,
    AutonomyPolicy,
    Deny,
)
from llm_orc.agentic.orchestrator_chunk import VisibilityEvent


def _fixed_level(level: str) -> AutonomyPolicy:
    return AutonomyPolicy(level_provider=lambda: level)


class TestBaselineLevel:
    """BASELINE_LEVEL — the operator-as-tool-user default per ADR-008."""

    @pytest.mark.parametrize(
        "tool_name",
        [
            "invoke_ensemble",
            "compose_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        ],
    )
    def test_all_five_tools_allow_silently(self, tool_name: str) -> None:
        policy = _fixed_level(BASELINE_LEVEL)
        decision = policy.decide(tool_name=tool_name, arguments={})
        assert isinstance(decision, Allow)
        assert decision.events == ()


class TestPureToolUserVisibleLevel:
    """PURE_TOOL_USER_VISIBLE_LEVEL — tightened level that surfaces compose events.

    Per ADR-008's negative consequence note: when the tool user is not the
    operator, silent composition is surprising. This level makes composition
    observable to the tool user via the response stream.
    """

    def test_compose_ensemble_allow_with_composition_event(self) -> None:
        policy = _fixed_level(PURE_TOOL_USER_VISIBLE_LEVEL)
        arguments = {"name": "new-ensemble", "profiles": ["default"]}
        decision = policy.decide(tool_name="compose_ensemble", arguments=arguments)
        assert isinstance(decision, Allow)
        assert len(decision.events) == 1
        event = decision.events[0]
        assert isinstance(event, VisibilityEvent)
        assert event.kind == "composition"
        assert event.payload == {"tool": "compose_ensemble", "arguments": arguments}

    @pytest.mark.parametrize(
        "tool_name",
        [
            "invoke_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        ],
    )
    def test_other_tools_allow_silently(self, tool_name: str) -> None:
        """Visibility at this level is composition-specific; invoke stays quiet."""
        policy = _fixed_level(PURE_TOOL_USER_VISIBLE_LEVEL)
        decision = policy.decide(tool_name=tool_name, arguments={})
        assert isinstance(decision, Allow)
        assert decision.events == ()


class TestLevelProviderReadPerDecision:
    """The level is resolved per-decide so config changes take effect immediately.

    Phase 1 has no mid-session level mutation, but this read contract lets
    later WPs add per-session overrides without changing the decide signature.
    """

    def test_two_decisions_read_level_twice(self) -> None:
        observed: list[str] = []
        levels = iter([BASELINE_LEVEL, PURE_TOOL_USER_VISIBLE_LEVEL])

        def provider() -> str:
            level = next(levels)
            observed.append(level)
            return level

        policy = AutonomyPolicy(level_provider=provider)
        first = policy.decide(tool_name="compose_ensemble", arguments={})
        second = policy.decide(tool_name="compose_ensemble", arguments={})

        assert observed == [BASELINE_LEVEL, PURE_TOOL_USER_VISIBLE_LEVEL]
        assert isinstance(first, Allow)
        assert first.events == ()
        assert isinstance(second, Allow)
        assert len(second.events) == 1


class TestUnknownLevelFallsBackToBaseline:
    """An unrecognized level falls back to baseline silent behavior.

    Safer than raising — an operator typo or a future level name leaking
    into config without corresponding policy code should not lock orchestrator
    sessions out; they continue at baseline semantics and the missing
    surfacing is a visible hint to the operator.
    """

    def test_unknown_level_allows_silently(self) -> None:
        policy = _fixed_level("some-future-level-not-yet-implemented")
        decision = policy.decide(tool_name="compose_ensemble", arguments={})
        assert isinstance(decision, Allow)
        assert decision.events == ()


class TestAutonomyDecisionTypes:
    """Both decision variants are first-class — Deny is reserved for WP-H.

    ADR-008 tighter-level semantics (approve-before-uncalibrated) need Deny;
    Phase 1 never returns it, but the type is part of the contract so WP-H
    can extend policy code without widening the decision union.
    """

    def test_deny_constructs_with_reason(self) -> None:
        deny = Deny(reason="placeholder")
        assert deny.reason == "placeholder"

    def test_allow_defaults_to_empty_events(self) -> None:
        allow = Allow()
        assert allow.events == ()
