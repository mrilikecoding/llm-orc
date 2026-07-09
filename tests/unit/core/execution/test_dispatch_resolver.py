"""Tests for the phase-layer dispatch resolver.

The resolver is the routing half of the dynamic-dispatch primitive, sibling to
the guard partition: it resolves each dispatch node's ``${dep.field}`` target
against accumulated upstream results and records the resolved ensemble name on a
runtime copy of the config (``dispatch_resolved``), which the runner then reads.
"""

from __future__ import annotations

import json
from typing import Any

from llm_orc.core.execution.phases.dispatch_resolver import DispatchResolver
from llm_orc.schemas.agent_config import (
    DynamicDispatchAgentConfig,
    LlmAgentConfig,
)


class TestDispatchResolver:
    def test_resolves_target_from_upstream_results(self) -> None:
        resolver = DispatchResolver()
        seat = DynamicDispatchAgentConfig(
            name="seat", dispatch="${classify.target}", depends_on=["classify"]
        )
        results: dict[str, Any] = {
            "classify": {
                "status": "success",
                "response": json.dumps({"target": "seat-a"}),
            }
        }

        resolved = resolver.resolve_targets([seat], results)

        assert len(resolved) == 1
        assert isinstance(resolved[0], DynamicDispatchAgentConfig)
        assert resolved[0].dispatch_resolved == "seat-a"

    def test_does_not_mutate_original_config(self) -> None:
        resolver = DispatchResolver()
        seat = DynamicDispatchAgentConfig(
            name="seat", dispatch="${classify.target}", depends_on=["classify"]
        )
        results: dict[str, Any] = {
            "classify": {
                "status": "success",
                "response": json.dumps({"target": "seat-a"}),
            }
        }

        resolver.resolve_targets([seat], results)

        assert seat.dispatch_resolved is None

    def test_passes_through_non_dispatch_agents(self) -> None:
        resolver = DispatchResolver()
        plain = LlmAgentConfig(name="plain", model_profile="gpt4")

        resolved = resolver.resolve_targets([plain], {})

        assert resolved == [plain]

    def test_dispatch_reuses_the_shared_reference_resolver_not_a_parallel_one(
        self,
    ) -> None:
        """Preservation (scenarios.md "the ${dep.field} resolver behaves
        identically for guard siblings and dispatch nodes"): a dispatch node
        resolves its target via the same ``phases.reference.resolve_reference``
        the guard partition uses, so both resolve a ``${dep.field}`` reference
        against ``results_dict`` identically — not through a parallel resolver.
        """
        from llm_orc.core.execution.phases.reference import resolve_reference

        results: dict[str, Any] = {
            "classify": {
                "status": "success",
                "response": json.dumps({"target": "seat-a"}),
            }
        }
        seat = DynamicDispatchAgentConfig(
            name="seat", dispatch="${classify.target}", depends_on=["classify"]
        )

        resolved_seat = DispatchResolver().resolve_targets([seat], results)[0]
        assert isinstance(resolved_seat, DynamicDispatchAgentConfig)
        guard_value = resolve_reference("${classify.target}", results)

        assert resolved_seat.dispatch_resolved == guard_value == "seat-a"


def test_resolve_reference_returns_none_for_unparseable_upstream() -> None:
    """A failed/skipped/prose upstream must resolve to None (guard skips,
    dispatch surfaces its typed error) — not crash the whole ensemble run."""
    from llm_orc.core.execution.phases.reference import resolve_reference

    assert resolve_reference("${dep.field}", {"dep": {"response": "not json"}}) is None
    assert resolve_reference("${dep.field}", {"dep": {"response": None}}) is None
    assert resolve_reference("${dep.field}", {}) is None
    assert (
        resolve_reference("${dep.missing}", {"dep": {"response": '{"other": 1}'}})
        is None
    )
