"""Phase-layer dispatch target resolution (control-flow primitive).

Sibling to the guard partition. Where the guard decides *whether* a node runs,
the dispatch resolver decides *which* ensemble a dynamic-dispatch node runs: it
resolves the node's ``dispatch`` ``${dep.field}`` reference against accumulated
upstream results (the same resolution the guard uses) and records the resolved
ensemble name on a runtime copy of the config. Deterministic, no model
involvement. The runner reads ``dispatch_resolved`` and executes it.
"""

from __future__ import annotations

from typing import Any

from llm_orc.core.execution.phases.reference import resolve_reference
from llm_orc.schemas.agent_config import AgentConfig, DynamicDispatchAgentConfig


class DispatchResolver:
    """Resolves dynamic-dispatch targets from accumulated upstream results."""

    def resolve_targets(
        self, phase_agents: list[AgentConfig], results_dict: dict[str, Any]
    ) -> list[AgentConfig]:
        """Return the phase agents with each dispatch target resolved.

        Dispatch nodes are replaced with a runtime copy carrying the resolved
        ensemble name (``dispatch_resolved``); other agents pass through
        unchanged. The originals are not mutated.
        """
        resolved: list[AgentConfig] = []
        for agent in phase_agents:
            if isinstance(agent, DynamicDispatchAgentConfig):
                target = resolve_reference(agent.dispatch, results_dict)
                resolved.append(agent.model_copy(update={"dispatch_resolved": target}))
            else:
                resolved.append(agent)
        return resolved
