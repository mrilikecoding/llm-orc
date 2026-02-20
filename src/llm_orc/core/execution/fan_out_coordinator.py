"""Fan-out coordination for ensemble execution."""

import json
from typing import Any

from llm_orc.core.execution.fan_out_expander import FanOutExpander
from llm_orc.core.execution.fan_out_gatherer import FanOutGatherer
from llm_orc.schemas.agent_config import AgentConfig


def _dep_name(dep: str | dict[str, Any]) -> str:
    """Extract the agent name from a dependency entry."""
    if isinstance(dep, dict):
        return str(dep["agent_name"])
    return dep


class FanOutCoordinator:
    """Coordinates fan-out expansion and result gathering for phases.

    Owns FanOutExpander and FanOutGatherer, providing a unified interface
    for detecting, expanding, and gathering fan-out agent results.
    """

    def __init__(self, expander: FanOutExpander, gatherer: FanOutGatherer) -> None:
        self._expander = expander
        self._gatherer = gatherer

    def detect_in_phase(
        self,
        phase_agents: list[AgentConfig],
        results_dict: dict[str, Any],
    ) -> list[tuple[AgentConfig, list[Any]]]:
        """Detect fan-out agents in phase with array upstream results."""
        fan_out_agents: list[tuple[AgentConfig, list[Any]]] = []

        for agent_config in phase_agents:
            if not agent_config.fan_out:
                continue

            if not agent_config.depends_on:
                continue

            upstream_name = _dep_name(agent_config.depends_on[0])
            upstream_result = results_dict.get(upstream_name, {})

            if upstream_result.get("status") != "success":
                continue

            response = upstream_result.get("response", "")

            # Apply input_key selection (ADR-014)
            if agent_config.input_key:
                array_result = self._select_key_array(
                    response, agent_config.input_key
                )
            else:
                array_result = self._expander.parse_array_from_result(
                    response
                )

            if array_result is not None and len(array_result) > 0:
                fan_out_agents.append((agent_config, array_result))

        return fan_out_agents

    @staticmethod
    def _select_key_array(
        response: str, input_key: str
    ) -> list[Any] | None:
        """Select array value from keyed upstream output (ADR-014)."""
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        selected = parsed.get(input_key)
        if isinstance(selected, list):
            return selected
        return None

    def expand_agent(
        self,
        agent_config: AgentConfig,
        upstream_array: list[Any],
    ) -> list[AgentConfig]:
        """Expand a fan-out agent into N instances."""
        return self._expander.expand_fan_out_agent(agent_config, upstream_array)

    def gather_results(
        self,
        original_agent_name: str,
        instance_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Gather results from fan-out instances into ordered array."""
        self._gatherer.clear(original_agent_name)

        for instance_name, result in instance_results.items():
            if not self._expander.is_fan_out_instance_name(instance_name):
                continue

            original = self._expander.get_original_agent_name(instance_name)
            if original != original_agent_name:
                continue

            success = result.get("status") == "success"
            self._gatherer.record_instance_result(
                instance_name=instance_name,
                result=result.get("response"),
                success=success,
                error=result.get("error"),
            )

        return self._gatherer.gather_results(original_agent_name)
