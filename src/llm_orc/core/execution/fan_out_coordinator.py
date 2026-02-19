"""Fan-out coordination for ensemble execution."""

from typing import Any

from llm_orc.core.execution.fan_out_expander import FanOutExpander
from llm_orc.core.execution.fan_out_gatherer import FanOutGatherer


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
        phase_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> list[tuple[dict[str, Any], list[Any]]]:
        """Detect fan-out agents in phase with array upstream results."""
        fan_out_agents: list[tuple[dict[str, Any], list[Any]]] = []

        for agent_config in phase_agents:
            if not agent_config.get("fan_out"):
                continue

            depends_on = agent_config.get("depends_on", [])
            if not depends_on:
                continue

            upstream_name = depends_on[0]
            upstream_result = results_dict.get(upstream_name, {})

            if upstream_result.get("status") != "success":
                continue

            response = upstream_result.get("response", "")
            array_result = self._expander.parse_array_from_result(response)

            if array_result is not None and len(array_result) > 0:
                fan_out_agents.append((agent_config, array_result))

        return fan_out_agents

    def expand_agent(
        self,
        agent_config: dict[str, Any],
        upstream_array: list[Any],
    ) -> list[dict[str, Any]]:
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
