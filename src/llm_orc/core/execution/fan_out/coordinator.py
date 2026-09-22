"""Fan-out coordination for ensemble execution."""

import json
from typing import Any

from llm_orc.core.execution.fan_out.expander import FanOutExpander
from llm_orc.core.execution.fan_out.gatherer import FanOutGatherer
from llm_orc.core.execution.utils import dep_name
from llm_orc.schemas.agent_config import AgentConfig


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
    ) -> tuple[list[tuple[AgentConfig, list[Any]]], list[tuple[AgentConfig, str]]]:
        """Detect fan-out agents in phase, partitioned by contract outcome.

        A fan-out agent's contract is: the upstream must have succeeded,
        and its response must yield a JSON array (whole, or at
        ``input_key``). A genuinely empty array is a legitimate
        zero-instance success. Anything else — unparseable response,
        non-object response with ``input_key``, missing or non-list
        key, or an upstream that didn't succeed — fails the agent
        closed rather than letting it silently run un-expanded on
        garbage input (no lenient parsing, no fence stripping).

        Returns:
            (ready, failed):
            - ready: agents paired with their (possibly empty) upstream
              array, for expansion.
            - failed: agents paired with an error message naming the
              upstream agent and what went wrong.
        """
        ready: list[tuple[AgentConfig, list[Any]]] = []
        failed: list[tuple[AgentConfig, str]] = []

        for agent_config in phase_agents:
            if not agent_config.fan_out:
                continue

            if not agent_config.depends_on:
                continue

            upstream_name = dep_name(agent_config.depends_on[0])
            upstream_result = results_dict.get(upstream_name, {})
            upstream_status = upstream_result.get("status")

            if upstream_status != "success":
                failed.append(
                    (
                        agent_config,
                        f"Fan-out agent '{agent_config.name}' cannot run: "
                        f"upstream agent '{upstream_name}' did not succeed "
                        f"(status: {upstream_status!r})",
                    )
                )
                continue

            response = upstream_result.get("response", "")

            # Apply input_key selection (ADR-014)
            if agent_config.input_key:
                array_result = self._select_key_array(response, agent_config.input_key)
                contract = f"a list at key '{agent_config.input_key}'"
            else:
                array_result = self._expander.parse_array_from_result(response)
                contract = "a JSON array"

            if array_result is None:
                failed.append(
                    (
                        agent_config,
                        f"Fan-out agent '{agent_config.name}' cannot run: "
                        f"upstream agent '{upstream_name}'s response did not "
                        f"contain {contract}",
                    )
                )
                continue

            ready.append((agent_config, array_result))

        return ready, failed

    @staticmethod
    def _select_key_array(response: str, input_key: str) -> list[Any] | None:
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
