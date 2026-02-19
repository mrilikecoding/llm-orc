"""Fan-out agent expansion for parallel execution over arrays (issue #73)."""

import json
from typing import Any

from llm_orc.core.execution.patterns import INSTANCE_PATTERN


class FanOutExpander:
    """Expands fan_out agents into N parallel instances at runtime."""

    def detect_fan_out_agents(self, agent_configs: list[dict[str, Any]]) -> list[str]:
        """Return names of agents with fan_out: true.

        Args:
            agent_configs: List of agent configurations

        Returns:
            List of agent names that have fan_out: true
        """
        return [
            agent["name"] for agent in agent_configs if agent.get("fan_out") is True
        ]

    def is_array_result(self, result: Any) -> bool:
        """Check if upstream result is a non-empty array.

        Args:
            result: Result from upstream agent (may be JSON string or Python list)

        Returns:
            True if result is a non-empty array, False otherwise
        """
        # Handle Python list directly
        if isinstance(result, list):
            return len(result) > 0

        # Try parsing as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return isinstance(parsed, list) and len(parsed) > 0
            except (json.JSONDecodeError, TypeError):
                return False

        return False

    def expand_fan_out_agent(
        self,
        agent_config: dict[str, Any],
        upstream_array: list[Any],
    ) -> list[dict[str, Any]]:
        """Create N copies of agent config with indexed names.

        Args:
            agent_config: Original agent configuration with fan_out: true
            upstream_array: Array from upstream agent to fan out over

        Returns:
            List of agent configs, one per array element, with indexed names
        """
        original_name = agent_config["name"]
        instances = []

        for index, chunk in enumerate(upstream_array):
            # Create copy of config without fan_out field
            instance_config = {
                key: value for key, value in agent_config.items() if key != "fan_out"
            }

            # Set indexed name
            instance_config["name"] = f"{original_name}[{index}]"

            # Store chunk metadata for input preparation
            instance_config["_fan_out_chunk"] = chunk
            instance_config["_fan_out_index"] = index
            instance_config["_fan_out_total"] = len(upstream_array)
            instance_config["_fan_out_original"] = original_name

            instances.append(instance_config)

        return instances

    def prepare_instance_input(
        self,
        chunk: Any,
        chunk_index: int,
        total_chunks: int,
        base_input: str,
    ) -> dict[str, Any]:
        """Prepare input for a single fan-out instance.

        Args:
            chunk: The chunk content for this instance
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks
            base_input: Original ensemble input

        Returns:
            Input dict with chunk content and metadata
        """
        return {
            "input": chunk,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "base_input": base_input,
        }

    def is_fan_out_instance_name(self, name: str) -> bool:
        """Check if name matches the fan-out instance pattern 'agent[N]'.

        Args:
            name: Agent name to check

        Returns:
            True if name matches instance pattern, False otherwise
        """
        return bool(INSTANCE_PATTERN.match(name))

    def get_original_agent_name(self, instance_name: str) -> str:
        """Extract original agent name from instance name.

        Args:
            instance_name: Instance name like 'extractor[0]'

        Returns:
            Original name like 'extractor', or input unchanged if not instance
        """
        match = INSTANCE_PATTERN.match(instance_name)
        if match:
            return match.group(1)
        return instance_name

    def get_instance_index(self, instance_name: str) -> int | None:
        """Extract index from instance name.

        Args:
            instance_name: Instance name like 'extractor[42]'

        Returns:
            Index as int, or None if not an instance name
        """
        match = INSTANCE_PATTERN.match(instance_name)
        if match:
            return int(match.group(2))
        return None

    def parse_array_from_result(self, result: Any) -> list[Any] | None:
        """Parse array from upstream agent result.

        Handles:
        - Python list directly
        - JSON array string
        - ScriptAgentOutput format with data field

        Args:
            result: Result from upstream agent

        Returns:
            Parsed array, or None if result is not an array
        """
        # Handle Python list directly
        if isinstance(result, list):
            return result

        # Try parsing as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)

                # Direct array
                if isinstance(parsed, list):
                    return parsed

                # ScriptAgentOutput format: {"success": true, "data": [...]}
                if isinstance(parsed, dict):
                    data = parsed.get("data")
                    if isinstance(data, list):
                        return data

            except (json.JSONDecodeError, TypeError):
                pass

        return None
