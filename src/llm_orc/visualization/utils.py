"""Shared utilities for visualization modules."""

from typing import Any


def calculate_agent_level(
    agent_name: str,
    dependencies: list[str],
    agents_state: dict[str, dict[str, Any]],
) -> int:
    """Calculate the dependency level of an agent (0 = no dependencies).

    Args:
        agent_name: Name of the agent
        dependencies: Direct dependencies of the agent
        agents_state: Dict mapping agent names to their info dicts
            (each must have a "dependencies" key)

    Returns:
        Integer dependency level (0 for no dependencies)
    """
    if not dependencies:
        return 0

    max_dep_level = 0
    for dep_name in dependencies:
        if dep_name in agents_state:
            dep_info = agents_state[dep_name]
            dep_dependencies = dep_info.get("dependencies", [])
            dep_level = calculate_agent_level(dep_name, dep_dependencies, agents_state)
            max_dep_level = max(max_dep_level, dep_level)

    return max_dep_level + 1
