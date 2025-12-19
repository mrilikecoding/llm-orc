"""Ensemble configuration loading and management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _find_agent_by_name(
    agents: list[dict[str, Any]], agent_name: str
) -> dict[str, Any] | None:
    """Find an agent by name in the agents list.

    Args:
        agents: List of agent configurations
        agent_name: Name of agent to find

    Returns:
        Agent configuration dictionary if found, None otherwise
    """
    return next((a for a in agents if a["name"] == agent_name), None)


def _perform_cycle_detection(
    agent_name: str,
    agents: list[dict[str, Any]],
    visited: set[str],
    recursion_stack: set[str],
) -> bool:
    """Perform cycle detection using depth-first search for a specific agent.

    Args:
        agent_name: Name of agent to check for cycles
        agents: List of agent configurations
        visited: Set of visited agent names
        recursion_stack: Set of agents currently in recursion stack

    Returns:
        True if cycle detected, False otherwise
    """
    if agent_name in recursion_stack:
        return True
    if agent_name in visited:
        return False

    visited.add(agent_name)
    recursion_stack.add(agent_name)

    # Find agent config
    agent_config = _find_agent_by_name(agents, agent_name)
    if agent_config:
        dependencies = agent_config.get("depends_on", [])
        for dep in dependencies:
            # Handle both simple string and conditional dict dependencies
            dep_name = dep if isinstance(dep, str) else dep.get("agent_name")
            if dep_name and _perform_cycle_detection(
                dep_name, agents, visited, recursion_stack
            ):
                return True

    recursion_stack.remove(agent_name)
    return False


def _check_agents_for_cycles(agents: list[dict[str, Any]]) -> None:
    """Check all agents for cycles and raise error if found.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If circular dependencies are detected
    """
    visited: set[str] = set()

    # Check each agent for cycles
    for agent in agents:
        if agent["name"] not in visited:
            recursion_stack: set[str] = set()
            if _perform_cycle_detection(
                agent["name"], agents, visited, recursion_stack
            ):
                raise ValueError(
                    f"Circular dependency detected involving agent: '{agent['name']}'"
                )


def _validate_fan_out_dependencies(agents: list[dict[str, Any]]) -> None:
    """Validate that fan_out agents have required dependencies.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If any agent has fan_out: true without depends_on
    """
    for agent in agents:
        if agent.get("fan_out") is True:
            depends_on = agent.get("depends_on", [])
            if not depends_on:
                raise ValueError(
                    f"Agent '{agent['name']}' has fan_out: true but requires "
                    f"depends_on to specify the upstream agent providing the array"
                )


def _check_missing_dependencies(agents: list[dict[str, Any]]) -> None:
    """Check for missing dependencies in agent configurations.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If any agent depends on a non-existent agent
    """
    # Create a map of agent names for quick lookup
    agent_names = {agent["name"] for agent in agents}

    # Check for missing dependencies
    for agent in agents:
        dependencies = agent.get("depends_on", [])
        for dep in dependencies:
            # Handle both simple string dependencies and conditional dict dependencies
            dep_name = dep if isinstance(dep, str) else dep.get("agent_name")
            if dep_name and dep_name not in agent_names:
                raise ValueError(
                    f"Agent '{agent['name']}' has missing dependency: '{dep_name}'"
                )


def _detect_circular_dependencies(agents: list[dict[str, Any]]) -> None:
    """Detect circular dependencies using depth-first search.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If circular dependencies are detected
    """
    _check_agents_for_cycles(agents)


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble of agents with dependency support."""

    name: str
    description: str
    agents: list[dict[str, Any]]
    default_task: str | None = None
    task: str | None = None  # Backward compatibility
    relative_path: str | None = None  # For hierarchical display
    validation: dict[str, Any] | None = None  # Validation configuration
    test_mode: dict[str, Any] | None = None  # Test mode configuration


class EnsembleLoader:
    """Loads ensemble configurations from files."""

    def load_from_file(self, file_path: str) -> EnsembleConfig:
        """Load ensemble configuration from a YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {file_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Support both default_task (preferred) and task (backward compatibility)
        default_task = data.get("default_task") or data.get("task")

        config = EnsembleConfig(
            name=data["name"],
            description=data["description"],
            agents=data["agents"],
            default_task=default_task,
            task=data.get("task"),  # Keep for backward compatibility
            validation=data.get("validation"),
            test_mode=data.get("test_mode"),
        )

        # Validate agent dependencies
        self._validate_dependencies(config.agents)

        return config

    def list_ensembles(self, directory: str) -> list[EnsembleConfig]:
        """List all ensemble configurations in a directory and subdirectories."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        ensembles = []
        # Use rglob for recursive search in subdirectories
        for yaml_file in dir_path.rglob("*.yaml"):
            try:
                config = self.load_from_file(str(yaml_file))
                # Store relative path for hierarchical display
                relative_path = yaml_file.relative_to(dir_path)
                config.relative_path = (
                    str(relative_path.parent)
                    if relative_path.parent != Path(".")
                    else None
                )
                ensembles.append(config)
            except Exception:
                # Skip invalid files
                continue

        # Also check for .yml files
        for yml_file in dir_path.rglob("*.yml"):
            try:
                config = self.load_from_file(str(yml_file))
                # Store relative path for hierarchical display
                relative_path = yml_file.relative_to(dir_path)
                config.relative_path = (
                    str(relative_path.parent)
                    if relative_path.parent != Path(".")
                    else None
                )
                ensembles.append(config)
            except Exception:
                # Skip invalid files
                continue

        return ensembles

    def _validate_dependencies(self, agents: list[dict[str, Any]]) -> None:
        """Validate agent dependencies for cycles and missing dependencies."""
        # Check for missing dependencies first
        _check_missing_dependencies(agents)

        # Validate fan_out agents have dependencies
        _validate_fan_out_dependencies(agents)

        # Then check for circular dependencies
        _detect_circular_dependencies(agents)

    def find_ensemble(self, directory: str, name: str) -> EnsembleConfig | None:
        """Find an ensemble by name in a directory, supporting hierarchical names.

        Supports matching by:
        - Simple name: "my-ensemble"
        - Full hierarchical name: "examples/my-ensemble/my-ensemble"
        - Directory path (if name matches): "examples/my-ensemble"
        """
        ensembles = self.list_ensembles(directory)
        for ensemble in ensembles:
            # Build potential matching patterns
            display_name = (
                f"{ensemble.relative_path}/{ensemble.name}"
                if ensemble.relative_path
                else ensemble.name
            )

            # Also support matching by directory path (relative_path alone)
            # if the last component matches the ensemble name
            directory_path = ensemble.relative_path

            # Check all matching patterns
            if ensemble.name == name or display_name == name:
                return ensemble

            # Support "examples/neon-shadows" matching an ensemble at
            # examples/neon-shadows/ensemble.yaml with name: neon-shadows
            if (
                directory_path
                and directory_path == name
                and name.endswith(ensemble.name)
            ):
                return ensemble

        return None
