"""Ensemble configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_orc.schemas.agent_config import AgentConfig, parse_agent_config


def _find_agent_by_name(
    agents: list[AgentConfig], agent_name: str
) -> AgentConfig | None:
    """Find an agent by name in the agents list.

    Args:
        agents: List of agent configurations
        agent_name: Name of agent to find

    Returns:
        Agent configuration if found, None otherwise
    """
    return next((a for a in agents if a.name == agent_name), None)


def _find_cycle_from(
    agent_name: str,
    agents: list[AgentConfig],
    visited: set[str],
    in_stack: set[str],
    path: list[str],
) -> list[str] | None:
    """DFS from agent_name, returning cycle path if found.

    Handles both simple string and conditional dict dependencies.

    Args:
        agent_name: Starting agent name
        agents: List of agent configurations
        visited: Set of already-visited agent names
        in_stack: Set of agents in the current recursion stack
        path: Ordered list of agents in the current DFS path

    Returns:
        List of agent names forming the cycle, or None
    """
    if agent_name in in_stack:
        cycle_start = path.index(agent_name)
        return path[cycle_start:]
    if agent_name in visited:
        return None

    visited.add(agent_name)
    in_stack.add(agent_name)
    path.append(agent_name)

    agent_config = _find_agent_by_name(agents, agent_name)
    if agent_config:
        for dep in agent_config.depends_on:
            dep_name = dep if isinstance(dep, str) else dep.get("agent_name")
            if dep_name:
                cycle = _find_cycle_from(dep_name, agents, visited, in_stack, path)
                if cycle is not None:
                    return cycle

    path.pop()
    in_stack.remove(agent_name)
    return None


def detect_cycle(agents: list[AgentConfig]) -> list[str] | None:
    """Detect cycles in agent dependencies using DFS.

    Handles both string and dict-form dependencies.

    Args:
        agents: List of agent configurations

    Returns:
        List of agent names forming the cycle, or None if acyclic
    """
    visited: set[str] = set()

    for agent in agents:
        if agent.name not in visited:
            cycle = _find_cycle_from(agent.name, agents, visited, set(), [])
            if cycle is not None:
                return cycle
    return None


def assert_no_cycles(agents: list[AgentConfig]) -> None:
    """Raise ValueError if any cycle exists in agent dependencies.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: With the cycle path (e.g. "a -> b -> a")
    """
    cycle = detect_cycle(agents)
    if cycle is not None:
        cycle_str = " -> ".join([*cycle, cycle[0]])
        raise ValueError(f"Circular dependency detected: {cycle_str}")


def _validate_fan_out_dependencies(agents: list[AgentConfig]) -> None:
    """Validate that fan_out agents have required dependencies.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If any agent has fan_out: true without depends_on
    """
    for agent in agents:
        if agent.fan_out is True:
            if not agent.depends_on:
                raise ValueError(
                    f"Agent '{agent.name}' has fan_out: true but requires "
                    f"depends_on to specify the upstream agent providing "
                    f"the array"
                )


def _check_missing_dependencies(agents: list[AgentConfig]) -> None:
    """Check for missing dependencies in agent configurations.

    Args:
        agents: List of agent configurations

    Raises:
        ValueError: If any agent depends on a non-existent agent
    """
    agent_names = {agent.name for agent in agents}

    for agent in agents:
        for dep in agent.depends_on:
            dep_name = dep if isinstance(dep, str) else dep.get("agent_name")
            if dep_name and dep_name not in agent_names:
                raise ValueError(
                    f"Agent '{agent.name}' has missing dependency: '{dep_name}'"
                )


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble of agents with dependency support."""

    name: str
    description: str
    agents: list[AgentConfig] = field(default_factory=list)
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

        # Parse each agent dict into typed AgentConfig (ADR-012)
        agents = [parse_agent_config(a) for a in data["agents"]]

        config = EnsembleConfig(
            name=data["name"],
            description=data["description"],
            agents=agents,
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

    def _validate_dependencies(self, agents: list[AgentConfig]) -> None:
        """Validate agent dependencies for cycles and missing dependencies."""
        _check_missing_dependencies(agents)
        _validate_fan_out_dependencies(agents)
        assert_no_cycles(agents)

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
