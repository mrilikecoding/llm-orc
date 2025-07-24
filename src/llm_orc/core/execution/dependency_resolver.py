"""Dependency resolution for agent execution chains."""

from collections.abc import Callable
from typing import Any


class DependencyResolver:
    """Resolves agent dependencies and enhances input with dependency results."""

    def __init__(
        self,
        role_resolver: Callable[[str], str | None],
    ) -> None:
        """Initialize resolver with role description function."""
        self._get_agent_role_description = role_resolver

    def enhance_input_with_dependencies(
        self,
        base_input: str,
        dependent_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> dict[str, str]:
        """Enhance input with dependency results for each dependent agent.

        Returns a dictionary mapping agent names to their enhanced input.
        Each agent gets only the results from their specific dependencies.
        """
        enhanced_inputs = {}

        for agent_config in dependent_agents:
            agent_name = agent_config["name"]
            dependencies = agent_config.get("depends_on", [])

            if not dependencies:
                enhanced_inputs[agent_name] = base_input
                continue

            # Build structured dependency results for this specific agent
            dependency_results = []
            for dep_name in dependencies:
                if (
                    dep_name in results_dict
                    and results_dict[dep_name].get("status") == "success"
                ):
                    response = results_dict[dep_name]["response"]
                    # Get agent role/profile for better attribution
                    dep_role = self._get_agent_role_description(dep_name)
                    role_text = f" ({dep_role})" if dep_role else ""

                    dependency_results.append(
                        f"Agent {dep_name}{role_text}:\n{response}"
                    )

            if dependency_results:
                deps_text = "\n\n".join(dependency_results)
                enhanced_inputs[agent_name] = (
                    f"You are {agent_name}. Please respond to the following input, "
                    f"taking into account the results from the previous agents "
                    f"in the dependency chain.\n\n"
                    f"Original Input:\n{base_input}\n\n"
                    f"Previous Agent Results (for your reference):\n"
                    f"{deps_text}\n\n"
                    f"Please provide your own analysis as {agent_name}, building upon "
                    f"(but not simply repeating) the previous results."
                )
            else:
                enhanced_inputs[agent_name] = (
                    f"You are {agent_name}. Please respond to: {base_input}"
                )

        return enhanced_inputs

    def has_dependencies(self, agent_config: dict[str, Any]) -> bool:
        """Check if an agent has dependencies."""
        dependencies = agent_config.get("depends_on", [])
        return bool(dependencies)

    def get_dependencies(self, agent_config: dict[str, Any]) -> list[str]:
        """Get list of dependencies for an agent."""
        dependencies = agent_config.get("depends_on", [])
        return dependencies if isinstance(dependencies, list) else []

    def dependencies_satisfied(
        self, agent_config: dict[str, Any], completed_agents: set[str]
    ) -> bool:
        """Check if all dependencies for an agent are satisfied."""
        dependencies = self.get_dependencies(agent_config)
        return all(dep in completed_agents for dep in dependencies)

    def filter_by_dependency_status(
        self,
        agents: list[dict[str, Any]],
        completed_agents: set[str],
        with_dependencies: bool = True,
    ) -> list[dict[str, Any]]:
        """Filter agents based on dependency satisfaction status.

        Args:
            agents: List of agent configurations
            completed_agents: Set of agent names that have completed
            with_dependencies: If True, return agents WITH satisfied dependencies.
                             If False, return agents WITHOUT dependencies.
        """
        if with_dependencies:
            return [
                agent
                for agent in agents
                if self.has_dependencies(agent)
                and self.dependencies_satisfied(agent, completed_agents)
            ]
        else:
            return [agent for agent in agents if not self.has_dependencies(agent)]

    def validate_dependency_chain(self, agents: list[dict[str, Any]]) -> list[str]:
        """Validate dependency chain and return list of validation errors."""
        # Perform basic validation (self-deps and missing deps)
        errors = _validate_basic_dependencies(agents)

        # Check for circular dependencies only if basic validation passes
        if not errors:
            circular_errors = _detect_circular_dependencies(agents, self)
            errors.extend(circular_errors)

        return errors


def _validate_basic_dependencies(agents: list[dict[str, Any]]) -> list[str]:
    """Validate basic dependency requirements (self-deps and missing deps).

    Args:
        agents: List of agent configurations

    Returns:
        List of validation error messages
    """
    errors = []
    agent_names = {agent["name"] for agent in agents}

    for agent in agents:
        agent_name = agent["name"]
        dependencies = agent.get("depends_on", [])

        # Check for self-dependency
        if agent_name in dependencies:
            errors.append(f"Agent '{agent_name}' cannot depend on itself")

        # Check for missing dependencies
        for dep in dependencies:
            if dep not in agent_names:
                errors.append(
                    f"Agent '{agent_name}' depends on non-existent agent '{dep}'"
                )

    return errors


def _detect_circular_dependencies(
    agents: list[dict[str, Any]], resolver: DependencyResolver
) -> list[str]:
    """Detect circular dependencies using depth-first search.

    Args:
        agents: List of agent configurations
        resolver: Dependency resolver instance for getting dependencies

    Returns:
        List of error messages (empty if no cycles detected)
    """
    errors = []
    visited = set()
    rec_stack = set()

    def has_cycle(agent_name: str) -> bool:
        if agent_name in rec_stack:
            return True
        if agent_name in visited:
            return False

        visited.add(agent_name)
        rec_stack.add(agent_name)

        # Find agent config by name
        agent_config = next(
            (a for a in agents if a["name"] == agent_name), None
        )
        if agent_config:
            for dep in resolver.get_dependencies(agent_config):
                if has_cycle(dep):
                    return True

        rec_stack.remove(agent_name)
        return False

    for agent in agents:
        if agent["name"] not in visited:
            if has_cycle(agent["name"]):
                errors.append("Circular dependency detected in agent chain")
                break

    return errors
