"""Dependency resolution for agent execution chains."""

import json
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

        For script agents (those with 'script' key), returns JSON-formatted
        input with a 'dependencies' dict containing upstream results.

        For LLM agents, returns text-formatted input with natural language
        context about previous agent results.
        """
        enhanced_inputs = {}

        for agent_config in dependent_agents:
            agent_name = agent_config["name"]
            dependencies = agent_config.get("depends_on", [])
            is_script_agent = "script" in agent_config

            if not dependencies:
                if is_script_agent:
                    enhanced_inputs[agent_name] = self._build_script_input(
                        agent_name, base_input, {}
                    )
                else:
                    enhanced_inputs[agent_name] = base_input
                continue

            # Extract dependency results as dict for script agents
            dep_results_dict = self._extract_dependency_results_as_dict(
                dependencies, results_dict
            )

            if is_script_agent:
                # Script agents get JSON with dependencies dict
                enhanced_inputs[agent_name] = self._build_script_input(
                    agent_name, base_input, dep_results_dict
                )
            else:
                # LLM agents get text format
                dependency_results = self._extract_successful_dependency_results(
                    dependencies, results_dict
                )
                if dependency_results:
                    enhanced_inputs[agent_name] = (
                        self._build_enhanced_input_with_dependencies(
                            agent_name, base_input, dependency_results
                        )
                    )
                else:
                    enhanced_inputs[agent_name] = (
                        self._build_enhanced_input_no_dependencies(
                            agent_name, base_input
                        )
                    )

        return enhanced_inputs

    def _extract_successful_dependency_results(
        self, dependencies: list[str], results_dict: dict[str, Any]
    ) -> list[str]:
        """Extract successful dependency results with role attribution.

        Args:
            dependencies: List of dependency names
            results_dict: Dictionary of previous agent results

        Returns:
            List of formatted dependency result strings
        """
        dependency_results = []
        for dep_name in dependencies:
            if (
                dep_name in results_dict
                and results_dict[dep_name].get("status") == "success"
            ):
                response = results_dict[dep_name]["response"]
                dep_role = self._get_agent_role_description(dep_name)
                role_text = f" ({dep_role})" if dep_role else ""

                dependency_results.append(f"Agent {dep_name}{role_text}:\n{response}")

        return dependency_results

    def _extract_dependency_results_as_dict(
        self, dependencies: list[str], results_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract successful dependency results as a dict.

        Args:
            dependencies: List of dependency agent names
            results_dict: Dictionary of previous agent results

        Returns:
            Dictionary mapping dependency names to their results
        """
        dep_results = {}
        for dep_name in dependencies:
            if (
                dep_name in results_dict
                and results_dict[dep_name].get("status") == "success"
            ):
                dep_results[dep_name] = results_dict[dep_name]
        return dep_results

    def _build_script_input(
        self, agent_name: str, base_input: str, dependencies: dict[str, Any]
    ) -> str:
        """Build JSON-formatted input for script agents.

        Args:
            agent_name: Name of the script agent
            base_input: Original input text
            dependencies: Dict of dependency results

        Returns:
            JSON string conforming to ScriptAgentInput schema
        """
        script_input = {
            "agent_name": agent_name,
            "input_data": base_input,
            "context": {},
            "dependencies": dependencies,
        }
        return json.dumps(script_input)

    def _build_enhanced_input_with_dependencies(
        self, agent_name: str, base_input: str, dependency_results: list[str]
    ) -> str:
        """Build enhanced input with dependency results.

        Args:
            agent_name: Name of the target agent
            base_input: Original input text
            dependency_results: List of formatted dependency result strings

        Returns:
            Enhanced input string with dependencies
        """
        deps_text = "\n\n".join(dependency_results)
        return (
            f"You are {agent_name}. Please respond to the following input, "
            f"taking into account the results from the previous agents "
            f"in the dependency chain.\n\n"
            f"Original Input:\n{base_input}\n\n"
            f"Previous Agent Results (for your reference):\n"
            f"{deps_text}\n\n"
            f"Please provide your own analysis as {agent_name}, building upon "
            f"(but not simply repeating) the previous results."
        )

    def _build_enhanced_input_no_dependencies(
        self, agent_name: str, base_input: str
    ) -> str:
        """Build enhanced input for agent without dependencies.

        Args:
            agent_name: Name of the target agent
            base_input: Original input text

        Returns:
            Simple enhanced input string
        """
        return f"You are {agent_name}. Please respond to: {base_input}"

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


def _find_agent_by_name(
    agents: list[dict[str, Any]], agent_name: str
) -> dict[str, Any] | None:
    """Find agent configuration by name.

    Args:
        agents: List of agent configurations
        agent_name: Name of agent to find

    Returns:
        Agent configuration or None if not found
    """
    return next((a for a in agents if a["name"] == agent_name), None)


def _check_cycle_from_node(
    agent_name: str,
    agents: list[dict[str, Any]],
    resolver: DependencyResolver,
    visited: set[str],
    rec_stack: set[str],
) -> bool:
    """Check for cycles starting from a specific agent node.

    Args:
        agent_name: Starting agent name
        agents: List of agent configurations
        resolver: Dependency resolver for getting dependencies
        visited: Set of already visited nodes
        rec_stack: Set of nodes in current recursion stack

    Returns:
        True if cycle detected, False otherwise
    """
    if agent_name in rec_stack:
        return True
    if agent_name in visited:
        return False

    visited.add(agent_name)
    rec_stack.add(agent_name)

    agent_config = _find_agent_by_name(agents, agent_name)
    if agent_config:
        for dep in resolver.get_dependencies(agent_config):
            if _check_cycle_from_node(dep, agents, resolver, visited, rec_stack):
                return True

    rec_stack.remove(agent_name)
    return False


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
    visited: set[str] = set()
    rec_stack: set[str] = set()

    for agent in agents:
        agent_name = agent["name"]
        if agent_name not in visited:
            if _check_cycle_from_node(agent_name, agents, resolver, visited, rec_stack):
                return ["Circular dependency detected in agent chain"]

    return []
