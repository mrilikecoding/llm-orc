"""Input enhancement for agents with dependencies."""

from typing import Any


class InputEnhancer:
    """Enhances agent input with dependency results and role information."""

    def __init__(
        self, current_agent_configs: list[dict[str, Any]] | None = None
    ) -> None:
        """Initialize the input enhancer.

        Args:
            current_agent_configs: Current agent configurations for role lookup
        """
        self._current_agent_configs = current_agent_configs

    def enhance_input_with_dependencies(
        self,
        base_input: str,
        dependent_agents: list[dict[str, Any]],
        results_dict: dict[str, Any],
    ) -> dict[str, str]:
        """Enhance input with dependency results for each dependent agent.

        Returns a dictionary mapping agent names to their enhanced input.
        Each agent gets only the results from their specific dependencies.

        Args:
            base_input: Original input text
            dependent_agents: List of agent configurations with dependencies
            results_dict: Dictionary of previous agent results

        Returns:
            Dictionary mapping agent names to their enhanced input
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
                    dep_role = self.get_agent_role_description(dep_name)
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

    def get_agent_role_description(self, agent_name: str) -> str | None:
        """Get a human-readable role description for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Human-readable role description or None if not found
        """
        # Try to find the agent in the current ensemble config
        if self._current_agent_configs:
            for agent_config in self._current_agent_configs:
                if agent_config["name"] == agent_name:
                    # Try model_profile first, then infer from name
                    if "model_profile" in agent_config:
                        profile = str(agent_config["model_profile"])
                        # Convert kebab-case to title case
                        return profile.replace("-", " ").title()
                    else:
                        # Convert agent name to readable format
                        return agent_name.replace("-", " ").title()

        # Fallback: convert name to readable format
        return agent_name.replace("-", " ").title()

    def get_agent_input(
        self, input_data: str | dict[str, str], agent_name: str
    ) -> str:
        """Get appropriate input for an agent from uniform or per-agent input.

        Args:
            input_data: Either a string for uniform input, or a dict mapping
                       agent names to their specific enhanced input
            agent_name: Name of the agent to get input for

        Returns:
            Input string for the specified agent
        """
        if isinstance(input_data, dict):
            return input_data.get(agent_name, "")
        return input_data

    def create_enhanced_input_for_agent(
        self,
        agent_name: str,
        base_input: str,
        dependencies: list[str],
        results_dict: dict[str, Any],
    ) -> str:
        """Create enhanced input for a single agent with its dependencies.

        Args:
            agent_name: Name of the agent
            base_input: Original input text
            dependencies: List of dependency agent names
            results_dict: Dictionary of previous agent results

        Returns:
            Enhanced input string for the agent
        """
        if not dependencies:
            return base_input

        # Build structured dependency results
        dependency_results = []
        for dep_name in dependencies:
            if (
                dep_name in results_dict
                and results_dict[dep_name].get("status") == "success"
            ):
                response = results_dict[dep_name]["response"]
                dep_role = self.get_agent_role_description(dep_name)
                role_text = f" ({dep_role})" if dep_role else ""

                dependency_results.append(
                    f"Agent {dep_name}{role_text}:\n{response}"
                )

        if dependency_results:
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
        else:
            return f"You are {agent_name}. Please respond to: {base_input}"

    def update_agent_configs(self, agent_configs: list[dict[str, Any]]) -> None:
        """Update the current agent configurations for role lookup.

        Args:
            agent_configs: New agent configurations
        """
        self._current_agent_configs = agent_configs
