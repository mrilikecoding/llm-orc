"""Dependency resolution for agent execution chains."""

import json
from collections.abc import Callable
from typing import Any

from llm_orc.core.execution.utils import dep_name
from llm_orc.schemas.agent_config import AgentConfig, ScriptAgentConfig


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
        dependent_agents: list[AgentConfig],
        results_dict: dict[str, Any],
    ) -> dict[str, str]:
        """Enhance input with dependency results for each dependent agent.

        Returns a dictionary mapping agent names to their enhanced input.
        Each agent gets only the results from their specific dependencies.

        For script agents, returns JSON-formatted input with a
        'dependencies' dict containing upstream results.

        For LLM agents, returns text-formatted input with natural language
        context about previous agent results.
        """
        return {
            agent_config.name: self._compute_agent_input(
                agent_config, base_input, results_dict
            )
            for agent_config in dependent_agents
        }

    def _compute_agent_input(
        self,
        agent_config: AgentConfig,
        base_input: str,
        results_dict: dict[str, Any],
    ) -> str:
        """Compute the enhanced input string for a single agent.

        Args:
            agent_config: Agent to compute input for.
            base_input: Original ensemble input.
            results_dict: Accumulated results from earlier agents.

        Returns:
            Input string for the agent.
        """
        agent_name = agent_config.name
        dependencies = agent_config.depends_on
        is_script_agent = isinstance(agent_config, ScriptAgentConfig)

        if not dependencies:
            if is_script_agent:
                return self._build_script_input(agent_name, base_input, {})
            return base_input

        # Apply input_key selection (ADR-014)
        effective_results, input_key_error = self._apply_input_key_selection(
            agent_config, results_dict
        )
        if input_key_error:
            return input_key_error

        dep_results_dict = self._extract_dependency_results_as_dict(
            dependencies, effective_results
        )

        if is_script_agent:
            return self._build_script_input(agent_name, base_input, dep_results_dict)

        dependency_results = self._extract_successful_dependency_results(
            dependencies, effective_results
        )
        if dependency_results:
            return self._build_enhanced_input_with_dependencies(
                agent_name, base_input, dependency_results
            )
        return self._build_enhanced_input_no_dependencies(agent_name, base_input)

    def _apply_input_key_selection(
        self,
        agent_config: AgentConfig,
        results_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Apply input_key selection to upstream results (ADR-014).

        Returns (effective_results, error_or_none). When input_key is
        set, the first dependency's response is replaced with the
        selected key's value. On error, returns an error message.
        """
        input_key = agent_config.input_key
        if not input_key or not agent_config.depends_on:
            return results_dict, None

        first_dep = dep_name(agent_config.depends_on[0])
        dep_result = results_dict.get(first_dep, {})

        if dep_result.get("status") != "success":
            return results_dict, None

        response = dep_result.get("response", "")

        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return results_dict, (
                f"input_key error: upstream '{first_dep}' output "
                f"is not dict-shaped (not valid JSON)"
            )

        if not isinstance(parsed, dict):
            return results_dict, (
                f"input_key error: upstream '{first_dep}' output "
                f"is not dict (got {type(parsed).__name__})"
            )

        if input_key not in parsed:
            available = list(parsed.keys())
            return results_dict, (
                f"input_key error: key '{input_key}' not found "
                f"in upstream '{first_dep}' output. "
                f"Available keys: {available}"
            )

        selected = parsed[input_key]
        new_response = selected if isinstance(selected, str) else json.dumps(selected)

        modified = dict(results_dict)
        modified[first_dep] = {
            **dep_result,
            "response": new_response,
        }
        return modified, None

    def _extract_successful_dependency_results(
        self, dependencies: list[str | dict[str, Any]], results_dict: dict[str, Any]
    ) -> list[str]:
        """Extract successful dependency results with role attribution.

        Args:
            dependencies: List of dependency names (str or dict form)
            results_dict: Dictionary of previous agent results

        Returns:
            List of formatted dependency result strings
        """
        dependency_results = []
        for dep in dependencies:
            agent_dep_name = dep_name(dep)
            if (
                agent_dep_name in results_dict
                and results_dict[agent_dep_name].get("status") == "success"
            ):
                response = results_dict[agent_dep_name]["response"]
                dep_role = self._get_agent_role_description(agent_dep_name)
                role_text = f" ({dep_role})" if dep_role else ""

                dependency_results.append(
                    f"Agent {agent_dep_name}{role_text}:\n{response}"
                )

        return dependency_results

    def _extract_dependency_results_as_dict(
        self, dependencies: list[str | dict[str, Any]], results_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract successful dependency results as a dict.

        Args:
            dependencies: List of dependency agent names (str or dict form)
            results_dict: Dictionary of previous agent results

        Returns:
            Dictionary mapping dependency names to their results
        """
        dep_results = {}
        for dep in dependencies:
            agent_dep_name = dep_name(dep)
            if (
                agent_dep_name in results_dict
                and results_dict[agent_dep_name].get("status") == "success"
            ):
                dep_results[agent_dep_name] = results_dict[agent_dep_name]
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
            f"Please respond to the following input, "
            f"taking into account the results from the previous agents "
            f"in the dependency chain.\n\n"
            f"Original Input:\n{base_input}\n\n"
            f"Previous Agent Results (for your reference):\n"
            f"{deps_text}\n\n"
            f"Please provide your own analysis, building upon "
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
        return base_input

    def has_dependencies(self, agent_config: AgentConfig) -> bool:
        """Check if an agent has dependencies."""
        return bool(agent_config.depends_on)

    def get_dependencies(self, agent_config: AgentConfig) -> list[str | dict[str, Any]]:
        """Get list of dependencies for an agent."""
        return agent_config.depends_on

    def dependencies_satisfied(
        self, agent_config: AgentConfig, completed_agents: set[str]
    ) -> bool:
        """Check if all dependencies for an agent are satisfied."""
        dependencies = self.get_dependencies(agent_config)
        return all(dep_name(dep) in completed_agents for dep in dependencies)

    @staticmethod
    def is_fan_out_instance_config(agent_config: AgentConfig) -> bool:
        """Check if an agent config is a fan-out instance.

        Args:
            agent_config: Agent configuration to check

        Returns:
            True if this is a fan-out instance configuration
        """
        return agent_config.fan_out_original is not None

    def prepare_fan_out_instance_input(
        self,
        instance_config: AgentConfig,
        base_input: str,
    ) -> str:
        """Prepare input for a fan-out instance.

        Args:
            instance_config: Instance configuration with fan_out_* metadata
            base_input: Original ensemble input

        Returns:
            Prepared input string (JSON for scripts, text for LLMs)
        """
        chunk = instance_config.fan_out_chunk
        index = instance_config.fan_out_index or 0
        total = instance_config.fan_out_total or 1
        name = instance_config.name
        is_script = isinstance(instance_config, ScriptAgentConfig)

        if is_script:
            return self._build_fan_out_script_input(
                name, chunk, index, total, base_input
            )
        else:
            return self._build_fan_out_llm_input(chunk, index, total, base_input)

    def _build_fan_out_script_input(
        self,
        agent_name: str,
        chunk: Any,
        chunk_index: int,
        total_chunks: int,
        base_input: str,
    ) -> str:
        """Build JSON input for a fan-out script agent instance."""
        script_input = {
            "agent_name": agent_name,
            "input": chunk,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "base_input": base_input,
            "context": {},
        }
        return json.dumps(script_input)

    def _build_fan_out_llm_input(
        self,
        chunk: Any,
        chunk_index: int,
        total_chunks: int,
        base_input: str,
    ) -> str:
        """Build text input for a fan-out LLM agent instance."""
        # Convert chunk to string if needed
        if isinstance(chunk, dict):
            chunk_text = json.dumps(chunk)
        else:
            chunk_text = str(chunk)

        return (
            f"Processing chunk {chunk_index + 1} of {total_chunks}.\n\n"
            f"Original task: {base_input}\n\n"
            f"Chunk content:\n{chunk_text}"
        )

    def filter_by_dependency_status(
        self,
        agents: list[AgentConfig],
        completed_agents: set[str],
        with_dependencies: bool = True,
    ) -> list[AgentConfig]:
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

    @staticmethod
    def get_agent_input(input_data: str | dict[str, str], agent_name: str) -> str:
        """Get appropriate input for an agent from uniform or per-agent input.

        Args:
            input_data: A string for uniform input, or a dict mapping
                       agent names to their specific enhanced input
            agent_name: Name of the agent to get input for

        Returns:
            Input string for the specified agent
        """
        if isinstance(input_data, dict):
            return input_data.get(agent_name, "")
        return input_data
