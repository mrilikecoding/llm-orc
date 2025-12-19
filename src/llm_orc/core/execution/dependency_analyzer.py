"""Agent dependency analysis for execution planning."""

import re
from typing import Any


class DependencyAnalyzer:
    """Analyzes agent dependencies to determine execution order."""

    def analyze_enhanced_dependency_graph(
        self, agent_configs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze agent dependencies and organize into execution phases.

        Uses topological sort to determine optimal execution order for agents
        with dependencies. Agents with no dependencies can run in parallel.

        Args:
            agent_configs: List of agent configurations with optional dependencies

        Returns:
            Dictionary containing:
            - phases: List of agent lists, each phase can run in parallel
            - dependency_map: Mapping of agent names to their dependencies
            - total_phases: Number of execution phases

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build dependency map for efficient lookup
        dependency_map = {
            agent_config["name"]: agent_config.get("depends_on", [])
            for agent_config in agent_configs
        }

        # Topological sort to determine execution phases
        phases = []
        remaining_agents = agent_configs.copy()
        processed_agents: set[str] = set()

        while remaining_agents:
            current_phase = []
            agents_to_remove = []

            # Find agents whose dependencies have been processed
            for agent_config in remaining_agents:
                agent_name = agent_config["name"]
                dependencies = dependency_map[agent_name]

                # Agent is ready if it has no dependencies or all are processed
                if self.agent_dependencies_satisfied(dependencies, processed_agents):
                    current_phase.append(agent_config)
                    agents_to_remove.append(agent_config)

            # Update processed agents and remove from remaining
            for agent_config in agents_to_remove:
                processed_agents.add(agent_config["name"])
                remaining_agents.remove(agent_config)

            # Detect circular dependencies
            if not current_phase:
                raise ValueError("Circular dependency detected in agent configuration")

            phases.append(current_phase)

        return {
            "phases": phases,
            "dependency_map": dependency_map,
            "total_phases": len(phases),
        }

    def agent_dependencies_satisfied(
        self, dependencies: list[str], processed_agents: set[str]
    ) -> bool:
        """Check if an agent's dependencies have been processed.

        Args:
            dependencies: List of dependency names
            processed_agents: Set of already processed agent names

        Returns:
            True if all dependencies are satisfied
        """
        return len(dependencies) == 0 or all(
            dep in processed_agents for dep in dependencies
        )

    def group_agents_by_level(
        self, agent_configs: list[dict[str, Any]]
    ) -> dict[int, list[dict[str, Any]]]:
        """Group agents by their dependency level for parallel execution.

        Args:
            agent_configs: List of agent configurations

        Returns:
            Dictionary mapping level numbers to lists of agents
        """
        dependency_graph = self.analyze_enhanced_dependency_graph(agent_configs)
        levels = {}

        for level, phase_agents in enumerate(dependency_graph["phases"]):
            levels[level] = phase_agents

        return levels

    def calculate_agent_level(
        self, agent_name: str, dependency_map: dict[str, list[str]]
    ) -> int:
        """Calculate the dependency level of an agent.

        Args:
            agent_name: Name of the agent
            dependency_map: Mapping of agent names to their dependencies

        Returns:
            Dependency level (0 for no dependencies, higher for more levels)
        """
        if agent_name not in dependency_map:
            return 0

        dependencies = dependency_map[agent_name]
        if not dependencies:
            return 0

        # Level is 1 + max level of dependencies
        max_dep_level = max(
            self.calculate_agent_level(dep, dependency_map) for dep in dependencies
        )
        return max_dep_level + 1

    def get_execution_phases(
        self, agent_configs: list[dict[str, Any]]
    ) -> list[list[str]]:
        """Get execution phases as lists of agent names.

        Args:
            agent_configs: List of agent configurations

        Returns:
            List of phases, each containing agent names that can run in parallel
        """
        dependency_graph = self.analyze_enhanced_dependency_graph(agent_configs)
        return [
            [agent["name"] for agent in phase] for phase in dependency_graph["phases"]
        ]

    def validate_dependencies(self, agent_configs: list[dict[str, Any]]) -> list[str]:
        """Validate agent dependencies and return any issues found.

        Args:
            agent_configs: List of agent configurations

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        agent_names = {config["name"] for config in agent_configs}

        for agent_config in agent_configs:
            agent_name = agent_config["name"]
            dependencies = agent_config.get("depends_on", [])

            # Check for self-dependency
            if agent_name in dependencies:
                errors.append(f"Agent '{agent_name}' cannot depend on itself")

            # Check for missing dependencies
            for dep in dependencies:
                if dep not in agent_names:
                    errors.append(
                        f"Agent '{agent_name}' depends on missing agent '{dep}'"
                    )

        # Check for circular dependencies by attempting analysis
        try:
            self.analyze_enhanced_dependency_graph(agent_configs)
        except ValueError as e:
            errors.append(str(e))

        return errors

    # ========== Fan-Out Support (Issue #73) ==========

    # Pattern for instance names: agent_name[index]
    _INSTANCE_PATTERN = re.compile(r"^(.+)\[(\d+)\]$")

    def normalize_agent_name(self, name: str) -> str:
        """Normalize agent name by removing instance index if present.

        Converts 'extractor[0]' to 'extractor' for dependency checking.

        Args:
            name: Agent name, possibly with instance index

        Returns:
            Original agent name without index
        """
        match = self._INSTANCE_PATTERN.match(name)
        if match:
            return match.group(1)
        return name

    def is_fan_out_instance_name(self, name: str) -> bool:
        """Check if name matches the fan-out instance pattern 'agent[N]'.

        Args:
            name: Agent name to check

        Returns:
            True if name matches instance pattern, False otherwise
        """
        return bool(self._INSTANCE_PATTERN.match(name))
