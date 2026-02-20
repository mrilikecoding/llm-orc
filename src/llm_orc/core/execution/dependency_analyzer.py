"""Agent dependency analysis for execution planning."""

from typing import Any

from llm_orc.core.execution.patterns import INSTANCE_PATTERN
from llm_orc.schemas.agent_config import AgentConfig


class DependencyAnalyzer:
    """Analyzes agent dependencies to determine execution order."""

    def analyze_enhanced_dependency_graph(
        self, agent_configs: list[AgentConfig]
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
        dependency_map = {
            agent_config.name: agent_config.depends_on for agent_config in agent_configs
        }

        phases = []
        remaining_agents = agent_configs.copy()
        processed_agents: set[str] = set()

        while remaining_agents:
            current_phase = []

            for agent_config in remaining_agents:
                agent_name = agent_config.name
                dependencies = dependency_map[agent_name]

                if self.agent_dependencies_satisfied(dependencies, processed_agents):
                    current_phase.append(agent_config)

            for agent_config in current_phase:
                processed_agents.add(agent_config.name)
            remaining_agents = [
                a for a in remaining_agents if a.name not in processed_agents
            ]

            if not current_phase:
                raise ValueError("Circular dependency detected in agent configuration")

            phases.append(current_phase)

        return {
            "phases": phases,
            "dependency_map": dependency_map,
            "total_phases": len(phases),
        }

    def agent_dependencies_satisfied(
        self, dependencies: list[str | dict[str, Any]], processed_agents: set[str]
    ) -> bool:
        """Check if an agent's dependencies have been processed.

        Args:
            dependencies: List of dependency names (str or dict form)
            processed_agents: Set of already processed agent names

        Returns:
            True if all dependencies are satisfied
        """
        return len(dependencies) == 0 or all(
            self._dep_name(dep) in processed_agents for dep in dependencies
        )

    @staticmethod
    def _dep_name(dep: str | dict[str, Any]) -> str:
        """Extract the agent name from a dependency entry.

        Dependency entries are either a plain string or a dict with a single
        ``"agent_name"`` key, e.g. ``{"agent_name": "b"}`` for conditional deps.
        """
        if isinstance(dep, dict):
            return str(dep["agent_name"])
        return dep

    def group_agents_by_level(
        self, agent_configs: list[AgentConfig]
    ) -> dict[int, list[AgentConfig]]:
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
        self,
        agent_name: str,
        dependency_map: dict[str, list[str | dict[str, Any]]],
        _cache: dict[str, int] | None = None,
    ) -> int:
        """Calculate the dependency level of an agent.

        Args:
            agent_name: Name of the agent
            dependency_map: Mapping of agent names to their dependencies
            _cache: Internal memoization cache (created automatically)

        Returns:
            Dependency level (0 for no dependencies, higher for more levels)
        """
        if _cache is None:
            _cache = {}

        if agent_name in _cache:
            return _cache[agent_name]

        if agent_name not in dependency_map:
            _cache[agent_name] = 0
            return 0

        dependencies = dependency_map[agent_name]
        if not dependencies:
            _cache[agent_name] = 0
            return 0

        # Level is 1 + max level of dependencies
        max_dep_level = max(
            self.calculate_agent_level(self._dep_name(dep), dependency_map, _cache)
            for dep in dependencies
        )
        level = max_dep_level + 1
        _cache[agent_name] = level
        return level

    def get_execution_phases(self, agent_configs: list[AgentConfig]) -> list[list[str]]:
        """Get execution phases as lists of agent names.

        Args:
            agent_configs: List of agent configurations

        Returns:
            List of phases, each containing agent names that can run in parallel
        """
        dependency_graph = self.analyze_enhanced_dependency_graph(agent_configs)
        return [[agent.name for agent in phase] for phase in dependency_graph["phases"]]

    def validate_dependencies(self, agent_configs: list[AgentConfig]) -> list[str]:
        """Validate agent dependencies and return any issues found.

        Args:
            agent_configs: List of agent configurations

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        agent_names = {config.name for config in agent_configs}

        for agent_config in agent_configs:
            agent_name = agent_config.name
            dep_names = [self._dep_name(d) for d in agent_config.depends_on]

            if agent_name in dep_names:
                errors.append(f"Agent '{agent_name}' cannot depend on itself")

            for dep_name in dep_names:
                if dep_name not in agent_names:
                    errors.append(
                        f"Agent '{agent_name}' depends on missing agent '{dep_name}'"
                    )

        try:
            self.analyze_enhanced_dependency_graph(agent_configs)
        except ValueError as e:
            errors.append(str(e))

        return errors

    # ========== Fan-Out Support (Issue #73) ==========

    @staticmethod
    def normalize_agent_name(name: str) -> str:
        """Normalize agent name by removing instance index if present.

        Converts 'extractor[0]' to 'extractor' for dependency checking.

        Args:
            name: Agent name, possibly with instance index

        Returns:
            Original agent name without index
        """
        match = INSTANCE_PATTERN.match(name)
        if match:
            return match.group(1)
        return name

    @staticmethod
    def is_fan_out_instance_name(name: str) -> bool:
        """Check if name matches the fan-out instance pattern 'agent[N]'.

        Args:
            name: Agent name to check

        Returns:
            True if name matches instance pattern, False otherwise
        """
        return bool(INSTANCE_PATTERN.match(name))
