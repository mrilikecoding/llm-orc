"""Comprehensive tests for agent dependency analyzer."""

from typing import Any

import pytest

from llm_orc.core.execution.dependency_analyzer import DependencyAnalyzer


class TestDependencyAnalyzer:
    """Test dependency analyzer functionality."""

    def test_analyze_enhanced_dependency_graph_no_dependencies(self) -> None:
        """Test analysis with agents having no dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a"},
            {"name": "agent_b"},
            {"name": "agent_c"},
        ]

        # When
        result = analyzer.analyze_enhanced_dependency_graph(agent_configs)

        # Then
        assert result["total_phases"] == 1
        assert len(result["phases"]) == 1
        assert len(result["phases"][0]) == 3
        assert result["dependency_map"] == {
            "agent_a": [],
            "agent_b": [],
            "agent_c": [],
        }

    def test_analyze_enhanced_dependency_graph_linear_dependencies(self) -> None:
        """Test analysis with linear dependency chain."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
            {"name": "agent_c", "depends_on": ["agent_b"]},
        ]

        # When
        result = analyzer.analyze_enhanced_dependency_graph(agent_configs)

        # Then
        assert result["total_phases"] == 3
        assert len(result["phases"]) == 3
        assert len(result["phases"][0]) == 1  # agent_a
        assert len(result["phases"][1]) == 1  # agent_b
        assert len(result["phases"][2]) == 1  # agent_c
        assert result["phases"][0][0]["name"] == "agent_a"
        assert result["phases"][1][0]["name"] == "agent_b"
        assert result["phases"][2][0]["name"] == "agent_c"

    def test_analyze_enhanced_dependency_graph_parallel_dependencies(self) -> None:
        """Test analysis with parallel dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
            {"name": "agent_d", "depends_on": ["agent_a", "agent_b"]},
        ]

        # When
        result = analyzer.analyze_enhanced_dependency_graph(agent_configs)

        # Then
        assert result["total_phases"] == 2
        assert len(result["phases"]) == 2
        assert len(result["phases"][0]) == 2  # agent_a, agent_b
        assert len(result["phases"][1]) == 2  # agent_c, agent_d

        # Check first phase contains both independent agents
        phase_0_names = {agent["name"] for agent in result["phases"][0]}
        assert phase_0_names == {"agent_a", "agent_b"}

        # Check second phase contains both dependent agents
        phase_1_names = {agent["name"] for agent in result["phases"][1]}
        assert phase_1_names == {"agent_c", "agent_d"}

    def test_analyze_enhanced_dependency_graph_complex_dependencies(self) -> None:
        """Test analysis with complex dependency structure."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a"]},
            {"name": "agent_d", "depends_on": ["agent_b"]},
            {"name": "agent_e", "depends_on": ["agent_c", "agent_d"]},
        ]

        # When
        result = analyzer.analyze_enhanced_dependency_graph(agent_configs)

        # Then
        assert result["total_phases"] == 3
        assert len(result["phases"]) == 3

        # Phase 0: agent_a, agent_b (no dependencies)
        phase_0_names = {agent["name"] for agent in result["phases"][0]}
        assert phase_0_names == {"agent_a", "agent_b"}

        # Phase 1: agent_c, agent_d (depend on phase 0)
        phase_1_names = {agent["name"] for agent in result["phases"][1]}
        assert phase_1_names == {"agent_c", "agent_d"}

        # Phase 2: agent_e (depends on phase 1)
        assert len(result["phases"][2]) == 1
        assert result["phases"][2][0]["name"] == "agent_e"

    def test_analyze_enhanced_dependency_graph_circular_dependency(self) -> None:
        """Test detection of circular dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": ["agent_b"]},
            {"name": "agent_b", "depends_on": ["agent_c"]},
            {"name": "agent_c", "depends_on": ["agent_a"]},
        ]

        # When / Then
        with pytest.raises(ValueError, match="Circular dependency detected"):
            analyzer.analyze_enhanced_dependency_graph(agent_configs)

    def test_analyze_enhanced_dependency_graph_empty_list(self) -> None:
        """Test analysis with empty agent list."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs: list[dict[str, Any]] = []

        # When
        result = analyzer.analyze_enhanced_dependency_graph(agent_configs)

        # Then
        assert result["total_phases"] == 0
        assert result["phases"] == []
        assert result["dependency_map"] == {}

    def test_agent_dependencies_satisfied_no_dependencies(self) -> None:
        """Test satisfaction check with no dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        dependencies: list[str] = []
        processed_agents: set[str] = set()

        # When
        result = analyzer.agent_dependencies_satisfied(dependencies, processed_agents)

        # Then
        assert result is True

    def test_agent_dependencies_satisfied_all_satisfied(self) -> None:
        """Test satisfaction check with all dependencies satisfied."""
        # Given
        analyzer = DependencyAnalyzer()
        dependencies = ["agent_a", "agent_b"]
        processed_agents = {"agent_a", "agent_b", "agent_c"}

        # When
        result = analyzer.agent_dependencies_satisfied(dependencies, processed_agents)

        # Then
        assert result is True

    def test_agent_dependencies_satisfied_some_unsatisfied(self) -> None:
        """Test satisfaction check with some dependencies unsatisfied."""
        # Given
        analyzer = DependencyAnalyzer()
        dependencies = ["agent_a", "agent_b", "agent_c"]
        processed_agents = {"agent_a", "agent_b"}

        # When
        result = analyzer.agent_dependencies_satisfied(dependencies, processed_agents)

        # Then
        assert result is False

    def test_agent_dependencies_satisfied_none_satisfied(self) -> None:
        """Test satisfaction check with no dependencies satisfied."""
        # Given
        analyzer = DependencyAnalyzer()
        dependencies = ["agent_a", "agent_b"]
        processed_agents: set[str] = set()

        # When
        result = analyzer.agent_dependencies_satisfied(dependencies, processed_agents)

        # Then
        assert result is False

    def test_group_agents_by_level_simple(self) -> None:
        """Test grouping agents by dependency level."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
            {"name": "agent_c", "depends_on": ["agent_b"]},
        ]

        # When
        result = analyzer.group_agents_by_level(agent_configs)

        # Then
        assert len(result) == 3
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert len(result[2]) == 1
        assert result[0][0]["name"] == "agent_a"
        assert result[1][0]["name"] == "agent_b"
        assert result[2][0]["name"] == "agent_c"

    def test_group_agents_by_level_parallel(self) -> None:
        """Test grouping agents with parallel execution opportunities."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]

        # When
        result = analyzer.group_agents_by_level(agent_configs)

        # Then
        assert len(result) == 2
        assert len(result[0]) == 2  # agent_a, agent_b
        assert len(result[1]) == 1  # agent_c

        level_0_names = {agent["name"] for agent in result[0]}
        assert level_0_names == {"agent_a", "agent_b"}
        assert result[1][0]["name"] == "agent_c"

    def test_calculate_agent_level_no_dependencies(self) -> None:
        """Test calculating level for agent with no dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        dependency_map: dict[str, list[str]] = {"agent_a": []}

        # When
        result = analyzer.calculate_agent_level("agent_a", dependency_map)

        # Then
        assert result == 0

    def test_calculate_agent_level_missing_agent(self) -> None:
        """Test calculating level for agent not in dependency map."""
        # Given
        analyzer = DependencyAnalyzer()
        dependency_map: dict[str, list[str]] = {}

        # When
        result = analyzer.calculate_agent_level("agent_a", dependency_map)

        # Then
        assert result == 0

    def test_calculate_agent_level_single_dependency(self) -> None:
        """Test calculating level for agent with single dependency."""
        # Given
        analyzer = DependencyAnalyzer()
        dependency_map = {"agent_a": [], "agent_b": ["agent_a"]}

        # When
        result = analyzer.calculate_agent_level("agent_b", dependency_map)

        # Then
        assert result == 1

    def test_calculate_agent_level_nested_dependencies(self) -> None:
        """Test calculating level for agent with nested dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        dependency_map = {
            "agent_a": [],
            "agent_b": ["agent_a"],
            "agent_c": ["agent_b"],
            "agent_d": ["agent_c"],
        }

        # When
        result = analyzer.calculate_agent_level("agent_d", dependency_map)

        # Then
        assert result == 3

    def test_calculate_agent_level_multiple_dependencies_different_levels(
        self,
    ) -> None:
        """Test calculating level with dependencies at different levels."""
        # Given
        analyzer = DependencyAnalyzer()
        dependency_map = {
            "agent_a": [],
            "agent_b": ["agent_a"],
            "agent_c": ["agent_b"],
            "agent_d": ["agent_a", "agent_c"],  # Depends on level 0 and level 2
        }

        # When
        result = analyzer.calculate_agent_level("agent_d", dependency_map)

        # Then
        assert result == 3  # 1 + max(0, 2)

    def test_get_execution_phases_simple(self) -> None:
        """Test getting execution phases as agent names."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # When
        result = analyzer.get_execution_phases(agent_configs)

        # Then
        assert len(result) == 2
        assert result[0] == ["agent_a"]
        assert result[1] == ["agent_b"]

    def test_get_execution_phases_parallel(self) -> None:
        """Test getting execution phases with parallel agents."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]

        # When
        result = analyzer.get_execution_phases(agent_configs)

        # Then
        assert len(result) == 2
        assert len(result[0]) == 2
        assert set(result[0]) == {"agent_a", "agent_b"}
        assert result[1] == ["agent_c"]

    def test_validate_dependencies_valid_configuration(self) -> None:
        """Test validation of valid dependency configuration."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert result == []

    def test_validate_dependencies_self_dependency(self) -> None:
        """Test validation detects self-dependency."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": ["agent_a"]},
        ]

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert len(result) >= 1
        error_messages = " ".join(result)
        assert "Agent 'agent_a' cannot depend on itself" in error_messages

    def test_validate_dependencies_missing_dependency(self) -> None:
        """Test validation detects missing dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": ["missing_agent"]},
        ]

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert len(result) >= 1
        error_messages = " ".join(result)
        assert "depends on missing agent 'missing_agent'" in error_messages

    def test_validate_dependencies_circular_dependency(self) -> None:
        """Test validation detects circular dependencies."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": ["agent_b"]},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert len(result) == 1
        assert "Circular dependency detected" in result[0]

    def test_validate_dependencies_multiple_errors(self) -> None:
        """Test validation detects multiple types of errors."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs = [
            {"name": "agent_a", "depends_on": ["agent_a", "missing_agent"]},
            {"name": "agent_b", "depends_on": ["agent_c"]},
            {"name": "agent_c", "depends_on": ["agent_b"]},
        ]

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert len(result) >= 2
        # Should detect self-dependency and missing agent
        error_messages = " ".join(result)
        assert "cannot depend on itself" in error_messages
        assert "missing agent" in error_messages

    def test_validate_dependencies_no_depends_on_key(self) -> None:
        """Test validation with agents having no depends_on key."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs: list[dict[str, Any]] = [
            {"name": "agent_a"},  # No depends_on key
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert result == []  # Should be valid

    def test_validate_dependencies_empty_list(self) -> None:
        """Test validation with empty agent list."""
        # Given
        analyzer = DependencyAnalyzer()
        agent_configs: list[dict[str, Any]] = []

        # When
        result = analyzer.validate_dependencies(agent_configs)

        # Then
        assert result == []
