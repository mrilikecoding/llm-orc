"""Tests for dependency resolver."""

from typing import Any
from unittest.mock import Mock

from llm_orc.core.execution.dependency_resolver import DependencyResolver


class TestDependencyResolver:
    """Test dependency resolution functionality."""

    def setup_resolver(self) -> tuple[DependencyResolver, Mock]:
        """Set up resolver with mocked role description function."""
        mock_role_resolver = Mock()
        mock_role_resolver.return_value = "Test Role"

        resolver = DependencyResolver(role_resolver=mock_role_resolver)

        return resolver, mock_role_resolver

    def test_enhance_input_with_dependencies_no_dependencies(self) -> None:
        """Test enhancement for agents with no dependencies."""
        resolver, _ = self.setup_resolver()

        agents = [
            {"name": "agent1", "role": "test"},
            {"name": "agent2", "role": "test"},
        ]
        results_dict: dict[str, Any] = {}

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        assert enhanced["agent1"] == "base input"
        assert enhanced["agent2"] == "base input"

    def test_enhance_input_with_dependencies_with_successful_deps(self) -> None:
        """Test enhancement with successful dependency results."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.side_effect = lambda name: f"{name.title()} Role"

        agents = [
            {"name": "agent2", "depends_on": ["agent1"]},
            {"name": "agent3", "depends_on": ["agent1", "agent2"]},
        ]
        results_dict = {
            "agent1": {"response": "First result", "status": "success"},
            "agent2": {"response": "Second result", "status": "success"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Check agent2 enhancement
        assert "You are agent2" in enhanced["agent2"]
        assert "base input" in enhanced["agent2"]
        assert "Agent agent1 (Agent1 Role):" in enhanced["agent2"]
        assert "First result" in enhanced["agent2"]

        # Check agent3 enhancement
        assert "You are agent3" in enhanced["agent3"]
        assert "Agent agent1 (Agent1 Role):" in enhanced["agent3"]
        assert "Agent agent2 (Agent2 Role):" in enhanced["agent3"]
        assert "First result" in enhanced["agent3"]
        assert "Second result" in enhanced["agent3"]

    def test_enhance_input_with_dependencies_with_failed_deps(self) -> None:
        """Test enhancement when some dependencies failed."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.return_value = "Test Role"

        agents = [{"name": "agent2", "depends_on": ["agent1", "failed_agent"]}]
        results_dict = {
            "agent1": {"response": "Success result", "status": "success"},
            "failed_agent": {"error": "Agent failed", "status": "failed"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should only include successful dependency
        assert "Agent agent1 (Test Role):" in enhanced["agent2"]
        assert "Success result" in enhanced["agent2"]
        assert "failed_agent" not in enhanced["agent2"]

    def test_enhance_input_with_dependencies_no_successful_deps(self) -> None:
        """Test enhancement when no dependencies are successful."""
        resolver, _ = self.setup_resolver()

        agents = [{"name": "agent2", "depends_on": ["failed_agent"]}]
        results_dict = {
            "failed_agent": {"error": "Agent failed", "status": "failed"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should fall back to simple prompt
        assert enhanced["agent2"] == "You are agent2. Please respond to: base input"

    def test_enhance_input_with_dependencies_missing_deps(self) -> None:
        """Test enhancement when dependencies are missing from results."""
        resolver, _ = self.setup_resolver()

        agents = [{"name": "agent2", "depends_on": ["missing_agent"]}]
        results_dict: dict[str, Any] = {}

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should fall back to simple prompt
        assert enhanced["agent2"] == "You are agent2. Please respond to: base input"

    def test_has_dependencies_true(self) -> None:
        """Test has_dependencies returns True for agents with dependencies."""
        resolver, _ = self.setup_resolver()

        agent = {"name": "test", "depends_on": ["other"]}
        assert resolver.has_dependencies(agent) is True

    def test_has_dependencies_false(self) -> None:
        """Test has_dependencies returns False for agents without dependencies."""
        resolver, _ = self.setup_resolver()

        agent = {"name": "test"}
        assert resolver.has_dependencies(agent) is False

        agent_empty = {"name": "test", "depends_on": []}
        assert resolver.has_dependencies(agent_empty) is False

    def test_get_dependencies(self) -> None:
        """Test getting dependencies from agent config."""
        resolver, _ = self.setup_resolver()

        agent_with_deps = {"name": "test", "depends_on": ["dep1", "dep2"]}
        assert resolver.get_dependencies(agent_with_deps) == ["dep1", "dep2"]

        agent_without = {"name": "test"}
        assert resolver.get_dependencies(agent_without) == []

    def test_dependencies_satisfied_true(self) -> None:
        """Test dependencies_satisfied returns True when all deps completed."""
        resolver, _ = self.setup_resolver()

        agent = {"name": "test", "depends_on": ["dep1", "dep2"]}
        completed = {"dep1", "dep2", "other"}

        assert resolver.dependencies_satisfied(agent, completed) is True

    def test_dependencies_satisfied_false(self) -> None:
        """Test dependencies_satisfied returns False when deps missing."""
        resolver, _ = self.setup_resolver()

        agent = {"name": "test", "depends_on": ["dep1", "dep2"]}
        completed = {"dep1"}  # Missing dep2

        assert resolver.dependencies_satisfied(agent, completed) is False

    def test_dependencies_satisfied_no_deps(self) -> None:
        """Test dependencies_satisfied returns True when no dependencies."""
        resolver, _ = self.setup_resolver()

        agent = {"name": "test"}
        completed: set[str] = set()

        assert resolver.dependencies_satisfied(agent, completed) is True

    def test_filter_by_dependency_status_with_dependencies(self) -> None:
        """Test filtering agents with satisfied dependencies."""
        resolver, _ = self.setup_resolver()

        agents: list[dict[str, Any]] = [
            {"name": "independent", "role": "test"},
            {"name": "dependent1", "depends_on": ["independent"]},
            {"name": "dependent2", "depends_on": ["missing"]},
            {"name": "dependent3", "depends_on": ["independent", "dependent1"]},
        ]
        completed = {"independent", "dependent1"}

        with_deps = resolver.filter_by_dependency_status(
            agents, completed, with_dependencies=True
        )

        # Should return dependent1 (deps satisfied) and dependent3 (deps satisfied)
        names = {agent["name"] for agent in with_deps}
        assert names == {"dependent1", "dependent3"}

    def test_filter_by_dependency_status_without_dependencies(self) -> None:
        """Test filtering agents without dependencies."""
        resolver, _ = self.setup_resolver()

        agents: list[dict[str, Any]] = [
            {"name": "independent1", "role": "test"},
            {"name": "independent2", "role": "test"},
            {"name": "dependent", "depends_on": ["independent1"]},
        ]
        completed: set[str] = set()

        without_deps = resolver.filter_by_dependency_status(
            agents, completed, with_dependencies=False
        )

        names = {agent["name"] for agent in without_deps}
        assert names == {"independent1", "independent2"}

    def test_validate_dependency_chain_valid(self) -> None:
        """Test validation of valid dependency chain."""
        resolver, _ = self.setup_resolver()

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
            {"name": "agent2", "depends_on": ["agent1"]},
            {"name": "agent3", "depends_on": ["agent1", "agent2"]},
        ]

        errors = resolver.validate_dependency_chain(agents)
        assert errors == []

    def test_validate_dependency_chain_self_dependency(self) -> None:
        """Test validation detects self-dependency."""
        resolver, _ = self.setup_resolver()

        agents = [{"name": "agent1", "depends_on": ["agent1"]}]

        errors = resolver.validate_dependency_chain(agents)
        assert len(errors) == 1
        assert "cannot depend on itself" in errors[0]

    def test_validate_dependency_chain_missing_dependency(self) -> None:
        """Test validation detects missing dependencies."""
        resolver, _ = self.setup_resolver()

        agents = [{"name": "agent1", "depends_on": ["missing_agent"]}]

        errors = resolver.validate_dependency_chain(agents)
        assert len(errors) == 1
        assert "non-existent agent" in errors[0]

    def test_validate_dependency_chain_circular_dependency(self) -> None:
        """Test validation detects circular dependencies."""
        resolver, _ = self.setup_resolver()

        agents = [
            {"name": "agent1", "depends_on": ["agent2"]},
            {"name": "agent2", "depends_on": ["agent1"]},
        ]

        errors = resolver.validate_dependency_chain(agents)
        assert len(errors) == 1
        assert "Circular dependency detected" in errors[0]

    def test_validate_dependency_chain_complex_circular(self) -> None:
        """Test validation detects complex circular dependencies."""
        resolver, _ = self.setup_resolver()

        agents = [
            {"name": "agent1", "depends_on": ["agent3"]},
            {"name": "agent2", "depends_on": ["agent1"]},
            {"name": "agent3", "depends_on": ["agent2"]},
        ]

        errors = resolver.validate_dependency_chain(agents)
        assert len(errors) == 1
        assert "Circular dependency detected" in errors[0]

    def test_validate_dependency_chain_multiple_errors(self) -> None:
        """Test validation returns multiple errors when present."""
        resolver, _ = self.setup_resolver()

        agents = [
            {"name": "agent1", "depends_on": ["agent1", "missing"]},  # Self + missing
            {"name": "agent2", "depends_on": ["missing2"]},  # Missing
        ]

        errors = resolver.validate_dependency_chain(agents)
        assert len(errors) == 3  # Self-dep, missing1, missing2

        error_text = " ".join(errors)
        assert "cannot depend on itself" in error_text
        assert "non-existent agent 'missing'" in error_text
        assert "non-existent agent 'missing2'" in error_text

    def test_enhance_input_role_resolver_called(self) -> None:
        """Test that role resolver is called for dependency attribution."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.return_value = "Custom Role"

        agents = [{"name": "agent2", "depends_on": ["agent1"]}]
        results_dict = {
            "agent1": {"response": "Result", "status": "success"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        mock_role_resolver.assert_called_once_with("agent1")
        assert "Agent agent1 (Custom Role):" in enhanced["agent2"]

    def test_enhance_input_role_resolver_none(self) -> None:
        """Test enhancement when role resolver returns None."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.return_value = None

        agents = [{"name": "agent2", "depends_on": ["agent1"]}]
        results_dict = {
            "agent1": {"response": "Result", "status": "success"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should not include role text when None
        assert "Agent agent1:" in enhanced["agent2"]
        assert "(None)" not in enhanced["agent2"]

    def test_filter_empty_agent_list(self) -> None:
        """Test filtering with empty agent list."""
        resolver, _ = self.setup_resolver()

        result = resolver.filter_by_dependency_status([], set(), with_dependencies=True)
        assert result == []

        result = resolver.filter_by_dependency_status(
            [], set(), with_dependencies=False
        )
        assert result == []

    def test_validate_empty_agent_list(self) -> None:
        """Test validation with empty agent list."""
        resolver, _ = self.setup_resolver()

        errors = resolver.validate_dependency_chain([])
        assert errors == []
