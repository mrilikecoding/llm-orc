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

    def test_enhance_input_with_mixed_dependency_statuses(self) -> None:
        """Test enhancement with mix of successful, failed, and missing deps."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.side_effect = lambda name: f"{name.title()} Role"

        agents = [{"name": "agent4", "depends_on": ["success", "failed", "missing"]}]
        results_dict = {
            "success": {"response": "Good result", "status": "success"},
            "failed": {"error": "Failed", "status": "failed"},
            # "missing" not in results_dict
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should only include successful dependency
        assert "Agent success (Success Role):" in enhanced["agent4"]
        assert "Good result" in enhanced["agent4"]
        assert "failed" not in enhanced["agent4"]
        assert "missing" not in enhanced["agent4"]

    def test_enhance_input_empty_dependencies_list(self) -> None:
        """Test enhancement with explicitly empty dependencies list."""
        resolver, _ = self.setup_resolver()

        agents = [{"name": "agent1", "depends_on": []}]
        results_dict: dict[str, Any] = {}

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        assert enhanced["agent1"] == "base input"

    def test_enhance_input_multiple_agents_various_dependencies(self) -> None:
        """Test enhancement with multiple agents having different patterns."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.side_effect = lambda name: f"{name.title()} Role"

        agents: list[dict[str, Any]] = [
            {"name": "no_deps", "role": "test"},
            {"name": "single_dep", "depends_on": ["success1"]},
            {"name": "multi_deps", "depends_on": ["success1", "success2"]},
            {"name": "partial_deps", "depends_on": ["success1", "failed1"]},
        ]
        results_dict = {
            "success1": {"response": "First success", "status": "success"},
            "success2": {"response": "Second success", "status": "success"},
            "failed1": {"error": "Failed", "status": "failed"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # No dependencies agent
        assert enhanced["no_deps"] == "base input"

        # Single dependency agent
        assert "Agent success1 (Success1 Role):" in enhanced["single_dep"]
        assert "First success" in enhanced["single_dep"]

        # Multiple dependencies agent
        assert "Agent success1 (Success1 Role):" in enhanced["multi_deps"]
        assert "Agent success2 (Success2 Role):" in enhanced["multi_deps"]
        assert "First success" in enhanced["multi_deps"]
        assert "Second success" in enhanced["multi_deps"]

        # Partial successful dependencies agent
        assert "Agent success1 (Success1 Role):" in enhanced["partial_deps"]
        assert "First success" in enhanced["partial_deps"]
        assert "failed1" not in enhanced["partial_deps"]

    def test_extract_successful_dependency_results_helper(self) -> None:
        """Test the extracted helper method for getting successful results."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.side_effect = lambda name: f"{name.title()} Role"

        dependencies = ["success", "failed", "missing"]
        results_dict = {
            "success": {"response": "Good result", "status": "success"},
            "failed": {"error": "Failed", "status": "failed"},
            # "missing" not in results_dict
        }

        # This method should be extracted during refactoring
        dependency_results = resolver._extract_successful_dependency_results(
            dependencies, results_dict
        )

        assert len(dependency_results) == 1
        assert "Agent success (Success Role):\nGood result" in dependency_results

    def test_build_enhanced_input_with_dependencies_helper(self) -> None:
        """Test the extracted helper method for building enhanced input."""
        resolver, _ = self.setup_resolver()

        agent_name = "test_agent"
        base_input = "test input"
        dependency_results = [
            "Agent dep1 (Dep1 Role):\nResult 1",
            "Agent dep2 (Dep2 Role):\nResult 2",
        ]

        # This method should be extracted during refactoring
        enhanced_input = resolver._build_enhanced_input_with_dependencies(
            agent_name, base_input, dependency_results
        )

        assert "You are test_agent" in enhanced_input
        assert "test input" in enhanced_input
        assert "Agent dep1 (Dep1 Role):\nResult 1" in enhanced_input
        assert "Agent dep2 (Dep2 Role):\nResult 2" in enhanced_input

    def test_build_enhanced_input_no_dependencies_helper(self) -> None:
        """Test the extracted helper method for building input without dependencies."""
        resolver, _ = self.setup_resolver()

        agent_name = "test_agent"
        base_input = "test input"

        # This method should be extracted during refactoring
        enhanced_input = resolver._build_enhanced_input_no_dependencies(
            agent_name, base_input
        )

        assert enhanced_input == "You are test_agent. Please respond to: test input"

    def test_enhance_input_script_agent_gets_json_with_dependencies(self) -> None:
        """Test script agents receive JSON-formatted input with dependencies."""
        import json

        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.return_value = "Test Role"

        agents = [
            {
                "name": "aggregator",
                "script": "aggregator.py",
                "depends_on": ["extractor"],
            },
        ]
        results_dict = {
            "extractor": {"status": "success", "response": "Extracted data here"},
        }

        enhanced = resolver.enhance_input_with_dependencies(
            "Process this data", agents, results_dict
        )

        # Script agent should get JSON with dependencies dict
        assert "aggregator" in enhanced
        parsed = json.loads(enhanced["aggregator"])
        assert parsed["agent_name"] == "aggregator"
        assert parsed["input_data"] == "Process this data"
        assert "dependencies" in parsed
        assert "extractor" in parsed["dependencies"]
        assert parsed["dependencies"]["extractor"]["response"] == "Extracted data here"

    def test_enhance_input_script_agent_without_dependencies(self) -> None:
        """Test script agent without dependencies still gets JSON format."""
        import json

        resolver, _ = self.setup_resolver()

        agents = [
            {"name": "processor", "script": "process.py"},
        ]
        results_dict: dict[str, Any] = {}

        enhanced = resolver.enhance_input_with_dependencies(
            "Input data", agents, results_dict
        )

        # Script agent with no deps should get JSON with empty dependencies
        assert "processor" in enhanced
        parsed = json.loads(enhanced["processor"])
        assert parsed["agent_name"] == "processor"
        assert parsed["input_data"] == "Input data"
        assert parsed["dependencies"] == {}

    def test_enhance_input_mixed_script_and_llm_agents(self) -> None:
        """Test mixed ensemble with both script and LLM agents."""
        import json

        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.return_value = "Source Role"

        agents = [
            {"name": "script_agent", "script": "process.py", "depends_on": ["source"]},
            {
                "name": "llm_agent",
                "model_profile": "ollama-llama3",
                "depends_on": ["source"],
            },
        ]
        results_dict = {"source": {"status": "success", "response": "Source output"}}

        enhanced = resolver.enhance_input_with_dependencies(
            "Analyze data", agents, results_dict
        )

        # Script agent gets JSON
        script_input = enhanced["script_agent"]
        parsed = json.loads(script_input)
        assert parsed["dependencies"]["source"]["response"] == "Source output"

        # LLM agent gets text format
        llm_input = enhanced["llm_agent"]
        assert "You are llm_agent" in llm_input
        assert "Previous Agent Results" in llm_input


class TestValidateDependencyChainHelperMethods:
    """Test helper methods extracted from validate_dependency_chain for complexity."""

    def setup_resolver(self) -> DependencyResolver:
        """Set up resolver for testing."""
        mock_role_resolver = Mock()
        mock_role_resolver.return_value = "Test Role"
        return DependencyResolver(role_resolver=mock_role_resolver)

    def test_validate_basic_dependencies_no_errors(self) -> None:
        """Test basic dependency validation with no errors."""
        from llm_orc.core.execution.dependency_resolver import (
            _validate_basic_dependencies,
        )

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
            {"name": "agent2", "depends_on": ["agent1"]},
        ]

        errors = _validate_basic_dependencies(agents)
        assert errors == []

    def test_validate_basic_dependencies_self_dependency(self) -> None:
        """Test basic validation detects self-dependency."""
        from llm_orc.core.execution.dependency_resolver import (
            _validate_basic_dependencies,
        )

        agents: list[dict[str, Any]] = [{"name": "agent1", "depends_on": ["agent1"]}]

        errors = _validate_basic_dependencies(agents)
        assert len(errors) == 1
        assert "cannot depend on itself" in errors[0]

    def test_validate_basic_dependencies_missing_dependency(self) -> None:
        """Test basic validation detects missing dependencies."""
        from llm_orc.core.execution.dependency_resolver import (
            _validate_basic_dependencies,
        )

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "depends_on": ["missing_agent"]}
        ]

        errors = _validate_basic_dependencies(agents)
        assert len(errors) == 1
        assert "non-existent agent" in errors[0]

    def test_validate_basic_dependencies_multiple_errors(self) -> None:
        """Test basic validation returns multiple errors."""
        from llm_orc.core.execution.dependency_resolver import (
            _validate_basic_dependencies,
        )

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "depends_on": ["agent1", "missing"]},
            {"name": "agent2", "depends_on": ["missing2"]},
        ]

        errors = _validate_basic_dependencies(agents)
        assert len(errors) == 3  # Self-dep, missing1, missing2

    def test_detect_circular_dependencies_no_cycles(self) -> None:
        """Test circular dependency detection with no cycles."""
        from llm_orc.core.execution.dependency_resolver import (
            _detect_circular_dependencies,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
            {"name": "agent2", "depends_on": ["agent1"]},
            {"name": "agent3", "depends_on": ["agent1", "agent2"]},
        ]

        errors = _detect_circular_dependencies(agents, resolver)
        assert errors == []

    def test_detect_circular_dependencies_simple_cycle(self) -> None:
        """Test detection of simple circular dependency."""
        from llm_orc.core.execution.dependency_resolver import (
            _detect_circular_dependencies,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "depends_on": ["agent2"]},
            {"name": "agent2", "depends_on": ["agent1"]},
        ]

        errors = _detect_circular_dependencies(agents, resolver)
        assert len(errors) == 1
        assert "Circular dependency detected" in errors[0]

    def test_detect_circular_dependencies_complex_cycle(self) -> None:
        """Test detection of complex circular dependency."""
        from llm_orc.core.execution.dependency_resolver import (
            _detect_circular_dependencies,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "depends_on": ["agent3"]},
            {"name": "agent2", "depends_on": ["agent1"]},
            {"name": "agent3", "depends_on": ["agent2"]},
        ]

        errors = _detect_circular_dependencies(agents, resolver)
        assert len(errors) == 1
        assert "Circular dependency detected" in errors[0]

    def test_find_agent_by_name_found(self) -> None:
        """Test finding agent by name when agent exists."""
        from llm_orc.core.execution.dependency_resolver import _find_agent_by_name

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
            {"name": "agent2", "depends_on": ["agent1"]},
        ]

        result = _find_agent_by_name(agents, "agent2")
        assert result is not None
        assert result["name"] == "agent2"
        assert result["depends_on"] == ["agent1"]

    def test_find_agent_by_name_not_found(self) -> None:
        """Test finding agent by name when agent doesn't exist."""
        from llm_orc.core.execution.dependency_resolver import _find_agent_by_name

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
        ]

        result = _find_agent_by_name(agents, "missing_agent")
        assert result is None

    def test_find_agent_by_name_empty_list(self) -> None:
        """Test finding agent by name in empty agent list."""
        from llm_orc.core.execution.dependency_resolver import _find_agent_by_name

        result = _find_agent_by_name([], "any_agent")
        assert result is None

    def test_check_cycle_from_node_no_cycle(self) -> None:
        """Test cycle checking from node with no cycles."""
        from llm_orc.core.execution.dependency_resolver import (
            _check_cycle_from_node,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
            {"name": "agent2", "depends_on": ["agent1"]},
        ]
        visited: set[str] = set()
        rec_stack: set[str] = set()

        result = _check_cycle_from_node("agent2", agents, resolver, visited, rec_stack)
        assert result is False
        assert "agent1" in visited
        assert "agent2" in visited
        assert len(rec_stack) == 0  # Should be empty after completion

    def test_check_cycle_from_node_direct_cycle(self) -> None:
        """Test cycle checking detects direct cycle."""
        from llm_orc.core.execution.dependency_resolver import (
            _check_cycle_from_node,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "depends_on": ["agent2"]},
            {"name": "agent2", "depends_on": ["agent1"]},
        ]
        visited: set[str] = set()
        rec_stack: set[str] = set()

        result = _check_cycle_from_node("agent1", agents, resolver, visited, rec_stack)
        assert result is True

    def test_check_cycle_from_node_already_visited(self) -> None:
        """Test cycle checking with already visited node."""
        from llm_orc.core.execution.dependency_resolver import (
            _check_cycle_from_node,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
        ]
        visited: set[str] = {"agent1"}  # Already visited
        rec_stack: set[str] = set()

        result = _check_cycle_from_node("agent1", agents, resolver, visited, rec_stack)
        assert result is False

    def test_check_cycle_from_node_in_recursion_stack(self) -> None:
        """Test cycle checking detects node in recursion stack."""
        from llm_orc.core.execution.dependency_resolver import (
            _check_cycle_from_node,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "test"},
        ]
        visited: set[str] = set()
        rec_stack: set[str] = {"agent1"}  # Already in recursion stack

        result = _check_cycle_from_node("agent1", agents, resolver, visited, rec_stack)
        assert result is True

    def test_check_cycle_from_node_missing_agent(self) -> None:
        """Test cycle checking with missing agent config."""
        from llm_orc.core.execution.dependency_resolver import (
            _check_cycle_from_node,
        )

        resolver = self.setup_resolver()
        agents: list[dict[str, Any]] = []  # Empty agents list
        visited: set[str] = set()
        rec_stack: set[str] = set()

        result = _check_cycle_from_node(
            "missing_agent", agents, resolver, visited, rec_stack
        )
        assert result is False
        assert "missing_agent" in visited
        assert len(rec_stack) == 0
