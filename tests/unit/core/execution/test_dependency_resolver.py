"""Tests for dependency resolver."""

from typing import Any
from unittest.mock import Mock

from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig, ScriptAgentConfig


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

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test-profile"),
            LlmAgentConfig(name="agent2", model_profile="test-profile"),
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

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent2", model_profile="test-profile", depends_on=["agent1"]
            ),
            LlmAgentConfig(
                name="agent3",
                model_profile="test-profile",
                depends_on=["agent1", "agent2"],
            ),
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

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent2",
                model_profile="test-profile",
                depends_on=["agent1", "failed_agent"],
            )
        ]
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

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent2",
                model_profile="test-profile",
                depends_on=["failed_agent"],
            )
        ]
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

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent2",
                model_profile="test-profile",
                depends_on=["missing_agent"],
            )
        ]
        results_dict: dict[str, Any] = {}

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        # Should fall back to simple prompt
        assert enhanced["agent2"] == "You are agent2. Please respond to: base input"

    def test_has_dependencies_true(self) -> None:
        """Test has_dependencies returns True for agents with dependencies."""
        resolver, _ = self.setup_resolver()

        agent = LlmAgentConfig(
            name="test", model_profile="test-profile", depends_on=["other"]
        )
        assert resolver.has_dependencies(agent) is True

    def test_has_dependencies_false(self) -> None:
        """Test has_dependencies returns False for agents without dependencies."""
        resolver, _ = self.setup_resolver()

        agent = LlmAgentConfig(name="test", model_profile="test-profile")
        assert resolver.has_dependencies(agent) is False

        agent_empty = LlmAgentConfig(
            name="test", model_profile="test-profile", depends_on=[]
        )
        assert resolver.has_dependencies(agent_empty) is False

    def test_get_dependencies(self) -> None:
        """Test getting dependencies from agent config."""
        resolver, _ = self.setup_resolver()

        agent_with_deps = LlmAgentConfig(
            name="test", model_profile="test-profile", depends_on=["dep1", "dep2"]
        )
        assert resolver.get_dependencies(agent_with_deps) == ["dep1", "dep2"]

        agent_without = LlmAgentConfig(name="test", model_profile="test-profile")
        assert resolver.get_dependencies(agent_without) == []

    def test_dependencies_satisfied_true(self) -> None:
        """Test dependencies_satisfied returns True when all deps completed."""
        resolver, _ = self.setup_resolver()

        agent = LlmAgentConfig(
            name="test", model_profile="test-profile", depends_on=["dep1", "dep2"]
        )
        completed = {"dep1", "dep2", "other"}

        assert resolver.dependencies_satisfied(agent, completed) is True

    def test_dependencies_satisfied_false(self) -> None:
        """Test dependencies_satisfied returns False when deps missing."""
        resolver, _ = self.setup_resolver()

        agent = LlmAgentConfig(
            name="test", model_profile="test-profile", depends_on=["dep1", "dep2"]
        )
        completed = {"dep1"}  # Missing dep2

        assert resolver.dependencies_satisfied(agent, completed) is False

    def test_dependencies_satisfied_no_deps(self) -> None:
        """Test dependencies_satisfied returns True when no dependencies."""
        resolver, _ = self.setup_resolver()

        agent = LlmAgentConfig(name="test", model_profile="test-profile")
        completed: set[str] = set()

        assert resolver.dependencies_satisfied(agent, completed) is True

    def test_filter_by_dependency_status_with_dependencies(self) -> None:
        """Test filtering agents with satisfied dependencies."""
        resolver, _ = self.setup_resolver()

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="independent", model_profile="test-profile"),
            LlmAgentConfig(
                name="dependent1",
                model_profile="test-profile",
                depends_on=["independent"],
            ),
            LlmAgentConfig(
                name="dependent2",
                model_profile="test-profile",
                depends_on=["missing"],
            ),
            LlmAgentConfig(
                name="dependent3",
                model_profile="test-profile",
                depends_on=["independent", "dependent1"],
            ),
        ]
        completed = {"independent", "dependent1"}

        with_deps = resolver.filter_by_dependency_status(
            agents, completed, with_dependencies=True
        )

        # Should return dependent1 (deps satisfied) and dependent3 (deps satisfied)
        names = {agent.name for agent in with_deps}
        assert names == {"dependent1", "dependent3"}

    def test_filter_by_dependency_status_without_dependencies(self) -> None:
        """Test filtering agents without dependencies."""
        resolver, _ = self.setup_resolver()

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="independent1", model_profile="test-profile"),
            LlmAgentConfig(name="independent2", model_profile="test-profile"),
            LlmAgentConfig(
                name="dependent",
                model_profile="test-profile",
                depends_on=["independent1"],
            ),
        ]
        completed: set[str] = set()

        without_deps = resolver.filter_by_dependency_status(
            agents, completed, with_dependencies=False
        )

        names = {agent.name for agent in without_deps}
        assert names == {"independent1", "independent2"}

    def test_enhance_input_role_resolver_called(self) -> None:
        """Test that role resolver is called for dependency attribution."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.return_value = "Custom Role"

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent2", model_profile="test-profile", depends_on=["agent1"]
            )
        ]
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

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent2", model_profile="test-profile", depends_on=["agent1"]
            )
        ]
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

    def test_enhance_input_with_mixed_dependency_statuses(self) -> None:
        """Test enhancement with mix of successful, failed, and missing deps."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.side_effect = lambda name: f"{name.title()} Role"

        agents: list[AgentConfig] = [
            LlmAgentConfig(
                name="agent4",
                model_profile="test-profile",
                depends_on=["success", "failed", "missing"],
            )
        ]
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

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test-profile", depends_on=[])
        ]
        results_dict: dict[str, Any] = {}

        enhanced = resolver.enhance_input_with_dependencies(
            "base input", agents, results_dict
        )

        assert enhanced["agent1"] == "base input"

    def test_enhance_input_multiple_agents_various_dependencies(self) -> None:
        """Test enhancement with multiple agents having different patterns."""
        resolver, mock_role_resolver = self.setup_resolver()
        mock_role_resolver.side_effect = lambda name: f"{name.title()} Role"

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="no_deps", model_profile="test-profile"),
            LlmAgentConfig(
                name="single_dep",
                model_profile="test-profile",
                depends_on=["success1"],
            ),
            LlmAgentConfig(
                name="multi_deps",
                model_profile="test-profile",
                depends_on=["success1", "success2"],
            ),
            LlmAgentConfig(
                name="partial_deps",
                model_profile="test-profile",
                depends_on=["success1", "failed1"],
            ),
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

        dependencies: list[str | dict[str, Any]] = [
            "success",
            "failed",
            "missing",
        ]
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

        agents: list[AgentConfig] = [
            ScriptAgentConfig(
                name="aggregator",
                script="aggregator.py",
                depends_on=["extractor"],
            ),
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

        agents: list[AgentConfig] = [
            ScriptAgentConfig(name="processor", script="process.py"),
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

        agents: list[AgentConfig] = [
            ScriptAgentConfig(
                name="script_agent", script="process.py", depends_on=["source"]
            ),
            LlmAgentConfig(
                name="llm_agent",
                model_profile="ollama-llama3",
                depends_on=["source"],
            ),
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


class TestFanOutInputPreparation:
    """Test fan-out instance input preparation (issue #73)."""

    def setup_resolver(self) -> DependencyResolver:
        """Set up resolver for testing."""
        mock_role_resolver = Mock()
        mock_role_resolver.return_value = "Test Role"
        return DependencyResolver(role_resolver=mock_role_resolver)

    def test_prepare_fan_out_instance_input_script_agent(self) -> None:
        """Test preparing input for a fan-out script agent instance."""
        import json

        resolver = self.setup_resolver()

        instance_config = ScriptAgentConfig(
            name="extractor[0]",
            script="extract.py",
            fan_out_chunk="Scene 1 content",
            fan_out_index=0,
            fan_out_total=3,
            fan_out_original="extractor",
        )

        result = resolver.prepare_fan_out_instance_input(
            instance_config=instance_config,
            base_input="Analyze this play",
        )

        # Should be JSON for script agent
        parsed = json.loads(result)
        assert parsed["agent_name"] == "extractor[0]"
        assert parsed["input"] == "Scene 1 content"
        assert parsed["chunk_index"] == 0
        assert parsed["total_chunks"] == 3
        assert parsed["base_input"] == "Analyze this play"

    def test_prepare_fan_out_instance_input_llm_agent(self) -> None:
        """Test preparing input for a fan-out LLM agent instance."""
        resolver = self.setup_resolver()

        instance_config = LlmAgentConfig(
            name="extractor[1]",
            model_profile="ollama-llama3",
            fan_out_chunk="Scene 2 dialogue",
            fan_out_index=1,
            fan_out_total=3,
            fan_out_original="extractor",
        )

        result = resolver.prepare_fan_out_instance_input(
            instance_config=instance_config,
            base_input="Extract themes",
        )

        # Should be text format for LLM agent
        assert "Scene 2 dialogue" in result
        assert "chunk 2 of 3" in result
        assert "Extract themes" in result

    def test_prepare_fan_out_instance_input_with_dict_chunk(self) -> None:
        """Test preparing input when chunk is a dict."""
        import json

        resolver = self.setup_resolver()

        chunk_data = {"scene": "Act 1", "text": "Dialogue here"}
        instance_config = ScriptAgentConfig(
            name="extractor[0]",
            script="extract.py",
            fan_out_chunk=chunk_data,
            fan_out_index=0,
            fan_out_total=2,
            fan_out_original="extractor",
        )

        result = resolver.prepare_fan_out_instance_input(
            instance_config=instance_config,
            base_input="Process scenes",
        )

        parsed = json.loads(result)
        assert parsed["input"] == chunk_data
        assert parsed["chunk_index"] == 0

    def test_is_fan_out_instance_config(self) -> None:
        """Test detecting fan-out instance configurations."""
        resolver = self.setup_resolver()

        instance_config = LlmAgentConfig(
            name="extractor[0]",
            model_profile="test-profile",
            fan_out_chunk="data",
            fan_out_index=0,
            fan_out_total=1,
            fan_out_original="extractor",
        )

        regular_config = LlmAgentConfig(
            name="extractor",
            model_profile="test",
        )

        assert resolver.is_fan_out_instance_config(instance_config) is True
        assert resolver.is_fan_out_instance_config(regular_config) is False


class TestGetAgentInput:
    """Test get_agent_input static method."""

    def test_uniform_string_input(self) -> None:
        """String input is returned unchanged for any agent."""
        assert DependencyResolver.get_agent_input("hello", "agent1") == "hello"

    def test_dict_input_returns_matching_agent(self) -> None:
        """Dict input returns the value for the named agent."""
        data = {"agent1": "input for 1", "agent2": "input for 2"}
        assert DependencyResolver.get_agent_input(data, "agent1") == "input for 1"

    def test_dict_input_returns_empty_for_missing_agent(self) -> None:
        """Dict input returns empty string for an absent agent name."""
        data = {"agent1": "input for 1"}
        assert DependencyResolver.get_agent_input(data, "missing") == ""
