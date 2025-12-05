"""Comprehensive tests for input enhancer."""

from typing import Any, cast

from llm_orc.core.execution.input_enhancer import InputEnhancer


class TestInputEnhancer:
    """Test input enhancer functionality."""

    def test_init_with_configs(self) -> None:
        """Test initialization with agent configurations."""
        # Given
        configs = [{"name": "agent_a", "model_profile": "gpt-4"}]

        # When
        enhancer = InputEnhancer(configs)

        # Then
        assert enhancer._current_agent_configs == configs

    def test_init_without_configs(self) -> None:
        """Test initialization without agent configurations."""
        # When
        enhancer = InputEnhancer()

        # Then
        assert enhancer._current_agent_configs is None

    def test_enhance_input_with_dependencies_no_dependencies(self) -> None:
        """Test enhancement with agents having no dependencies."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b"},  # No depends_on key
        ]
        results_dict: dict[str, Any] = {}

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        assert result == {
            "agent_a": "Analyze this data",
            "agent_b": "Analyze this data",
        }

    def test_enhance_input_with_dependencies_successful_dependencies(self) -> None:
        """Test enhancement with successful dependency results."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        results_dict = {
            "agent_a": {"status": "success", "response": "Data analysis complete"}
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        assert "agent_b" in result
        enhanced_input = result["agent_b"]
        assert "You are agent_b" in enhanced_input
        assert "Original Input:\nAnalyze this data" in enhanced_input
        assert "Agent agent_a (Agent_A):\nData analysis complete" in enhanced_input
        assert "Previous Agent Results" in enhanced_input

    def test_enhance_input_with_dependencies_failed_dependencies(self) -> None:
        """Test enhancement with failed dependency results."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        results_dict = {
            "agent_a": {"status": "failed", "error": "Something went wrong"}
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        assert result == {
            "agent_b": "You are agent_b. Please respond to: Analyze this data"
        }

    def test_enhance_input_with_dependencies_missing_dependencies(self) -> None:
        """Test enhancement when dependency results are missing."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        results_dict: dict[str, Any] = {}

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        assert result == {
            "agent_b": "You are agent_b. Please respond to: Analyze this data"
        }

    def test_enhance_input_with_dependencies_multiple_dependencies(self) -> None:
        """Test enhancement with multiple dependencies."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]
        results_dict = {
            "agent_a": {"status": "success", "response": "First analysis"},
            "agent_b": {"status": "success", "response": "Second analysis"},
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        assert "agent_c" in result
        enhanced_input = result["agent_c"]
        assert "Agent agent_a (Agent_A):\nFirst analysis" in enhanced_input
        assert "Agent agent_b (Agent_B):\nSecond analysis" in enhanced_input

    def test_enhance_input_with_dependencies_with_roles(self) -> None:
        """Test enhancement with agent roles included."""
        # Given
        configs = [
            {"name": "agent_a", "model_profile": "data-analyst"},
        ]
        enhancer = InputEnhancer(configs)
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        results_dict = {
            "agent_a": {"status": "success", "response": "Analysis complete"}
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        enhanced_input = result["agent_b"]
        assert "Agent agent_a (Data Analyst):\nAnalysis complete" in enhanced_input

    def test_get_agent_role_description_with_model_profile(self) -> None:
        """Test getting role description from model profile."""
        # Given
        configs = [
            {"name": "agent_a", "model_profile": "data-analyst"},
        ]
        enhancer = InputEnhancer(configs)

        # When
        result = enhancer.get_agent_role_description("agent_a")

        # Then
        assert result == "Data Analyst"

    def test_get_agent_role_description_without_model_profile(self) -> None:
        """Test getting role description without model profile."""
        # Given
        configs = [
            {"name": "agent_a"},
        ]
        enhancer = InputEnhancer(configs)

        # When
        result = enhancer.get_agent_role_description("agent_a")

        # Then
        assert result == "Agent_A"

    def test_get_agent_role_description_agent_not_found(self) -> None:
        """Test getting role description for agent not in configs."""
        # Given
        configs = [
            {"name": "agent_a", "model_profile": "analyst"},
        ]
        enhancer = InputEnhancer(configs)

        # When
        result = enhancer.get_agent_role_description("agent_b")

        # Then
        assert result == "Agent_B"

    def test_get_agent_role_description_no_configs(self) -> None:
        """Test getting role description when no configs are set."""
        # Given
        enhancer = InputEnhancer()

        # When
        result = enhancer.get_agent_role_description("agent-name")

        # Then
        assert result == "Agent Name"

    def test_get_agent_input_string_input(self) -> None:
        """Test getting agent input from string input."""
        # Given
        enhancer = InputEnhancer()
        input_data = "Uniform input for all agents"

        # When
        result = enhancer.get_agent_input(input_data, "agent_a")

        # Then
        assert result == "Uniform input for all agents"

    def test_get_agent_input_dict_input_found(self) -> None:
        """Test getting agent input from dict input when agent is found."""
        # Given
        enhancer = InputEnhancer()
        input_data = {
            "agent_a": "Specific input for agent A",
            "agent_b": "Specific input for agent B",
        }

        # When
        result = enhancer.get_agent_input(input_data, "agent_a")

        # Then
        assert result == "Specific input for agent A"

    def test_get_agent_input_dict_input_not_found(self) -> None:
        """Test getting agent input from dict input when agent is not found."""
        # Given
        enhancer = InputEnhancer()
        input_data = {
            "agent_a": "Specific input for agent A",
        }

        # When
        result = enhancer.get_agent_input(input_data, "agent_b")

        # Then
        assert result == ""

    def test_create_enhanced_input_for_agent_no_dependencies(self) -> None:
        """Test creating enhanced input for agent with no dependencies."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Original input"

        # When
        result = enhancer.create_enhanced_input_for_agent("agent_a", base_input, [], {})

        # Then
        assert result == "Original input"

    def test_create_enhanced_input_for_agent_with_dependencies(self) -> None:
        """Test creating enhanced input for agent with dependencies."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Original input"
        dependencies = ["agent_a"]
        results_dict = {
            "agent_a": {"status": "success", "response": "Dependency result"}
        }

        # When
        result = enhancer.create_enhanced_input_for_agent(
            "agent_b", base_input, dependencies, results_dict
        )

        # Then
        assert "You are agent_b" in result
        assert "Original Input:\nOriginal input" in result
        assert "Agent agent_a (Agent_A):\nDependency result" in result

    def test_create_enhanced_input_for_agent_failed_dependencies(self) -> None:
        """Test creating enhanced input when dependencies failed."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Original input"
        dependencies = ["agent_a"]
        results_dict = {"agent_a": {"status": "failed", "error": "Failed"}}

        # When
        result = enhancer.create_enhanced_input_for_agent(
            "agent_b", base_input, dependencies, results_dict
        )

        # Then
        assert result == "You are agent_b. Please respond to: Original input"

    def test_create_enhanced_input_for_agent_with_role(self) -> None:
        """Test creating enhanced input with agent role."""
        # Given
        configs = [
            {"name": "agent_a", "model_profile": "data-scientist"},
        ]
        enhancer = InputEnhancer(configs)
        base_input = "Original input"
        dependencies = ["agent_a"]
        results_dict = {
            "agent_a": {"status": "success", "response": "Scientific analysis"}
        }

        # When
        result = enhancer.create_enhanced_input_for_agent(
            "agent_b", base_input, dependencies, results_dict
        )

        # Then
        assert "Agent agent_a (Data Scientist):\nScientific analysis" in result

    def test_update_agent_configs(self) -> None:
        """Test updating agent configurations."""
        # Given
        enhancer = InputEnhancer()
        new_configs = [
            {"name": "agent_a", "model_profile": "analyst"},
            {"name": "agent_b", "model_profile": "writer"},
        ]

        # When
        enhancer.update_agent_configs(new_configs)

        # Then
        assert enhancer._current_agent_configs == new_configs

    def test_get_agent_role_description_complex_names(self) -> None:
        """Test role description with complex agent names."""
        # Given
        enhancer = InputEnhancer()

        # When
        result = enhancer.get_agent_role_description("data-analysis-agent")

        # Then
        assert result == "Data Analysis Agent"

    def test_get_agent_role_description_numeric_profile(self) -> None:
        """Test role description with numeric model profile."""
        # Given
        configs = [
            {"name": "agent_a", "model_profile": 123},
        ]
        enhancer = InputEnhancer(configs)

        # When
        result = enhancer.get_agent_role_description("agent_a")

        # Then
        assert result == "123"

    def test_enhance_input_mixed_dependency_statuses(self) -> None:
        """Test enhancement with mixed dependency statuses."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze this data"
        dependent_agents = [
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]
        results_dict = {
            "agent_a": {"status": "success", "response": "Success result"},
            "agent_b": {"status": "failed", "error": "Failed"},
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        enhanced_input = result["agent_c"]
        # Should only include successful dependency
        assert "Agent agent_a (Agent_A):\nSuccess result" in enhanced_input
        assert "agent_b" not in enhanced_input

    def test_enhance_input_empty_agent_list(self) -> None:
        """Test enhancement with empty agent list."""
        # Given
        enhancer = InputEnhancer()
        base_input = "Test input"
        dependent_agents: list[dict[str, Any]] = []
        results_dict: dict[str, Any] = {}

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, dependent_agents, results_dict
        )

        # Then
        assert result == {}

    def test_enhance_input_script_agent_gets_json_dependencies(self) -> None:
        """Test script agents receive JSON-formatted input with dependencies."""
        import json

        # Given
        enhancer = InputEnhancer()
        base_input = "Process this data"
        dependent_agents = [
            {
                "name": "aggregator",
                "script": "aggregator.py",
                "depends_on": ["agent_a"],
            },
        ]
        results_dict = {
            "agent_a": {"status": "success", "response": "Analysis result from agent A"}
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then - script agent should get JSON with dependencies dict
        assert "aggregator" in result
        enhanced_input = result["aggregator"]

        # Parse as JSON
        parsed = json.loads(enhanced_input)
        assert parsed["agent_name"] == "aggregator"
        assert parsed["input_data"] == "Process this data"
        assert "dependencies" in parsed
        assert "agent_a" in parsed["dependencies"]
        dep_response = parsed["dependencies"]["agent_a"]["response"]
        assert dep_response == "Analysis result from agent A"

    def test_enhance_input_script_agent_with_multiple_dependencies(self) -> None:
        """Test script agent with multiple dependencies gets all in JSON."""
        import json

        # Given
        enhancer = InputEnhancer()
        base_input = "Combine results"
        dependent_agents = [
            {
                "name": "combiner",
                "script": "combine.py",
                "depends_on": ["extractor_1", "extractor_2"],
            },
        ]
        results_dict = {
            "extractor_1": {"status": "success", "response": "Result 1"},
            "extractor_2": {"status": "success", "response": "Result 2"},
        }

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then
        parsed = json.loads(result["combiner"])
        assert len(parsed["dependencies"]) == 2
        assert parsed["dependencies"]["extractor_1"]["response"] == "Result 1"
        assert parsed["dependencies"]["extractor_2"]["response"] == "Result 2"

    def test_enhance_input_mixed_script_and_llm_agents(self) -> None:
        """Test mixed ensemble with both script and LLM agents."""
        import json

        # Given
        enhancer = InputEnhancer()
        base_input = "Analyze data"
        dependent_agents = [
            {
                "name": "script_agent",
                "script": "process.py",
                "depends_on": ["source"],
            },
            {
                "name": "llm_agent",
                "model_profile": "ollama-llama3",
                "depends_on": ["source"],
            },
        ]
        results_dict = {"source": {"status": "success", "response": "Source data here"}}

        # When
        result = enhancer.enhance_input_with_dependencies(
            base_input, cast(list[dict[str, Any]], dependent_agents), results_dict
        )

        # Then - script agent gets JSON
        script_input = result["script_agent"]
        parsed = json.loads(script_input)
        assert parsed["dependencies"]["source"]["response"] == "Source data here"

        # LLM agent gets text format
        llm_input = result["llm_agent"]
        assert "You are llm_agent" in llm_input
        assert "Previous Agent Results" in llm_input
