"""End-to-end test for story generator → user input agent flow (BDD scenario).

This test validates the complete AgentRequest processing flow as described in
the BDD feature: story generator outputs AgentRequest, ensemble processes it,
and coordinates user input agent execution with dynamic parameters.
"""

import json
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.schemas.agent_config import AgentConfig, ScriptAgentConfig


class TestStoryGeneratorFlow:
    """Test the complete story generator to user input agent flow."""

    @pytest.mark.asyncio
    async def test_story_generator_to_user_input_agent_coordination(self) -> None:
        """Test the BDD scenario: story generator → user input agent flow.

        This test validates that:
        1. Story generator outputs AgentRequest objects
        2. EnsembleExecutor processes the AgentRequest objects
        3. AgentRequest metadata is properly stored and available
        4. The flow enables dynamic parameter generation for inter-agent communication
        """
        config = EnsembleConfig(
            name="story_user_input_flow",
            description="Test story generator to user input coordination",
            agents=[
                ScriptAgentConfig(
                    name="cyberpunk_story_generator",
                    script="story_generator_with_requests.py",
                )
            ],
        )

        executor = EnsembleExecutor()

        # Mock artifact manager to prevent real artifact creation
        mock_artifact_manager = Mock()
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            # Create ScriptAgentInput for the story generator
            story_input = {
                "agent_name": "cyberpunk_story_generator",
                "input_data": "Generate a cyberpunk story beginning",
                "context": {"theme": "cyberpunk", "setting": "Neo-Tokyo"},
                "dependencies": {},
            }

            result = await executor.execute(config, json.dumps(story_input))

        # Verify the execution completed successfully
        assert result["status"] in ["completed", "completed_with_errors"]
        assert "cyberpunk_story_generator" in result["results"]

        # Verify story generator executed successfully
        story_result = result["results"]["cyberpunk_story_generator"]
        assert story_result["status"] == "success"

        # Parse story generator response to check AgentRequest output
        story_response = story_result["response"]
        if isinstance(story_response, str):
            story_data = json.loads(story_response)
        else:
            story_data = story_response

        # Verify the story generator produced valid ScriptAgentOutput with AgentRequest
        assert story_data["success"] is True
        assert "story_fragment" in story_data["data"]
        assert "agent_requests" in story_data
        assert len(story_data["agent_requests"]) == 1

        # Verify the AgentRequest is properly structured
        agent_request = story_data["agent_requests"][0]
        assert agent_request["target_agent_type"] == "user_input"
        assert "prompt" in agent_request["parameters"]
        assert "cyberpunk" in agent_request["parameters"]["prompt"]
        assert agent_request["priority"] == 1

        # CRITICAL: Verify EnsembleExecutor processed the AgentRequest
        # This confirms AgentRequestProcessor and EnsembleExecutor integration
        assert "processed_agent_requests" in result["metadata"]
        processed_requests = result["metadata"]["processed_agent_requests"]
        assert len(processed_requests) == 1
        assert processed_requests[0]["target_agent_type"] == "user_input"

        # Verify that the AgentRequest was stored in the agent result metadata
        assert "agent_requests" in story_result
        assert len(story_result["agent_requests"]) == 1

        # This confirms that the critical gap from ADR-001 has been filled:
        # Scripts can now output AgentRequest objects and the ensemble processes them
        # for inter-agent communication and dynamic parameter generation

    @pytest.mark.asyncio
    async def test_agent_request_enables_dynamic_parameter_generation(self) -> None:
        """Test that AgentRequest objects enable dynamic parameter generation.

        This test focuses on the dynamic parameter generation capability
        that was missing from ADR-001 implementation.
        """
        # Test the AgentRequestProcessor directly to show dynamic parameter generation
        from llm_orc.core.execution.agent_request_processor import AgentRequestProcessor
        from llm_orc.core.execution.dependency_resolver import DependencyResolver
        from llm_orc.schemas.script_agent import AgentRequest

        # Create components
        mock_role_resolver = Mock(return_value="User Input Agent")
        dependency_resolver = DependencyResolver(mock_role_resolver)
        processor = AgentRequestProcessor(dependency_resolver)

        # Create an AgentRequest (as would be output by story generator)
        agent_request = AgentRequest(
            target_agent_type="user_input",
            parameters={
                "prompt": "Enter the protagonist's cybernetic enhancement",
                "validation_pattern": r"^[a-zA-Z\s]{3,50}$",
                "multiline": False,
            },
            priority=1,
        )

        # Test dynamic parameter generation
        dynamic_params = processor.generate_dynamic_parameters(
            agent_request,
            context={"story_theme": "cyberpunk", "character_type": "protagonist"},
        )

        # Verify parameters were generated correctly
        assert "prompt" in dynamic_params
        assert "validation_pattern" in dynamic_params
        assert "multiline" in dynamic_params
        assert (
            dynamic_params["prompt"] == "Enter the protagonist's cybernetic enhancement"
        )
        assert dynamic_params["multiline"] is False

        # Test coordination with available agents
        phase_agents: list[AgentConfig] = [
            ScriptAgentConfig(
                name="user_input_collector",
                script="primitives/user_input.py",
            )
        ]

        coordinated_agents = processor.coordinate_agent_execution(
            [agent_request], {}, phase_agents
        )

        # Verify coordination produced updated agent configurations
        assert len(coordinated_agents) == 1
        coordinated_agent = coordinated_agents[0]
        assert coordinated_agent["name"] == "user_input_collector"
        assert "dynamic_parameters" in coordinated_agent
        assert (
            coordinated_agent["dynamic_parameters"]["prompt"]
            == "Enter the protagonist's cybernetic enhancement"
        )

        # This demonstrates that AgentRequest objects successfully enable
        # dynamic parameter generation for target agents, completing the ADR-001 gap
