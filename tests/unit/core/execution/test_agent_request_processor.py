"""Tests for AgentRequestProcessor (ADR-001) - TDD Red Phase.

This test file validates the AgentRequestProcessor class that processes
AgentRequest objects from ScriptAgentOutput to enable dynamic parameter
generation and inter-agent communication.
"""

import json
from unittest.mock import Mock

import pytest

from llm_orc.core.execution.agent_request_processor import AgentRequestProcessor
from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.schemas.agent_config import ScriptAgentConfig
from llm_orc.schemas.script_agent import AgentRequest


class TestAgentRequestProcessor:
    """Test AgentRequestProcessor for dynamic parameter generation."""

    def test_agent_request_processor_initialization(self) -> None:
        """Test AgentRequestProcessor can be initialized with DependencyResolver."""
        # RED PHASE: This will fail because AgentRequestProcessor doesn't exist yet
        dependency_resolver = Mock(spec=DependencyResolver)
        processor = AgentRequestProcessor(dependency_resolver)

        assert processor is not None
        assert processor._dependency_resolver is dependency_resolver

    def test_generate_dynamic_parameters_creates_parameters_for_target_agent(
        self,
    ) -> None:
        """Test dynamic parameter generation for target agents."""
        # RED PHASE: This will fail because the method doesn't exist
        dependency_resolver = Mock(spec=DependencyResolver)
        processor = AgentRequestProcessor(dependency_resolver)

        agent_request = AgentRequest(
            target_agent_type="user_input",
            parameters={"prompt": "Enter character name", "multiline": False},
            priority=1,
        )

        # Generate parameters for the target agent
        dynamic_params = processor.generate_dynamic_parameters(
            agent_request, context={"story_theme": "cyberpunk"}
        )

        assert "prompt" in dynamic_params
        assert "multiline" in dynamic_params
        assert dynamic_params["prompt"] == "Enter character name"
        assert dynamic_params["multiline"] is False

    def test_coordinate_with_dependency_resolver_integrates_with_ensemble_flow(
        self,
    ) -> None:
        """Test integration with DependencyResolver for agent coordination."""
        # RED PHASE: This will fail because the integration method doesn't exist
        mock_resolver = Mock(spec=DependencyResolver)
        processor = AgentRequestProcessor(mock_resolver)

        agent_requests = [
            AgentRequest(
                target_agent_type="user_input",
                parameters={"prompt": "Character name?"},
                priority=1,
            )
        ]

        # Coordinate with dependency resolver
        result = processor.coordinate_agent_execution(
            agent_requests,
            results_dict={"story_generator": {"response": "story data"}},
            phase_agents=[
                ScriptAgentConfig(name="user_input", script="primitives/user_input.py")
            ],
        )

        # Should return updated agent configurations or execution plan
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_process_script_agent_output_with_agent_requests(self) -> None:
        """Test full processing of ScriptAgentOutput containing AgentRequest objects."""
        # RED PHASE: This will fail because the full processing method doesn't exist
        dependency_resolver = Mock(spec=DependencyResolver)
        processor = AgentRequestProcessor(dependency_resolver)

        # JSON response from a script agent that includes agent_requests
        script_response = json.dumps(
            {
                "success": True,
                "data": {
                    "story_fragment": "In Neo-Tokyo 2087, the rain never stops...",
                    "character_prompt_generated": True,
                },
                "error": None,
                "agent_requests": [
                    {
                        "target_agent_type": "user_input",
                        "parameters": {
                            "prompt": "What is the protagonist's enhancement?",
                            "validation_pattern": r"^[a-zA-Z\s]{3,50}$",
                            "retry_message": "Please enter a valid enhancement type",
                        },
                        "priority": 1,
                    }
                ],
            }
        )

        # Process the full script output
        processed_result = await processor.process_script_output_with_requests(
            script_response,
            source_agent="story_generator",
            current_phase_agents=[
                ScriptAgentConfig(
                    name="user_input",
                    script="primitives/user_input.py",
                )
            ],
        )

        assert processed_result is not None
        assert "agent_requests" in processed_result
        assert len(processed_result["agent_requests"]) == 1

        # Verify the request was properly processed
        request = processed_result["agent_requests"][0]
        assert request["target_agent_type"] == "user_input"
        assert "prompt" in request["parameters"]

    @pytest.mark.asyncio
    async def test_plain_text_response_returns_empty_agent_requests(self) -> None:
        """Plain text (non-JSON) output returns empty agent_requests, not an error."""
        dependency_resolver = Mock(spec=DependencyResolver)
        processor = AgentRequestProcessor(dependency_resolver)

        result = await processor.process_script_output_with_requests(
            "Authentication working",
            source_agent="validator",
            current_phase_agents=[],
        )

        assert result["agent_requests"] == []
        assert result["coordinated_agents"] == []
        assert result["source_agent"] == "validator"
