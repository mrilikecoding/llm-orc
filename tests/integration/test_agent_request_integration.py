"""Integration tests for AgentRequestProcessor with EnsembleExecutor.

Tests the complete flow from script agent outputting AgentRequest objects
to the ensemble processing them for inter-agent communication.
"""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.execution.scripting.resolver import ScriptResolver
from llm_orc.schemas.agent_config import AgentConfig, ScriptAgentConfig
from llm_orc.schemas.script_agent import AgentRequest
from tests.fixtures.test_primitives import (
    TestPrimitiveFactory,
    create_test_primitive_with_json_contract,
)


@pytest.fixture
def test_primitives_with_script_resolver(
    tmp_path: Path,
) -> Generator[dict[str, Path], None, None]:
    """Provide test primitives and patch ScriptResolver to find them.

    Per ADR-006: Tests should be independent of library submodule.
    This fixture creates minimal test primitives and configures the
    ScriptResolver to find them before checking library locations.
    """
    # Create test primitives directory
    primitives_dir = TestPrimitiveFactory.setup_test_primitives_dir(tmp_path)

    # Create test script that outputs AgentRequest objects
    create_test_primitive_with_json_contract(
        tmp_path,
        "test_json_contract_agent",
        mock_agent_requests=[
            {
                "target_agent_type": "user_input",
                "parameters": {
                    "prompt": "What is the character's name?",
                    "validation_pattern": "^[A-Za-z\\s]{2,30}$",
                    "retry_message": "Please enter a valid name",
                    "mock_user_input": "test_character_name",  # For testing
                },
                "priority": 1,
            }
        ],
    )

    # Patch ScriptResolver to use test directories
    # Use custom search paths to include test primitives
    original_init = ScriptResolver.__init__

    def mock_init(
        self: ScriptResolver,
        search_paths: list[str] | None = None,
        project_dir: Path | None = None,
    ) -> None:
        """Initialize with test search paths."""
        test_search_paths = [
            str(primitives_dir),
            str(tmp_path),
        ]
        original_init(self, test_search_paths)

    with patch.object(ScriptResolver, "__init__", mock_init):
        yield {"primitives_dir": primitives_dir, "tmp_path": tmp_path}


class TestAgentRequestIntegration:
    """Integration tests for AgentRequest processing in ensemble execution."""

    @pytest.mark.asyncio
    async def test_ensemble_processes_agent_requests_from_script_output(self) -> None:
        """Test that ensemble can process AgentRequest objects from script output.

        RED PHASE: This will fail because EnsembleExecutor doesn't process requests.
        """
        # Create a test script that outputs AgentRequest objects
        test_script_content = """#!/usr/bin/env python3
import json
import os

# Get input
input_data = os.environ.get("INPUT_DATA", "{}")
parsed_input = json.loads(input_data)

# Create output with AgentRequest
output = {
    "success": True,
    "data": {
        "story_fragment": "A cyberpunk detective story begins...",
        "prompt_generated": True
    },
    "error": None,
    "agent_requests": [
        {
            "target_agent_type": "user_input",
            "parameters": {
                "prompt": "What is the detective's name?",
                "validation_pattern": "^[A-Za-z\\s]{2,30}$",
                "retry_message": "Please enter a valid name (2-30 characters)"
            },
            "priority": 1
        }
    ]
}

print(json.dumps(output))
"""

        # Create temporary script file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script_content)
            script_path = f.name

        try:
            Path(script_path).chmod(0o755)

            # Create ensemble config with script that outputs AgentRequest
            config = EnsembleConfig(
                name="agent_request_test",
                description="Test AgentRequest processing",
                agents=[
                    ScriptAgentConfig(
                        name="story_generator",
                        script=script_path,
                    ),
                    ScriptAgentConfig(
                        name="user_input_collector",
                        script=(
                            'echo \'{"success": true, "data": {"user_input": "J"}}\''
                        ),
                    ),
                ],
            )

            executor = ExecutorFactory.create_root_executor()

            # Mock artifact manager to prevent real artifact creation
            mock_artifact_manager = Mock()
            mock_artifact_manager.save_execution_results = Mock()

            with patch.object(executor, "_artifact_manager", mock_artifact_manager):
                result = await executor.execute(config, '{"story_theme": "cyberpunk"}')

            # Verify execution completed
            assert result["status"] in ["completed", "completed_with_errors"]
            assert "story_generator" in result["results"]

            # Verify story generator output contains agent_requests
            story_result = result["results"]["story_generator"]
            assert story_result["status"] == "success"

            # Parse the JSON response to check for agent_requests
            story_response = story_result["response"]
            if isinstance(story_response, str):
                story_data = json.loads(story_response)
            else:
                story_data = story_response

            # This assertion should FAIL initially because EnsembleExecutor
            # doesn't process agent_requests from script output yet
            assert "agent_requests" in story_data
            assert len(story_data["agent_requests"]) == 1

            request = story_data["agent_requests"][0]
            assert request["target_agent_type"] == "user_input"
            assert "prompt" in request["parameters"]

            # Verify the ensemble processes the agent requests for coordination
            # This should also FAIL because AgentRequestProcessor isn't integrated
            # with EnsembleExecutor yet
            assert "processed_agent_requests" in result.get("metadata", {})

        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_story_generator_to_user_input_agent_flow(
        self, test_primitives_with_script_resolver: dict[str, Path]
    ) -> None:
        """Test the complete flow from story generator to user input agent.

        This validates the BDD scenario: story generator → user input agent flow.
        Uses test fixtures per ADR-006 to avoid library dependency.
        """
        config = EnsembleConfig(
            name="story_to_user_input_flow",
            description="Test story generator to user input agent coordination",
            agents=[
                ScriptAgentConfig(
                    name="cyberpunk_story_generator",
                    script="test_json_contract_agent.py",
                    parameters={
                        "theme": "cyberpunk",
                        "character_type": "protagonist",
                    },
                ),
                ScriptAgentConfig(
                    name="user_input_agent",
                    script="primitives/user_input.py",
                ),
            ],
        )

        executor = ExecutorFactory.create_root_executor()

        # Mock artifact manager
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

        # Verify the execution completed
        assert result["status"] in ["completed", "completed_with_errors"]
        assert "cyberpunk_story_generator" in result["results"]

        # Verify story generator produced AgentRequest for user input
        story_result = result["results"]["cyberpunk_story_generator"]
        assert story_result["status"] == "success"

        # This should FAIL because AgentRequest processing isn't integrated
        # The ensemble should coordinate the user_input_agent based on the AgentRequest
        assert "user_input_agent" in result["results"]

        # Verify the user input agent received dynamic parameters
        user_input_result = result["results"]["user_input_agent"]
        assert user_input_result["status"] == "success"

        # The user input agent should have received the dynamically generated prompt
        # from the story generator's AgentRequest
        user_input_response = user_input_result["response"]
        if isinstance(user_input_response, str):
            user_input_data = json.loads(user_input_response)
        else:
            user_input_data = user_input_response

        # This assertion should FAIL because dynamic parameter passing
        # from AgentRequest to target agent isn't implemented yet
        assert "received_dynamic_parameters" in user_input_data

    def test_agent_request_processor_integration_with_dependency_resolver(self) -> None:
        """Test AgentRequestProcessor integrates with DependencyResolver.

        RED PHASE: Will fail because integration isn't implemented yet.
        """
        from llm_orc.core.execution.phases.agent_request_processor import (
            AgentRequestProcessor,
        )
        from llm_orc.core.execution.phases.dependency_resolver import DependencyResolver

        # Create mock role resolver
        mock_role_resolver = Mock(return_value="Test Role")
        dependency_resolver = DependencyResolver(mock_role_resolver)

        # Create AgentRequestProcessor with DependencyResolver
        processor = AgentRequestProcessor(dependency_resolver)

        # Create test AgentRequest
        agent_request = AgentRequest(
            target_agent_type="user_input",
            parameters={
                "prompt": "Enter protagonist name",
                "context": "cyberpunk_story",
            },
            priority=1,
        )

        # Test coordination with dependency resolver
        phase_agents: list[AgentConfig] = [
            ScriptAgentConfig(
                name="user_input_collector",
                script="primitives/user_input.py",
            )
        ]

        results_dict = {
            "story_generator": {
                "response": json.dumps(
                    {
                        "success": True,
                        "data": {"story": "Generated story"},
                        "agent_requests": [agent_request.model_dump()],
                    }
                )
            }
        }

        # This should work because we implemented the basic coordination
        coordinated_agents = processor.coordinate_agent_execution(
            [agent_request], results_dict, phase_agents
        )

        assert len(coordinated_agents) > 0
        assert "dynamic_parameters" in coordinated_agents[0]

        # This assertion will be implemented in future iterations
        # For now, the basic coordination is sufficient
        # TODO: Implement enhanced_phases integration with DependencyResolver
        # enhanced_phases = processor.generate_enhanced_phases_with_requests(
        #     [agent_request], phase_agents, dependency_resolver
        # )
        # assert enhanced_phases is not None
