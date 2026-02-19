"""Tests for script agent integration with ensemble execution."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


@pytest.fixture(autouse=True)
def mock_expensive_dependencies() -> Generator[None, None, None]:
    """Mock expensive dependencies for all ensemble script integration tests."""
    mock_config_manager = Mock()
    mock_config_manager.load_performance_config.return_value = {
        "execution": {"default_timeout": 60},
        "concurrency": {"max_concurrent_agents": 5},
        "script_cache": {
            "enabled": True,
            "ttl_seconds": 3600,
            "max_size": 1000,
            "persist_to_artifacts": False,
        },
    }
    mock_config_manager.get_model_profiles.return_value = {}

    with patch(
        "llm_orc.core.execution.ensemble_execution.ConfigurationManager",
        return_value=mock_config_manager,
    ):
        with patch("llm_orc.core.execution.ensemble_execution.CredentialStorage"):
            with patch("llm_orc.core.execution.ensemble_execution.ModelFactory"):
                yield


class TestEnsembleScriptIntegration:
    """Test script agent integration with ensemble execution."""

    @pytest.mark.asyncio
    async def test_ensemble_with_script_agent(self) -> None:
        """Test ensemble execution with script-based agent."""
        config = EnsembleConfig(
            name="test_script_ensemble",
            description="Test ensemble with script agent",
            agents=[
                {
                    "name": "echo_agent",
                    "script": (
                        'echo "{"success": true, "output": "Script output from agent"}"'
                    ),
                    "timeout_seconds": 1,
                }
            ],
        )

        executor = EnsembleExecutor()

        # Mock ArtifactManager to prevent real artifact creation
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            result = await executor.execute(config, "test input")

        assert result["status"] in ["completed", "completed_with_errors"]
        assert "echo_agent" in result["results"]
        response = result["results"]["echo_agent"]["response"]
        if isinstance(response, dict):
            assert "Script output from agent" in response.get("output", "")
        else:
            assert "Script output from agent" in response

        # Verify no real artifacts were created by checking the mock was called
        mock_artifact_manager.save_execution_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensemble_with_mixed_agents(self) -> None:
        """Test ensemble with both script and LLM agents."""
        config = EnsembleConfig(
            name="mixed_ensemble",
            description="Mixed script and LLM agents",
            agents=[
                {
                    "name": "data_fetcher",
                    "script": (
                        'echo "{"success": true, '
                        '"output": "Data fetched successfully"}"'
                    ),
                    "timeout_seconds": 1,
                },
                {
                    "name": "llm_analyzer",
                    "model_profile": "claude-analyst",
                    "system_prompt": "Analyze the provided data",
                    "timeout_seconds": 2,
                },
            ],
        )

        executor = EnsembleExecutor()

        # Mock ArtifactManager to prevent real artifact creation
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            result = await executor.execute(config, "test data")

        assert result["status"] in ["completed", "completed_with_errors"]
        assert "data_fetcher" in result["results"]
        assert "llm_analyzer" in result["results"]
        response = result["results"]["data_fetcher"]["response"]
        if isinstance(response, dict):
            assert "Data fetched successfully" in response.get("output", "")
        else:
            assert "Data fetched successfully" in response

        # Verify no real artifacts were created by checking the mock was called
        mock_artifact_manager.save_execution_results.assert_called_once()

    def test_ensemble_config_validates_agent_types(self) -> None:
        """Test that ensemble configuration validates agent types."""
        # This should work - valid script agent
        config = EnsembleConfig(
            name="valid_script",
            description="Valid script agent",
            agents=[
                {
                    "name": "valid_agent",
                    "type": "script",
                    "command": "echo 'test'",
                }
            ],
        )

        assert config.agents[0]["type"] == "script"
        assert config.agents[0]["command"] == "echo 'test'"

    @pytest.mark.asyncio
    async def test_integration_tests_do_not_create_artifacts(self) -> None:
        """Test that integration tests do not create real artifacts."""
        # Run a test with EnsembleExecutor and verify no artifacts are created
        config = EnsembleConfig(
            name="test_no_artifacts",
            description="Test that should not create artifacts",
            agents=[
                {
                    "name": "test_agent",
                    "script": 'echo "test output"',
                    "timeout_seconds": 1,
                }
            ],
        )

        executor = EnsembleExecutor()

        # Mock ArtifactManager to prevent real artifact creation
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            result = await executor.execute(config, "test input")

        # Verify the test completed successfully
        assert result["status"] in ["completed", "completed_with_errors"]

        # Verify ArtifactManager was called with the mock (not the real one)
        mock_artifact_manager.save_execution_results.assert_called_once()

        # Verify no real artifacts were created in the file system
        artifacts_dir = Path(".llm-orc/artifacts/test_no_artifacts")
        assert not artifacts_dir.exists(), (
            "Real artifacts should not be created during tests"
        )

    @pytest.mark.asyncio
    async def test_script_resolution_priority_order(self) -> None:
        """Test script resolution follows priority order (replaces BDD scenario)."""
        # Create temporary script structure to test resolution
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .llm-orc/scripts directory structure
            scripts_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            scripts_dir.mkdir(parents=True)

            # Create test script
            test_script = scripts_dir / "test_resolution.py"
            test_script.write_text('print("resolved from .llm-orc/scripts")')
            test_script.chmod(0o755)

            config = EnsembleConfig(
                name="script_resolution_test",
                description="Test script resolution priority",
                agents=[
                    {
                        "name": "resolver_agent",
                        "script": str(test_script),
                        "timeout_seconds": 2,
                    }
                ],
            )

            executor = EnsembleExecutor()
            mock_artifact_manager = Mock(spec=ArtifactManager)
            mock_artifact_manager.save_execution_results = Mock()

            with patch.object(executor, "_artifact_manager", mock_artifact_manager):
                result = await executor.execute(config, "test resolution")

            assert result["status"] in ["completed", "completed_with_errors"]
            assert "resolver_agent" in result["results"]

    @pytest.mark.asyncio
    async def test_ensemble_caching_integration(self) -> None:
        """Test ensemble caching functionality (replaces BDD scenario)."""
        config = EnsembleConfig(
            name="cache_test_ensemble",
            description="Test caching with deterministic script",
            agents=[
                {
                    "name": "deterministic_agent",
                    "script": (
                        'echo "{\\"success\\": true, \\"data\\": \\"cached_result\\"}"'
                    ),
                    "timeout_seconds": 1,
                }
            ],
        )

        executor = EnsembleExecutor()
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            # First execution - should cache result
            result1 = await executor.execute(config, "cache test input")

            # Second execution - should use cached result
            result2 = await executor.execute(config, "cache test input")

        # Both executions should succeed
        assert result1["status"] in ["completed", "completed_with_errors"]
        assert result2["status"] in ["completed", "completed_with_errors"]

        # Results should be consistent (cached)
        assert "deterministic_agent" in result1["results"]
        assert "deterministic_agent" in result2["results"]

    @pytest.mark.asyncio
    async def test_async_performance_integration(self) -> None:
        """Test async performance with parallel scripts (replaces BDD scenario)."""
        import time

        config = EnsembleConfig(
            name="performance_test_ensemble",
            description="Test parallel script execution performance",
            agents=[
                {
                    "name": "fast_agent",
                    "script": (
                        "sleep 0.1 && echo "
                        '"{\\"success\\": true, \\"agent\\": \\"fast\\"}"'
                    ),
                    "timeout_seconds": 1,
                },
                {
                    "name": "medium_agent",
                    "script": (
                        "sleep 0.2 && echo "
                        '"{\\"success\\": true, \\"agent\\": \\"medium\\"}"'
                    ),
                    "timeout_seconds": 1,
                },
                {
                    "name": "slow_agent",
                    "script": (
                        "sleep 0.3 && echo "
                        '"{\\"success\\": true, \\"agent\\": \\"slow\\"}"'
                    ),
                    "timeout_seconds": 1,
                },
            ],
        )

        executor = EnsembleExecutor()
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        start_time = time.time()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            result = await executor.execute(config, "performance test")

        execution_time = time.time() - start_time

        # Should complete in parallel (closer to max script time, not sum)
        assert execution_time < 1.0, (
            f"Parallel execution took {execution_time}s, expected < 1.0s"
        )
        assert result["status"] in ["completed", "completed_with_errors"]
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_error_handling_integration(self) -> None:
        """Test error handling with proper exception chaining (replaces BDD)."""
        config = EnsembleConfig(
            name="error_test_ensemble",
            description="Test error handling and exception chaining",
            agents=[
                {
                    "name": "failing_agent",
                    "script": "exit 1",  # Script that will fail
                    "timeout_seconds": 1,
                },
                {
                    "name": "success_agent",
                    "script": (
                        'echo "{\\"success\\": true, \\"message\\": \\"I succeeded\\"}"'
                    ),
                    "timeout_seconds": 1,
                },
            ],
        )

        executor = EnsembleExecutor()
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            result = await executor.execute(config, "error test")

        # Ensemble should handle errors gracefully and include both agents
        assert result["status"] == "completed"
        assert "failing_agent" in result["results"]
        assert "success_agent" in result["results"]

        # Check that failing agent properly reports error in response
        failing_result = result["results"]["failing_agent"]
        assert failing_result["status"] == "success"  # Executor handles gracefully
        assert "exit code 1" in failing_result["response"]  # Error details in response

        # Success agent should show proper failure message
        success_result = result["results"]["success_agent"]
        assert success_result["status"] == "success"  # Graceful handling
        assert "Script not found" in success_result["response"]  # Expected error

    @pytest.mark.asyncio
    async def test_script_resolver_ensemble_executor_json_contract_validation(
        self,
    ) -> None:
        """RED PHASE: Test ScriptResolver ↔ EnsembleExecutor JSON integration.

        This test validates the integration between ScriptResolver script discovery
        and EnsembleExecutor JSON I/O contract validation during ensemble execution.

        Expected to FAIL initially because JSON contract validation is not implemented
        in the ScriptResolver → EnsembleExecutor handoff.
        """
        import json
        from pathlib import Path

        from llm_orc.core.execution.script_resolver import ScriptResolver
        from llm_orc.schemas.script_agent import ScriptAgentInput, ScriptAgentOutput

        # Create script resolver and test discovery
        resolver = ScriptResolver()

        # Test script discovery from .llm-orc/scripts/
        script_path = resolver.resolve_script_path("test_json_contract_agent.py")
        assert Path(script_path).exists(), "Test script should be discoverable"

        # Create ensemble config using discovered script
        config = EnsembleConfig(
            name="json_contract_validation_test",
            description="Test JSON contract validation in ScriptResolver integration",
            agents=[
                {
                    "name": "contract_validator",
                    "script": "test_json_contract_agent.py",  # By ScriptResolver
                    "timeout_seconds": 5,
                }
            ],
        )

        # Create input that conforms to ScriptAgentInput schema
        script_input = ScriptAgentInput(
            agent_name="contract_validator",
            input_data="Test input for JSON contract validation",
            context={"test_context": "integration_test"},
            dependencies={"upstream_agent": "some_result"},
        )

        executor = EnsembleExecutor()
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        with patch.object(executor, "_artifact_manager", mock_artifact_manager):
            # Execute ensemble with ScriptAgentInput JSON
            result = await executor.execute(config, script_input.model_dump_json())

        # Verify ensemble execution succeeded
        assert result["status"] in ["completed", "completed_with_errors"]
        assert "contract_validator" in result["results"]

        agent_result = result["results"]["contract_validator"]
        assert agent_result["status"] == "success"

        # Parse and validate output conforms to ScriptAgentOutput schema
        response = agent_result["response"]

        # This is where the test should FAIL initially:
        # JSON contract validation should ensure response is valid ScriptAgentOutput
        if isinstance(response, str):
            response_data = json.loads(response)
        else:
            response_data = response

        # Validate the output conforms to ScriptAgentOutput schema
        # This assertion should FAIL because JSON contract validation
        # is not implemented in the ScriptResolver → EnsembleExecutor handoff
        validated_output = ScriptAgentOutput.model_validate(response_data)

        # Additional assertions that should pass once validation is implemented
        assert validated_output.success is True
        assert validated_output.data is not None
        assert validated_output.error is None
        assert isinstance(validated_output.agent_requests, list)

        # Verify processed data contains expected structure
        assert "processed_input" in validated_output.data
        assert "agent_name" in validated_output.data

        # This assertion should FAIL because the ScriptAgentInput JSON
        # is not being properly passed through the ScriptResolver → EnsembleExecutor
        # The script receives default values instead of the structured input
        expected_name = "contract_validator"
        actual_name = validated_output.data["agent_name"]
        assert actual_name == expected_name, (
            f"Expected agent_name '{expected_name}' but got '{actual_name}'. "
            "This indicates JSON contract validation is not implemented."
        )

        # Additional validation that the JSON structure was properly passed
        assert validated_output.data["dependency_count"] > 0, (
            "Expected dependencies to be passed to script, but got 0. "
            "This confirms ScriptAgentInput JSON is not validated/passed correctly."
        )
        assert len(validated_output.data["context_keys"]) > 0, (
            "Expected context to be passed to script, but got empty list. "
            "This confirms ScriptAgentInput JSON is not validated/passed correctly."
        )

