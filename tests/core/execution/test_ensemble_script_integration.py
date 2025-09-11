"""Tests for script agent integration with ensemble execution."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


@pytest.fixture(autouse=True)
def mock_expensive_dependencies():
    """Mock expensive dependencies for all ensemble script integration tests."""
    with patch("llm_orc.core.execution.ensemble_execution.ConfigurationManager"):
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
