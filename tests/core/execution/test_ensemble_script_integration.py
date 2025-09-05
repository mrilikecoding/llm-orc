"""Tests for script agent integration with ensemble execution."""

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


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
        result = await executor.execute(config, "test input")

        assert result["status"] in ["completed", "completed_with_errors"]
        assert "echo_agent" in result["results"]
        response = result["results"]["echo_agent"]["response"]
        if isinstance(response, dict):
            assert "Script output from agent" in response.get("output", "")
        else:
            assert "Script output from agent" in response

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
        result = await executor.execute(config, "test data")

        assert result["status"] in ["completed", "completed_with_errors"]
        assert "data_fetcher" in result["results"]
        assert "llm_analyzer" in result["results"]
        response = result["results"]["data_fetcher"]["response"]
        if isinstance(response, dict):
            assert "Data fetched successfully" in response.get("output", "")
        else:
            assert "Data fetched successfully" in response

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
