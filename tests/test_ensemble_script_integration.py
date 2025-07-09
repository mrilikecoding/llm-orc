"""Tests for script agent integration with ensemble execution."""

import pytest

from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.ensemble_execution import EnsembleExecutor


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
                    "type": "script",
                    "command": 'echo "Script output: $INPUT_DATA"',
                    "timeout_seconds": 10,
                }
            ],
            coordinator={
                "synthesis_prompt": "Summarize the script output",
                "output_format": "json",
            },
        )

        executor = EnsembleExecutor()
        result = await executor.execute(config, "test input")

        assert result["status"] in ["completed", "completed_with_errors"]
        assert "echo_agent" in result["results"]
        assert (
            "Script output: test input" in result["results"]["echo_agent"]["response"]
        )

    @pytest.mark.asyncio
    async def test_ensemble_with_mixed_agents(self) -> None:
        """Test ensemble with both script and LLM agents."""
        config = EnsembleConfig(
            name="mixed_ensemble",
            description="Mixed script and LLM agents",
            agents=[
                {
                    "name": "data_fetcher",
                    "type": "script",
                    "script": 'echo "Data: $INPUT_DATA"',
                },
                {
                    "name": "llm_analyzer",
                    "type": "llm",
                    "role": "analyst",
                    "model": "claude-3-sonnet",
                    "prompt": "Analyze the provided data",
                },
            ],
            coordinator={
                "synthesis_prompt": "Combine script and LLM outputs",
                "output_format": "json",
            },
        )

        executor = EnsembleExecutor()
        result = await executor.execute(config, "test data")

        assert result["status"] in ["completed", "completed_with_errors"]
        assert "data_fetcher" in result["results"]
        assert "llm_analyzer" in result["results"]
        assert "Data: test data" in result["results"]["data_fetcher"]["response"]

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
            coordinator={"synthesis_prompt": "test"},
        )

        assert config.agents[0]["type"] == "script"
        assert config.agents[0]["command"] == "echo 'test'"
