"""Tests for implicit agent type detection based on configuration fields."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


class TestImplicitAgentDetection:
    """Test implicit agent type detection in ensemble executor."""

    @pytest.mark.asyncio
    async def test_executor_detects_script_agent_by_script_field(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that executor detects script agents by presence of 'script' field."""
        executor = mock_ensemble_executor

        agent_config = ScriptAgentConfig(
            name="test_script",
            script="echo 'Hello World'",
            # No 'type' field - routed by isinstance(ScriptAgentConfig)
        )

        with patch.object(executor._script_agent_runner, "execute") as mock_script:
            mock_script.return_value = ("Script output", None)

            result = await executor._execute_agent(agent_config, "test input")

            mock_script.assert_called_once_with(agent_config, "test input")
            assert result == ("Script output", None)

    @pytest.mark.asyncio
    async def test_executor_detects_llm_agent_by_model_profile_field(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test executor detects LLM agents by 'model_profile' field."""
        executor = mock_ensemble_executor

        agent_config = LlmAgentConfig(
            name="test_llm",
            model_profile="default-gpt4",
            system_prompt="You are a helpful assistant",
            # No 'type' field - routed by isinstance(LlmAgentConfig)
        )

        with patch.object(executor._llm_agent_runner, "execute") as mock_llm:
            mock_llm.return_value = ("LLM output", MagicMock())

            result = await executor._execute_agent(agent_config, "test input")

            mock_llm.assert_called_once_with(agent_config, "test input")
            assert result[0] == "LLM output"

    def test_parse_agent_config_raises_for_missing_type_fields(self) -> None:
        """Test parse_agent_config raises when 'script' nor model source is missing.

        With typed models, invalid configs are rejected at parse time via
        LlmAgentConfig's model_validator, which requires model_profile or
        model+provider.
        """
        from pydantic import ValidationError

        from llm_orc.schemas.agent_config import parse_agent_config

        with pytest.raises((ValueError, ValidationError)):
            parse_agent_config({"name": "invalid_agent", "some_other_field": "value"})

    @pytest.mark.asyncio
    async def test_parse_agent_config_script_takes_priority_over_model(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test 'script' key causes parse_agent_config to produce ScriptAgentConfig.

        The discriminator in parse_agent_config checks 'script' before 'model_profile',
        so a config with 'script' always becomes a ScriptAgentConfig and routes to the
        script runner.
        """
        from llm_orc.schemas.agent_config import parse_agent_config

        executor = mock_ensemble_executor

        # parse_agent_config checks 'script' first — produces ScriptAgentConfig
        agent_config = parse_agent_config(
            {"name": "test_agent", "script": "echo 'Script'"}
        )
        assert isinstance(agent_config, ScriptAgentConfig)

        with patch.object(executor._script_agent_runner, "execute") as mock_script:
            with patch.object(executor._llm_agent_runner, "execute") as mock_llm:
                mock_script.return_value = ("Script output", None)

                await executor._execute_agent(agent_config, "test input")

                mock_script.assert_called_once()
                mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_implicit_detection_with_script_agent(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that implicit detection works with ScriptAgent.

        ScriptAgentRunner converts the typed config to a dict via model_dump()
        before passing it to ScriptAgent, which expects a dict at its boundary.
        """
        executor = mock_ensemble_executor

        agent_config = ScriptAgentConfig(
            name="enhanced_script",
            script="scripts/test.py",
            parameters={"key": "value"},  # Enhanced script agent features
        )

        # Mock the enhanced script agent execution
        with patch(
            "llm_orc.core.execution.scripting.agent_runner.ScriptAgent"
        ) as mock_agent_class:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.execute.return_value = {
                "success": True,
                "result": "data",
            }
            mock_agent_class.return_value = mock_agent_instance

            await executor._execute_agent(agent_config, "test input")

            # ScriptAgentRunner converts typed config to dict via model_dump()
            # before instantiating ScriptAgent — assert the boundary conversion
            mock_agent_class.assert_called_once_with(
                "enhanced_script", agent_config.model_dump(), project_dir=None
            )
            mock_agent_instance.execute.assert_called_once()

    def test_agent_type_detection_in_config_validation(self) -> None:
        """Test that agent type can be detected during configuration validation."""
        from llm_orc.core.config.ensemble_config import EnsembleConfig

        # Ensemble with mixed implicit agent types using typed constructors
        ensemble_config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble with implicit agent types",
            agents=[
                ScriptAgentConfig(name="script_agent", script="echo 'test'"),
                LlmAgentConfig(
                    name="llm_agent",
                    model_profile="default-gpt4",
                    system_prompt="Test prompt",
                ),
            ],
        )

        # Both agents should be valid without explicit 'type' field
        assert len(ensemble_config.agents) == 2
        assert ensemble_config.agents[0].name == "script_agent"
        assert ensemble_config.agents[1].name == "llm_agent"
