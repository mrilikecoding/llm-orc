"""Tests for implicit agent type detection based on configuration fields."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestImplicitAgentDetection:
    """Test implicit agent type detection in ensemble executor."""

    @pytest.mark.asyncio
    async def test_executor_detects_script_agent_by_script_field(
        self, mock_ensemble_executor
    ) -> None:
        """Test that executor detects script agents by presence of 'script' field."""
        executor = mock_ensemble_executor

        agent_config = {
            "name": "test_script",
            "script": "echo 'Hello World'",
            # No 'type' field - should be detected as script agent
        }

        with patch.object(executor, "_execute_script_agent") as mock_script:
            mock_script.return_value = ("Script output", None)

            result = await executor._execute_agent(agent_config, "test input")

            mock_script.assert_called_once_with(agent_config, "test input")
            assert result == ("Script output", None)

    @pytest.mark.asyncio
    async def test_executor_detects_llm_agent_by_model_profile_field(
        self, mock_ensemble_executor
    ) -> None:
        """Test executor detects LLM agents by 'model_profile' field."""
        executor = mock_ensemble_executor

        agent_config = {
            "name": "test_llm",
            "model_profile": "default-gpt4",
            "system_prompt": "You are a helpful assistant",
            # No 'type' field - should be detected as LLM agent
        }

        with patch.object(executor, "_execute_llm_agent") as mock_llm:
            mock_llm.return_value = ("LLM output", MagicMock())

            result = await executor._execute_agent(agent_config, "test input")

            mock_llm.assert_called_once_with(agent_config, "test input")
            assert result[0] == "LLM output"

    @pytest.mark.asyncio
    async def test_executor_raises_for_missing_type_fields(
        self, mock_ensemble_executor
    ) -> None:
        """Test executor raises error when neither field is present."""
        executor = mock_ensemble_executor

        agent_config = {
            "name": "invalid_agent",
            "some_other_field": "value",
            # Neither 'script' nor 'model_profile' - should raise error
        }

        with pytest.raises(
            ValueError, match="must have either 'script' or 'model_profile'"
        ):
            await executor._execute_agent(agent_config, "test input")

    @pytest.mark.asyncio
    async def test_executor_maintains_backward_compatibility(
        self, mock_ensemble_executor
    ) -> None:
        """Test executor maintains backward compatibility with 'type' field."""
        executor = mock_ensemble_executor

        # Script agent with explicit type
        script_config = {
            "name": "test_script",
            "type": "script",
            "script": "echo 'Hello'",
        }

        with patch.object(executor, "_execute_script_agent") as mock_script:
            mock_script.return_value = ("Script output", None)

            await executor._execute_agent(script_config, "test input")
            mock_script.assert_called_once()

        # LLM agent with explicit type (existing behavior)
        llm_config = {
            "name": "test_llm",
            "type": "llm",
            "model_profile": "default-gpt4",
        }

        with patch.object(executor, "_execute_llm_agent") as mock_llm:
            mock_llm.return_value = ("LLM output", MagicMock())

            await executor._execute_agent(llm_config, "test input")
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_script_field_takes_priority_over_model_profile(
        self, mock_ensemble_executor
    ) -> None:
        """Test 'script' field takes priority if both are present."""
        executor = mock_ensemble_executor

        agent_config = {
            "name": "test_agent",
            "script": "echo 'Script'",
            "model_profile": "default-gpt4",  # Both present - script should win
        }

        with patch.object(executor, "_execute_script_agent") as mock_script:
            with patch.object(executor, "_execute_llm_agent") as mock_llm:
                mock_script.return_value = ("Script output", None)

                await executor._execute_agent(agent_config, "test input")

                mock_script.assert_called_once()
                mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_implicit_detection_with_enhanced_script_agent(
        self, mock_ensemble_executor
    ) -> None:
        """Test that implicit detection works with EnhancedScriptAgent."""
        executor = mock_ensemble_executor

        agent_config = {
            "name": "enhanced_script",
            "script": "scripts/test.py",
            "parameters": {"key": "value"},  # Enhanced script agent features
        }

        # Mock the enhanced script agent execution
        with patch(
            "llm_orc.core.execution.ensemble_execution.EnhancedScriptAgent"
        ) as mock_agent_class:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.execute.return_value = {
                "success": True,
                "result": "data",
            }
            mock_agent_class.return_value = mock_agent_instance

            await executor._execute_agent(agent_config, "test input")

            # Should use EnhancedScriptAgent for script agents
            mock_agent_class.assert_called_once_with("enhanced_script", agent_config)
            mock_agent_instance.execute.assert_called_once()

    def test_agent_type_detection_in_config_validation(self) -> None:
        """Test that agent type can be detected during configuration validation."""
        from llm_orc.core.config.ensemble_config import EnsembleConfig

        # Ensemble with mixed implicit agent types

        ensemble_config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble with implicit agent types",
            agents=[
                {"name": "script_agent", "script": "echo 'test'"},
                {
                    "name": "llm_agent",
                    "model_profile": "default-gpt4",
                    "system_prompt": "Test prompt",
                },
            ],
        )

        # Both agents should be valid without explicit 'type' field
        assert len(ensemble_config.agents) == 2
        assert ensemble_config.agents[0]["name"] == "script_agent"
        assert ensemble_config.agents[1]["name"] == "llm_agent"
