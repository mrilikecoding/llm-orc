"""Tests for configurable coordinator model functionality."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.models import OAuthClaudeModel, OllamaModel


class TestConfigurableCoordinator:
    """Test configurable coordinator model and role injection."""

    @pytest.fixture
    def executor(self) -> EnsembleExecutor:
        """Create ensemble executor."""
        return EnsembleExecutor()

    @pytest.fixture
    def basic_config(self) -> EnsembleConfig:
        """Create basic config without coordinator model."""
        return EnsembleConfig(
            name="test",
            description="Test ensemble",
            agents=[],
            coordinator={
                "synthesis_prompt": "Synthesize the results",
                "timeout_seconds": 30,
            },
        )

    @pytest.fixture
    def oauth_coordinator_config(self) -> EnsembleConfig:
        """Create config with OAuth coordinator model."""
        return EnsembleConfig(
            name="test-oauth-coordinator",
            description="Test ensemble with OAuth coordinator",
            agents=[],
            coordinator={
                "model": "anthropic-claude-pro-max",
                "system_prompt": (
                    "You are a strategic advisor with expertise in business analysis."
                ),
                "synthesis_prompt": (
                    "Synthesize the expert recommendations into actionable strategy."
                ),
                "timeout_seconds": 60,
            },
        )

    @pytest.mark.asyncio
    async def test_get_synthesis_model_defaults_to_configured_fallback(
        self, executor: EnsembleExecutor, basic_config: EnsembleConfig
    ) -> None:
        """Test that synthesis model uses configured default when no model specified."""
        synthesis_model = await executor._get_synthesis_model(basic_config)

        assert isinstance(synthesis_model, OllamaModel)
        # Should use configured default (claude-3-5-sonnet) as Ollama model
        # when no authentication is configured
        assert synthesis_model.model_name == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_get_synthesis_model_uses_configured_model(
        self,
        executor: EnsembleExecutor,
        oauth_coordinator_config: EnsembleConfig,
        monkeypatch: Any,
    ) -> None:
        """Test that synthesis model uses configured coordinator model."""
        # Mock the _load_model method to return a mock OAuth model
        mock_oauth_model = AsyncMock(spec=OAuthClaudeModel)
        mock_load_model = AsyncMock(return_value=mock_oauth_model)
        monkeypatch.setattr(executor, "_load_model", mock_load_model)

        synthesis_model = await executor._get_synthesis_model(oauth_coordinator_config)

        # Should call _load_model with the coordinator model
        mock_load_model.assert_called_once_with("anthropic-claude-pro-max")
        assert synthesis_model == mock_oauth_model

    @pytest.mark.asyncio
    async def test_synthesize_results_with_coordinator_role_injection(
        self,
        executor: EnsembleExecutor,
        oauth_coordinator_config: EnsembleConfig,
        monkeypatch: Any,
    ) -> None:
        """Test that synthesis uses coordinator role injection properly."""
        # Mock the synthesis model
        mock_synthesis_model = AsyncMock()
        mock_synthesis_model.generate_response.return_value = (
            "Synthesized result with role"
        )
        mock_synthesis_model.get_last_usage.return_value = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        # Mock _get_synthesis_model to return our mock
        mock_get_synthesis_model = AsyncMock(return_value=mock_synthesis_model)
        monkeypatch.setattr(executor, "_get_synthesis_model", mock_get_synthesis_model)

        # Test agent results
        agent_results = {
            "agent1": {"status": "success", "response": "Analysis from agent 1"},
            "agent2": {"status": "success", "response": "Analysis from agent 2"},
        }

        # Call synthesis
        result, model = await executor._synthesize_results(
            oauth_coordinator_config, agent_results
        )

        # Verify the call
        assert result == "Synthesized result with role"
        assert model == mock_synthesis_model

        # Check that generate_response was called with correct role and message
        mock_synthesis_model.generate_response.assert_called_once()
        call_args = mock_synthesis_model.generate_response.call_args

        # Role should be the coordinator's system_prompt
        expected_role = (
            "You are a strategic advisor with expertise in business analysis."
        )
        assert call_args.kwargs["role_prompt"] == expected_role

        # Message should contain synthesis instructions and agent results
        message = call_args.kwargs["message"]
        expected_synthesis = (
            "Synthesize the expert recommendations into actionable strategy."
        )
        assert expected_synthesis in message
        assert "agent1: Analysis from agent 1" in message
        assert "agent2: Analysis from agent 2" in message

    @pytest.mark.asyncio
    async def test_synthesize_results_backward_compatibility(
        self,
        executor: EnsembleExecutor,
        basic_config: EnsembleConfig,
        monkeypatch: Any,
    ) -> None:
        """Test synthesis with legacy config (no coordinator system_prompt)."""
        # Mock the synthesis model
        mock_synthesis_model = AsyncMock()
        mock_synthesis_model.generate_response.return_value = "Legacy synthesis result"
        mock_synthesis_model.get_last_usage.return_value = {
            "input_tokens": 80,
            "output_tokens": 40,
        }

        # Mock _get_synthesis_model to return our mock
        mock_get_synthesis_model = AsyncMock(return_value=mock_synthesis_model)
        monkeypatch.setattr(executor, "_get_synthesis_model", mock_get_synthesis_model)

        # Test agent results
        agent_results = {
            "agent1": {"status": "success", "response": "Result 1"},
        }

        # Call synthesis
        result, model = await executor._synthesize_results(basic_config, agent_results)

        # Verify the call
        assert result == "Legacy synthesis result"

        # Check that generate_response was called with legacy structure
        call_args = mock_synthesis_model.generate_response.call_args

        # Role should be the synthesis prompt itself (legacy behavior)
        assert call_args.kwargs["role_prompt"] == "Synthesize the results"

        # Message should be the fallback format
        message = call_args.kwargs["message"]
        assert "Please synthesize these results:" in message
        assert "agent1: Result 1" in message

    @pytest.mark.asyncio
    async def test_synthesize_results_handles_agent_errors(
        self,
        executor: EnsembleExecutor,
        oauth_coordinator_config: EnsembleConfig,
        monkeypatch: Any,
    ) -> None:
        """Test that synthesis properly handles agent errors in results."""
        # Mock the synthesis model
        mock_synthesis_model = AsyncMock()
        mock_synthesis_model.generate_response.return_value = (
            "Synthesis with error handling"
        )
        mock_synthesis_model.get_last_usage.return_value = {
            "input_tokens": 90,
            "output_tokens": 45,
        }

        mock_get_synthesis_model = AsyncMock(return_value=mock_synthesis_model)
        monkeypatch.setattr(executor, "_get_synthesis_model", mock_get_synthesis_model)

        # Test agent results with mixed success and error
        agent_results = {
            "agent1": {"status": "success", "response": "Successful analysis"},
            "agent2": {"status": "error", "error": "Failed to connect"},
        }

        # Call synthesis
        await executor._synthesize_results(oauth_coordinator_config, agent_results)

        # Check that the message includes both success and error results
        call_args = mock_synthesis_model.generate_response.call_args
        message = call_args.kwargs["message"]

        assert "agent1: Successful analysis" in message
        assert "agent2: [Error: Failed to connect]" in message

    @pytest.mark.asyncio
    async def test_coordinator_model_fallback_on_load_failure(
        self,
        executor: EnsembleExecutor,
        oauth_coordinator_config: EnsembleConfig,
        monkeypatch: Any,
    ) -> None:
        """Test that coordinator falls back to Ollama if model loading fails."""
        # Mock _load_model to raise an exception
        mock_load_model = AsyncMock(side_effect=Exception("Model loading failed"))
        monkeypatch.setattr(executor, "_load_model", mock_load_model)

        # Should fall back to Ollama without raising
        synthesis_model = await executor._get_synthesis_model(oauth_coordinator_config)

        # Should be Ollama fallback
        assert isinstance(synthesis_model, OllamaModel)
        assert synthesis_model.model_name == "llama3"

    def test_coordinator_config_validation(self) -> None:
        """Test that coordinator configurations are properly structured."""
        # Test with all coordinator options
        config = EnsembleConfig(
            name="full-coordinator-test",
            description="Test all coordinator options",
            agents=[],
            coordinator={
                "model": "anthropic-claude-pro-max",
                "system_prompt": "You are a CEO making strategic decisions.",
                "synthesis_prompt": (
                    "Synthesize all perspectives into a final decision."
                ),
                "timeout_seconds": 120,
                "output_format": "json",
            },
        )

        # Verify all fields are accessible
        assert config.coordinator["model"] == "anthropic-claude-pro-max"
        assert "CEO" in config.coordinator["system_prompt"]
        assert "Synthesize" in config.coordinator["synthesis_prompt"]
        assert config.coordinator["timeout_seconds"] == 120
        assert config.coordinator["output_format"] == "json"

    @pytest.mark.asyncio
    async def test_coordinator_with_oauth_model_integration(
        self, executor: EnsembleExecutor, monkeypatch: Any
    ) -> None:
        """Test end-to-end coordinator with OAuth model (integration test)."""
        # Create config with OAuth coordinator
        config = EnsembleConfig(
            name="oauth-integration-test",
            description="OAuth coordinator integration",
            agents=[],
            coordinator={
                "model": "anthropic-claude-pro-max",
                "system_prompt": (
                    "You are a venture capital partner evaluating investments."
                ),
                "synthesis_prompt": (
                    "Provide an investment recommendation with risk assessment."
                ),
                "timeout_seconds": 90,
            },
        )

        # Mock the OAuth model creation
        mock_oauth_model = AsyncMock(spec=OAuthClaudeModel)
        response_text = (
            "INVESTMENT RECOMMENDATION: PASS - "
            "High risk, insufficient market validation."
        )
        mock_oauth_model.generate_response.return_value = response_text
        mock_oauth_model.get_last_usage.return_value = {
            "input_tokens": 200,
            "output_tokens": 100,
            "cost_usd": 0.0,
        }

        mock_load_model = AsyncMock(return_value=mock_oauth_model)
        monkeypatch.setattr(executor, "_load_model", mock_load_model)

        # Test synthesis with realistic agent results
        agent_results = {
            "market_analyst": {
                "status": "success",
                "response": "Market size: $2B TAM, growing 15% annually",
            },
            "financial_analyst": {
                "status": "success",
                "response": "Unit economics unclear, high burn rate",
            },
            "tech_analyst": {
                "status": "success",
                "response": "Strong technical team, but unproven architecture",
            },
        }

        result, model = await executor._synthesize_results(config, agent_results)

        # Verify OAuth model was used
        assert model == mock_oauth_model
        assert "INVESTMENT RECOMMENDATION" in result

        # Verify proper role injection
        call_args = mock_oauth_model.generate_response.call_args
        assert "venture capital partner" in call_args.kwargs["role_prompt"]
        assert "investment recommendation" in call_args.kwargs["message"]
        assert "market_analyst:" in call_args.kwargs["message"]
        assert "financial_analyst:" in call_args.kwargs["message"]
        assert "tech_analyst:" in call_args.kwargs["message"]
