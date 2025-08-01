"""Tests for ensemble execution."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.models.anthropic import ClaudeCLIModel, ClaudeModel, OAuthClaudeModel
from llm_orc.models.base import ModelInterface


class TestEnsembleExecutor:
    """Test ensemble execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_ensemble(self) -> None:
        """Test executing a simple ensemble with mock agents."""
        # Create ensemble config
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble",
            agents=[
                {"name": "agent1", "role": "tester", "model": "mock-model"},
                {"name": "agent2", "role": "reviewer", "model": "mock-model"},
            ],
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.side_effect = [
            "Agent 1 response: This looks good",
            "Agent 2 response: I found some issues",
        ]
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        # Create role definitions
        role1 = RoleDefinition(name="tester", prompt="You are a tester")
        role2 = RoleDefinition(name="reviewer", prompt="You are a reviewer")

        # Create executor with mock dependencies
        executor = EnsembleExecutor()

        # No synthesis model needed in dependency-based architecture

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.side_effect = [role1, role2]
            mock_load_model.return_value = mock_model

            # Execute ensemble
            result = await executor.execute(config, input_data="Test this code")

        # Verify result structure
        assert result["ensemble"] == "test_ensemble"
        assert result["status"] == "completed"
        assert "input" in result
        assert "results" in result
        assert "metadata" in result
        # Synthesis field exists but is None in new architecture
        assert result["synthesis"] is None

        # Verify agent results
        agent_results = result["results"]
        assert len(agent_results) == 2
        assert "agent1" in agent_results
        assert "agent2" in agent_results

        # Verify metadata
        metadata = result["metadata"]
        assert "duration" in metadata
        assert metadata["agents_used"] == 2

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_different_models(self) -> None:
        """Test executing ensemble with different models per agent."""
        config = EnsembleConfig(
            name="multi_model_ensemble",
            description="Ensemble with different models",
            agents=[
                {"name": "claude_agent", "role": "analyst", "model": "claude-3-sonnet"},
                {"name": "local_agent", "role": "checker", "model": "llama3"},
            ],
        )

        # Mock different models
        claude_model = AsyncMock(spec=ModelInterface)
        claude_model.generate_response.return_value = "Claude analysis result"
        claude_model.get_last_usage.return_value = {
            "total_tokens": 40,
            "input_tokens": 25,
            "output_tokens": 15,
            "cost_usd": 0.008,
            "duration_ms": 80,
        }

        llama_model = AsyncMock(spec=ModelInterface)
        llama_model.generate_response.return_value = "Llama check result"
        llama_model.get_last_usage.return_value = {
            "total_tokens": 25,
            "input_tokens": 15,
            "output_tokens": 10,
            "cost_usd": 0.003,
            "duration_ms": 60,
        }

        # Mock role
        analyst_role = RoleDefinition(name="analyst", prompt="Analyze this")
        checker_role = RoleDefinition(name="checker", prompt="Check this")

        executor = EnsembleExecutor()

        # No synthesis model needed in dependency-based architecture

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.side_effect = [analyst_role, checker_role]
            mock_load_model.side_effect = [claude_model, llama_model]

            result = await executor.execute(config, input_data="Analyze this feature")

        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert "claude_agent" in result["results"]
        assert "local_agent" in result["results"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_handles_agent_failure(self) -> None:
        """Test that ensemble execution handles individual agent failures."""
        config = EnsembleConfig(
            name="test_ensemble_with_failure",
            description="Test ensemble with one failing agent",
            agents=[
                {"name": "working_agent", "role": "tester", "model": "mock-model"},
                {"name": "failing_agent", "role": "reviewer", "model": "mock-model"},
            ],
        )

        # Create mock models - one works, one fails
        working_model = AsyncMock(spec=ModelInterface)
        working_model.generate_response.return_value = "Working agent response"

        failing_model = AsyncMock(spec=ModelInterface)
        failing_model.generate_response.side_effect = Exception("Model failed")

        # Mock roles
        role = RoleDefinition(name="tester", prompt="You are a tester")

        executor = EnsembleExecutor()

        # No synthesis model needed in dependency-based architecture

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch.object(
                executor._model_factory,
                "get_fallback_model",
                new_callable=AsyncMock,
            ) as mock_fallback_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.side_effect = [working_model, failing_model]

            # Make fallback also fail to test original error handling
            fallback_model = AsyncMock(spec=ModelInterface)
            fallback_model.generate_response.side_effect = Exception(
                "Fallback also failed"
            )
            mock_fallback_model.return_value = fallback_model

            result = await executor.execute(config, input_data="Test input")

        # Should still complete but mark failures
        assert result["status"] == "completed_with_errors"
        assert len(result["results"]) == 2
        assert "working_agent" in result["results"]
        assert "failing_agent" in result["results"]

        # Working agent should have response
        assert "response" in result["results"]["working_agent"]

        # Failing agent should have error
        assert "error" in result["results"]["failing_agent"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_dependency_based(self) -> None:
        """Test that ensemble execution works with dependency-based approach."""
        config = EnsembleConfig(
            name="dependency_test",
            description="Test dependency-based functionality",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "mock-model"},
            ],
        )

        # Mock agent response
        agent_model = AsyncMock(spec=ModelInterface)
        agent_model.generate_response.return_value = "Detailed analysis result"

        role = RoleDefinition(name="analyst", prompt="Analyze")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = agent_model

            result = await executor.execute(config, input_data="Test analysis")

        # In dependency-based architecture, synthesis is None
        assert result["synthesis"] is None
        assert result["results"]["agent1"]["response"] == "Detailed analysis result"

    @pytest.mark.asyncio
    async def test_execute_ensemble_tracks_usage_metrics(self) -> None:
        """Test that ensemble execution tracks LLM usage metrics."""
        config = EnsembleConfig(
            name="usage_tracking_test",
            description="Test usage tracking",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "claude-3-sonnet"},
                {"name": "agent2", "role": "reviewer", "model": "llama3"},
            ],
        )

        # Mock models with usage tracking
        claude_model = AsyncMock(spec=ModelInterface)
        claude_model.generate_response.return_value = "Claude response"
        claude_model.get_last_usage.return_value = {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
            "cost_usd": 0.0045,
            "duration_ms": 1200,
            "model": "claude-3-sonnet",
        }

        llama_model = AsyncMock(spec=ModelInterface)
        llama_model.generate_response.return_value = "Llama response"
        llama_model.get_last_usage.return_value = {
            "input_tokens": 45,
            "output_tokens": 80,
            "total_tokens": 125,
            "cost_usd": 0.0,  # Local model, no cost
            "duration_ms": 800,
            "model": "llama3",
        }

        # No synthesis model needed in dependency-based architecture

        role = RoleDefinition(name="test", prompt="Test role")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.side_effect = [claude_model, llama_model]

            result = await executor.execute(config, input_data="Test usage tracking")

        # Verify usage tracking is included in results
        assert "usage" in result["metadata"]
        usage = result["metadata"]["usage"]

        # Check individual agent usage
        assert "agents" in usage
        agent_usage = usage["agents"]

        assert "agent1" in agent_usage
        assert agent_usage["agent1"]["model"] == "claude-3-sonnet"
        assert agent_usage["agent1"]["input_tokens"] == 50
        assert agent_usage["agent1"]["output_tokens"] == 100
        assert agent_usage["agent1"]["total_tokens"] == 150
        assert agent_usage["agent1"]["cost_usd"] == 0.0045
        assert agent_usage["agent1"]["duration_ms"] == 1200

        assert "agent2" in agent_usage
        assert agent_usage["agent2"]["model"] == "llama3"
        assert agent_usage["agent2"]["total_tokens"] == 125
        assert agent_usage["agent2"]["cost_usd"] == 0.0

        # Check totals (no synthesis in dependency-based architecture)
        assert "totals" in usage
        totals = usage["totals"]
        assert totals["total_tokens"] == 275  # 150 + 125 (no synthesis)
        assert totals["total_cost_usd"] == 0.0045  # 0.0045 + 0.0 (no synthesis)
        assert totals["total_duration_ms"] == 2000  # 1200 + 800 (no synthesis)
        assert totals["agents_count"] == 2

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_global_timeout(self) -> None:
        """Test that ensemble execution respects global timeout settings."""
        config = EnsembleConfig(
            name="timeout_test",
            description="Test timeout functionality",
            agents=[
                {
                    "name": "slow_agent",
                    "role": "analyst",
                    "model": "slow-model",
                    "timeout_seconds": 0.1,  # 100ms timeout at agent level
                },
            ],
        )

        # Mock model that takes too long
        slow_model = AsyncMock(spec=ModelInterface)

        async def slow_response(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(0.2)  # Takes 200ms, longer than 100ms timeout
            return "This should timeout"

        slow_model.generate_response = slow_response
        slow_model.get_last_usage.return_value = {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
            "cost_usd": 0.001,
            "duration_ms": 200,
            "model": "slow-model",
        }

        role = RoleDefinition(name="analyst", prompt="Analyze")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = slow_model

            result = await executor.execute(config, input_data="Test timeout")

        # Should complete with errors due to timeout
        assert result["status"] == "completed_with_errors"
        assert "slow_agent" in result["results"]
        agent_result = result["results"]["slow_agent"]
        assert agent_result["status"] == "failed"
        assert "timed out" in agent_result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_per_agent_timeout(self) -> None:
        """Test that individual agents can have their own timeout settings."""
        config = EnsembleConfig(
            name="per_agent_timeout_test",
            description="Test per-agent timeout functionality",
            agents=[
                {
                    "name": "fast_agent",
                    "role": "analyst",
                    "model": "fast-model",
                    "timeout_seconds": 1.0,
                },
                {
                    "name": "slow_agent",
                    "role": "reviewer",
                    "model": "slow-model",
                    "timeout_seconds": 0.05,  # 50ms timeout
                },
            ],
        )

        # Fast model
        fast_model = AsyncMock(spec=ModelInterface)
        fast_model.generate_response.return_value = "Fast response"
        fast_model.get_last_usage.return_value = {
            "input_tokens": 20,
            "output_tokens": 30,
            "total_tokens": 50,
            "cost_usd": 0.001,
            "duration_ms": 500,
            "model": "fast-model",
        }

        # Slow model that exceeds its timeout
        slow_model = AsyncMock(spec=ModelInterface)

        async def slow_response(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(0.1)  # Takes 100ms, longer than 50ms timeout
            return "This should timeout"

        slow_model.generate_response = slow_response
        slow_model.get_last_usage.return_value = {
            "input_tokens": 30,
            "output_tokens": 40,
            "total_tokens": 70,
            "cost_usd": 0.002,
            "duration_ms": 100,
            "model": "slow-model",
        }

        role = RoleDefinition(name="test", prompt="Test")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.side_effect = [fast_model, slow_model]

            result = await executor.execute(config, input_data="Test per-agent timeout")

        # Should complete with errors due to one agent timing out
        assert result["status"] == "completed_with_errors"

        # Fast agent should succeed
        assert result["results"]["fast_agent"]["status"] == "success"
        assert result["results"]["fast_agent"]["response"] == "Fast response"

        # Slow agent should fail with timeout
        assert result["results"]["slow_agent"]["status"] == "failed"
        assert "timed out" in result["results"]["slow_agent"]["error"].lower()

    # Synthesis timeout test removed - no synthesis in dependency-based arch

    @pytest.mark.asyncio
    async def test_load_model_with_authentication_configurations(self) -> None:
        """Test _load_model resolves auth configurations to model instances."""
        executor = EnsembleExecutor()

        # Mock the model factory's credential storage (not executor's)
        with (
            patch.object(
                executor._model_factory, "_credential_storage"
            ) as mock_storage,
        ):
            # Test 1: Load model for "anthropic-api" auth configuration
            mock_storage.get_auth_method.return_value = "api_key"
            mock_storage.get_api_key.return_value = "sk-ant-test123"

            model = await executor._model_factory.load_model("anthropic-api")

            # Should create ClaudeModel with API key
            assert isinstance(model, ClaudeModel)
            assert model.api_key == "sk-ant-test123"

            # Test 2: Load model for "anthropic-claude-pro-max" OAuth configuration
            mock_storage.get_auth_method.return_value = "oauth"
            mock_storage.get_oauth_token.return_value = {
                "access_token": "oauth_access_token",
                "refresh_token": "oauth_refresh_token",
                "client_id": "oauth_client_id",
            }

            model = await executor._model_factory.load_model("anthropic-claude-pro-max")

            # Should create OAuthClaudeModel
            assert isinstance(model, OAuthClaudeModel)
            assert model.access_token == "oauth_access_token"
            assert model.refresh_token == "oauth_refresh_token"
            assert model.client_id == "oauth_client_id"

            # Test 3: Load model for "claude-cli" configuration
            # claude-cli stores path as "api_key"
            mock_storage.get_auth_method.return_value = "api_key"
            mock_storage.get_api_key.return_value = "/usr/local/bin/claude"

            model = await executor._model_factory.load_model("claude-cli")

            # Should create ClaudeCLIModel
            assert isinstance(model, ClaudeCLIModel)
            assert model.claude_path == "/usr/local/bin/claude"

    @pytest.mark.asyncio
    async def test_load_model_prompts_for_auth_setup_when_not_configured(self) -> None:
        """Test that _load_model prompts user to set up auth when not configured."""
        executor = EnsembleExecutor()

        # Mock authentication system - no auth method configured
        with (
            patch(
                "llm_orc.core.models.model_factory._should_prompt_for_auth",
                return_value=True,
            ),
            patch(
                "llm_orc.core.models.model_factory._prompt_auth_setup"
            ) as mock_prompt_setup,
            patch.object(
                executor._model_factory, "_credential_storage"
            ) as mock_storage,
        ):
            # Simulate no auth method configured
            mock_storage.get_auth_method.return_value = None

            # Mock successful auth setup
            mock_prompt_setup.return_value = True

            # After setup, mock the configured auth method
            # First call: None, second call: oauth
            mock_storage.get_auth_method.side_effect = [None, "oauth"]
            mock_storage.get_oauth_token.return_value = {
                "access_token": "new_oauth_token",
                "refresh_token": "new_refresh_token",
                "client_id": "new_client_id",
            }

            model = await executor._model_factory.load_model("anthropic-claude-pro-max")

            # Should prompt for auth setup
            mock_prompt_setup.assert_called_once_with(
                "anthropic-claude-pro-max", mock_storage
            )

            # Should create OAuthClaudeModel after setup
            assert isinstance(model, OAuthClaudeModel)

    @pytest.mark.asyncio
    async def test_load_model_fallback_when_user_declines_auth_setup(self) -> None:
        """Test that _load_model falls back to Ollama when user declines auth setup."""
        executor = EnsembleExecutor()

        # Mock authentication system - no auth method configured
        with (
            patch(
                "llm_orc.core.models.model_factory._should_prompt_for_auth",
                return_value=True,
            ),
            patch(
                "llm_orc.core.models.model_factory._prompt_auth_setup"
            ) as mock_prompt_setup,
            patch.object(
                executor._model_factory, "_credential_storage"
            ) as mock_storage,
        ):
            # Simulate no auth method configured
            mock_storage.get_auth_method.return_value = None

            # User declines to set up authentication
            mock_prompt_setup.return_value = False

            model = await executor._model_factory.load_model("anthropic-claude-pro-max")

            # Should prompt user for auth setup
            mock_prompt_setup.assert_called_once_with(
                "anthropic-claude-pro-max", mock_storage
            )

            # Should fall back to Ollama
            from llm_orc.models.ollama import OllamaModel

            assert isinstance(model, OllamaModel)

    def test_should_prompt_for_auth_returns_true_for_known_providers(self) -> None:
        """Test that _should_prompt_for_auth returns True for known provider configs."""
        from llm_orc.core.models.model_factory import _should_prompt_for_auth

        # Should return True for known provider configurations
        assert _should_prompt_for_auth("anthropic-api") is True
        assert _should_prompt_for_auth("anthropic-claude-pro-max") is True
        assert _should_prompt_for_auth("claude-cli") is True
        assert _should_prompt_for_auth("openai-api") is True
        assert _should_prompt_for_auth("google-gemini") is True

    def test_should_prompt_for_auth_returns_false_for_mock_models(self) -> None:
        """Test that _should_prompt_for_auth returns False for mock/local models."""
        from llm_orc.core.models.model_factory import _should_prompt_for_auth

        # Should return False for mock models and local models
        assert _should_prompt_for_auth("mock-model") is False
        assert _should_prompt_for_auth("mock-claude") is False
        assert _should_prompt_for_auth("llama3") is False
        assert _should_prompt_for_auth("llama2") is False
        assert _should_prompt_for_auth("unknown-model") is False

    @pytest.mark.asyncio
    async def test_ensemble_execution_with_model_profile(self) -> None:
        """Test ensemble execution using model_profile.

        Uses model_profile instead of explicit model+provider.
        """
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import yaml

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config with model profiles
            global_dir = temp_path / "global"
            global_dir.mkdir()
            config_data = {
                "model_profiles": {
                    "test-profile": {
                        "model": "claude-3-5-sonnet-20241022",
                        "provider": "anthropic-claude-pro-max",
                    }
                }
            }
            config_file = global_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            executor = EnsembleExecutor()

            # Test the _load_model_from_agent_config method directly
            with patch.object(
                executor._config_manager, "resolve_model_profile"
            ) as mock_resolve_model_profile:
                mock_resolve_model_profile.return_value = (
                    "claude-3-5-sonnet-20241022",
                    "anthropic-claude-pro-max",
                )

                with patch.object(
                    executor._model_factory, "_credential_storage"
                ) as mock_credential_storage:
                    # Mock auth method to prevent fallback logic
                    mock_credential_storage.get_auth_method.return_value = "oauth"
                    mock_credential_storage.get_oauth_token.return_value = {
                        "access_token": "test_token",
                        "refresh_token": "test_refresh",
                        "client_id": "test_client_id",
                    }

                    # This should call resolve_model_profile and use the resolved
                    # model+provider
                    # Note: The method may not raise an error due to fallback logic,
                    # but should call resolve_model_profile
                    await executor._model_factory.load_model_from_agent_config(
                        {"name": "agent1", "model_profile": "test-profile"}
                    )

                    # Verify that resolve_model_profile was called
                    mock_resolve_model_profile.assert_called_once_with("test-profile")

    @pytest.mark.asyncio
    async def test_ensemble_execution_fallback_to_explicit_model_provider(
        self,
    ) -> None:
        """Test fallback to explicit model+provider when no model_profile."""
        executor = EnsembleExecutor()

        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager"
        ) as mock_config_manager_class:
            mock_config_manager = mock_config_manager_class.return_value

            with patch(
                "llm_orc.core.auth.authentication.CredentialStorage"
            ) as mock_credential_storage:
                mock_storage_instance = mock_credential_storage.return_value
                mock_storage_instance.get_auth_method.return_value = None

                # This should use explicit model+provider, not call
                # resolve_model_profile
                await executor._model_factory.load_model_from_agent_config(
                    {
                        "name": "agent1",
                        "model": "claude-3-5-sonnet-20241022",
                        "provider": "anthropic-claude-pro-max",
                    }
                )

                # Verify that resolve_model_profile was NOT called
                mock_config_manager.resolve_model_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_dependency_based_ensemble(self) -> None:
        """Test executing ensemble with agent dependencies instead of coordinator."""
        # RED: This test should fail until we implement dependency execution
        config = EnsembleConfig(
            name="dependency_test",
            description="Test dependency-based execution",
            agents=[
                {
                    "name": "researcher",
                    "role": "researcher",
                    "model": "mock-model",
                },
                {
                    "name": "analyzer",
                    "role": "analyzer",
                    "model": "mock-model",
                },
                {
                    "name": "synthesizer",
                    "role": "synthesizer",
                    "model": "mock-model",
                    "depends_on": ["researcher", "analyzer"],
                },
            ],
        )

        # Create mock models with predictable responses
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.side_effect = [
            "Research findings: Data collected",
            "Analysis results: Patterns identified",
            "Synthesis: Combined research and analysis",
        ]
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        # Create role definitions
        role = RoleDefinition(name="test_role", prompt="You are an agent")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            result = await executor.execute(config, input_data="Test input")

        # Verify dependency-based execution
        assert result["status"] == "completed"
        assert len(result["results"]) == 3
        assert "researcher" in result["results"]
        assert "analyzer" in result["results"]
        assert "synthesizer" in result["results"]

        # Verify synthesizer executed after dependencies
        expected_response = "Synthesis: Combined research and analysis"
        assert result["results"]["synthesizer"]["response"] == expected_response
        assert result["results"]["synthesizer"]["status"] == "success"

        # Should not have old coordinator-style synthesis
        assert result["synthesis"] is None

    @pytest.mark.asyncio
    async def test_load_model_from_agent_config_delegation(self) -> None:
        """Test _load_model_from_agent_config delegates to model factory."""
        executor = EnsembleExecutor()

        mock_model = AsyncMock(spec=ModelInterface)
        agent_config = {"name": "test_agent", "model": "test-model"}

        with patch.object(
            executor._model_factory,
            "load_model_from_agent_config",
            new_callable=AsyncMock,
        ) as mock_load:
            mock_load.return_value = mock_model

            result = await executor._load_model_from_agent_config(agent_config)

            assert result == mock_model
            mock_load.assert_called_once_with(agent_config)

    @pytest.mark.asyncio
    async def test_execute_streaming_with_progress_updates(self) -> None:
        """Test streaming execution yields progress updates."""
        config = EnsembleConfig(
            name="streaming_test",
            description="Test streaming execution",
            agents=[
                {"name": "agent1", "role": "tester", "model": "mock-model"},
                {"name": "agent2", "role": "reviewer", "model": "mock-model"},
            ],
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.side_effect = [
            "Agent 1 response",
            "Agent 2 response",
        ]
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="test", prompt="Test role")
        executor = EnsembleExecutor()

        # Mock dependencies
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            # Collect streaming events
            events = []
            async for event in executor.execute_streaming(config, "Test input"):
                events.append(event)

        # Verify we got the expected events
        assert len(events) >= 2  # At least started and completed

        # Check execution started event
        start_event = events[0]
        assert start_event["type"] == "execution_started"
        assert start_event["data"]["ensemble"] == "streaming_test"
        assert start_event["data"]["total_agents"] == 2

        # Check execution completed event (may not be last due to performance events)
        completion_events = [e for e in events if e["type"] == "execution_completed"]
        assert len(completion_events) == 1, (
            f"Expected 1 execution_completed event, got {len(completion_events)}"
        )

        completion_event = completion_events[0]
        assert completion_event["data"]["ensemble"] == "streaming_test"
        assert completion_event["data"]["status"] == "completed"
        assert "results" in completion_event["data"]

    @pytest.mark.asyncio
    async def test_resolve_model_profile_to_config(self) -> None:
        """Test model profile resolution to enhanced config."""
        executor = EnsembleExecutor()

        # Mock model profiles in config manager
        mock_profiles = {
            "test-profile": {
                "model": "claude-3-5-sonnet",
                "provider": "anthropic-claude-pro-max",
                "temperature": 0.7,
            }
        }

        with patch.object(
            executor._config_manager, "get_model_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = mock_profiles

            # Test with model_profile specified
            agent_config = {
                "name": "test_agent",
                "model_profile": "test-profile",
                "temperature": 0.9,  # Should override profile
            }

            enhanced = await executor._resolve_model_profile_to_config(agent_config)

            # Should merge profile with agent config (agent takes precedence)
            assert enhanced["model"] == "claude-3-5-sonnet"
            assert enhanced["provider"] == "anthropic-claude-pro-max"
            assert enhanced["temperature"] == 0.9  # Agent override
            assert enhanced["name"] == "test_agent"

    @pytest.mark.asyncio
    async def test_resolve_model_profile_nonexistent_profile(self) -> None:
        """Test model profile resolution with nonexistent profile."""
        executor = EnsembleExecutor()

        with patch.object(
            executor._config_manager, "get_model_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {}  # No profiles

            agent_config = {
                "name": "test_agent",
                "model_profile": "nonexistent-profile",
            }

            # Should return original config when profile doesn't exist
            enhanced = await executor._resolve_model_profile_to_config(agent_config)
            assert enhanced == agent_config

    @pytest.mark.asyncio
    async def test_execute_script_agents(self) -> None:
        """Test execution of script agents."""
        config = EnsembleConfig(
            name="script_test",
            description="Test script agent execution",
            agents=[
                {
                    "name": "script_agent",
                    "type": "script",
                    "script": "echo 'Script output'",
                    "role": "data_collector",
                },
                {"name": "llm_agent", "role": "analyzer", "model": "mock-model"},
            ],
        )

        executor = EnsembleExecutor()

        # Create mock results dict to collect script results
        results_dict: dict[str, Any] = {}

        # Mock script agent execution
        with patch.object(
            executor, "_execute_agent_with_timeout", new_callable=AsyncMock
        ) as mock_execute_timeout:
            mock_execute_timeout.return_value = ("Script output", None)

            with patch.object(
                executor,
                "_resolve_model_profile_to_config",
                return_value={
                    "name": "script_agent",
                    "type": "script",
                    "script": "echo 'Script output'",
                    "timeout_seconds": 60,
                },
            ):
                context_data, has_errors = await executor._execute_script_agents(
                    config, "Test input", results_dict
                )

        # Verify script results
        assert has_errors is False
        assert "script_agent" in results_dict
        assert results_dict["script_agent"]["status"] == "success"
        assert results_dict["script_agent"]["response"] == "Script output"

        # Verify context data contains script results
        assert "script_agent" in context_data
        assert context_data["script_agent"] == "Script output"

    @pytest.mark.asyncio
    async def test_execute_script_agents_with_error(self) -> None:
        """Test script agent execution with error handling."""
        config = EnsembleConfig(
            name="script_error_test",
            description="Test script agent error handling",
            agents=[
                {
                    "name": "failing_script",
                    "type": "script",
                    "script": "exit 1",
                    "role": "data_collector",
                },
            ],
        )

        executor = EnsembleExecutor()
        results_dict: dict[str, Any] = {}

        # Mock script agent failure
        with patch.object(
            executor, "_execute_agent_with_timeout", new_callable=AsyncMock
        ) as mock_execute_timeout:
            mock_execute_timeout.side_effect = RuntimeError("Script failed")

            with patch.object(executor, "_resolve_model_profile_to_config"):
                context_data, has_errors = await executor._execute_script_agents(
                    config, "Test input", results_dict
                )

        # Verify error handling
        assert has_errors is True
        assert "failing_script" in results_dict
        assert results_dict["failing_script"]["status"] == "failed"
        assert "Script failed" in results_dict["failing_script"]["error"]

        # Context data should be empty when script fails
        assert context_data == {}

    @pytest.mark.asyncio
    async def test_load_role_creates_default_role(self) -> None:
        """Test _load_role creates default role definition."""
        # Mock dependencies to avoid YAML loading affected by test contamination
        with (
            patch("llm_orc.core.execution.ensemble_execution.ConfigurationManager"),
            patch("llm_orc.core.execution.ensemble_execution.CredentialStorage"),
        ):
            executor = EnsembleExecutor()

            role = await executor._load_role("test_analyst")

            assert isinstance(role, RoleDefinition)
            assert role.name == "test_analyst"
            assert "test_analyst" in role.prompt
            assert "helpful analysis" in role.prompt

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_no_timeout(self) -> None:
        """Test _execute_agent_with_timeout with no timeout specified."""
        executor = EnsembleExecutor()

        agent_config = {"name": "test_agent", "model": "mock-model"}
        input_data = "Test input"

        # Mock _execute_agent to return expected result
        with patch.object(
            executor, "_execute_agent", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = ("Agent response", None)

            result, model = await executor._execute_agent_with_timeout(
                agent_config, input_data, None
            )

            assert result == "Agent response"
            assert model is None
            mock_execute.assert_called_once_with(agent_config, input_data)

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_timeout_occurs(self) -> None:
        """Test _execute_agent_with_timeout when timeout occurs."""
        executor = EnsembleExecutor()

        agent_config = {"name": "test_agent", "model": "mock-model"}
        input_data = "Test input"

        # Mock the execution coordinator to raise the timeout exception directly
        with patch.object(
            executor._execution_coordinator,
            "execute_agent_with_timeout",
            new_callable=AsyncMock,
        ) as mock_execute:
            # Make the coordinator raise the timeout exception
            timeout_msg = "Agent execution timed out after 1 seconds"
            mock_execute.side_effect = Exception(timeout_msg)

            # Test with timeout that should trigger the timeout exception
            with pytest.raises(Exception, match="timed out after 1 seconds"):
                await executor._execute_agent_with_timeout(
                    agent_config,
                    input_data,
                    1,  # 1 second timeout
                )

    @pytest.mark.asyncio
    async def test_analyze_dependencies(self) -> None:
        """Test _analyze_dependencies separates agents correctly."""
        executor = EnsembleExecutor()

        llm_agents: list[dict[str, Any]] = [
            {"name": "independent1", "role": "analyst", "model": "mock-model"},
            {"name": "independent2", "role": "reviewer", "model": "mock-model"},
            {
                "name": "dependent1",
                "role": "synthesizer",
                "model": "mock-model",
                "depends_on": ["independent1", "independent2"],
            },
            {
                "name": "dependent2",
                "role": "summarizer",
                "model": "mock-model",
                "depends_on": ["dependent1"],
            },
        ]

        independent, dependent = executor._analyze_dependencies(llm_agents)

        assert len(independent) == 2
        assert len(dependent) == 2

        # Check independent agents
        independent_names = [agent["name"] for agent in independent]
        assert "independent1" in independent_names
        assert "independent2" in independent_names

        # Check dependent agents
        dependent_names = [agent["name"] for agent in dependent]
        assert "dependent1" in dependent_names
        assert "dependent2" in dependent_names

    @pytest.mark.asyncio
    async def test_analyze_dependencies_empty_depends_on(self) -> None:
        """Test _analyze_dependencies with empty depends_on list."""
        executor = EnsembleExecutor()

        llm_agents: list[dict[str, Any]] = [
            {"name": "agent1", "role": "analyst", "model": "mock-model"},
            {
                "name": "agent2",
                "role": "reviewer",
                "model": "mock-model",
                "depends_on": [],  # Empty dependencies
            },
        ]

        independent, dependent = executor._analyze_dependencies(llm_agents)

        # Both should be independent since empty depends_on means no dependencies
        assert len(independent) == 2
        assert len(dependent) == 0

    @pytest.mark.asyncio
    async def test_resolve_model_profile_to_config_without_profile(self) -> None:
        """Test model profile resolution without model_profile key."""
        executor = EnsembleExecutor()

        agent_config = {"name": "test_agent", "model": "claude-3-sonnet"}

        # Should return copy of original config when no model_profile
        enhanced = await executor._resolve_model_profile_to_config(agent_config)
        assert enhanced == agent_config
        assert enhanced is not agent_config  # Should be a copy

    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self) -> None:
        """Test that parallel execution is significantly faster than sequential.

        RED: This test should fail until we implement parallel execution.
        """
        import time

        # Create ensemble with multiple agents that have simulated latency
        config = EnsembleConfig(
            name="parallel_test",
            description="Test parallel execution performance",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "mock-model"},
                {"name": "agent2", "role": "reviewer", "model": "mock-model"},
                {"name": "agent3", "role": "synthesizer", "model": "mock-model"},
            ],
        )

        # Create mock model with simulated latency
        mock_model = AsyncMock(spec=ModelInterface)

        async def slow_response(*args: Any, **kwargs: Any) -> str:
            # Simulate 0.5 second LLM API call latency
            await asyncio.sleep(0.5)
            return "Agent response after delay"

        mock_model.generate_response = slow_response
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 500,
        }

        role = RoleDefinition(name="test", prompt="Test role")
        executor = EnsembleExecutor()

        # Mock dependencies
        with (
            patch.object(
                executor, "_load_role_from_config", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            # Measure execution time
            start_time = time.time()
            result = await executor.execute(config, "Test input")
            execution_time = time.time() - start_time

        # Verify results
        assert result["status"] == "completed"
        assert len(result["results"]) == 3

        # With 3 agents each taking 0.5s:
        # - Sequential execution should take ~1.5s
        # - Parallel execution should take ~0.5s
        # We expect execution time to be closer to parallel time (< 1.0s)
        expected_parallel_time = 0.5
        expected_sequential_time = 1.5

        # RED: This assertion should fail with current sequential implementation
        # The test expects parallel execution performance
        assert execution_time < (expected_parallel_time + 0.3), (
            f"Execution took {execution_time:.2f}s, "
            f"expected ~{expected_parallel_time:.2f}s for parallel execution "
            f"(sequential would take ~{expected_sequential_time:.2f}s)"
        )

    @pytest.mark.asyncio
    async def test_oauth_fallback_display_enhancement(self) -> None:
        """Test enhanced OAuth fallback display with specific error messaging.

        This test ensures that OAuth token refresh failures are properly caught
        and displayed with clear model profile and fallback information.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleConfig
        from llm_orc.models.anthropic import OAuthClaudeModel

        executor = EnsembleExecutor()

        # Create test ensemble config with OAuth model
        config = EnsembleConfig(
            name="oauth-fallback-test",
            description="Test OAuth fallback display enhancement",
            agents=[
                {
                    "name": "oauth-agent",
                    "model_profile": "premium-claude",  # Uses OAuth authentication
                    "system_prompt": "You are a test agent.",
                }
            ],
        )

        # Create mock OAuth model that will fail with token refresh error
        mock_oauth_model = Mock(spec=OAuthClaudeModel)
        mock_oauth_model.generate_response = AsyncMock()
        mock_oauth_model.generate_response.side_effect = Exception(
            "OAuth token refresh failed with status 400: "
            '{"error": "invalid_grant", "error_description": "Refresh token not found"}'
        )
        # Mock the usage collection method to avoid TypeError
        mock_oauth_model.get_last_usage.return_value = {
            "input_tokens": 10,
            "output_tokens": 5,
            "duration_ms": 100,
            "cost_usd": 0.01,
        }

        # Create mock fallback model that succeeds
        mock_fallback_model = Mock()
        mock_fallback_model.generate_response = AsyncMock(
            return_value="Fallback response"
        )
        mock_fallback_model.model_name = "llama3"
        mock_fallback_model.get_last_usage.return_value = {
            "input_tokens": 15,
            "output_tokens": 10,
            "duration_ms": 200,
            "cost_usd": 0.02,
        }

        # Track streaming events (Phase 5: unified event system)
        streaming_events: list[dict[str, Any]] = []

        with (
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                return_value=mock_oauth_model,
            ),
            patch.object(
                executor._model_factory,
                "get_fallback_model",
                return_value=mock_fallback_model,
            ),
            patch.object(
                executor,
                "_load_role_from_config",
                new_callable=AsyncMock,
                return_value=Mock(name="oauth-agent", prompt="Test prompt"),
            ),
        ):
            # Use streaming execution to capture events (Phase 5: unified event system)
            async for event in executor.execute_streaming(
                config, "Test OAuth fallback"
            ):
                streaming_events.append(event)

        # Verify the ensemble completed successfully with fallback
        completion_events = [
            e for e in streaming_events if e["type"] == "execution_completed"
        ]
        assert len(completion_events) == 1
        result = completion_events[0]["data"]
        assert result["status"] == "completed"
        assert result["results"]["oauth-agent"]["status"] == "success"
        assert result["results"]["oauth-agent"]["response"] == "Fallback response"

        # Verify enhanced OAuth fallback event was emitted through unified system
        oauth_fallback_events = [
            e for e in streaming_events if e["type"] == "agent_fallback_started"
        ]

        assert len(oauth_fallback_events) == 1, (
            f"Expected 1 OAuth fallback event, got {len(oauth_fallback_events)}. "
            f"Events: {[e['type'] for e in streaming_events]}"
        )

        event_data = oauth_fallback_events[0]["data"]

        # Verify enhanced OAuth fallback event contains detailed information
        assert event_data["agent_name"] == "oauth-agent"
        assert event_data["original_model_profile"] == "premium-claude"
        assert event_data["fallback_model_name"] == "llama3"
        assert "OAuth token refresh failed" in event_data["original_error"]
        assert "invalid_grant" in event_data["original_error"]
        assert event_data["failure_type"] == "oauth_error"

    @pytest.mark.asyncio
    async def test_model_loading_fallback_display_enhancement(self) -> None:
        """Test enhanced model loading fallback display with specific error messaging.

        This test ensures that model loading failures (provider unavailable,
        model not found, etc.) are properly caught and displayed with clear
        model profile and fallback information.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleConfig

        executor = EnsembleExecutor()

        # Create test ensemble config with model that will fail to load
        config = EnsembleConfig(
            name="model-loading-fallback-test",
            description="Test model loading fallback display enhancement",
            agents=[
                {
                    "name": "failing-agent",
                    "model_profile": "guaranteed-fail",  # Profile designed to fail
                    "system_prompt": "You are a test agent.",
                }
            ],
        )

        # Create mock fallback model that succeeds
        mock_fallback_model = Mock()
        mock_fallback_model.generate_response = AsyncMock(
            return_value="Model loading fallback response"
        )
        mock_fallback_model.model_name = "llama3"
        mock_fallback_model.get_last_usage.return_value = {
            "input_tokens": 15,
            "output_tokens": 10,
            "duration_ms": 200,
            "cost_usd": 0.02,
        }

        # Track streaming events (Phase 5: unified event system)
        streaming_events: list[dict[str, Any]] = []

        with (
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                side_effect=Exception(
                    "Model provider 'nonexistent-provider' not available. "
                    "Model 'this-model-definitely-does-not-exist' not found."
                ),
            ),
            patch.object(
                executor._model_factory,
                "get_fallback_model",
                return_value=mock_fallback_model,
            ),
            patch.object(
                executor,
                "_load_role_from_config",
                new_callable=AsyncMock,
                return_value=Mock(name="failing-agent", prompt="Test prompt"),
            ),
        ):
            # Use streaming execution to capture events (Phase 5: unified event system)
            async for event in executor.execute_streaming(
                config, "Test model loading fallback"
            ):
                streaming_events.append(event)

        # Verify the ensemble completed successfully with fallback
        completion_events = [
            e for e in streaming_events if e["type"] == "execution_completed"
        ]
        assert len(completion_events) == 1
        result = completion_events[0]["data"]
        assert result["status"] == "completed"
        assert result["results"]["failing-agent"]["status"] == "success"
        assert (
            result["results"]["failing-agent"]["response"]
            == "Model loading fallback response"
        )

        # Verify enhanced model loading fallback event was emitted through unified
        # system
        loading_fallback_events = [
            e for e in streaming_events if e["type"] == "agent_fallback_started"
        ]

        assert len(loading_fallback_events) == 1, (
            f"Expected 1 model loading fallback event, "
            f"got {len(loading_fallback_events)}. "
            f"Events: {[e['type'] for e in streaming_events]}"
        )

        event_data = loading_fallback_events[0]["data"]

        # Verify enhanced model loading fallback event contains detailed information
        assert event_data["agent_name"] == "failing-agent"
        assert event_data["original_model_profile"] == "guaranteed-fail"
        assert event_data["fallback_model_name"] == "llama3"
        assert "Model provider" in event_data["original_error"]
        assert "not available" in event_data["original_error"]
        assert event_data["failure_type"] == "model_loading"

    @pytest.mark.asyncio
    async def test_emit_performance_event_queue_full_exception(self) -> None:
        """Test _emit_performance_event handles QueueFull exception gracefully.

        This covers lines 97-99.
        """
        executor = EnsembleExecutor()

        # Mock the streaming event queue to be full
        with patch.object(executor, "_streaming_event_queue") as mock_queue:
            mock_queue.put_nowait.side_effect = asyncio.QueueFull("Queue is full")

            # Should not raise exception, should silently ignore
            executor._emit_performance_event("test_event", {"test": "data"})

            # Verify that put_nowait was attempted
            mock_queue.put_nowait.assert_called_once()
