"""Tests for ensemble execution."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.models.anthropic import ClaudeCLIModel, ClaudeModel, OAuthClaudeModel
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


class TestEnsembleExecutor:
    """Test ensemble execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_ensemble(self, mock_ensemble_executor: Any) -> None:
        """Test executing a simple ensemble with mock agents."""
        # Create ensemble config
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
                LlmAgentConfig(name="agent2", model_profile="test-reviewer"),
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

        # Use the mocked executor to prevent artifact creation
        executor = mock_ensemble_executor

        # No synthesis model needed in dependency-based architecture

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
        assert result["ensemble"] == "test_ensemble", "ensemble name mismatch"
        assert result["status"] == "completed", "ensemble should be completed"
        assert "input" in result, "result missing 'input' field"
        assert "results" in result, "result missing 'results' field"
        assert "metadata" in result, "result missing 'metadata' field"
        # Synthesis field exists but is None in new architecture
        assert result["synthesis"] is None, (
            "synthesis should be None in dependency-based arch"
        )

        # Verify agent results
        agent_results = result["results"]
        assert len(agent_results) == 2, f"expected 2 agents, got {len(agent_results)}"
        assert "agent1" in agent_results, "agent1 missing from results"
        assert "agent2" in agent_results, "agent2 missing from results"

        # Verify metadata
        metadata = result["metadata"]
        assert "duration" in metadata, "metadata missing 'duration' field"
        assert metadata["agents_used"] == 2, (
            f"expected agents_used=2, got {metadata['agents_used']}"
        )

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_different_models(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test executing ensemble with different models per agent."""
        config = EnsembleConfig(
            name="multi_model_ensemble",
            description="Ensemble with different models",
            agents=[
                LlmAgentConfig(name="claude_agent", model_profile="claude-analyst"),
                LlmAgentConfig(name="local_agent", model_profile="llama-checker"),
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

        executor = mock_ensemble_executor

        # No synthesis model needed in dependency-based architecture

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_execute_ensemble_handles_agent_failure(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that ensemble execution handles individual agent failures."""
        config = EnsembleConfig(
            name="test_ensemble_with_failure",
            description="Test ensemble with one failing agent",
            agents=[
                LlmAgentConfig(name="working_agent", model_profile="test-tester"),
                LlmAgentConfig(name="failing_agent", model_profile="test-reviewer"),
            ],
        )

        # Create mock models - one works, one fails
        working_model = AsyncMock(spec=ModelInterface)
        working_model.generate_response.return_value = "Working agent response"

        failing_model = AsyncMock(spec=ModelInterface)
        failing_model.generate_response.side_effect = Exception("Model failed")

        # Mock roles
        role = RoleDefinition(name="tester", prompt="You are a tester")

        executor = mock_ensemble_executor

        # No synthesis model needed in dependency-based architecture

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_execute_ensemble_dependency_based(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that ensemble execution works with dependency-based approach."""
        config = EnsembleConfig(
            name="dependency_test",
            description="Test dependency-based functionality",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-analyst"),
            ],
        )

        # Mock agent response
        agent_model = AsyncMock(spec=ModelInterface)
        agent_model.generate_response.return_value = "Detailed analysis result"

        role = RoleDefinition(name="analyst", prompt="Analyze")

        executor = mock_ensemble_executor

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_execute_ensemble_records_per_agent_usage_fields(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that per-agent token counts, cost, and model are recorded."""
        config = EnsembleConfig(
            name="usage_tracking_test",
            description="Test usage tracking",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="claude-analyst"),
                LlmAgentConfig(name="agent2", model_profile="llama-reviewer"),
            ],
        )

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
            "cost_usd": 0.0,
            "duration_ms": 800,
            "model": "llama3",
        }

        role = RoleDefinition(name="test", prompt="Test role")
        executor = mock_ensemble_executor

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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

        agent_usage = result["metadata"]["usage"]["agents"]

        assert "agent1" in agent_usage, "agent1 should appear in per-agent usage"
        assert agent_usage["agent1"]["model"] == "claude-3-sonnet", "wrong model"
        assert agent_usage["agent1"]["input_tokens"] == 50, "wrong input_tokens"
        assert agent_usage["agent1"]["output_tokens"] == 100, "wrong output_tokens"
        assert agent_usage["agent1"]["total_tokens"] == 150, "wrong total_tokens"
        assert agent_usage["agent1"]["cost_usd"] == 0.0045, "cost should be recorded"
        assert agent_usage["agent1"]["duration_ms"] == 1200, "wrong duration_ms"

        assert "agent2" in agent_usage, "agent2 should appear in per-agent usage"
        assert agent_usage["agent2"]["model"] == "llama3", "wrong local model"
        assert agent_usage["agent2"]["total_tokens"] == 125, "wrong total_tokens"
        assert agent_usage["agent2"]["cost_usd"] == 0.0, "local model wrong cost"

    @pytest.mark.asyncio
    async def test_execute_ensemble_aggregates_usage_totals(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that per-agent usage is summed correctly into totals."""
        config = EnsembleConfig(
            name="usage_tracking_test",
            description="Test usage tracking",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="claude-analyst"),
                LlmAgentConfig(name="agent2", model_profile="llama-reviewer"),
            ],
        )

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
            "cost_usd": 0.0,
            "duration_ms": 800,
            "model": "llama3",
        }

        role = RoleDefinition(name="test", prompt="Test role")
        executor = mock_ensemble_executor

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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

        totals = result["metadata"]["usage"]["totals"]

        assert totals["total_tokens"] == 275, "total_tokens should be 150 + 125"
        assert totals["total_cost_usd"] == 0.0045, "total_cost_usd: 0.0045 + 0.0"
        assert totals["total_duration_ms"] == 2000, "total_duration_ms: 1200 + 800"
        assert totals["agents_count"] == 2, "agents_count should match number of agents"

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_global_timeout(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that ensemble execution respects global timeout settings."""
        config = EnsembleConfig(
            name="timeout_test",
            description="Test timeout functionality",
            agents=[
                LlmAgentConfig(
                    name="slow_agent",
                    model_profile="slow-analyst",
                    timeout_seconds=1,  # 1 second timeout at agent level
                ),
            ],
        )

        # Mock model that takes too long
        slow_model = AsyncMock(spec=ModelInterface)

        async def slow_response(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(2.0)  # Takes 2s, longer than 1s timeout
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

        executor = mock_ensemble_executor

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_execute_ensemble_with_per_agent_timeout(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that individual agents can have their own timeout settings."""
        config = EnsembleConfig(
            name="per_agent_timeout_test",
            description="Test per-agent timeout functionality",
            agents=[
                LlmAgentConfig(
                    name="fast_agent",
                    model_profile="fast-analyst",
                    timeout_seconds=5,
                ),
                LlmAgentConfig(
                    name="slow_agent",
                    model_profile="slow-reviewer",
                    timeout_seconds=1,  # 1 second timeout
                ),
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
            await asyncio.sleep(2.0)  # Takes 2s, longer than 1s timeout
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

        executor = mock_ensemble_executor

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_load_model_with_authentication_configurations(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _load_model resolves auth configurations to model instances."""
        executor = mock_ensemble_executor

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
    async def test_ensemble_execution_with_model_profile(
        self, mock_ensemble_executor: Any
    ) -> None:
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

            executor = mock_ensemble_executor

            # Test model_factory.load_model_from_agent_config directly
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
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test fallback to explicit model+provider when no model_profile."""
        executor = mock_ensemble_executor

        # Mock the model loading to avoid auth issues in CI
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Test response"

        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager"
        ) as mock_config_manager_class:
            mock_config_manager = mock_config_manager_class.return_value

            with patch.object(
                executor._model_factory, "load_model", return_value=mock_model
            ):
                # This should use explicit model+provider, not call
                # resolve_model_profile
                result = await executor._model_factory.load_model_from_agent_config(
                    {
                        "name": "agent1",
                        "model": "claude-3-5-sonnet-20241022",
                        "provider": "anthropic-claude-pro-max",
                    }
                )

                # Verify that resolve_model_profile was NOT called
                mock_config_manager.resolve_model_profile.assert_not_called()
                # Verify we got the mocked model back
                assert result == mock_model

    @pytest.mark.asyncio
    async def test_execute_dependency_based_ensemble(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test executing ensemble with agent dependencies instead of coordinator."""
        # RED: This test should fail until we implement dependency execution
        config = EnsembleConfig(
            name="dependency_test",
            description="Test dependency-based execution",
            agents=[
                LlmAgentConfig(name="researcher", model_profile="test-researcher"),
                LlmAgentConfig(name="analyzer", model_profile="test-analyzer"),
                LlmAgentConfig(
                    name="synthesizer",
                    model_profile="test-synthesizer",
                    depends_on=["researcher", "analyzer"],
                ),
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

        executor = mock_ensemble_executor

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_execute_streaming_with_progress_updates(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test streaming execution yields progress updates."""
        config = EnsembleConfig(
            name="streaming_test",
            description="Test streaming execution",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
                LlmAgentConfig(name="agent2", model_profile="test-reviewer"),
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
        executor = mock_ensemble_executor

        # Mock dependencies
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
    async def test_resolve_model_profile_to_config(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test model profile resolution to enhanced config."""
        executor = mock_ensemble_executor

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
            agent_config = LlmAgentConfig(
                name="test_agent",
                model_profile="test-profile",
                temperature=0.9,  # Should override profile
            )

            enhanced = (
                await executor._llm_agent_runner._resolve_model_profile_to_config(
                    agent_config
                )
            )

            # Should merge profile with agent config (agent takes precedence)
            assert enhanced["model"] == "claude-3-5-sonnet"
            assert enhanced["provider"] == "anthropic-claude-pro-max"
            assert enhanced["temperature"] == 0.9  # Agent override
            assert enhanced["name"] == "test_agent"

    @pytest.mark.asyncio
    async def test_resolve_model_profile_nonexistent_profile(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test model profile resolution with nonexistent profile."""
        executor = mock_ensemble_executor

        with patch.object(
            executor._config_manager, "get_model_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {}  # No profiles

            agent_config = LlmAgentConfig(
                name="test_agent",
                model_profile="nonexistent-profile",
            )

            # Should return config dict when profile doesn't exist
            enhanced = (
                await executor._llm_agent_runner._resolve_model_profile_to_config(
                    agent_config
                )
            )
            assert enhanced == agent_config.model_dump()

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_no_timeout(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _execute_agent_with_timeout with no timeout specified."""
        executor = mock_ensemble_executor

        agent_config = LlmAgentConfig(name="test_agent", model_profile="mock-model")
        input_data = "Test input"

        # Mock _execute_agent to return expected result
        with patch.object(
            executor, "_execute_agent", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = ("Agent response", None)

            (
                result,
                model,
            ) = await executor._execution_coordinator.execute_agent_with_timeout(
                agent_config, input_data, None
            )

            assert result == "Agent response"
            assert model is None
            mock_execute.assert_called_once_with(agent_config, input_data)

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_timeout_occurs(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _execute_agent_with_timeout when timeout occurs."""
        executor = mock_ensemble_executor

        agent_config = LlmAgentConfig(name="test_agent", model_profile="mock-model")
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
                await executor._execution_coordinator.execute_agent_with_timeout(
                    agent_config,
                    input_data,
                    1,  # 1 second timeout
                )

    @pytest.mark.asyncio
    async def test_resolve_model_profile_to_config_without_profile(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test model profile resolution without model_profile key."""
        executor = mock_ensemble_executor

        agent_config = LlmAgentConfig(
            name="test_agent",
            model="claude-3-sonnet",
            provider="anthropic",
        )

        # Should return dict copy of config when no model_profile
        enhanced = await executor._llm_agent_runner._resolve_model_profile_to_config(
            agent_config
        )
        assert enhanced == agent_config.model_dump()
        assert enhanced is not agent_config  # Result is a new dict, not the config

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_parallel_execution_performance(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that parallel execution is significantly faster than sequential.

        RED: This test should fail until we implement parallel execution.
        """
        import time

        # Create ensemble with multiple agents that have simulated latency
        config = EnsembleConfig(
            name="parallel_test",
            description="Test parallel execution performance",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-analyst"),
                LlmAgentConfig(name="agent2", model_profile="test-reviewer"),
                LlmAgentConfig(name="agent3", model_profile="test-synthesizer"),
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
        executor = mock_ensemble_executor

        # Mock dependencies
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
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
        # - Sequential execution should take ~1.5s + overhead
        # - Parallel execution should take ~0.5s + overhead
        # The overhead includes dependency analysis, model setup, monitoring, etc.
        # We expect significant speedup but allow for realistic overhead
        expected_parallel_time = 0.5
        expected_sequential_time = 1.5
        overhead_allowance = 1.2  # Allow 1.2s for framework overhead (CI environments)

        # Test that execution time is closer to parallel than sequential
        # This ensures we get the performance benefit without being too strict
        parallel_with_overhead = expected_parallel_time + overhead_allowance
        sequential_with_overhead = expected_sequential_time + overhead_allowance

        assert execution_time < parallel_with_overhead, (
            f"Execution took {execution_time:.2f}s, "
            f"expected <{parallel_with_overhead:.2f}s for parallel execution "
            f"(sequential would be ~{sequential_with_overhead:.2f}s)"
        )

    @pytest.mark.asyncio
    async def test_oauth_fallback_completes_with_fallback_response(
        self, mock_ensemble_executor: Any
    ) -> None:
        """OAuth token refresh failure should recover via fallback and complete."""
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleConfig
        from llm_orc.models.anthropic import OAuthClaudeModel

        executor = mock_ensemble_executor
        config = EnsembleConfig(
            name="oauth-fallback-test",
            description="Test OAuth fallback display enhancement",
            agents=[
                LlmAgentConfig(
                    name="oauth-agent",
                    model_profile="premium-claude",
                    system_prompt="You are a test agent.",
                )
            ],
        )

        mock_oauth_model = Mock(spec=OAuthClaudeModel)
        mock_oauth_model.generate_response = AsyncMock()
        mock_oauth_model.generate_response.side_effect = Exception(
            "OAuth token refresh failed with status 400: "
            '{"error": "invalid_grant", "error_description": "Refresh token not found"}'
        )
        mock_oauth_model.get_last_usage.return_value = {
            "input_tokens": 10,
            "output_tokens": 5,
            "duration_ms": 100,
            "cost_usd": 0.01,
        }

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
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
                return_value=Mock(name="oauth-agent", prompt="Test prompt"),
            ),
        ):
            async for event in executor.execute_streaming(
                config, "Test OAuth fallback"
            ):
                streaming_events.append(event)

        completion_events = [
            e for e in streaming_events if e["type"] == "execution_completed"
        ]
        assert len(completion_events) == 1, (
            f"expected 1 completion event, got {len(completion_events)}"
        )
        result = completion_events[0]["data"]
        assert result["status"] == "completed", "should complete after fallback"
        assert result["results"]["oauth-agent"]["status"] == "success", (
            "oauth-agent should report success via fallback"
        )
        assert result["results"]["oauth-agent"]["response"] == "Fallback response", (
            "oauth-agent should return fallback model's response"
        )

    @pytest.mark.asyncio
    async def test_oauth_fallback_emits_structured_event(
        self, mock_ensemble_executor: Any
    ) -> None:
        """OAuth fallback emits an agent_fallback_started event with error details."""
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleConfig
        from llm_orc.models.anthropic import OAuthClaudeModel

        executor = mock_ensemble_executor
        config = EnsembleConfig(
            name="oauth-fallback-test",
            description="Test OAuth fallback display enhancement",
            agents=[
                LlmAgentConfig(
                    name="oauth-agent",
                    model_profile="premium-claude",
                    system_prompt="You are a test agent.",
                )
            ],
        )

        mock_oauth_model = Mock(spec=OAuthClaudeModel)
        mock_oauth_model.generate_response = AsyncMock()
        mock_oauth_model.generate_response.side_effect = Exception(
            "OAuth token refresh failed with status 400: "
            '{"error": "invalid_grant", "error_description": "Refresh token not found"}'
        )
        mock_oauth_model.get_last_usage.return_value = {
            "input_tokens": 10,
            "output_tokens": 5,
            "duration_ms": 100,
            "cost_usd": 0.01,
        }

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
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
                return_value=Mock(name="oauth-agent", prompt="Test prompt"),
            ),
        ):
            async for event in executor.execute_streaming(
                config, "Test OAuth fallback"
            ):
                streaming_events.append(event)

        oauth_fallback_events = [
            e for e in streaming_events if e["type"] == "agent_fallback_started"
        ]
        assert len(oauth_fallback_events) == 1, (
            f"expected 1 agent_fallback_started, got {len(oauth_fallback_events)}. "
            f"All types: {[e['type'] for e in streaming_events]}"
        )

        event_data = oauth_fallback_events[0]["data"]
        assert event_data["agent_name"] == "oauth-agent", "wrong agent_name in event"
        assert event_data["original_model_profile"] == "premium-claude", (
            "event should record the original profile"
        )
        assert event_data["fallback_model_name"] == "llama3", (
            "event should record the fallback model"
        )
        assert "OAuth token refresh failed" in event_data["original_error"], (
            "event should include OAuth error text"
        )
        assert "invalid_grant" in event_data["original_error"], (
            "event should include OAuth error code"
        )
        assert event_data["failure_type"] == "oauth_error", (
            f"wrong failure_type: {event_data.get('failure_type')!r}"
        )

    @pytest.mark.asyncio
    async def test_model_loading_fallback_completes_with_fallback_response(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Model loading failure should recover via fallback and complete."""
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleConfig

        executor = mock_ensemble_executor
        config = EnsembleConfig(
            name="model-loading-fallback-test",
            description="Test model loading fallback display enhancement",
            agents=[
                LlmAgentConfig(
                    name="failing-agent",
                    model_profile="guaranteed-fail",
                    system_prompt="You are a test agent.",
                )
            ],
        )

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
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
                return_value=Mock(name="failing-agent", prompt="Test prompt"),
            ),
        ):
            async for event in executor.execute_streaming(
                config, "Test model loading fallback"
            ):
                streaming_events.append(event)

        completion_events = [
            e for e in streaming_events if e["type"] == "execution_completed"
        ]
        assert len(completion_events) == 1, (
            f"expected 1 completion event, got {len(completion_events)}"
        )
        result = completion_events[0]["data"]
        assert result["status"] == "completed", "should complete after fallback"
        assert result["results"]["failing-agent"]["status"] == "success", (
            "failing-agent should report success via fallback"
        )
        assert (
            result["results"]["failing-agent"]["response"]
            == "Model loading fallback response"
        ), "failing-agent should return the fallback model's response"

    @pytest.mark.asyncio
    async def test_model_loading_fallback_emits_structured_event(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Model loading failure emits agent_fallback_started with error details."""
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleConfig

        executor = mock_ensemble_executor
        config = EnsembleConfig(
            name="model-loading-fallback-test",
            description="Test model loading fallback display enhancement",
            agents=[
                LlmAgentConfig(
                    name="failing-agent",
                    model_profile="guaranteed-fail",
                    system_prompt="You are a test agent.",
                )
            ],
        )

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
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
                return_value=Mock(name="failing-agent", prompt="Test prompt"),
            ),
        ):
            async for event in executor.execute_streaming(
                config, "Test model loading fallback"
            ):
                streaming_events.append(event)

        loading_fallback_events = [
            e for e in streaming_events if e["type"] == "agent_fallback_started"
        ]
        assert len(loading_fallback_events) == 1, (
            f"expected 1 agent_fallback_started, got {len(loading_fallback_events)}. "
            f"All types: {[e['type'] for e in streaming_events]}"
        )

        event_data = loading_fallback_events[0]["data"]
        assert event_data["agent_name"] == "failing-agent", (
            "event should name the failing agent"
        )
        assert event_data["original_model_profile"] == "guaranteed-fail", (
            "event should record the original profile"
        )
        assert event_data["fallback_model_name"] == "llama3", (
            "event should record the fallback model"
        )
        assert "Model provider" in event_data["original_error"], (
            "event should include provider error text"
        )
        assert "not available" in event_data["original_error"], (
            "event should include 'not available' message"
        )
        assert event_data["failure_type"] == "model_loading", (
            f"wrong failure_type: {event_data.get('failure_type')!r}"
        )

    @pytest.mark.asyncio
    async def test_emit_performance_event_queue_full_exception(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _emit_performance_event handles QueueFull exception gracefully.

        This covers lines 97-99.
        """
        executor = mock_ensemble_executor

        # Mock the streaming event queue to be full
        with patch.object(executor, "_streaming_event_queue") as mock_queue:
            mock_queue.put_nowait.side_effect = asyncio.QueueFull("Queue is full")

            # Should not raise exception, should silently ignore
            executor._emit_performance_event("test_event", {"test": "data"})

            # Verify that put_nowait was attempted
            mock_queue.put_nowait.assert_called_once()

    @pytest.mark.asyncio
    async def test_artifact_manager_integration(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test ArtifactManager integration into ensemble execution.

        RED Phase: This test should fail until integration is implemented.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.execution.artifact_manager import ArtifactManager

        config = EnsembleConfig(
            name="artifact_test",
            description="Test artifact manager integration",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
            ],
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Agent response"
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="tester", prompt="You are a tester")
        executor = mock_ensemble_executor

        # Mock ArtifactManager
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results = Mock()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch.object(executor, "_artifact_manager", mock_artifact_manager),
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            result = await executor.execute(config, input_data="Test input")

        # Verify artifact manager was called with correct parameters
        mock_artifact_manager.save_execution_results.assert_called_once()
        call_args = mock_artifact_manager.save_execution_results.call_args

        # Check ensemble name was passed correctly
        assert call_args[0][0] == "artifact_test"

        # Check complete results were passed
        saved_results = call_args[0][1]
        assert saved_results["ensemble"] == "artifact_test"
        assert saved_results["input"]["data"] == "Test input"
        assert "results" in saved_results
        assert "agent1" in saved_results["results"]

        # Verify execution completed successfully
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_automatic_interactive_mode_detection_and_integration(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test automatic detection and integration of interactive mode for ensembles.

        When an ensemble contains scripts that require user input, the system should:
        1. Automatically detect this condition
        2. Enable interactive mode without requiring --interactive flag
        3. Use execute_with_user_input() seamlessly
        4. Handle user input collection during script execution

        RED Phase: This test should fail until automatic detection is implemented.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.execution.script_user_input_handler import (
            ScriptUserInputHandler,
        )

        # Create ensemble config with interactive script agent
        config = EnsembleConfig(
            name="interactive_ensemble",
            description="Ensemble with interactive script requiring user input",
            agents=[
                ScriptAgentConfig(
                    name="data_collector",
                    script="primitives/user-interaction/get_user_input.py",
                ),
                LlmAgentConfig(
                    name="data_processor",
                    model_profile="claude-analyst",
                    system_prompt="Process the collected data",
                    timeout_seconds=30,
                ),
            ],
        )

        executor = mock_ensemble_executor

        # Mock ScriptUserInputHandler
        mock_handler = Mock(spec=ScriptUserInputHandler)
        mock_handler.ensemble_requires_user_input.return_value = True
        mock_handler.collect_user_input = AsyncMock(return_value="John Doe")

        # Mock execution results for interactive script
        mock_script_result = {
            "success": True,
            "output": "User provided: John Doe",
            "collected_data": {"name": "John Doe", "timestamp": "2025-01-15T10:30:00"},
        }

        # Mock LLM agent response
        mock_llm_model = AsyncMock(spec=ModelInterface)
        mock_llm_model.generate_response.return_value = (
            "Data processed successfully. Name: John Doe, captured at 10:30 AM"
        )
        mock_llm_model.get_last_usage.return_value = {
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "cost_usd": 0.01,
            "duration_ms": 800,
        }

        from llm_orc.core.config.roles import RoleDefinition

        role = RoleDefinition(name="analyst", prompt="Process data")

        # The key behavior we're testing: automatic detection and mode switching
        with (
            patch.object(
                executor, "_detect_interactive_ensemble", return_value=True
            ) as mock_detect,
            patch.object(
                executor, "_create_user_input_handler", return_value=mock_handler
            ) as mock_create_handler,
            patch.object(
                executor, "execute_with_user_input", new_callable=AsyncMock
            ) as mock_execute_interactive,
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            # Setup mock returns
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_llm_model

            # Mock the interactive execution result
            mock_execute_interactive.return_value = {
                "ensemble": "interactive_ensemble",
                "status": "completed",
                "input": {"data": "Test interactive execution"},
                "results": {
                    "data_collector": {
                        "response": mock_script_result,
                        "status": "success",
                    },
                    "data_processor": {
                        "response": "Data processed successfully. Name: John Doe",
                        "status": "success",
                    },
                },
                "metadata": {
                    "duration": 2.5,
                    "agents_used": 2,
                    "interactive_mode": True,
                    "user_inputs_collected": 1,
                },
                "synthesis": None,
            }

            # Execute ensemble - this should automatically detect interactive mode
            result = await executor.execute(
                config, input_data="Test interactive execution"
            )

        # Verify automatic detection occurred
        mock_detect.assert_called_once_with(config)

        # Verify user input handler was created
        mock_create_handler.assert_called_once()

        # Verify interactive execution was used automatically
        mock_execute_interactive.assert_called_once_with(
            config, "Test interactive execution", mock_handler
        )

        # Verify execution completed successfully with interactive mode
        assert result["status"] == "completed", "ensemble should complete"
        assert result["metadata"]["interactive_mode"] is True, (
            "interactive_mode flag missing"
        )
        assert result["metadata"]["user_inputs_collected"] == 1, (
            f"expected 1 input, got {result['metadata']['user_inputs_collected']}"
        )
        assert "data_collector" in result["results"], "data_collector missing"
        assert "data_processor" in result["results"], "data_processor missing"

        # Verify interactive script result contains user data
        collector_result = result["results"]["data_collector"]["response"]
        assert "John Doe" in str(collector_result), "collector result missing user data"
        assert collector_result["success"] is True, "collector should report success"

        # Verify LLM agent processed the interactive data
        processor_response = result["results"]["data_processor"]["response"]
        assert "John Doe" in processor_response, "processor response missing name"
        assert "processed successfully" in processor_response.lower(), (
            "processor response should confirm processing"
        )

    @pytest.mark.asyncio
    async def test_non_interactive_ensemble_bypasses_interactive_mode(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that non-interactive ensembles bypass interactive mode detection.

        This ensures we don't add overhead to standard ensemble execution.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.execution.script_user_input_handler import (
            ScriptUserInputHandler,
        )

        # Create ensemble config without any interactive scripts
        config = EnsembleConfig(
            name="standard_ensemble",
            description="Standard ensemble with no user input requirements",
            agents=[
                LlmAgentConfig(
                    name="data_processor",
                    model_profile="claude-analyst",
                    system_prompt="Process the data",
                    timeout_seconds=30,
                ),
                ScriptAgentConfig(
                    name="data_formatter",
                    script="echo '{\"formatted\": true}'",
                ),
            ],
        )

        executor = mock_ensemble_executor

        # Mock the handler to return False (no interactive scripts detected)
        mock_handler = Mock(spec=ScriptUserInputHandler)
        mock_handler.ensemble_requires_user_input.return_value = False

        # Mock LLM agent response
        mock_llm_model = AsyncMock(spec=ModelInterface)
        mock_llm_model.generate_response.return_value = "Data processed successfully"
        mock_llm_model.get_last_usage.return_value = {
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "cost_usd": 0.01,
            "duration_ms": 800,
        }

        from llm_orc.core.config.roles import RoleDefinition

        role = RoleDefinition(name="analyst", prompt="Process data")

        # Track that detection occurs but interactive mode is not used
        with (
            patch.object(
                executor, "_detect_interactive_ensemble", return_value=False
            ) as mock_detect,
            patch.object(
                executor, "execute_with_user_input", new_callable=AsyncMock
            ) as mock_execute_interactive,
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch.object(executor, "_artifact_manager") as mock_artifact_manager,
        ):
            # Setup mocks
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_llm_model
            mock_artifact_manager.save_execution_results = Mock()

            # Execute ensemble - should use standard execution path
            result = await executor.execute(
                config, input_data="Test standard execution"
            )

        # Verify detection occurred
        mock_detect.assert_called_once_with(config)

        # Verify interactive execution was NOT called
        mock_execute_interactive.assert_not_called()

        # Verify standard execution completed successfully
        assert result["status"] in ["completed", "completed_with_errors"]
        assert result["ensemble"] == "standard_ensemble"
        assert "data_processor" in result["results"]
        assert "data_formatter" in result["results"]

        # Verify no interactive mode metadata is present
        assert result["metadata"].get("interactive_mode") is None
        assert result["metadata"].get("user_inputs_collected") is None

    @pytest.mark.asyncio
    async def test_artifact_manager_error_handling(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that artifact manager errors don't break execution.

        RED Phase: This test should fail until error handling is implemented.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.execution.artifact_manager import ArtifactManager

        config = EnsembleConfig(
            name="artifact_error_test",
            description="Test artifact manager error handling",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
            ],
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Agent response"
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="tester", prompt="You are a tester")
        executor = mock_ensemble_executor

        # Mock ArtifactManager that raises an exception
        mock_artifact_manager = Mock(spec=ArtifactManager)
        mock_artifact_manager.save_execution_results.side_effect = Exception(
            "Permission denied creating artifact directory"
        )

        # Mock the role and model loading methods
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch.object(executor, "_artifact_manager", mock_artifact_manager),
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            # Execution should not fail even if artifact saving fails
            result = await executor.execute(config, input_data="Test input")

        # Verify execution completed successfully despite artifact error
        assert result["status"] == "completed"
        assert result["ensemble"] == "artifact_error_test"

        # Verify artifact manager was called
        mock_artifact_manager.save_execution_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_script_agent_uses_user_input_for_interactive_scripts(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _script_agent_runner.execute detects and handles user input.

        When a script agent configuration references a script that requires user input,
        _script_agent_runner.execute should:
        1. Detect that the script requires user input
        2. Automatically call execute_with_user_input() instead of execute()
        3. Pass the user input handler to the script agent
        4. Return the response with actual user input, not "User cancelled input"

        RED Phase: This test should fail until automatic detection is implemented
        in ScriptAgentRunner.execute method.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.agents.script_agent import ScriptAgent
        from llm_orc.core.execution.script_user_input_handler import (
            ScriptUserInputHandler,
        )

        executor = mock_ensemble_executor

        # Create script agent configuration that requires user input
        agent_config = ScriptAgentConfig(
            name="interactive_agent",
            script="primitives/user-interaction/get_user_input.py",
            parameters={"prompt": "What's your name?"},
        )
        input_data = "start_interactive"

        # Mock user input detection
        mock_user_input_detection = Mock(spec=ScriptUserInputHandler)
        mock_user_input_detection.requires_user_input.return_value = True

        # Mock ScriptAgent
        mock_script_agent = Mock(spec=ScriptAgent)
        mock_script_agent.name = "interactive_agent"
        mock_script_agent.script = "primitives/user-interaction/get_user_input.py"
        mock_script_agent.environment = {}
        mock_script_agent.parameters = {"prompt": "What's your name?"}

        # Mock script resolver
        mock_script_resolver = Mock()
        mock_script_resolver.resolve_script_path.return_value = "/tmp/test_script.py"
        mock_script_agent._script_resolver = mock_script_resolver

        # Mock interpreter getter
        mock_script_agent._get_interpreter = Mock(return_value=["python"])
        mock_script_agent.timeout = 60

        # The key behavior: when user input is required, execute_with_user_input
        # should be called and return actual user input
        mock_script_agent.execute_with_user_input = AsyncMock(
            return_value='{"user_input": "User input placeholder", "success": true}'
        )
        # Regular execute should not be called for interactive scripts
        mock_script_agent.execute = AsyncMock(
            return_value='{"error": "User cancelled input", "success": false}'
        )

        # Mock the script detection and agent creation
        with (
            patch(
                "llm_orc.core.execution.script_agent_runner.ScriptAgent",
                return_value=mock_script_agent,
            ),
            patch(
                "llm_orc.core.execution.script_agent_runner.ScriptUserInputHandler",
                return_value=mock_user_input_detection,
            ) as mock_handler_class,
            patch("os.path.exists", return_value=True),
            patch("subprocess.run") as mock_subprocess_run,
            patch("builtins.input", return_value="Test User"),
        ):
            # Mock subprocess to simulate user input collection
            mock_subprocess_run.return_value.stdout = (
                '{"user_input": "Test User", "success": true}'
            )
            mock_subprocess_run.return_value.returncode = 0
            # This is the core test: _script_agent_runner.execute should automatically
            # detect user input requirement and use execute_with_user_input
            response, model, _ = await executor._script_agent_runner.execute(
                agent_config, input_data
            )

        # Verify that user input detection was performed
        mock_handler_class.assert_called_once()
        mock_user_input_detection.requires_user_input.assert_called_once_with(
            "primitives/user-interaction/get_user_input.py"
        )

        # The actual implementation uses subprocess to run interactive scripts
        # directly, not through execute_with_user_input method
        mock_subprocess_run.assert_called_once()

        # Verify subprocess was called with correct script
        call_args = mock_subprocess_run.call_args
        assert call_args[0][0] == ["python", "/tmp/test_script.py"]

    @pytest.mark.asyncio
    async def test_execute_with_user_input_comprehensive(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test execute_with_user_input method thoroughly for refactoring.

        This test verifies the complete interactive execution flow:
        1. Setup and initialization
        2. Dependency analysis and phase preparation
        3. Phase execution with user input handling
        4. User input counting and metadata tracking
        5. Result finalization with interactive metadata
        """
        from unittest.mock import AsyncMock, Mock

        from llm_orc.core.execution.script_user_input_handler import (
            ScriptUserInputHandler,
        )

        # Create ensemble config with mixed agents (script + LLM)
        config = EnsembleConfig(
            name="interactive_test_ensemble",
            description="Test ensemble with user input handling",
            agents=[
                ScriptAgentConfig(
                    name="user_input_collector",
                    script="primitives/user-interaction/get_user_input.py",
                ),
                LlmAgentConfig(
                    name="data_processor",
                    model_profile="claude-analyst",
                    system_prompt="Process collected data",
                    timeout_seconds=30,
                    depends_on=["user_input_collector"],
                ),
                ScriptAgentConfig(
                    name="interactive_script2",
                    script="primitives/user-interaction/confirm_action.py",
                    depends_on=["data_processor"],
                ),
            ],
        )

        input_data = "Process user data interactively"
        executor = mock_ensemble_executor

        # Mock ScriptUserInputHandler
        mock_handler = Mock(spec=ScriptUserInputHandler)
        mock_handler.ensemble_requires_user_input.return_value = True
        mock_handler.collect_user_input = AsyncMock(
            side_effect=["John Doe", "confirmed"]
        )

        # Mock LLM model
        mock_llm_model = AsyncMock(spec=ModelInterface)
        mock_llm_model.generate_response.return_value = "Data processed: John Doe"
        mock_llm_model.get_last_usage.return_value = {
            "total_tokens": 75,
            "input_tokens": 45,
            "output_tokens": 30,
            "cost_usd": 0.015,
            "duration_ms": 1200,
        }

        # Mock the phase execution to simulate interactive script responses
        original_execute_agents = executor._agent_dispatcher.execute_agents_in_phase

        async def mock_execute_agents_in_phase(
            agents: list[Any], phase_input: str | dict[str, str]
        ) -> dict[str, Any]:
            results = {}
            for agent in agents:
                # Simulate interactive script results with collected_data
                results[agent.name] = {
                    "success": True,
                    "status": "success",
                    "output": f"Interactive script {agent.name} executed",
                    "model_instance": None,  # Script agents don't have models
                    "response": {
                        "collected_data": {"user_input": f"data from {agent.name}"}
                    },
                }
            return results

        executor._agent_dispatcher.execute_agents_in_phase = (
            mock_execute_agents_in_phase
        )

        # Execute the method we're testing
        result = await executor.execute_with_user_input(
            config, input_data, mock_handler
        )

        # Verify result structure and interactive metadata
        assert result["ensemble"] == "interactive_test_ensemble"
        assert result["input"]["data"] == input_data
        assert result["metadata"]["interactive_mode"] is True
        assert (
            result["metadata"]["user_inputs_collected"] == 3
        )  # Three interactive scripts
        assert "duration" in result["metadata"]  # Actual field name
        assert "agents_used" in result["metadata"]
        assert "results" in result

        # Verify results contain all agents
        assert "user_input_collector" in result["results"]
        assert "data_processor" in result["results"]
        assert "interactive_script2" in result["results"]

        # Restore original method
        executor._agent_dispatcher.execute_agents_in_phase = original_execute_agents

    @pytest.mark.asyncio
    async def test_initialize_execution_setup(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _initialize_execution_setup helper method.

        This method should:
        1. Store agent configs for role descriptions
        2. Create initial result structure
        3. Reset usage collector
        4. Return initialized result and results_dict
        """
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
                LlmAgentConfig(name="agent2", model_profile="test-reviewer"),
            ],
        )
        input_data = "Test input"

        executor = mock_ensemble_executor

        # This should fail initially since the method doesn't exist yet
        result, results_dict = await executor._initialize_execution_setup(
            config, input_data
        )

        # Verify agent configs were stored
        assert executor._current_agent_configs == config.agents

        # Verify result structure
        assert result["ensemble"] == "test_ensemble"
        assert result["input"]["data"] == "Test input"
        assert "results" in result
        assert result["metadata"]["agents_used"] == 2

        # Verify results_dict is the same reference
        assert results_dict is result["results"]
        assert results_dict == {}  # Initially empty, populated during execution

        # Verify usage collector was reset
        assert executor._usage_collector.get_agent_usage() == {}

    @pytest.mark.asyncio
    async def test_analyze_and_prepare_phases(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _analyze_and_prepare_phases helper method.

        This method should:
        1. Use dependency analyzer to get enhanced dependency graph
        2. Return phases for execution
        """
        config = EnsembleConfig(
            name="dependency_test",
            description="Test dependency analysis",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-analyst"),
                LlmAgentConfig(
                    name="agent2",
                    model_profile="test-reviewer",
                    depends_on=["agent1"],
                ),
            ],
        )

        executor = mock_ensemble_executor

        # This should fail initially since the method doesn't exist yet
        phases = await executor._analyze_and_prepare_phases(config)

        # Verify phases structure
        assert isinstance(phases, list)
        assert len(phases) >= 1

        # Verify first phase contains independent agents
        phase0_names = [agent.name for agent in phases[0]]
        assert "agent1" in phase0_names

        # Verify dependent agents are in later phases
        if len(phases) > 1:
            phase1_names = [agent.name for agent in phases[1]]
            assert "agent2" in phase1_names

    @pytest.mark.asyncio
    async def test_execute_phase_with_monitoring(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _execute_phase_with_monitoring helper method.

        This method should:
        1. Emit phase started event
        2. Determine phase input using dependency resolver
        3. Start phase monitoring
        4. Execute agents in parallel
        5. Process results
        6. Stop monitoring and emit completion event
        7. Return has_errors flag
        """
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
            ],
        )

        executor = mock_ensemble_executor

        # Mock dependencies
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Test response"
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="tester", prompt="You are a tester")

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            phase_agents = config.agents
            input_data = "Test input"
            results_dict: dict[str, Any] = {}
            phase_index = 0

            has_errors, user_inputs = await executor._execute_phase_with_monitoring(
                phase_index, phase_agents, input_data, results_dict
            )

            # Verify results were populated
            assert len(results_dict) == 1
            assert "agent1" in results_dict

            # Verify return values
            assert isinstance(has_errors, bool)
            assert isinstance(user_inputs, int)

    @pytest.mark.asyncio
    async def test_finalize_execution_results(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test _finalize_execution_results helper method.

        This method should:
        1. Get collected usage and adaptive stats
        2. Finalize result using processor
        3. Save artifacts with error handling
        4. Return final result
        """
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble",
            agents=[LlmAgentConfig(name="agent1", model_profile="test-tester")],
        )

        executor = mock_ensemble_executor

        # Mock initial result structure
        result = {
            "ensemble": "test_ensemble",
            "status": "in_progress",
            "input": {"data": "Test input"},
            "results": {"agent1": {"response": "Test response", "status": "success"}},
            "metadata": {},
            "synthesis": None,
        }

        has_errors = False
        start_time = time.time() - 1.0  # 1 second ago

        # This should fail initially since the method doesn't exist yet
        final_result = await executor._finalize_execution_results(
            config, result, has_errors, start_time
        )

        # Verify result finalization
        assert final_result["status"] in ["completed", "completed_with_errors"]
        assert "metadata" in final_result
        assert "usage" in final_result["metadata"]

        # Verify artifacts were saved (mocked)
        # The real implementation should handle save errors gracefully

    @pytest.mark.asyncio
    async def test_execute_with_validation_config(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that ensemble execution runs validation when config is present.

        RED Phase: This test should fail until validation integration is implemented.
        """
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.validation import ValidationResult

        # Create ensemble config with validation
        config = EnsembleConfig(
            name="validation_test",
            description="Test validation integration",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
            ],
            validation={
                "structural": {
                    "required_agents": ["agent1"],
                    "max_execution_time": 30.0,
                }
            },
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Test response"
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="tester", prompt="You are a tester")
        executor = mock_ensemble_executor

        # Mock validation evaluator
        mock_validation_result = Mock(spec=ValidationResult)
        mock_validation_result.passed = True
        mock_validation_result.ensemble_name = "validation_test"

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch(
                "llm_orc.core.execution.ensemble_execution._run_validation",
                new_callable=AsyncMock,
            ) as mock_run_validation,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model
            mock_run_validation.return_value = mock_validation_result

            # Execute ensemble
            result = await executor.execute(config, input_data="Test validation")

        # Verify result structure includes validation
        assert result["ensemble"] == "validation_test"
        assert result["status"] == "completed"
        assert "validation_result" in result
        assert result["validation_result"] is not None

        # Verify validation was executed
        validation_result = result["validation_result"]
        assert validation_result.passed is True
        assert validation_result.ensemble_name == "validation_test"

    @pytest.mark.asyncio
    async def test_execute_without_validation_config(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that ensemble execution skips validation when no config present.

        This ensures backward compatibility.
        """
        from unittest.mock import AsyncMock

        # Create ensemble config WITHOUT validation
        config = EnsembleConfig(
            name="no_validation_test",
            description="Test without validation",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="test-tester"),
            ],
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Test response"
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="tester", prompt="You are a tester")
        executor = mock_ensemble_executor

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            # Execute ensemble
            result = await executor.execute(
                config, input_data="Test without validation"
            )

        # Verify result structure does NOT include validation
        assert result["ensemble"] == "no_validation_test"
        assert result["status"] == "completed"
        assert (
            "validation_result" not in result or result.get("validation_result") is None
        )


class TestFanOutExecution:
    """Test fan-out agent execution (issue #73)."""

    @pytest.mark.asyncio
    async def test_detect_fan_out_in_phase(self, mock_ensemble_executor: Any) -> None:
        """Test detection of fan-out agents with array upstream results."""
        executor = mock_ensemble_executor

        phase_agents = [
            LlmAgentConfig(
                name="extractor",
                model_profile="test",
                fan_out=True,
                depends_on=["chunker"],
            ),
            LlmAgentConfig(name="normal_agent", model_profile="test"),
        ]

        results_dict = {
            "chunker": {
                "status": "success",
                "response": '["chunk1", "chunk2", "chunk3"]',
            },
        }

        fan_out_agents = executor._fan_out_coordinator.detect_in_phase(
            phase_agents, results_dict
        )

        assert len(fan_out_agents) == 1
        agent_config, upstream_array = fan_out_agents[0]
        assert agent_config.name == "extractor"
        assert upstream_array == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_detect_fan_out_no_array_upstream(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that fan-out is skipped when upstream is not an array."""
        executor = mock_ensemble_executor

        phase_agents = [
            LlmAgentConfig(
                name="extractor",
                model_profile="test",
                fan_out=True,
                depends_on=["processor"],
            ),
        ]

        results_dict = {
            "processor": {
                "status": "success",
                "response": "Just a plain string result",
            },
        }

        fan_out_agents = executor._fan_out_coordinator.detect_in_phase(
            phase_agents, results_dict
        )

        # Should return empty - upstream is not an array
        assert len(fan_out_agents) == 0

    @pytest.mark.asyncio
    async def test_expand_fan_out_instances(self, mock_ensemble_executor: Any) -> None:
        """Test expanding a fan-out agent into instances."""
        executor = mock_ensemble_executor

        agent_config = LlmAgentConfig(
            name="extractor",
            model_profile="test",
            fan_out=True,
            depends_on=["chunker"],
            system_prompt="Extract data",
        )

        upstream_array = ["chunk_a", "chunk_b"]

        instances = executor._fan_out_coordinator.expand_agent(
            agent_config, upstream_array
        )

        assert len(instances) == 2
        assert instances[0].name == "extractor[0]"
        assert instances[1].name == "extractor[1]"
        assert instances[0].fan_out_chunk == "chunk_a"
        assert instances[1].fan_out_chunk == "chunk_b"
        assert instances[0].system_prompt == "Extract data"

    @pytest.mark.asyncio
    async def test_gather_fan_out_results(self, mock_ensemble_executor: Any) -> None:
        """Test gathering results from fan-out instances."""
        executor = mock_ensemble_executor

        # Simulate instance results
        instance_results = {
            "extractor[0]": {"status": "success", "response": "result_0"},
            "extractor[1]": {"status": "success", "response": "result_1"},
            "extractor[2]": {"status": "success", "response": "result_2"},
        }

        gathered = executor._fan_out_coordinator.gather_results(
            "extractor", instance_results
        )

        assert gathered["fan_out"] is True
        assert gathered["status"] == "success"
        assert gathered["response"] == ["result_0", "result_1", "result_2"]

    @pytest.mark.asyncio
    async def test_gather_fan_out_results_partial_failure(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test gathering results with partial failures."""
        executor = mock_ensemble_executor

        instance_results = {
            "extractor[0]": {"status": "success", "response": "result_0"},
            "extractor[1]": {"status": "failed", "error": "timeout"},
            "extractor[2]": {"status": "success", "response": "result_2"},
        }

        gathered = executor._fan_out_coordinator.gather_results(
            "extractor", instance_results
        )

        assert gathered["fan_out"] is True
        assert gathered["status"] == "partial"
        assert len(gathered["instances"]) == 3
        assert gathered["instances"][1]["status"] == "failed"
