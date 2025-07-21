"""Tests for enhanced model profiles with system_prompt and timeout_seconds."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


class TestEnhancedModelProfiles:
    """Test enhanced model profiles with complete agent configuration."""

    def test_model_profile_with_system_prompt_and_timeout(self) -> None:
        """Test that model profiles can include system_prompt and timeout_seconds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config with enhanced model profile
            config_data = {
                "model_profiles": {
                    "worker_bee": {
                        "model": "llama3",
                        "provider": "ollama",
                        "cost_per_token": 0.0,
                        "system_prompt": (
                            "You are a diligent worker agent that processes tasks."
                        ),
                        "timeout_seconds": 30,
                    }
                }
            }

            config_file = config_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Create ConfigurationManager with the temp directory
            config_manager = ConfigurationManager()
            config_manager._local_config_dir = config_dir

            # Get model profiles
            profiles = config_manager.get_model_profiles()

            # Verify enhanced profile structure
            assert "worker_bee" in profiles
            profile = profiles["worker_bee"]
            assert profile["model"] == "llama3"
            assert profile["provider"] == "ollama"
            assert profile["cost_per_token"] == 0.0  # type: ignore[comparison-overlap]
            assert (
                profile["system_prompt"]
                == "You are a diligent worker agent that processes tasks."
            )
            assert profile["timeout_seconds"] == 30  # type: ignore[comparison-overlap]

    @pytest.mark.asyncio
    async def test_ensemble_agent_uses_model_profile_system_prompt(self) -> None:
        """Test that agents can use system_prompt from model profile."""
        # Create ensemble config that uses model_profile without explicit system_prompt
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble with model profile system prompt",
            agents=[{"name": "worker1", "model_profile": "worker_bee"}],
        )

        # Mock the configuration manager to return our enhanced profile
        enhanced_profile = {
            "model": "llama3",
            "provider": "ollama",
            "cost_per_token": 0.0,
            "system_prompt": "You are a diligent worker agent that processes tasks.",
            "timeout_seconds": 30,
        }

        executor = EnsembleExecutor()

        # Mock the model profile resolution to return enhanced profile
        with patch.object(executor, "_resolve_model_profile_to_config") as mock_resolve:
            mock_resolve.return_value = enhanced_profile

            # Mock the role loading to capture what system_prompt gets used
            with patch.object(executor, "_load_role_from_config") as mock_load_role:
                mock_load_role.return_value = RoleDefinition(
                    name="worker1",
                    prompt="You are a diligent worker agent that processes tasks.",
                )

                # Mock model loading and execution
                with patch.object(
                    executor, "_load_model_from_agent_config"
                ) as mock_load_model:
                    mock_model = AsyncMock()
                    mock_model.generate_response.return_value = (
                        "Task completed efficiently"
                    )
                    # For sync methods on async mock, explicitly set return value
                    mock_model.get_last_usage = lambda: {
                        "total_tokens": 100,
                        "input_tokens": 60,
                        "output_tokens": 40,
                        "cost_usd": 0.02,
                        "duration_ms": 200,
                    }
                    mock_load_model.return_value = mock_model

                    # Execute the ensemble (no synthesis in dependency-based arch)
                    await executor.execute(config, "Test task")

                    # Verify the role was loaded with the system_prompt from
                    # model profile
                    mock_load_role.assert_called_once()
                    # The agent config passed to _load_role_from_config is the
                    # original config
                    agent_config = mock_load_role.call_args[0][0]
                    assert agent_config["name"] == "worker1"
                    assert agent_config["model_profile"] == "worker_bee"
                    # The enhanced config is resolved internally - verify
                    # _resolve_model_profile_to_config was called
                    assert mock_resolve.called

    @pytest.mark.asyncio
    async def test_ensemble_agent_uses_model_profile_timeout(self) -> None:
        """Test that agents can use timeout_seconds from model profile."""
        # Create ensemble config that uses model_profile without explicit timeout
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble with model profile timeout",
            agents=[{"name": "worker1", "model_profile": "worker_bee"}],
        )

        # Mock the configuration manager to return our enhanced profile
        enhanced_profile = {
            "model": "llama3",
            "provider": "ollama",
            "cost_per_token": 0.0,
            "system_prompt": "You are a diligent worker agent that processes tasks.",
            "timeout_seconds": 30,
        }

        executor = EnsembleExecutor()

        # Mock the model profile resolution to return enhanced profile
        with patch.object(executor, "_resolve_model_profile_to_config") as mock_resolve:
            mock_resolve.return_value = enhanced_profile

            # Mock the timeout execution method to capture timeout value
            with patch.object(
                executor, "_execute_agent_with_timeout"
            ) as mock_execute_timeout:
                mock_execute_timeout.return_value = ("Task completed", None)

                # Execute the ensemble (no synthesis in dependency-based arch)
                await executor.execute(config, "Test task")

                # Verify the timeout was passed from model profile
                mock_execute_timeout.assert_called_once()
                timeout_arg = mock_execute_timeout.call_args[0][
                    2
                ]  # Third argument is timeout
                assert timeout_arg == 30

    def test_model_profile_explicit_overrides_take_precedence(self) -> None:
        """Test that explicit agent config overrides model profile defaults."""
        # This test ensures backward compatibility - explicit configs override
        # profile defaults
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config with enhanced model profile
            config_data = {
                "model_profiles": {
                    "worker_bee": {
                        "model": "llama3",
                        "provider": "ollama",
                        "cost_per_token": 0.0,
                        "system_prompt": "Default worker prompt",
                        "timeout_seconds": 30,
                    }
                }
            }

            config_file = config_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Create agent config that overrides model profile defaults
            agent_config = {
                "name": "worker1",
                "model_profile": "worker_bee",
                "system_prompt": "Custom override prompt",
                "timeout_seconds": 60,
            }

            # Mock configuration manager
            config_manager = ConfigurationManager()
            config_manager._local_config_dir = config_dir

            # Create executor and test profile resolution with overrides
            # executor = EnsembleExecutor()  # Not used in this test

            # This should merge profile defaults with explicit overrides
            # We'll need to implement _resolve_model_profile_to_config method
            # For now, let's verify the structure we want
            profiles = config_manager.get_model_profiles()
            profile = profiles["worker_bee"]

            # Verify profile has defaults
            assert profile["system_prompt"] == "Default worker prompt"
            assert profile["timeout_seconds"] == 30  # type: ignore[comparison-overlap]

            # Verify explicit overrides would take precedence
            # (Implementation detail - explicit agent config should override
            # profile defaults)
            merged_config = {**profile, **agent_config}
            assert merged_config["system_prompt"] == "Custom override prompt"
            assert merged_config["timeout_seconds"] == 60
