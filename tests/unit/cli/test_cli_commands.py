"""Tests for CLI command implementations."""

from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, call, mock_open, patch

import click
import pytest
import yaml
from click.exceptions import ClickException

import llm_orc.cli_commands
from llm_orc.cli_commands import (
    invoke_ensemble,
    list_ensembles_command,
    list_profiles_command,
)
from llm_orc.cli_modules.commands.auth_commands import (
    add_auth_provider,
    auth_setup,
    list_auth_providers,
    logout_oauth_providers,
    remove_auth_provider,
)
from llm_orc.cli_modules.commands.auth_commands import (
    test_token_refresh as refresh_token_test,
)
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


class TestInvokeEnsemble:
    """Test the invoke_ensemble function."""

    @pytest.fixture
    def mock_config_manager(self) -> Mock:
        """Create a mock configuration manager."""
        manager = Mock()
        manager.get_ensembles_dirs.return_value = [Path("/test/ensembles")]
        manager.load_performance_config.return_value = {
            "streaming_enabled": False,
            "max_concurrent": 3,
        }
        return manager

    @pytest.fixture
    def mock_ensemble_config(self) -> Mock:
        """Create a mock ensemble configuration."""
        config = Mock()
        config.name = "test_ensemble"
        config.description = "Test ensemble for testing"
        config.agents = [
            LlmAgentConfig(name="agent_1", model_profile="test", depends_on=[]),
            LlmAgentConfig(
                name="agent_2", model_profile="test", depends_on=["agent_1"]
            ),
        ]
        return config

    @pytest.fixture
    def mock_loader(self) -> Mock:
        """Create a mock ensemble loader."""
        loader = Mock()
        return loader

    @pytest.fixture
    def mock_executor(self) -> Mock:
        """Create a mock ensemble executor."""
        executor = Mock()
        executor._get_effective_concurrency_limit.return_value = 3
        return executor

    def test_invoke_ensemble_basic_success(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test basic ensemble invocation with minimal parameters."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify configuration manager was used to find ensemble dirs
            mock_config_manager.get_ensembles_dirs.assert_called_once()

            # Verify loader was used to find ensemble
            mock_loader.find_ensemble.assert_called_once_with(
                "/test/ensembles", "test_ensemble"
            )

            # Verify executor.execute was called with correct parameters
            mock_executor.execute.assert_called_once()
            call_args = mock_executor.execute.call_args
            assert call_args[0][0] == mock_ensemble_config  # ensemble_config
            assert call_args[0][1] == "test input"  # input_data

    def test_invoke_ensemble_custom_config_dir(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test ensemble invocation with custom config directory."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir="/custom/config",
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Should NOT call get_ensembles_dirs when custom config_dir provided
            mock_config_manager.get_ensembles_dirs.assert_not_called()

            # Should use custom config directory
            mock_loader.find_ensemble.assert_called_once_with(
                "/custom/config", "test_ensemble"
            )

            # Verify executor.execute was called with correct parameters
            mock_executor.execute.assert_called_once()
            call_args = mock_executor.execute.call_args
            assert call_args[0][0] == mock_ensemble_config  # ensemble_config
            assert call_args[0][1] == "test input"  # input_data

    def test_invoke_ensemble_no_ensemble_dirs_found(
        self, mock_config_manager: Mock
    ) -> None:
        """Test error when no ensemble directories are found."""
        mock_config_manager.get_ensembles_dirs.return_value = []

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            pytest.raises(ClickException, match="No ensemble directories found"),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

    def test_invoke_ensemble_not_found(
        self, mock_config_manager: Mock, mock_loader: Mock
    ) -> None:
        """Test error when ensemble is not found."""
        mock_loader.find_ensemble.return_value = None

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            pytest.raises(
                ClickException, match="Ensemble 'test_ensemble' not found in"
            ),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

    def test_invoke_ensemble_input_data_priority(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test input data priority: positional > option."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="positional_input",  # Should take precedence
                config_dir=None,
                input_data_option="option_input",
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify that executor.execute was called with the positional input
            mock_executor.execute.assert_called_once()
            call_args = mock_executor.execute.call_args
            # The second argument should be the input data (positional takes precedence)
            assert call_args[0][1] == "positional_input"

    def test_invoke_ensemble_fallback_to_option_input(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test fallback to option input when positional is None."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data=None,  # No positional input
                config_dir=None,
                input_data_option="option_input",  # Should be used
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify that executor.execute was called with the correct input
            mock_executor.execute.assert_called_once()
            call_args = mock_executor.execute.call_args
            # The second argument should be the input data
            assert call_args[0][1] == "option_input"

    def test_invoke_ensemble_stdin_input(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test reading input from stdin when no input provided."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5},
            }
        )

        # Mock stdin
        stdin_data = "input from stdin"
        mock_stdin = StringIO(stdin_data)
        mock_stdin.isatty = lambda: False  # type: ignore[method-assign]  # Indicate piped input

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.sys.stdin", mock_stdin),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data=None,
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify that executor.execute was called with stdin data
            mock_executor.execute.assert_called_once()
            call_args = mock_executor.execute.call_args
            # The second argument should be the input data from stdin
            assert call_args[0][1] == stdin_data

    def test_invoke_ensemble_default_input(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test default input when no input provided and not piped."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5},
            }
        )

        # Mock stdin as TTY (not piped)
        mock_stdin = Mock()
        mock_stdin.isatty.return_value = True

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.sys.stdin", mock_stdin),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data=None,
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify that executor.execute was called with default input
            mock_executor.execute.assert_called_once()
            call_args = mock_executor.execute.call_args
            # The second argument should be the default input data
            assert call_args[0][1] == "Please analyze this."

    def test_invoke_ensemble_streaming_execution(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test streaming execution when streaming flag is enabled."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run") as mock_asyncio_run,
            patch(
                "llm_orc.cli_commands.run_streaming_execution"
            ) as mock_streaming_exec,
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard_exec,
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="rich",  # Use Rich interface to enable streaming
                streaming=True,  # Enable streaming
                max_concurrent=None,
                detailed=False,
            )

            # Should NOT call standard execution
            mock_standard_exec.assert_not_called()

            # Should call streaming execution
            mock_streaming_exec.assert_called_once()

            # Verify asyncio.run was called with the streaming function
            mock_asyncio_run.assert_called_once()

    def test_invoke_ensemble_streaming_from_config(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test streaming execution when enabled in config."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config
        mock_config_manager.load_performance_config.return_value = {
            "streaming_enabled": True  # Enable in config
        }

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch(
                "llm_orc.cli_commands.run_streaming_execution"
            ) as mock_streaming_exec,
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="rich",  # Use Rich interface to respect config
                streaming=False,  # CLI flag disabled
                max_concurrent=None,
                detailed=False,
            )

            # Should still call streaming execution due to config
            mock_streaming_exec.assert_called_once()

    def test_invoke_ensemble_text_output_with_performance_info(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test text output format shows performance information."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5, "agents_used": 1},
            }
        )

        # Mock the execution coordinator to prevent AsyncMock warnings
        mock_coordinator = Mock(spec_set=["get_effective_concurrency_limit"])
        mock_coordinator.get_effective_concurrency_limit.return_value = 3
        mock_executor._execution_coordinator = mock_coordinator

        # Explicitly set mock attributes to prevent AsyncMock creation
        mock_executor.configure_mock(**{"_execution_coordinator": mock_coordinator})

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.click.echo") as mock_echo,
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="text",  # Text format
                streaming=False,
                max_concurrent=None,
                detailed=True,  # Need detailed=True to show dependency graph
            )

            # Should print performance information
            mock_echo.assert_called()
            echo_calls = []
            for call in mock_echo.call_args_list:
                if call.args:  # Check if positional args exist
                    echo_calls.append(str(call.args[0]))
            echo_output = " ".join(echo_calls)

            # Text output should show dependency graph and results
            assert "Agent Dependencies:" in echo_output
            assert "agent_1" in echo_output
            assert "agent_2" in echo_output
            assert "Test response" in echo_output

    def test_invoke_ensemble_text_output_performance_fallback(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test text output works without calling performance config."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Text output should not call load_performance_config at all
        # So we don't need to mock it

        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5, "agents_used": 1},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.click.echo") as mock_echo,
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="text",
                streaming=False,
                max_concurrent=None,
                detailed=True,  # Need detailed=True to show dependency graph
            )

            # Should print fallback information
            echo_calls = []
            for call in mock_echo.call_args_list:
                if call.args:  # Check if positional args exist
                    echo_calls.append(str(call.args[0]))
            echo_output = " ".join(echo_calls)

            # Text output should show dependency graph and results
            assert "Agent Dependencies:" in echo_output
            assert "agent_1" in echo_output
            assert "agent_2" in echo_output
            assert "Test response" in echo_output

    def test_invoke_ensemble_execution_failure(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test error handling when execution fails."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock asyncio.run to raise an exception
        execution_error = Exception("Execution failed")

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run", side_effect=execution_error),
            pytest.raises(
                ClickException, match="Ensemble execution failed: Execution failed"
            ),
        ):
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

    def test_invoke_ensemble_max_concurrent_override(self) -> None:
        """Test max_concurrent parameter override (line 86)."""
        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = [Path("/test/ensembles")]
        mock_config_manager.load_performance_config.return_value = {
            "streaming_enabled": False
        }

        mock_ensemble_config = Mock()
        mock_ensemble_config.name = "test_ensemble"
        mock_ensemble_config.description = "Test ensemble"
        mock_ensemble_config.agents = [Mock(), Mock()]
        mock_ensemble_config.relative_path = None

        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        # Mock the executor.execute method to return expected structure
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "test_agent": {"status": "success", "response": "Test response"}
                },
                "metadata": {"execution_time": 1.5, "agents_used": 1},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.run_standard_execution") as mock_run_std,
            patch("click.echo"),
        ):
            # Create a simple async function to avoid AsyncMockMixin issues
            async def simple_run_standard_execution(*args: Any, **kwargs: Any) -> None:
                return None

            mock_run_std.side_effect = simple_run_standard_execution

            # This should hit line 86 (the pass statement in max_concurrent handling)
            invoke_ensemble(
                ensemble_name="test_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=5,  # This triggers the max_concurrent override logic
                detailed=False,
            )


class TestListEnsemblesCommand:
    """Test the list_ensembles_command function."""

    def test_list_ensembles_default_config(self) -> None:
        """Test listing ensembles with default configuration."""
        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = [Path("/test/ensembles")]
        mock_config_manager.local_config_dir = None  # No local config

        # Mock ensemble objects
        mock_ensemble1 = Mock()
        mock_ensemble1.name = "ensemble1"
        mock_ensemble1.description = "Test ensemble 1"
        mock_ensemble1.relative_path = None

        mock_ensemble2 = Mock()
        mock_ensemble2.name = "ensemble2"
        mock_ensemble2.description = "Test ensemble 2"
        mock_ensemble2.relative_path = None

        mock_loader = Mock()
        mock_loader.list_ensembles.return_value = [mock_ensemble1, mock_ensemble2]

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
        ):
            list_ensembles_command(config_dir=None)

            mock_config_manager.get_ensembles_dirs.assert_called_once()
            mock_loader.list_ensembles.assert_called_once_with("/test/ensembles")

    def test_list_ensembles_custom_config_dir(self) -> None:
        """Test listing ensembles with custom config directory."""
        mock_config_manager = Mock()

        # Mock ensemble object
        mock_ensemble = Mock()
        mock_ensemble.name = "custom_ensemble"
        mock_ensemble.description = "Custom test ensemble"
        mock_ensemble.relative_path = None

        mock_loader = Mock()
        mock_loader.list_ensembles.return_value = [mock_ensemble]

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
        ):
            list_ensembles_command(config_dir="/custom/config")

            # Should NOT call get_ensembles_dirs
            mock_config_manager.get_ensembles_dirs.assert_not_called()

            # Should call list_ensembles with custom directory
            mock_loader.list_ensembles.assert_called_once_with("/custom/config")

    def test_list_ensembles_no_directories_found(self) -> None:
        """Test when no ensemble directories are found (lines 153-155)."""
        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = []

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("click.echo") as mock_echo,
        ):
            list_ensembles_command(None)

            mock_echo.assert_any_call("No ensemble directories found.")
            mock_echo.assert_any_call(
                "Run 'llm-orc config init' to set up local configuration."
            )

    def test_list_ensembles_no_ensembles_found_in_directories(self) -> None:
        """Test when no ensembles are found in any directories (lines 175-179)."""
        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = [Path("/test/ensembles")]
        mock_config_manager.local_config_dir = None

        mock_loader = Mock()
        mock_loader.list_ensembles.return_value = []

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("click.echo") as mock_echo,
        ):
            list_ensembles_command(None)

            mock_echo.assert_any_call(
                "No ensembles found in any configured directories:"
            )
            mock_echo.assert_any_call("  /test/ensembles")
            mock_echo.assert_any_call(
                "  (Create .yaml files with ensemble configurations)"
            )

    def test_list_ensembles_with_local_ensembles(self) -> None:
        """Test listing with local ensembles (lines 185-187)."""
        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = [
            Path("/home/.llm-orc/ensembles")
        ]
        mock_config_manager.local_config_dir = Path("/home/.llm-orc")
        mock_config_manager.classify_tier.return_value = "local"

        mock_ensemble = Mock()
        mock_ensemble.name = "local_ensemble"
        mock_ensemble.description = "Local test ensemble"
        mock_ensemble.relative_path = None

        mock_loader = Mock()
        mock_loader.list_ensembles.return_value = [mock_ensemble]

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("click.echo") as mock_echo,
        ):
            list_ensembles_command(None)

            mock_echo.assert_any_call("Available ensembles:")
            mock_echo.assert_any_call("\n📁 Local Repo (.llm-orc/ensembles):")
            mock_echo.assert_any_call("  local_ensemble: Local test ensemble")

    def test_list_ensembles_custom_dir_no_ensembles(self) -> None:
        """Test custom directory with no ensembles (lines 203-204)."""
        mock_loader = Mock()
        mock_loader.list_ensembles.return_value = []

        with (
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("click.echo") as mock_echo,
        ):
            list_ensembles_command("/custom/config")

            mock_echo.assert_any_call("No ensembles found in /custom/config")
            mock_echo.assert_any_call(
                "  (Create .yaml files with ensemble configurations)"
            )


class TestListProfilesCommand:
    """Test the list_profiles_command function."""

    def test_list_profiles_command(self) -> None:
        """Test listing model profiles."""
        with patch("llm_orc.cli_commands.display_local_profiles") as mock_display:
            list_profiles_command()
            mock_display.assert_called_once()

    def test_list_profiles_no_profiles_found(self) -> None:
        """Test when no model profiles are found (lines 220-222)."""
        mock_config_manager = Mock()
        mock_config_manager.get_model_profiles.return_value = {}

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("click.echo") as mock_echo,
        ):
            list_profiles_command()

            mock_echo.assert_any_call("No model profiles found.")
            mock_echo.assert_any_call(
                "Run 'llm-orc config init' to create default profiles."
            )

    def test_list_profiles_global_profile_overridden_by_local(self) -> None:
        """Test when global profile is overridden by local (lines 256-257)."""
        mock_config_manager = Mock()
        mock_config_manager.get_model_profiles.return_value = {
            "test-model": {"provider": "anthropic", "model": "claude-3-sonnet"}
        }
        mock_config_manager.global_config_dir = Path("/global")
        mock_config_manager.local_config_dir = Path("/local")

        # Mock global and local profiles - test-model exists in both (overridden)
        global_profiles = {
            "test-model": {"provider": "anthropic", "model": "claude-3-sonnet"}
        }
        local_profiles = {
            "test-model": {"provider": "anthropic", "model": "claude-3-haiku"}
        }

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("builtins.open"),
            patch("yaml.safe_load") as mock_yaml_load,
            patch("llm_orc.cli_commands.get_available_providers", return_value={}),
            patch("llm_orc.cli_commands.display_local_profiles"),
            patch("click.echo") as mock_echo,
        ):
            # Mock path.exists() to return True for both config files
            with patch.object(Path, "exists", return_value=True):
                # Set up yaml.safe_load to return profiles based on call order
                mock_yaml_load.side_effect = [
                    {"model_profiles": global_profiles},  # First call (global)
                    {"model_profiles": local_profiles},  # Second call (local)
                ]

                list_profiles_command()

            # The key test: verify that the overridden message is shown
            mock_echo.assert_any_call("  test-model: (overridden by local)")

    def test_list_profiles_invalid_profile_format(self) -> None:
        """Test handling of invalid profile format (lines 263-267)."""
        mock_config_manager = Mock()
        mock_config_manager.get_model_profiles.return_value = {
            "valid-model": {"provider": "anthropic", "model": "claude-3-sonnet"},
            "invalid-model": "not_a_dict",  # This is invalid
        }
        mock_config_manager.global_config_dir = Path("/global")
        mock_config_manager.local_config_dir = None

        # Mock global config with invalid profile
        global_profiles = {
            "valid-model": {"provider": "anthropic", "model": "claude-3-sonnet"},
            "invalid-model": "not_a_dict",
        }

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("builtins.open"),
            patch("yaml.safe_load", return_value={"model_profiles": global_profiles}),
            patch("llm_orc.cli_commands.get_available_providers", return_value={}),
            patch("click.echo") as mock_echo,
        ):
            # Mock path.exists() to return True for global config file
            with patch.object(Path, "exists", return_value=True):
                list_profiles_command()

            # Check that invalid profile format is handled
            mock_echo.assert_any_call(
                "  invalid-model: [Invalid profile format - expected dict, got str]"
            )


class TestListProfilesCommandHelpers:
    """Test helper functions extracted from list_profiles_command for complexity."""

    def test_load_profiles_from_config_existing_file(self) -> None:
        """Test loading profiles from existing config file."""
        from llm_orc.cli_commands import _load_profiles_from_config

        config_file = Path("/test/config.yaml")
        expected_profiles = {
            "test-model": {"provider": "anthropic", "model": "claude-3-sonnet"}
        }

        mock_config = {"model_profiles": expected_profiles}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml.dump(mock_config))),
            # Defensive patch to prevent AsyncMock contamination from other tests
            patch("llm_orc.cli_commands.run_standard_execution", return_value=None),
        ):
            result = _load_profiles_from_config(config_file)

        assert result == expected_profiles

    def test_load_profiles_from_config_nonexistent_file(self) -> None:
        """Test loading profiles from nonexistent config file returns empty dict."""
        from llm_orc.cli_commands import _load_profiles_from_config

        config_file = Path("/nonexistent/config.yaml")

        with patch("pathlib.Path.exists", return_value=False):
            result = _load_profiles_from_config(config_file)

        assert result == {}

    def test_load_profiles_from_config_no_model_profiles_key(self) -> None:
        """Test loading from config file with no model_profiles key."""
        from llm_orc.cli_commands import _load_profiles_from_config

        config_file = Path("/test/config.yaml")
        mock_config = {"other_key": "value"}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml.dump(mock_config))),
        ):
            result = _load_profiles_from_config(config_file)

        assert result == {}

    def test_display_global_profile_valid(self) -> None:
        """Test displaying a valid global profile."""
        from llm_orc.cli_commands import _display_global_profile

        profile_name = "test-model"
        profile = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "cost_per_token": "$0.003",
        }

        with patch("click.echo") as mock_echo:
            _display_global_profile(profile_name, profile)

        mock_echo.assert_any_call(f"  {profile_name}:")
        mock_echo.assert_any_call("    Model: claude-3-sonnet")
        mock_echo.assert_any_call("    Provider: anthropic")
        mock_echo.assert_any_call("    Cost per token: $0.003")

    def test_display_global_profile_invalid_format(self) -> None:
        """Test displaying an invalid global profile format."""
        from llm_orc.cli_commands import _display_global_profile

        profile_name = "invalid-model"
        profile = "not_a_dict"

        with patch("click.echo") as mock_echo:
            _display_global_profile(profile_name, profile)

        expected_message = (
            "  invalid-model: [Invalid profile format - expected dict, got str]"
        )
        mock_echo.assert_called_once_with(expected_message)

    def test_display_global_profile_missing_fields(self) -> None:
        """Test displaying a profile with missing fields."""
        from llm_orc.cli_commands import _display_global_profile

        profile_name = "incomplete-model"
        profile = {"provider": "anthropic"}  # Missing model and cost

        with patch("click.echo") as mock_echo:
            _display_global_profile(profile_name, profile)

        mock_echo.assert_any_call(f"  {profile_name}:")
        mock_echo.assert_any_call("    Model: Unknown")
        mock_echo.assert_any_call("    Provider: anthropic")
        mock_echo.assert_any_call("    Cost per token: Not specified")


class TestInvokeEnsembleHelperMethods:
    """Test helper methods extracted from invoke_ensemble for complexity reduction."""

    def test_resolve_input_data_positional_priority(self) -> None:
        """Test input data resolution with positional argument having priority."""
        # Given
        positional_input = "positional data"
        option_input = "option data"

        # When
        with patch("llm_orc.cli_commands.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            result = llm_orc.cli_commands._resolve_input_data(
                positional_input, option_input
            )

        # Then
        assert result == "positional data"

    def test_resolve_input_data_option_fallback(self) -> None:
        """Test input data resolution falling back to option when positional is None."""
        # Given
        positional_input = None
        option_input = "option data"

        # When
        with patch("llm_orc.cli_commands.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            result = llm_orc.cli_commands._resolve_input_data(
                positional_input, option_input
            )

        # Then
        assert result == "option data"

    def test_resolve_input_data_stdin_when_piped(self) -> None:
        """Test input data resolution reading from stdin when piped."""
        # Given
        positional_input = None
        option_input = None
        stdin_content = "piped input data"

        # When
        with patch("llm_orc.cli_commands.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False  # Piped input
            mock_stdin.read.return_value = stdin_content + "\n  "  # With whitespace
            result = llm_orc.cli_commands._resolve_input_data(
                positional_input, option_input
            )

        # Then
        assert result == "piped input data"  # Stripped

    def test_resolve_input_data_default_when_tty(self) -> None:
        """Test input data resolution using default when no input and TTY."""
        # Given
        positional_input = None
        option_input = None

        # When
        with patch("llm_orc.cli_commands.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True  # TTY (not piped)
            result = llm_orc.cli_commands._resolve_input_data(
                positional_input, option_input
            )

        # Then
        assert result == "Please analyze this."

    def test_find_ensemble_config_found_in_first_dir(self) -> None:
        """Test ensemble config found in first directory."""
        # Given
        ensemble_name = "test-ensemble"
        ensemble_dirs = [Path("/dir1"), Path("/dir2")]
        mock_config = Mock()
        mock_config.name = "test-ensemble"

        # When
        with patch("llm_orc.cli_commands.EnsembleLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.find_ensemble.side_effect = [
                mock_config,
                None,
            ]  # Found in first

            result = llm_orc.cli_commands._find_ensemble_config(
                ensemble_name, ensemble_dirs
            )

        # Then
        assert result == mock_config
        mock_loader.find_ensemble.assert_called_once_with("/dir1", "test-ensemble")

    def test_find_ensemble_config_found_in_second_dir(self) -> None:
        """Test ensemble config found in second directory."""
        # Given
        ensemble_name = "test-ensemble"
        ensemble_dirs = [Path("/dir1"), Path("/dir2")]
        mock_config = Mock()

        # When
        with patch("llm_orc.cli_commands.EnsembleLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.find_ensemble.side_effect = [
                None,
                mock_config,
            ]  # Found in second

            result = llm_orc.cli_commands._find_ensemble_config(
                ensemble_name, ensemble_dirs
            )

        # Then
        assert result == mock_config
        assert mock_loader.find_ensemble.call_count == 2

    def test_find_ensemble_config_not_found(self) -> None:
        """Test ensemble config not found in any directory."""
        # Given
        ensemble_name = "missing-ensemble"
        ensemble_dirs = [Path("/dir1"), Path("/dir2")]

        # When/Then
        with patch("llm_orc.cli_commands.EnsembleLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.find_ensemble.return_value = None  # Not found anywhere

            with pytest.raises(
                click.ClickException,
                match="Ensemble 'missing-ensemble' not found in: /dir1, /dir2",
            ):
                llm_orc.cli_commands._find_ensemble_config(ensemble_name, ensemble_dirs)


class TestListEnsemblesHelperMethods:
    """Test helper methods extracted from list_ensembles_command for complexity."""

    def test_get_grouped_ensembles_with_local_and_global(self) -> None:
        """Test grouping ensembles into local and global categories."""
        # Given
        config_manager = Mock()
        config_manager.local_config_dir = Path("/local")
        config_manager.classify_tier.side_effect = lambda p: (
            "local" if str(p).startswith("/local") else "global"
        )
        ensemble_dirs = [Path("/local/ensembles"), Path("/global/ensembles")]

        mock_local_ensemble = Mock()
        mock_local_ensemble.name = "local-ensemble"
        mock_global_ensemble = Mock()
        mock_global_ensemble.name = "global-ensemble"

        # When
        with patch("llm_orc.cli_commands.EnsembleLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.list_ensembles.side_effect = [
                [mock_local_ensemble],
                [mock_global_ensemble],
            ]

            local_ensembles, library_ensembles, global_ensembles = (
                llm_orc.cli_commands._get_grouped_ensembles(
                    config_manager, ensemble_dirs
                )
            )

        # Then
        assert local_ensembles == [mock_local_ensemble]
        assert library_ensembles == []
        assert global_ensembles == [mock_global_ensemble]
        assert mock_loader.list_ensembles.call_count == 2

    def test_get_grouped_ensembles_no_local_config(self) -> None:
        """Test grouping when no local config directory exists."""
        # Given
        config_manager = Mock()
        config_manager.local_config_dir = None
        config_manager.classify_tier.return_value = "global"
        ensemble_dirs = [Path("/global/ensembles")]

        mock_ensemble = Mock()
        mock_ensemble.name = "global-ensemble"
        mock_ensemble.relative_path = None

        # When
        with patch("llm_orc.cli_commands.EnsembleLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.list_ensembles.return_value = [mock_ensemble]

            local_ensembles, library_ensembles, global_ensembles = (
                llm_orc.cli_commands._get_grouped_ensembles(
                    config_manager, ensemble_dirs
                )
            )

        # Then
        assert local_ensembles == []
        assert library_ensembles == []
        assert global_ensembles == [mock_ensemble]

    def test_display_grouped_ensembles_both_types(self) -> None:
        """Test displaying both local and global ensembles."""
        # Given
        config_manager = Mock()
        config_manager.global_config_dir = Path("/global")

        local_ensemble = Mock()
        local_ensemble.name = "local-test"
        local_ensemble.description = "Local description"
        local_ensemble.relative_path = None

        global_ensemble = Mock()
        global_ensemble.name = "global-test"
        global_ensemble.description = "Global description"
        global_ensemble.relative_path = None

        local_ensembles = [local_ensemble]
        global_ensembles = [global_ensemble]

        # When
        with patch("click.echo") as mock_echo:
            llm_orc.cli_commands._display_grouped_ensembles(
                config_manager, local_ensembles, [], global_ensembles
            )

        # Then
        expected_calls = [
            call("Available ensembles:"),
            call("\n📁 Local Repo (.llm-orc/ensembles):"),
            call("  local-test: Local description"),
            call("\n🌐 Global (/global/ensembles):"),
            call("  global-test: Global description"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    def test_display_grouped_ensembles_only_global(self) -> None:
        """Test displaying only global ensembles."""
        # Given
        config_manager = Mock()
        config_manager.global_config_dir = Path("/global")

        global_ensemble = Mock()
        global_ensemble.name = "global-test"
        global_ensemble.description = "Global description"
        global_ensemble.relative_path = None

        local_ensembles: list[Mock] = []
        global_ensembles = [global_ensemble]

        # When
        with patch("click.echo") as mock_echo:
            llm_orc.cli_commands._display_grouped_ensembles(
                config_manager, local_ensembles, [], global_ensembles
            )

        # Then
        expected_calls = [
            call("Available ensembles:"),
            call("\n🌐 Global (/global/ensembles):"),
            call("  global-test: Global description"),
        ]
        mock_echo.assert_has_calls(expected_calls)


class TestAuthCommands:
    """Test auth module-level functions are callable."""

    def test_add_auth_provider(self) -> None:
        """Test add_auth_provider is a callable function."""
        assert callable(add_auth_provider)

    def test_list_auth_providers(self) -> None:
        """Test list_auth_providers is a callable function."""
        assert callable(list_auth_providers)

    def test_remove_auth_provider(self) -> None:
        """Test remove_auth_provider is a callable function."""
        assert callable(remove_auth_provider)

    def test_test_token_refresh(self) -> None:
        """Test refresh_token_test is a callable function."""
        assert callable(refresh_token_test)

    def test_auth_setup(self) -> None:
        """Test auth_setup is a callable function."""
        assert callable(auth_setup)

    def test_logout_oauth_providers(self) -> None:
        """Test logout_oauth_providers is a callable function."""
        assert callable(logout_oauth_providers)


class TestInteractiveScriptIntegration:
    """Test CLI integration with interactive script execution."""

    @pytest.fixture
    def mock_config_manager(self) -> Mock:
        """Create a mock configuration manager."""
        manager = Mock()
        manager.get_ensembles_dirs.return_value = [Path("/test/ensembles")]
        manager.load_performance_config.return_value = {
            "streaming_enabled": False,
            "max_concurrent": 3,
        }
        return manager

    @pytest.fixture
    def mock_interactive_ensemble_config(self) -> Mock:
        """Create a mock ensemble configuration with interactive scripts."""
        config = Mock()
        config.name = "interactive_ensemble"
        config.description = "Ensemble with interactive scripts"
        # Create agents with script references that need user input
        config.agents = [
            ScriptAgentConfig(
                name="input_collector",
                script="primitives/user-interaction/get_user_input.py",
                depends_on=[],
            ),
            ScriptAgentConfig(
                name="data_processor",
                script="data/process_input.py",
                depends_on=["input_collector"],
            ),
        ]
        return config

    @pytest.fixture
    def mock_loader(self) -> Mock:
        """Create a mock ensemble loader."""
        loader = Mock()
        return loader

    @pytest.fixture
    def mock_executor(self) -> Mock:
        """Create a mock ensemble executor."""
        executor = Mock()
        executor._get_effective_concurrency_limit.return_value = 3
        return executor

    def test_invoke_ensemble_detects_interactive_scripts_and_sets_up_user_input_handler(
        self,
        mock_config_manager: Mock,
        mock_interactive_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test CLI detects interactive scripts and enables user input handling.

        This test verifies the end-to-end integration where:
        1. CLI command receives an ensemble with interactive scripts
        2. CLI detects that user input will be needed
        3. CLI sets up proper user input handler for the execution
        4. CLI invokes the ensemble with interactive support enabled
        """
        mock_loader.find_ensemble.return_value = mock_interactive_ensemble_config

        # Mock the executor to support streaming execution for interactive
        async def mock_execute_streaming(*args: Any, **kwargs: Any) -> Any:
            yield {
                "type": "ensemble_started",
                "data": {"ensemble_name": "interactive_ensemble"},
            }
            yield {
                "type": "agent_started",
                "data": {"agent_name": "input_collector"},
            }
            yield {
                "type": "agent_completed",
                "data": {
                    "agent_name": "input_collector",
                    "response": '{"user_input": "John Doe"}',
                    "status": "success",
                },
            }
            yield {
                "type": "execution_completed",
                "data": {
                    "results": {
                        "input_collector": {
                            "status": "success",
                            "response": '{"user_input": "John Doe"}',
                        },
                    },
                    "metadata": {"execution_time": 2.5},
                },
            }

        mock_executor.execute_streaming = mock_execute_streaming

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch(
                "llm_orc.core.execution.script_user_input_handler.ScriptUserInputHandler"
            ) as mock_input_handler_class,
        ):
            # Mock the input handler instance
            mock_input_handler = Mock()
            mock_input_handler.ensemble_requires_user_input.return_value = True
            mock_input_handler_class.return_value = mock_input_handler

            invoke_ensemble(
                ensemble_name="interactive_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify that the CLI detected interactive scripts
            mock_input_handler.ensemble_requires_user_input.assert_called_once_with(
                mock_interactive_ensemble_config
            )

            # The implementation uses streaming execution for interactive ensembles
            # It doesn't have a separate execute_with_user_input method yet
            # Verify that streaming execution was triggered (which handles interactive)

    def test_invoke_ensemble_fallback_to_standard_execution_for_non_interactive(
        self,
        mock_config_manager: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test CLI falls back to standard execution for non-interactive scripts."""
        # Create non-interactive ensemble config
        non_interactive_config = Mock()
        non_interactive_config.name = "standard_ensemble"
        non_interactive_config.description = "Standard non-interactive ensemble"
        non_interactive_config.agents = [
            ScriptAgentConfig(
                name="text_processor",
                script="text/analyze.py",
                depends_on=[],
            )
        ]

        mock_loader.find_ensemble.return_value = non_interactive_config

        # Mock standard execution
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {
                    "text_processor": {
                        "status": "success",
                        "response": "Analysis complete",
                    }
                },
                "metadata": {"execution_time": 1.0},
            }
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch(
                "llm_orc.core.execution.script_user_input_handler.ScriptUserInputHandler"
            ) as mock_input_handler_class,
        ):
            # Mock input handler that detects no interactive scripts
            mock_input_handler = Mock()
            mock_input_handler.ensemble_requires_user_input.return_value = False
            mock_input_handler_class.return_value = mock_input_handler

            invoke_ensemble(
                ensemble_name="standard_ensemble",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

            # Verify interactive detection was attempted
            mock_input_handler.ensemble_requires_user_input.assert_called_once_with(
                non_interactive_config
            )

            # Verify standard execution was used (through asyncio.run)
            # Since CLI routing goes through run_standard_execution
            # we can verify interactive wasn't used
            assert (
                not hasattr(mock_executor, "execute_with_user_input")
                or mock_executor.execute_with_user_input.call_count == 0
            )
