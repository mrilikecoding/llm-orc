"""Tests for CLI command implementations."""

import asyncio
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, call, mock_open, patch

import click
import pytest
import yaml
from click.exceptions import ClickException

import llm_orc.cli_commands
from llm_orc.cli_commands import (
    add_auth_provider,
    auth_setup,
    check_global_config,
    check_local_config,
    init_local_config,
    invoke_ensemble,
    list_auth_providers,
    list_ensembles_command,
    list_profiles_command,
    logout_oauth_providers,
    refresh_token_test,
    remove_auth_provider,
    reset_global_config,
    reset_local_config,
    serve_ensemble,
)


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
        config.agents = [Mock(), Mock()]  # 2 agents
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

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run") as mock_run,
            patch("llm_orc.cli_commands.run_standard_execution"),
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

            # Verify asyncio.run was called with standard execution
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert asyncio.iscoroutine(call_args)

    def test_invoke_ensemble_custom_config_dir(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test ensemble invocation with custom config directory."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution"),
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

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard_exec,
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

            # The actual input data passed to execution should be positional
            mock_run_call = mock_standard_exec.call_args
            # input_data should be the third argument
            assert "positional_input" in str(mock_run_call)

    def test_invoke_ensemble_fallback_to_option_input(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test fallback to option input when positional is None."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard_exec,
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

            # The input data passed should be from option
            mock_run_call = mock_standard_exec.call_args
            assert "option_input" in str(mock_run_call)

    def test_invoke_ensemble_stdin_input(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test reading input from stdin when no input provided."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

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
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard_exec,
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

            # Should use stdin data
            mock_run_call = mock_standard_exec.call_args
            assert stdin_data in str(mock_run_call)

    def test_invoke_ensemble_default_input(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test default input when no input provided and not piped."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

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
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard_exec,
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

            # Should use default input
            mock_run_call = mock_standard_exec.call_args
            assert "Please analyze this." in str(mock_run_call)

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
            patch("llm_orc.cli_commands.asyncio.run"),
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
                output_format="json",
                streaming=True,  # Enable streaming
                max_concurrent=None,
                detailed=False,
            )

            # Should NOT call standard execution
            mock_standard_exec.assert_not_called()

            # Should call streaming execution
            mock_streaming_exec.assert_called_once()

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
                output_format="json",
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

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution"),
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
                detailed=False,
            )

            # Should print performance information
            mock_echo.assert_called()
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            echo_output = " ".join(echo_calls)

            assert "test_ensemble" in echo_output
            assert "agents" in echo_output
            assert "Performance" in echo_output

    def test_invoke_ensemble_text_output_performance_fallback(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
        mock_loader: Mock,
        mock_executor: Mock,
    ) -> None:
        """Test text output fallback when first performance config call fails."""
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        # Mock load_performance_config to fail on first call but succeed on second
        call_count = 0

        def performance_config_side_effect() -> dict[str, bool]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call (text output) fails
                raise Exception("Config error")
            else:  # Second call (streaming decision) succeeds
                return {"streaming_enabled": False}

        mock_config_manager.load_performance_config.side_effect = (
            performance_config_side_effect
        )

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("llm_orc.cli_commands.run_standard_execution"),
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
                detailed=False,
            )

            # Should print fallback information
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            echo_output = " ".join(echo_calls)

            assert "test_ensemble" in echo_output
            assert "Test ensemble for testing" in echo_output  # Description

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

        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.asyncio.run"),
            patch("click.echo"),
        ):
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

        mock_ensemble2 = Mock()
        mock_ensemble2.name = "ensemble2"
        mock_ensemble2.description = "Test ensemble 2"

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

        mock_ensemble = Mock()
        mock_ensemble.name = "local_ensemble"
        mock_ensemble.description = "Local test ensemble"

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


class TestServeEnsemble:
    """Test the serve_ensemble function."""

    def test_serve_ensemble(self) -> None:
        """Test serving an ensemble as MCP server."""
        mock_runner = Mock()
        mock_runner.run = Mock()

        with patch(
            "llm_orc.cli_commands.MCPServerRunner", return_value=mock_runner
        ) as mock_runner_class:
            serve_ensemble("test_ensemble", 8080)

            # Verify runner was created with correct parameters
            mock_runner_class.assert_called_once_with("test_ensemble", 8080)

            # Verify runner.run() was called
            mock_runner.run.assert_called_once()


class TestConfigCommands:
    """Test configuration-related command functions."""

    def test_init_local_config(self) -> None:
        """Test init local config command delegation."""
        with patch("llm_orc.cli_commands.ConfigCommands") as mock_config_class:
            init_local_config("test_project")
            mock_config_class.init_local_config.assert_called_once_with("test_project")

    def test_reset_global_config(self) -> None:
        """Test reset global config command delegation."""
        with patch("llm_orc.cli_commands.ConfigCommands") as mock_config_class:
            reset_global_config(True, False)
            mock_config_class.reset_global_config.assert_called_once_with(True, False)

    def test_check_global_config(self) -> None:
        """Test check global config command delegation."""
        with patch("llm_orc.cli_commands.ConfigCommands") as mock_config_class:
            check_global_config()
            mock_config_class.check_global_config.assert_called_once()

    def test_check_local_config(self) -> None:
        """Test check local config command delegation."""
        with patch("llm_orc.cli_commands.ConfigCommands") as mock_config_class:
            check_local_config()
            mock_config_class.check_local_config.assert_called_once()

    def test_reset_local_config(self) -> None:
        """Test reset local config command delegation."""
        with patch("llm_orc.cli_commands.ConfigCommands") as mock_config_class:
            reset_local_config(False, True, "project")
            mock_config_class.reset_local_config.assert_called_once_with(
                False, True, "project"
            )


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

            local_ensembles, global_ensembles = (
                llm_orc.cli_commands._get_grouped_ensembles(
                    config_manager, ensemble_dirs
                )
            )

        # Then
        assert local_ensembles == [mock_local_ensemble]
        assert global_ensembles == [mock_global_ensemble]
        assert mock_loader.list_ensembles.call_count == 2

    def test_get_grouped_ensembles_no_local_config(self) -> None:
        """Test grouping when no local config directory exists."""
        # Given
        config_manager = Mock()
        config_manager.local_config_dir = None
        ensemble_dirs = [Path("/global/ensembles")]

        mock_ensemble = Mock()
        mock_ensemble.name = "global-ensemble"

        # When
        with patch("llm_orc.cli_commands.EnsembleLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.list_ensembles.return_value = [mock_ensemble]

            local_ensembles, global_ensembles = (
                llm_orc.cli_commands._get_grouped_ensembles(
                    config_manager, ensemble_dirs
                )
            )

        # Then
        assert local_ensembles == []
        assert global_ensembles == [mock_ensemble]

    def test_display_grouped_ensembles_both_types(self) -> None:
        """Test displaying both local and global ensembles."""
        # Given
        config_manager = Mock()
        config_manager.global_config_dir = Path("/global")

        local_ensemble = Mock()
        local_ensemble.name = "local-test"
        local_ensemble.description = "Local description"

        global_ensemble = Mock()
        global_ensemble.name = "global-test"
        global_ensemble.description = "Global description"

        local_ensembles = [local_ensemble]
        global_ensembles = [global_ensemble]

        # When
        with patch("click.echo") as mock_echo:
            llm_orc.cli_commands._display_grouped_ensembles(
                config_manager, local_ensembles, global_ensembles
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

        local_ensembles: list[Mock] = []
        global_ensembles = [global_ensemble]

        # When
        with patch("click.echo") as mock_echo:
            llm_orc.cli_commands._display_grouped_ensembles(
                config_manager, local_ensembles, global_ensembles
            )

        # Then
        expected_calls = [
            call("Available ensembles:"),
            call("\n🌐 Global (/global/ensembles):"),
            call("  global-test: Global description"),
        ]
        mock_echo.assert_has_calls(expected_calls)


class TestAuthCommands:
    """Test authentication-related command functions."""

    def test_add_auth_provider(self) -> None:
        """Test add auth provider command delegation."""
        with patch("llm_orc.cli_commands.AuthCommands") as mock_auth_class:
            add_auth_provider("provider", "key", "client_id", "secret")
            mock_auth_class.add_auth_provider.assert_called_once_with(
                "provider", "key", "client_id", "secret"
            )

    def test_list_auth_providers(self) -> None:
        """Test list auth providers command delegation."""
        with patch("llm_orc.cli_commands.AuthCommands") as mock_auth_class:
            list_auth_providers(True)
            mock_auth_class.list_auth_providers.assert_called_once_with(True)

    def test_remove_auth_provider(self) -> None:
        """Test remove auth provider command delegation."""
        with patch("llm_orc.cli_commands.AuthCommands") as mock_auth_class:
            remove_auth_provider("provider")
            mock_auth_class.remove_auth_provider.assert_called_once_with("provider")

    def test_test_token_refresh(self) -> None:
        """Test token refresh command delegation."""
        with patch("llm_orc.cli_commands.AuthCommands") as mock_auth_class:
            refresh_token_test("provider")
            mock_auth_class.test_token_refresh.assert_called_once_with("provider")

    def test_auth_setup(self) -> None:
        """Test auth setup command delegation."""
        with patch("llm_orc.cli_commands.AuthCommands") as mock_auth_class:
            auth_setup()
            mock_auth_class.auth_setup.assert_called_once()

    def test_logout_oauth_providers(self) -> None:
        """Test logout OAuth providers command delegation."""
        with patch("llm_orc.cli_commands.AuthCommands") as mock_auth_class:
            logout_oauth_providers("provider", False)
            mock_auth_class.logout_oauth_providers.assert_called_once_with(
                "provider", False
            )
