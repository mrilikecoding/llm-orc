"""Tests for CLI commands complexity refactoring following strict TDD methodology.

This module contains tests specifically designed to verify the behavior
of complex functions before and after refactoring to reduce complexity.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.exceptions import ClickException

from llm_orc.cli_commands import invoke_ensemble


class TestInvokeEnsembleComplexityRefactor:
    """Test suite for invoke_ensemble complexity refactoring.

    These tests verify the exact behavior of the complex invoke_ensemble function
    before refactoring to ensure behavior is preserved.
    """

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
            {"name": "agent_1", "depends_on": []},
            {"name": "agent_2", "depends_on": ["agent_1"]},
        ]
        return config

    def test_invoke_ensemble_config_dir_none_uses_config_manager_dirs(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble with config_dir=None uses config manager directories.

        This tests the first complexity branch: config_dir determination.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
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
                ensemble_name="test",
                input_data="test input",
                config_dir=None,  # This triggers the complex branch
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

        # Verify config manager was used for dirs
        mock_config_manager.get_ensembles_dirs.assert_called_once()

    def test_invoke_ensemble_config_dir_provided_uses_custom_dir(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble with config_dir provided uses custom directory.

        This tests the else branch of config_dir determination.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        custom_dir = "/custom/config"

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
        ):
            invoke_ensemble(
                ensemble_name="test",
                input_data="test input",
                config_dir=custom_dir,  # This triggers the else branch
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

        # Verify config manager dirs was NOT called
        mock_config_manager.get_ensembles_dirs.assert_not_called()
        # Verify loader was called with custom dir
        mock_loader.find_ensemble.assert_called_once_with(custom_dir, "test")

    def test_invoke_ensemble_max_concurrent_override_branch(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble max_concurrent override logic.

        This tests the max_concurrent handling branch (currently a pass statement).
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
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
                ensemble_name="test",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=5,  # This triggers the max_concurrent branch
                detailed=False,
            )

        # This should complete without error (testing the pass statement)
        mock_executor.execute.assert_called_once()

    def test_invoke_ensemble_performance_config_display_with_rich_output(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble performance config display with Rich output format.

        This tests the complex performance display logic branch.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        # Mock coordinator for performance display
        mock_coordinator = Mock()
        mock_coordinator.get_effective_concurrency_limit.return_value = 3
        mock_executor._execution_coordinator = mock_coordinator

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
                ensemble_name="test",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format=None,  # Rich output triggers performance display
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

        # Verify performance config was loaded and displayed
        mock_config_manager.load_performance_config.assert_called()
        mock_echo.assert_called()

    def test_invoke_ensemble_performance_config_exception_fallback(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble performance config exception fallback branch.

        This tests the exception handling in performance display logic.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        # Mock performance config to raise exception
        mock_config_manager.load_performance_config.side_effect = Exception(
            "Config error"
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
                ensemble_name="test",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format=None,  # Rich output triggers performance display
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

        # Verify fallback output was used
        mock_echo.assert_called()
        # Should include fallback messages
        echo_calls = [
            str(call.args[0]) for call in mock_echo.call_args_list if call.args
        ]
        echo_output = " ".join(echo_calls)
        assert "Invoking ensemble: test" in echo_output

    def test_invoke_ensemble_streaming_determination_text_format(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble streaming determination for text format.

        This tests the streaming determination logic complexity branch.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
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
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard,
            patch("llm_orc.cli_commands.run_streaming_execution") as mock_streaming,
            patch("llm_orc.cli_commands.asyncio.run"),
        ):
            invoke_ensemble(
                ensemble_name="test",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="text",  # Text format should force standard execution
                streaming=True,  # Even with streaming=True
                max_concurrent=None,
                detailed=False,
            )

        # Should call standard execution, not streaming
        mock_standard.assert_called_once()
        mock_streaming.assert_not_called()

    def test_invoke_ensemble_streaming_determination_rich_format_with_config(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble streaming determination for Rich format with config.

        This tests the streaming configuration loading branch.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        mock_config_manager.load_performance_config.return_value = {
            "streaming_enabled": True  # Config enables streaming
        }

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard,
            patch("llm_orc.cli_commands.run_streaming_execution") as mock_streaming,
            patch("llm_orc.cli_commands.asyncio.run"),
        ):
            invoke_ensemble(
                ensemble_name="test",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format=None,  # Rich format respects config
                streaming=False,  # CLI flag disabled
                max_concurrent=None,
                detailed=False,
            )

        # Should call streaming execution due to config
        mock_streaming.assert_called_once()
        mock_standard.assert_not_called()

    def test_invoke_ensemble_interactive_execution_branch(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble interactive execution branch.

        This tests the interactive script detection and execution logic.
        """
        mock_loader = Mock()
        mock_loader.find_ensemble.return_value = mock_ensemble_config

        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        # Mock user input handler to detect interactive scripts
        mock_input_handler = Mock()
        mock_input_handler.ensemble_requires_user_input.return_value = True

        with (
            patch(
                "llm_orc.cli_commands.ConfigurationManager",
                return_value=mock_config_manager,
            ),
            patch("llm_orc.cli_commands.EnsembleLoader", return_value=mock_loader),
            patch("llm_orc.cli_commands.EnsembleExecutor", return_value=mock_executor),
            patch(
                "llm_orc.core.execution.script_user_input_handler.ScriptUserInputHandler",
                return_value=mock_input_handler,
            ),
            patch("llm_orc.cli_commands.run_streaming_execution") as mock_streaming,
            patch("llm_orc.cli_commands.asyncio.run"),
        ):
            invoke_ensemble(
                ensemble_name="test",
                input_data="test input",
                config_dir=None,
                input_data_option=None,
                output_format="json",
                streaming=False,
                max_concurrent=None,
                detailed=False,
            )

        # Should call streaming execution for interactive scripts
        mock_streaming.assert_called_once()
        mock_input_handler.ensemble_requires_user_input.assert_called_once()


class TestInvokeEnsembleRefactoredFunctions:
    """Test suite for the helper functions extracted from invoke_ensemble.

    These tests verify that the extracted helper functions work correctly
    and preserve the original behavior.
    """

    def test_determine_ensemble_directories_with_config_manager(self) -> None:
        """Test helper function to determine ensemble directories from config manager."""
        from llm_orc.cli_commands import _determine_ensemble_directories

        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = [Path("/test/ensembles")]

        result = _determine_ensemble_directories(mock_config_manager, None)

        assert result == [Path("/test/ensembles")]
        mock_config_manager.get_ensembles_dirs.assert_called_once()

    def test_determine_ensemble_directories_with_custom_dir(self) -> None:
        """Test helper function to determine ensemble directories from custom dir."""
        from llm_orc.cli_commands import _determine_ensemble_directories

        mock_config_manager = Mock()
        custom_dir = "/custom/config"

        result = _determine_ensemble_directories(mock_config_manager, custom_dir)

        assert result == [Path(custom_dir)]
        mock_config_manager.get_ensembles_dirs.assert_not_called()

    def test_determine_ensemble_directories_no_dirs_found_raises_exception(
        self,
    ) -> None:
        """Test helper function raises exception when no directories found."""
        from llm_orc.cli_commands import _determine_ensemble_directories

        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = []

        with pytest.raises(ClickException, match="No ensemble directories found"):
            _determine_ensemble_directories(mock_config_manager, None)

    def test_setup_performance_display_success(self) -> None:
        """Test helper function to setup performance display."""
        from llm_orc.cli_commands import _setup_performance_display

        mock_config_manager = Mock()
        mock_config_manager.load_performance_config.return_value = {
            "streaming_enabled": True
        }

        mock_executor = Mock()
        mock_coordinator = Mock()
        mock_coordinator.get_effective_concurrency_limit.return_value = 3
        mock_executor._execution_coordinator = mock_coordinator

        mock_ensemble_config = Mock()
        mock_ensemble_config.agents = [{"name": "agent1"}, {"name": "agent2"}]

        with patch("llm_orc.cli_commands.click.echo") as mock_echo:
            _setup_performance_display(
                mock_config_manager,
                mock_executor,
                "test_ensemble",
                mock_ensemble_config,
                False,
                None,  # Rich output format
                "test input",
            )

        mock_echo.assert_called()
        mock_config_manager.load_performance_config.assert_called_once()

    def test_setup_performance_display_skips_for_text_output(self) -> None:
        """Test helper function skips display for text/json output."""
        from llm_orc.cli_commands import _setup_performance_display

        mock_config_manager = Mock()
        mock_executor = Mock()
        mock_ensemble_config = Mock()

        with patch("llm_orc.cli_commands.click.echo") as mock_echo:
            _setup_performance_display(
                mock_config_manager,
                mock_executor,
                "test_ensemble",
                mock_ensemble_config,
                False,
                "text",  # Text output format
                "test input",
            )

        mock_echo.assert_not_called()
        mock_config_manager.load_performance_config.assert_not_called()

    def test_setup_performance_display_fallback(self) -> None:
        """Test helper function to setup performance display fallback."""
        from llm_orc.cli_commands import _setup_performance_display

        mock_config_manager = Mock()
        mock_config_manager.load_performance_config.side_effect = Exception(
            "Config error"
        )

        mock_executor = Mock()
        mock_ensemble_config = Mock()
        mock_ensemble_config.description = "Test ensemble"
        mock_ensemble_config.agents = [{"name": "agent1"}]

        with patch("llm_orc.cli_commands.click.echo") as mock_echo:
            _setup_performance_display(
                mock_config_manager,
                mock_executor,
                "test_ensemble",
                mock_ensemble_config,
                False,
                None,  # Rich output format
                "test input",
            )

        mock_echo.assert_called()
        # Should use fallback display
        echo_calls = [
            str(call.args[0]) for call in mock_echo.call_args_list if call.args
        ]
        echo_output = " ".join(echo_calls)
        assert "Invoking ensemble: test_ensemble" in echo_output

    def test_determine_effective_streaming_text_format(self) -> None:
        """Test helper function to determine effective streaming for text format."""
        from llm_orc.cli_commands import _determine_effective_streaming

        mock_config_manager = Mock()

        result = _determine_effective_streaming(mock_config_manager, "text", True)

        assert result is False  # Text format always uses standard execution
        mock_config_manager.load_performance_config.assert_not_called()

    def test_determine_effective_streaming_rich_format_with_config(self) -> None:
        """Test helper function to determine effective streaming for rich format."""
        from llm_orc.cli_commands import _determine_effective_streaming

        mock_config_manager = Mock()
        mock_config_manager.load_performance_config.return_value = {
            "streaming_enabled": True
        }

        result = _determine_effective_streaming(
            mock_config_manager,
            "rich",
            False,  # CLI flag disabled
        )

        assert result is True  # Config enables streaming
        mock_config_manager.load_performance_config.assert_called_once()

    def test_determine_effective_streaming_exception_fallback(self) -> None:
        """Test helper function falls back to CLI flag on config exception."""
        from llm_orc.cli_commands import _determine_effective_streaming

        mock_config_manager = Mock()
        mock_config_manager.load_performance_config.side_effect = Exception(
            "Config error"
        )

        result = _determine_effective_streaming(
            mock_config_manager,
            "rich",
            True,  # CLI flag enabled
        )

        assert result is True  # Falls back to CLI flag

    def test_execute_ensemble_interactive(self) -> None:
        """Test helper function to execute ensemble with interactive scripts."""
        from llm_orc.cli_commands import _execute_ensemble_with_mode

        mock_executor = Mock()
        mock_ensemble_config = Mock()

        with (
            patch("llm_orc.cli_commands.asyncio.run") as mock_asyncio,
            patch("llm_orc.cli_commands.run_streaming_execution") as mock_streaming,
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard,
        ):
            _execute_ensemble_with_mode(
                mock_executor,
                mock_ensemble_config,
                "test input",
                "json",
                False,
                True,  # requires_user_input = True
                False,  # effective_streaming = False
            )

        mock_streaming.assert_called_once()
        mock_standard.assert_not_called()
        mock_asyncio.assert_called_once()

    def test_execute_ensemble_streaming(self) -> None:
        """Test helper function to execute ensemble with streaming."""
        from llm_orc.cli_commands import _execute_ensemble_with_mode

        mock_executor = Mock()
        mock_ensemble_config = Mock()

        with (
            patch("llm_orc.cli_commands.asyncio.run") as mock_asyncio,
            patch("llm_orc.cli_commands.run_streaming_execution") as mock_streaming,
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard,
        ):
            _execute_ensemble_with_mode(
                mock_executor,
                mock_ensemble_config,
                "test input",
                "json",
                False,
                False,  # requires_user_input = False
                True,  # effective_streaming = True
            )

        mock_streaming.assert_called_once()
        mock_standard.assert_not_called()
        mock_asyncio.assert_called_once()

    def test_execute_ensemble_standard(self) -> None:
        """Test helper function to execute ensemble with standard execution."""
        from llm_orc.cli_commands import _execute_ensemble_with_mode

        mock_executor = Mock()
        mock_ensemble_config = Mock()

        with (
            patch("llm_orc.cli_commands.asyncio.run") as mock_asyncio,
            patch("llm_orc.cli_commands.run_streaming_execution") as mock_streaming,
            patch("llm_orc.cli_commands.run_standard_execution") as mock_standard,
        ):
            _execute_ensemble_with_mode(
                mock_executor,
                mock_ensemble_config,
                "test input",
                "json",
                False,
                False,  # requires_user_input = False
                False,  # effective_streaming = False
            )

        mock_standard.assert_called_once()
        mock_streaming.assert_not_called()
        mock_asyncio.assert_called_once()
