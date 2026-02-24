"""Tests for CLI commands complexity refactoring following strict TDD methodology.

This module contains tests specifically designed to verify the behavior
of complex functions before and after refactoring to reduce complexity.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.cli_commands import invoke_ensemble
from llm_orc.schemas.agent_config import LlmAgentConfig


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
            LlmAgentConfig(name="agent_1", model_profile="test", depends_on=[]),
            LlmAgentConfig(
                name="agent_2", model_profile="test", depends_on=["agent_1"]
            ),
        ]
        return config

    def test_invoke_ensemble_config_dir_none_uses_config_manager_dirs(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble with config_dir=None uses service find_ensemble.

        This tests the first complexity branch: config_dir determination.
        """
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service.find_ensemble_by_name.return_value = mock_ensemble_config
        mock_service._get_executor.return_value = mock_executor

        with patch("llm_orc.cli_commands._get_service", return_value=mock_service):
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

        # Verify service was used for ensemble lookup
        mock_service.find_ensemble_by_name.assert_called_once_with("test")

    def test_invoke_ensemble_config_dir_provided_uses_custom_dir(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble with config_dir provided uses custom directory.

        This tests the else branch of config_dir determination.
        """
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        custom_dir = "/custom/config"

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service._get_executor.return_value = mock_executor
        mock_service.find_ensemble_in_dir.return_value = mock_ensemble_config

        with patch("llm_orc.cli_commands._get_service", return_value=mock_service):
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

        # Verify service find_ensemble_by_name was NOT called for custom dir
        mock_service.find_ensemble_by_name.assert_not_called()
        # Verify service was called with custom dir via _find_ensemble_config
        mock_service.find_ensemble_in_dir.assert_called_once_with("test", custom_dir)

    def test_invoke_ensemble_max_concurrent_override_branch(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble max_concurrent override logic.

        This tests the max_concurrent handling branch (currently a pass statement).
        """
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service.find_ensemble_by_name.return_value = mock_ensemble_config
        mock_service._get_executor.return_value = mock_executor

        with patch("llm_orc.cli_commands._get_service", return_value=mock_service):
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

    def test_invoke_ensemble_performance_display_with_rich_output(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble performance display with Rich output format."""
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service.find_ensemble_by_name.return_value = mock_ensemble_config
        mock_service._get_executor.return_value = mock_executor

        with (
            patch("llm_orc.cli_commands._get_service", return_value=mock_service),
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

        mock_echo.assert_called()
        echo_calls = [
            str(call.args[0]) for call in mock_echo.call_args_list if call.args
        ]
        echo_output = " ".join(echo_calls)
        assert "Executing ensemble 'test'" in echo_output

    def test_invoke_ensemble_streaming_determination_text_format(
        self,
        mock_config_manager: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test invoke_ensemble streaming determination for text format.

        This tests the streaming determination logic complexity branch.
        """
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent": {"status": "success", "response": "Test"}},
                "metadata": {"execution_time": 1.5},
            }
        )

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service.find_ensemble_by_name.return_value = mock_ensemble_config
        mock_service._get_executor.return_value = mock_executor

        with (
            patch("llm_orc.cli_commands._get_service", return_value=mock_service),
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

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service.find_ensemble_by_name.return_value = mock_ensemble_config
        mock_service._get_executor.return_value = mock_executor

        with (
            patch("llm_orc.cli_commands._get_service", return_value=mock_service),
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

        mock_service = Mock()
        mock_service.config_manager = mock_config_manager
        mock_service.find_ensemble_by_name.return_value = mock_ensemble_config
        mock_service._get_executor.return_value = mock_executor

        with (
            patch("llm_orc.cli_commands._get_service", return_value=mock_service),
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

    def test_setup_performance_display_success(self) -> None:
        """Test helper function to setup performance display."""
        from llm_orc.cli_commands import _setup_performance_display

        mock_config_manager = Mock()
        mock_executor = Mock()
        mock_ensemble_config = Mock()
        mock_ensemble_config.agents = [
            LlmAgentConfig(name="agent1", model_profile="test"),
            LlmAgentConfig(name="agent2", model_profile="test"),
        ]

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
        echo_calls = [
            str(call.args[0]) for call in mock_echo.call_args_list if call.args
        ]
        echo_output = " ".join(echo_calls)
        assert "Executing ensemble 'test_ensemble'" in echo_output
        assert "2 agents" in echo_output

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
