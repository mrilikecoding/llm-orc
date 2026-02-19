"""Tests for CLI utility functions."""

from unittest.mock import Mock, patch

from llm_orc.cli_modules.utils.cli_utils import (
    echo_error,
    echo_info,
    echo_success,
)


class TestEchoFunctions:
    """Test echo utility functions."""

    @patch("click.echo")
    def test_echo_success(self, mock_echo: Mock) -> None:
        """Test success message echo."""
        # Given
        message = "Operation completed"

        # When
        echo_success(message)

        # Then
        mock_echo.assert_called_once_with("✅ Operation completed")

    @patch("click.echo")
    def test_echo_error(self, mock_echo: Mock) -> None:
        """Test error message echo."""
        # Given
        message = "Operation failed"

        # When
        echo_error(message)

        # Then
        mock_echo.assert_called_once_with("❌ Operation failed")

    @patch("click.echo")
    def test_echo_info(self, mock_echo: Mock) -> None:
        """Test info message echo."""
        # Given
        message = "Information message"

        # When
        echo_info(message)

        # Then
        mock_echo.assert_called_once_with("ℹ️  Information message")
