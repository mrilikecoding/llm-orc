"""Comprehensive tests for CLI utility functions."""

from typing import Any
from unittest.mock import Mock, patch

import click
import pytest

from llm_orc.cli_modules.utils.cli_utils import (
    confirm_destructive_action,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
    format_key_value_output,
    format_list_output,
    handle_cli_error,
    truncate_string,
    validate_required_param,
)


class TestHandleCliError:
    """Test error handling utility."""

    def test_handle_cli_error_basic(self) -> None:
        """Test basic error handling."""
        # Given
        operation = "test operation"
        error = ValueError("test error")

        # When / Then
        with pytest.raises(click.ClickException) as exc_info:
            handle_cli_error(operation, error)

        assert "test operation failed: test error" in str(exc_info.value)

    def test_handle_cli_error_chaining(self) -> None:
        """Test error chaining."""
        # Given
        operation = "complex operation"
        error = RuntimeError("runtime issue")

        # When / Then
        with pytest.raises(click.ClickException) as exc_info:
            handle_cli_error(operation, error)

        assert exc_info.value.__cause__ is error


class TestConfirmDestructiveAction:
    """Test destructive action confirmation."""

    @patch("click.confirm")
    def test_confirm_destructive_action_yes(self, mock_confirm: Mock) -> None:
        """Test user confirmation - yes."""
        # Given
        mock_confirm.return_value = True
        message = "Delete all data"

        # When
        result = confirm_destructive_action(message)

        # Then
        assert result is True
        mock_confirm.assert_called_once_with("⚠️  Delete all data", default=False)

    @patch("click.confirm")
    def test_confirm_destructive_action_no(self, mock_confirm: Mock) -> None:
        """Test user confirmation - no."""
        # Given
        mock_confirm.return_value = False
        message = "Delete all data"

        # When
        result = confirm_destructive_action(message)

        # Then
        assert result is False
        mock_confirm.assert_called_once_with("⚠️  Delete all data", default=False)


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

    @patch("click.echo")
    def test_echo_warning(self, mock_echo: Mock) -> None:
        """Test warning message echo."""
        # Given
        message = "Warning message"

        # When
        echo_warning(message)

        # Then
        mock_echo.assert_called_once_with("⚠️  Warning message")


class TestValidateRequiredParam:
    """Test parameter validation utility."""

    def test_validate_required_param_valid_string(self) -> None:
        """Test validation with valid string parameter."""
        # Given
        value = "valid value"
        param_name = "test_param"

        # When / Then - should not raise
        validate_required_param(value, param_name)

    def test_validate_required_param_valid_non_string(self) -> None:
        """Test validation with valid non-string parameter."""
        # Given
        value = 123
        param_name = "test_param"

        # When / Then - should not raise
        validate_required_param(value, param_name)

    def test_validate_required_param_none(self) -> None:
        """Test validation with None parameter."""
        # Given
        value = None
        param_name = "test_param"

        # When / Then
        with pytest.raises(click.ClickException) as exc_info:
            validate_required_param(value, param_name)

        assert "Missing required parameter: test_param" in str(exc_info.value)

    def test_validate_required_param_empty_string(self) -> None:
        """Test validation with empty string parameter."""
        # Given
        value = ""
        param_name = "test_param"

        # When / Then
        with pytest.raises(click.ClickException) as exc_info:
            validate_required_param(value, param_name)

        assert "Missing required parameter: test_param" in str(exc_info.value)

    def test_validate_required_param_whitespace_string(self) -> None:
        """Test validation with whitespace-only string parameter."""
        # Given
        value = "   "
        param_name = "test_param"

        # When / Then
        with pytest.raises(click.ClickException) as exc_info:
            validate_required_param(value, param_name)

        assert "Missing required parameter: test_param" in str(exc_info.value)


class TestFormatListOutput:
    """Test list output formatting."""

    @patch("click.echo")
    @patch("llm_orc.cli_modules.utils.cli_utils.echo_info")
    def test_format_list_output_empty_list(
        self, mock_echo_info: Mock, mock_echo: Mock
    ) -> None:
        """Test formatting empty list."""
        # Given
        items: list[str] = []

        # When
        format_list_output(items)

        # Then
        mock_echo_info.assert_called_once_with("No items found")
        mock_echo.assert_not_called()

    @patch("click.echo")
    @patch("llm_orc.cli_modules.utils.cli_utils.echo_info")
    def test_format_list_output_empty_list_custom_message(
        self, mock_echo_info: Mock, mock_echo: Mock
    ) -> None:
        """Test formatting empty list with custom message."""
        # Given
        items: list[str] = []
        empty_message = "No data available"

        # When
        format_list_output(items, empty_message=empty_message)

        # Then
        mock_echo_info.assert_called_once_with("No data available")
        mock_echo.assert_not_called()

    @patch("click.echo")
    def test_format_list_output_with_items(self, mock_echo: Mock) -> None:
        """Test formatting list with items."""
        # Given
        items = ["item1", "item2", "item3"]

        # When
        format_list_output(items)

        # Then
        expected_calls = [
            (("  item1",),),
            (("  item2",),),
            (("  item3",),),
        ]
        assert mock_echo.call_args_list == expected_calls

    @patch("click.echo")
    def test_format_list_output_custom_prefix(self, mock_echo: Mock) -> None:
        """Test formatting list with custom prefix."""
        # Given
        items = ["item1", "item2"]
        prefix = "* "

        # When
        format_list_output(items, prefix=prefix)

        # Then
        expected_calls = [
            (("* item1",),),
            (("* item2",),),
        ]
        assert mock_echo.call_args_list == expected_calls


class TestFormatKeyValueOutput:
    """Test key-value output formatting."""

    @patch("click.echo")
    def test_format_key_value_output_basic(self, mock_echo: Mock) -> None:
        """Test basic key-value formatting."""
        # Given
        data = {"key1": "value1", "key2": "value2"}

        # When
        format_key_value_output(data)

        # Then
        expected_calls = [
            (("  key1: value1",),),
            (("  key2: value2",),),
        ]
        assert mock_echo.call_args_list == expected_calls

    @patch("click.echo")
    def test_format_key_value_output_custom_format(self, mock_echo: Mock) -> None:
        """Test key-value formatting with custom formatting."""
        # Given
        data = {"name": "test", "status": "active"}
        prefix = ">> "
        separator = " = "

        # When
        format_key_value_output(data, prefix=prefix, separator=separator)

        # Then
        expected_calls = [
            ((">> name = test",),),
            ((">> status = active",),),
        ]
        assert mock_echo.call_args_list == expected_calls

    @patch("click.echo")
    def test_format_key_value_output_mixed_types(self, mock_echo: Mock) -> None:
        """Test key-value formatting with mixed value types."""
        # Given
        data: dict[str, Any] = {"count": 42, "enabled": True, "name": "test"}

        # When
        format_key_value_output(data)

        # Then
        expected_calls = [
            (("  count: 42",),),
            (("  enabled: True",),),
            (("  name: test",),),
        ]
        assert mock_echo.call_args_list == expected_calls

    @patch("click.echo")
    def test_format_key_value_output_empty_dict(self, mock_echo: Mock) -> None:
        """Test key-value formatting with empty dictionary."""
        # Given
        data: dict[str, Any] = {}

        # When
        format_key_value_output(data)

        # Then
        mock_echo.assert_not_called()


class TestTruncateString:
    """Test string truncation utility."""

    def test_truncate_string_short_text(self) -> None:
        """Test truncation with short text."""
        # Given
        text = "Short text"

        # When
        result = truncate_string(text)

        # Then
        assert result == "Short text"

    def test_truncate_string_exact_length(self) -> None:
        """Test truncation with text at exact max length."""
        # Given
        text = "A" * 80  # Exactly 80 characters

        # When
        result = truncate_string(text, max_length=80)

        # Then
        assert result == text
        assert len(result) == 80

    def test_truncate_string_long_text(self) -> None:
        """Test truncation with text longer than max length."""
        # Given
        text = "A" * 100  # 100 characters
        max_length = 80

        # When
        result = truncate_string(text, max_length=max_length)

        # Then
        assert result == ("A" * 77) + "..."
        assert len(result) == 80

    def test_truncate_string_custom_max_length(self) -> None:
        """Test truncation with custom max length."""
        # Given
        text = "This is a longer text that should be truncated"
        max_length = 20

        # When
        result = truncate_string(text, max_length=max_length)

        # Then
        assert result == "This is a longer ..."
        assert len(result) == 20

    def test_truncate_string_very_short_max_length(self) -> None:
        """Test truncation with very short max length."""
        # Given
        text = "Hello world"
        max_length = 5

        # When
        result = truncate_string(text, max_length=max_length)

        # Then
        assert result == "He..."
        assert len(result) == 5
