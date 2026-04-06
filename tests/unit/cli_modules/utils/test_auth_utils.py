"""Comprehensive tests for auth utility functions."""

import time
from unittest.mock import Mock, patch

import click
import pytest

from llm_orc.cli_modules.utils.auth_utils import (
    handle_anthropic_interactive_auth,
    handle_claude_cli_auth,
    show_auth_method_help,
    validate_provider_authentication,
)


class TestHandleClaudeCLIAuth:
    """Test Claude CLI authentication handling."""

    @patch("shutil.which")
    def test_handle_claude_cli_auth_success(self, mock_which: Mock) -> None:
        """Test successful Claude CLI auth setup."""
        # Given
        mock_which.return_value = "/usr/local/bin/claude"
        mock_storage = Mock()

        # When
        handle_claude_cli_auth(mock_storage)

        # Then
        mock_which.assert_called_once_with("claude")
        mock_storage.store_api_key.assert_called_once_with(
            "claude-cli", "/usr/local/bin/claude"
        )

    @patch("shutil.which")
    def test_handle_claude_cli_auth_not_found(self, mock_which: Mock) -> None:
        """Test Claude CLI auth when claude command not found."""
        # Given
        mock_which.return_value = None
        mock_storage = Mock()

        # When / Then
        with pytest.raises(click.ClickException, match="Claude CLI not found"):
            handle_claude_cli_auth(mock_storage)

        mock_storage.store_api_key.assert_not_called()


class TestHandleAnthropicInteractiveAuth:
    """Test interactive Anthropic authentication setup."""

    @patch("click.prompt")
    def test_prompts_for_api_key_and_stores_it(self, mock_prompt: Mock) -> None:
        """Test that the function prompts for an API key and stores it."""
        # Given
        mock_prompt.return_value = "test_api_key"
        mock_auth_manager = Mock()
        mock_storage = Mock()

        # When
        handle_anthropic_interactive_auth(mock_auth_manager, mock_storage)

        # Then
        mock_storage.store_api_key.assert_called_once_with(
            "anthropic-api", "test_api_key"
        )


class TestShowAuthMethodHelp:
    """Test authentication method help display."""

    @patch("click.echo")
    def test_show_auth_method_help(self, mock_echo: Mock) -> None:
        """Test that help information is displayed."""
        # When
        show_auth_method_help()

        # Then
        assert mock_echo.call_count > 0
        # Check that key terms are mentioned in the help output
        all_calls = [str(call[0][0]) for call in mock_echo.call_args_list if call[0]]
        help_text = " ".join(all_calls)
        assert "OAuth" in help_text or "API Key" in help_text or "Claude" in help_text


class TestProviderAuthentication:
    """Test provider authentication testing."""

    def test_no_auth_method(self) -> None:
        """Test when provider has no auth method."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = None
        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is False

    def test_api_key_auth_success(self) -> None:
        """Test successful API key authentication."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "api_key"
        mock_storage.get_api_key.return_value = "test_api_key"

        mock_auth_manager = Mock()
        mock_auth_manager.authenticate.return_value = True

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is True
        mock_auth_manager.authenticate.assert_called_once_with(
            "test_provider", "test_api_key"
        )

    def test_api_key_auth_failure(self) -> None:
        """Test failed API key authentication."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "api_key"
        mock_storage.get_api_key.return_value = "test_api_key"

        mock_auth_manager = Mock()
        mock_auth_manager.authenticate.return_value = False

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is False

    def test_api_key_auth_no_key(self) -> None:
        """Test API key auth when no key is stored."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "api_key"
        mock_storage.get_api_key.return_value = None

        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is False
        mock_auth_manager.authenticate.assert_not_called()

    def test_oauth_auth_valid_token(self) -> None:
        """Test OAuth authentication with valid token."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = {
            "access_token": "test_token",
            "expires_at": time.time() + 3600,  # Valid for 1 hour
        }

        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is True

    def test_oauth_auth_expired_token(self) -> None:
        """Test OAuth authentication with expired token."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = {
            "access_token": "test_token",
            "expires_at": time.time() - 3600,  # Expired 1 hour ago
        }

        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is False

    def test_oauth_auth_no_expiration(self) -> None:
        """Test OAuth authentication without expiration info."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = {
            "access_token": "test_token",
            # No expires_at field
        }

        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is True  # Assume valid without expiration info

    def test_oauth_auth_no_token(self) -> None:
        """Test OAuth auth when no token is stored."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = None

        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is False

    def test_unknown_auth_method(self) -> None:
        """Test with unknown authentication method."""
        # Given
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "unknown_method"

        mock_auth_manager = Mock()

        # When
        result = validate_provider_authentication(
            mock_storage, mock_auth_manager, "test_provider"
        )

        # Then
        assert result is False
