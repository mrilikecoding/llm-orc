"""Tests for OAuth model implementations."""

from unittest.mock import Mock, patch

import pytest

from llm_orc import __version__
from llm_orc.models import OAuthClaudeModel


class TestOAuthClaudeModel:
    """Test cases for OAuthClaudeModel."""

    def test_initialization(self) -> None:
        """Test OAuthClaudeModel initialization."""
        model = OAuthClaudeModel(
            access_token="test_token",
            refresh_token="test_refresh",
            client_id="test_client",
            model="claude-3-5-sonnet-20241022",
        )

        assert model.access_token == "test_token"
        assert model.refresh_token == "test_refresh"
        assert model.client_id == "test_client"
        assert model.model == "claude-3-5-sonnet-20241022"
        assert model.name == "oauth-claude-claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_generate_response_success(self) -> None:
        """Test successful response generation."""
        model = OAuthClaudeModel(
            access_token="test_token",
            refresh_token="test_refresh",
            client_id="test_client",
        )

        # Mock the OAuth client response
        mock_response = {
            "content": [{"text": "Hello! How can I help you today?"}],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }

        with patch.object(model.client, "create_message", return_value=mock_response):
            result = await model.generate_response(
                "Hello", "You are a helpful assistant"
            )

            assert result == "Hello! How can I help you today?"
            usage = model.get_last_usage()
            assert usage is not None
            assert usage["input_tokens"] == 10
            assert usage["output_tokens"] == 15

    @pytest.mark.asyncio
    async def test_generate_response_with_token_refresh(self) -> None:
        """Test response generation with token refresh."""
        model = OAuthClaudeModel(
            access_token="test_token",
            refresh_token="test_refresh",
            client_id="test_client",
        )

        # Mock the OAuth client to first fail with token expired, then succeed
        mock_client = Mock()
        mock_client.create_message.side_effect = [
            Exception("Token expired - refresh needed"),
            {
                "content": [{"text": "Response after refresh"}],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            },
        ]
        mock_client.refresh_access_token.return_value = True
        model.client = mock_client

        result = await model.generate_response("Hello", "You are a helpful assistant")

        assert result == "Response after refresh"
        assert mock_client.refresh_access_token.called
        assert mock_client.create_message.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_response_token_refresh_fails(self) -> None:
        """Test response generation when token refresh fails."""
        model = OAuthClaudeModel(
            access_token="test_token",
            refresh_token="test_refresh",
            client_id="test_client",
        )

        # Mock the OAuth client to fail with token expired
        mock_client = Mock()
        mock_client.create_message.side_effect = Exception(
            "Token expired - refresh needed"
        )
        mock_client.refresh_access_token.return_value = False
        model.client = mock_client

        with pytest.raises(Exception, match="Token expired") as exc_info:
            await model.generate_response("Hello", "You are a helpful assistant")

        assert "Token expired" in str(exc_info.value)
        assert mock_client.refresh_access_token.called

    @pytest.mark.asyncio
    async def test_generate_response_no_content(self) -> None:
        """Test response generation when API returns no content."""
        model = OAuthClaudeModel(access_token="test_token")

        # Mock the OAuth client response with no content
        mock_response = {
            "content": [],
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }

        with patch.object(model.client, "create_message", return_value=mock_response):
            result = await model.generate_response(
                "Hello", "You are a helpful assistant"
            )

            assert result == ""
            usage = model.get_last_usage()
            assert usage is not None
            assert usage["output_tokens"] == 0

    def test_oauth_client_uses_dynamic_version(self) -> None:
        """Test that OAuth client uses dynamic version in User-Agent header."""
        model = OAuthClaudeModel(access_token="test_token")

        headers = model.client._get_headers()
        expected_user_agent = f"LLM-Orchestra/Python {__version__}"

        assert headers["User-Agent"] == expected_user_agent
        assert headers["X-Stainless-Package-Version"] == __version__

    @pytest.mark.asyncio
    async def test_automatic_token_refresh_updates_stored_credentials(self) -> None:
        """Test that automatic token refresh updates stored credentials."""
        from llm_orc.authentication import CredentialStorage

        # Create a model with a credential storage callback
        model = OAuthClaudeModel(
            access_token="old_token",
            refresh_token="test_refresh",
            client_id="test_client",
        )

        # Mock credential storage
        mock_storage = Mock(spec=CredentialStorage)
        model._credential_storage = mock_storage
        model._provider_key = "anthropic-claude-pro-max"

        # Mock OAuth client: first fail with expired token, then succeed with new tokens
        mock_client = Mock()
        mock_client.create_message.side_effect = [
            Exception("Token expired - refresh needed"),
            {
                "content": [{"text": "Response with new token"}],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            },
        ]
        mock_client.refresh_access_token.return_value = True
        mock_client.access_token = "new_access_token"  # Simulates updated token
        mock_client.refresh_token = "new_refresh_token"  # Updated refresh token
        model.client = mock_client

        result = await model.generate_response("Hello", "You are a helpful assistant")

        # Verify successful response
        assert result == "Response with new token"

        # Verify token refresh was called
        assert mock_client.refresh_access_token.called

        # Verify credential storage was updated with new tokens
        mock_storage.store_oauth_token.assert_called_once()
        call_args = mock_storage.store_oauth_token.call_args
        assert call_args[0][0] == "anthropic-claude-pro-max"  # provider
        assert call_args[0][1] == "new_access_token"  # access_token
        assert call_args[0][2] == "new_refresh_token"  # refresh_token
        # expires_at should be a timestamp in the future
        import time

        assert call_args[1]["expires_at"] > time.time()
