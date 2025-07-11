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
