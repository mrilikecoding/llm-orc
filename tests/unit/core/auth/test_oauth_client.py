"""Tests for OAuth client functionality."""

import time
from unittest.mock import Mock, patch

import pytest
import requests

from llm_orc.core.auth.oauth_client import OAuthClaudeClient, OAuthTokenRefreshError


class TestOAuthClaudeClient:
    """Test OAuth Claude client functionality."""

    def test_init(self) -> None:
        """Test client initialization."""
        access_token = "test-access-token"
        refresh_token = "test-refresh-token"

        client = OAuthClaudeClient(access_token, refresh_token)

        assert client.access_token == access_token
        assert client.refresh_token == refresh_token
        assert client.base_url == "https://api.anthropic.com/v1"

    def test_init_without_refresh_token(self) -> None:
        """Test client initialization without refresh token."""
        access_token = "test-access-token"

        client = OAuthClaudeClient(access_token)

        assert client.access_token == access_token
        assert client.refresh_token is None

    def test_get_headers(self) -> None:
        """Test header generation."""
        client = OAuthClaudeClient("test-token")

        headers = client._get_headers()

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"
        assert headers["anthropic-beta"] == "oauth-2025-04-20"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "User-Agent" in headers
        assert "LLM-Orchestra" in headers["User-Agent"]

    def test_is_token_expired_no_expiration(self) -> None:
        """Test token expiration check when no expiration provided."""
        client = OAuthClaudeClient("test-token")

        assert client.is_token_expired() is False
        assert client.is_token_expired(None) is False

    def test_is_token_expired_valid_token(self) -> None:
        """Test token expiration check for valid token."""
        client = OAuthClaudeClient("test-token")

        # Token expires in 1 hour (3600 seconds)
        expires_at = int(time.time()) + 3600

        assert client.is_token_expired(expires_at) is False

    def test_is_token_expired_expired_token(self) -> None:
        """Test token expiration check for expired token."""
        client = OAuthClaudeClient("test-token")

        # Token expired 1 hour ago
        expires_at = int(time.time()) - 3600

        assert client.is_token_expired(expires_at) is True

    def test_is_token_expired_soon_to_expire(self) -> None:
        """Test token expiration check for token expiring soon."""
        client = OAuthClaudeClient("test-token")

        # Token expires in 2 minutes (less than 5 minute buffer)
        expires_at = int(time.time()) + 120

        assert client.is_token_expired(expires_at) is True

    def test_refresh_access_token_no_refresh_token(self) -> None:
        """Test refresh token failure when no refresh token available."""
        client = OAuthClaudeClient("test-token")  # No refresh token

        with pytest.raises(ValueError, match="No refresh token available"):
            client.refresh_access_token("client-id")

    @patch("requests.post")
    def test_refresh_access_token_success(self, mock_post: Mock) -> None:
        """Test successful token refresh."""
        client = OAuthClaudeClient("old-token", "refresh-token")

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_response

        result = client.refresh_access_token("client-id")

        assert result is True
        assert client.access_token == "new-access-token"
        assert client.refresh_token == "new-refresh-token"

        # Verify the API call
        mock_post.assert_called_once_with(
            "https://console.anthropic.com/v1/oauth/token",
            json={
                "grant_type": "refresh_token",
                "refresh_token": "refresh-token",
                "client_id": "client-id",
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

    @patch("requests.post")
    def test_refresh_access_token_success_no_new_refresh_token(
        self, mock_post: Mock
    ) -> None:
        """Test successful token refresh without new refresh token."""
        client = OAuthClaudeClient("old-token", "refresh-token")

        # Mock successful response without new refresh token
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_response

        result = client.refresh_access_token("client-id")

        assert result is True
        assert client.access_token == "new-access-token"
        assert client.refresh_token == "refresh-token"  # Unchanged

    @patch("requests.post")
    def test_refresh_access_token_http_error(self, mock_post: Mock) -> None:
        """Test token refresh with HTTP error."""
        client = OAuthClaudeClient("old-token", "refresh-token")

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        # Should raise OAuthTokenRefreshError instead of returning False
        with pytest.raises(OAuthTokenRefreshError) as exc_info:
            client.refresh_access_token("client-id")

        assert exc_info.value.status_code == 400
        assert "Bad request" in str(exc_info.value)
        assert client.access_token == "old-token"  # Unchanged

    @patch("requests.post")
    def test_refresh_access_token_network_error(self, mock_post: Mock) -> None:
        """Test token refresh with network error."""
        client = OAuthClaudeClient("old-token", "refresh-token")

        # Mock network error
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        # Should raise OAuthTokenRefreshError instead of returning False
        with pytest.raises(OAuthTokenRefreshError) as exc_info:
            client.refresh_access_token("client-id")

        assert exc_info.value.status_code == 0
        assert "Network error" in str(exc_info.value)

    @patch("requests.post")
    def test_refresh_access_token_unexpected_error(self, mock_post: Mock) -> None:
        """Test token refresh with unexpected error."""
        client = OAuthClaudeClient("old-token", "refresh-token")

        # Mock unexpected error
        mock_post.side_effect = Exception("Unexpected error")

        # Should raise OAuthTokenRefreshError instead of returning False
        with pytest.raises(OAuthTokenRefreshError) as exc_info:
            client.refresh_access_token("client-id")

        assert exc_info.value.status_code == 0
        assert "Unexpected error" in str(exc_info.value)

    def test_revoke_token_no_token(self) -> None:
        """Test token revocation when no token available."""
        client = OAuthClaudeClient("access-token")  # No refresh token

        # Try to revoke refresh token when none exists
        result = client.revoke_token("client-id", "refresh_token")

        assert result is False

    @patch("requests.post")
    def test_revoke_token_success(self, mock_post: Mock) -> None:
        """Test successful token revocation."""
        client = OAuthClaudeClient("access-token", "refresh-token")

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = client.revoke_token("client-id", "access_token")

        assert result is True

        # Verify the API call
        mock_post.assert_called_once_with(
            "https://console.anthropic.com/v1/oauth/revoke",
            json={
                "token": "access-token",
                "token_type_hint": "access_token",
                "client_id": "client-id",
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

    @patch("requests.post")
    def test_revoke_refresh_token_success(self, mock_post: Mock) -> None:
        """Test successful refresh token revocation."""
        client = OAuthClaudeClient("access-token", "refresh-token")

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = client.revoke_token("client-id", "refresh_token")

        assert result is True

        # Verify the API call
        mock_post.assert_called_once_with(
            "https://console.anthropic.com/v1/oauth/revoke",
            json={
                "token": "refresh-token",
                "token_type_hint": "refresh_token",
                "client_id": "client-id",
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

    @patch("requests.post")
    def test_revoke_token_error(self, mock_post: Mock) -> None:
        """Test token revocation with error."""
        client = OAuthClaudeClient("access-token")

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        result = client.revoke_token("client-id", "access_token")

        assert result is False

    @patch("requests.post")
    def test_revoke_token_exception(self, mock_post: Mock) -> None:
        """Test token revocation with exception."""
        client = OAuthClaudeClient("access-token")

        # Mock exception
        mock_post.side_effect = Exception("Network error")

        result = client.revoke_token("client-id", "access_token")

        assert result is False

    @patch("requests.post")
    def test_revoke_all_tokens_success(self, mock_post: Mock) -> None:
        """Test successful revocation of all tokens."""
        client = OAuthClaudeClient("access-token", "refresh-token")

        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = client.revoke_all_tokens("client-id")

        assert result is True
        assert mock_post.call_count == 2  # Two revocation calls

    @patch("requests.post")
    def test_revoke_all_tokens_no_refresh_token(self, mock_post: Mock) -> None:
        """Test revocation of all tokens when no refresh token."""
        client = OAuthClaudeClient("access-token")  # No refresh token

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = client.revoke_all_tokens("client-id")

        assert result is True
        assert mock_post.call_count == 1  # Only access token revocation

    @patch("requests.post")
    def test_revoke_all_tokens_partial_failure(self, mock_post: Mock) -> None:
        """Test revocation when one token fails."""
        client = OAuthClaudeClient("access-token", "refresh-token")

        # Mock responses: success for first, failure for second
        responses = [
            Mock(status_code=200),  # Access token success
            Mock(status_code=400),  # Refresh token failure
        ]
        mock_post.side_effect = responses

        result = client.revoke_all_tokens("client-id")

        assert result is False
        assert mock_post.call_count == 2

    @patch("requests.post")
    def test_create_message_success(self, mock_post: Mock) -> None:
        """Test successful message creation."""
        client = OAuthClaudeClient("access-token")

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "content": [{"text": "Hello!"}],
            "role": "assistant",
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = client.create_message("claude-3-sonnet", messages)

        assert result["id"] == "msg_123"

        # Verify the API call
        mock_post.assert_called_once_with(
            "https://api.anthropic.com/v1/messages",
            headers=client._get_headers(),
            json={
                "model": "claude-3-sonnet",
                "max_tokens": 4096,
                "messages": messages,
            },
            timeout=30,
        )

    @patch("requests.post")
    def test_create_message_with_system(self, mock_post: Mock) -> None:
        """Test message creation with system prompt."""
        client = OAuthClaudeClient("access-token")

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg_123"}
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "You are a helpful assistant."

        client.create_message(
            "claude-3-sonnet", messages, system=system_prompt, temperature=0.7
        )

        # Verify system prompt is included
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        assert request_data["system"] == system_prompt
        assert request_data["temperature"] == 0.7

    @patch("requests.post")
    def test_create_message_token_expired(self, mock_post: Mock) -> None:
        """Test message creation with expired token."""
        client = OAuthClaudeClient("access-token")

        # Mock 401 response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception, match="Token expired - refresh needed"):
            client.create_message("claude-3-sonnet", messages)

    @patch("requests.post")
    def test_create_message_api_error(self, mock_post: Mock) -> None:
        """Test message creation with API error."""
        client = OAuthClaudeClient("access-token")

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(
            Exception, match="API request failed: 500 - Internal server error"
        ):
            client.create_message("claude-3-sonnet", messages)
