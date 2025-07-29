"""Tests for OAuthClaudeModel generate_response helper methods."""

import time
from unittest.mock import Mock, patch

import pytest

from llm_orc.models.anthropic import OAuthClaudeModel


class TestOAuthGenerateResponseHelperMethods:
    """Test helper methods extracted from OAuthClaudeModel.generate_response."""

    def test_handle_proactive_token_refresh_expired_token(self) -> None:
        """Test proactive token refresh when token is expired."""
        from llm_orc.models.anthropic import _handle_proactive_token_refresh

        model = OAuthClaudeModel(
            access_token="old_token",
            refresh_token="test_refresh",
            client_id="test_client",
            expires_at=int(time.time()) - 100,  # Expired
        )

        # Mock successful refresh
        mock_client = Mock()
        mock_client.is_token_expired.return_value = True
        mock_client.refresh_access_token.return_value = True
        mock_client.access_token = "new_token"
        mock_client.refresh_token = "new_refresh"
        model.client = mock_client

        result = _handle_proactive_token_refresh(model)

        assert result is True
        assert model.access_token == "new_token"
        assert model.refresh_token == "new_refresh"

    def test_handle_proactive_token_refresh_not_expired(self) -> None:
        """Test proactive token refresh when token is not expired."""
        from llm_orc.models.anthropic import _handle_proactive_token_refresh

        model = OAuthClaudeModel(
            access_token="current_token",
            refresh_token="test_refresh",
            client_id="test_client",
            expires_at=int(time.time()) + 3600,  # Not expired
        )

        mock_client = Mock()
        mock_client.is_token_expired.return_value = False
        model.client = mock_client

        result = _handle_proactive_token_refresh(model)

        assert result is True  # No refresh needed, but success
        assert model.access_token == "current_token"  # Unchanged

    def test_handle_proactive_token_refresh_missing_requirements(self) -> None:
        """Test proactive token refresh when missing required fields."""
        from llm_orc.models.anthropic import _handle_proactive_token_refresh

        model = OAuthClaudeModel(
            access_token="token"
        )  # Missing refresh_token and client_id

        result = _handle_proactive_token_refresh(model)

        assert result is True  # Skip refresh, but success

    def test_prepare_oauth_message_request_basic(self) -> None:
        """Test preparing OAuth message request with basic parameters."""
        from llm_orc.models.anthropic import _prepare_oauth_message_request

        model = OAuthClaudeModel(access_token="token", model="claude-test")
        model.add_to_conversation("user", "Hello")

        request_params = _prepare_oauth_message_request(model, "Test role")

        assert request_params["model"] == "claude-test"
        assert request_params["max_tokens"] == 1000
        assert (
            request_params["system"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert len(request_params["messages"]) >= 1

    def test_prepare_oauth_message_request_no_role_injection(self) -> None:
        """Test preparing OAuth message request without role injection."""
        from llm_orc.models.anthropic import _prepare_oauth_message_request

        model = OAuthClaudeModel(access_token="token")
        # Add a user message to the conversation history to test
        model.add_to_conversation("user", "Test message")

        request_params = _prepare_oauth_message_request(
            model, "You are a helpful assistant"
        )

        # Role injection is now handled in generate_response, not here
        messages = request_params["messages"]
        assert len(messages) >= 1
        assert messages[0]["content"] == "Test message"
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_oauth_api_call_success(self) -> None:
        """Test successful OAuth API call execution."""
        from llm_orc.models.anthropic import _execute_oauth_api_call

        model = OAuthClaudeModel(access_token="token")
        request_params = {
            "model": "claude-test",
            "max_tokens": 1000,
            "system": "System prompt",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        mock_response = {
            "content": [{"text": "Hello response"}],
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }

        with patch.object(model.client, "create_message", return_value=mock_response):
            response = await _execute_oauth_api_call(model, request_params)

        assert response == mock_response

    def test_process_oauth_response_success(self) -> None:
        """Test processing successful OAuth response."""
        from llm_orc.models.anthropic import _process_oauth_response

        model = OAuthClaudeModel(access_token="token", model="claude-test")
        response = {
            "content": [{"text": "Test response"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        start_time = time.time() - 0.5  # 500ms ago

        result = _process_oauth_response(model, response, start_time)

        assert result == "Test response"

        # Verify usage was recorded
        usage = model.get_last_usage()
        assert usage is not None
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["model"] == "claude-test"

    def test_process_oauth_response_empty_content(self) -> None:
        """Test processing OAuth response with empty content."""
        from llm_orc.models.anthropic import _process_oauth_response

        model = OAuthClaudeModel(access_token="token")
        response = {
            "content": [],
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        start_time = time.time()

        result = _process_oauth_response(model, response, start_time)

        assert result == ""

    def test_handle_oauth_token_refresh_error_success(self) -> None:
        """Test handling OAuth token refresh error with successful refresh."""
        from llm_orc.models.anthropic import _handle_oauth_token_refresh_error

        model = OAuthClaudeModel(
            access_token="old_token",
            refresh_token="refresh_token",
            client_id="client_id",
        )

        mock_client = Mock()
        mock_client.refresh_access_token.return_value = True
        mock_client.access_token = "new_token"
        mock_client.refresh_token = "new_refresh"
        model.client = mock_client

        error = Exception("Token expired - refresh needed")

        result = _handle_oauth_token_refresh_error(model, error, "Hello", "Role")

        assert result is True  # Should return True indicating retry is possible

    def test_handle_oauth_token_refresh_error_not_token_error(self) -> None:
        """Test handling OAuth error that is not a token error."""
        from llm_orc.models.anthropic import _handle_oauth_token_refresh_error

        model = OAuthClaudeModel(access_token="token")
        error = Exception("Some other error")

        result = _handle_oauth_token_refresh_error(model, error, "Hello", "Role")
        assert result is False  # Should return False for non-token errors

    def test_handle_oauth_token_refresh_error_refresh_fails(self) -> None:
        """Test handling OAuth token refresh error when refresh fails."""
        from llm_orc.models.anthropic import _handle_oauth_token_refresh_error

        model = OAuthClaudeModel(
            access_token="token",
            refresh_token="refresh_token",
            client_id="client_id",
        )

        from llm_orc.core.auth.oauth_client import OAuthTokenRefreshError

        mock_client = Mock()
        mock_client.refresh_access_token.side_effect = OAuthTokenRefreshError(
            400, "Bad request"
        )
        model.client = mock_client

        error = Exception("Token expired - refresh needed")

        with pytest.raises(Exception, match="OAuth token refresh failed"):
            _handle_oauth_token_refresh_error(model, error, "Hello", "Role")
