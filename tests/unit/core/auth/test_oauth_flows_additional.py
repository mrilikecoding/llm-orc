"""Additional tests for OAuth flows to improve coverage."""

from unittest.mock import Mock, patch

import pytest

from llm_orc.core.auth.oauth_flows import (
    GoogleGeminiOAuthFlow,
    OAuthFlow,
    create_oauth_flow,
)

# OAuth callback handler tests are skipped due to complex HTTP server setup requirements
# The main functionality is already tested through integration tests


class TestOAuthFlowBase:
    """Test base OAuthFlow class functionality."""

    def test_oauth_flow_google_authorization_url(self) -> None:
        """Test Google authorization URL generation."""
        flow = OAuthFlow("google", "test_client_id", "test_secret")

        auth_url = flow.get_authorization_url()

        assert "accounts.google.com/o/oauth2/v2/auth" in auth_url
        assert "client_id=test_client_id" in auth_url
        assert "response_type=code" in auth_url
        assert "scope=https://www.googleapis.com/auth/userinfo.email" in auth_url

    def test_oauth_flow_github_authorization_url(self) -> None:
        """Test GitHub authorization URL generation."""
        flow = OAuthFlow("github", "test_client_id", "test_secret")

        auth_url = flow.get_authorization_url()

        assert "github.com/login/oauth/authorize" in auth_url
        assert "client_id=test_client_id" in auth_url
        assert "scope=user:email" in auth_url

    def test_oauth_flow_unsupported_provider(self) -> None:
        """Test unsupported provider raises error."""
        flow = OAuthFlow("unsupported", "test_client_id", "test_secret")

        with pytest.raises(ValueError, match="OAuth not supported for provider"):
            flow.get_authorization_url()

    @patch("socket.socket")
    def test_start_callback_server_no_available_port(self, mock_socket: Mock) -> None:
        """Test callback server when no ports are available."""
        flow = OAuthFlow("google", "test_client_id", "test_secret")

        # Mock socket to always raise OSError (port unavailable)
        mock_socket.side_effect = OSError("Port unavailable")

        with patch("llm_orc.core.auth.oauth_flows.HTTPServer") as mock_server:
            mock_server.side_effect = OSError("Port unavailable")

            with pytest.raises(
                RuntimeError, match="No available port for OAuth callback"
            ):
                flow.start_callback_server()

    def test_exchange_code_for_tokens_base_implementation(self) -> None:
        """Test base implementation of token exchange."""
        flow = OAuthFlow("google", "test_client_id", "test_secret")

        tokens = flow.exchange_code_for_tokens("test_auth_code_12345")

        assert tokens["access_token"] == "mock_access_token_test_auth_"
        assert tokens["refresh_token"] == "mock_refresh_token_test_auth_"
        assert tokens["expires_in"] == 3600
        assert tokens["token_type"] == "Bearer"


class TestGoogleGeminiOAuthFlow:
    """Test Google Gemini OAuth flow."""

    def test_init(self) -> None:
        """Test Google Gemini OAuth flow initialization."""
        flow = GoogleGeminiOAuthFlow("test_client_id", "test_secret")

        assert flow.provider == "google"
        assert flow.client_id == "test_client_id"
        assert flow.client_secret == "test_secret"

    def test_get_authorization_url(self) -> None:
        """Test Google Gemini authorization URL."""
        flow = GoogleGeminiOAuthFlow("test_client_id", "test_secret")

        auth_url = flow.get_authorization_url()

        assert "accounts.google.com/o/oauth2/v2/auth" in auth_url
        assert "client_id=test_client_id" in auth_url
        assert "generative-language.retriever" in auth_url

    def test_exchange_code_for_tokens(self) -> None:
        """Test Google token exchange."""
        flow = GoogleGeminiOAuthFlow("test_client_id", "test_secret")

        tokens = flow.exchange_code_for_tokens("google_test_code")

        assert tokens["access_token"] == "google_access_token_google_tes"
        assert tokens["refresh_token"] == "google_refresh_token_google_tes"
        assert tokens["token_type"] == "Bearer"


class TestCreateOAuthFlow:
    """Test OAuth flow factory function."""

    def test_create_google_oauth_flow(self) -> None:
        """Test creating Google OAuth flow."""
        flow = create_oauth_flow("google", "client_id", "client_secret")

        assert isinstance(flow, GoogleGeminiOAuthFlow)
        assert flow.provider == "google"

    def test_create_unsupported_oauth_flow(self) -> None:
        """Test creating OAuth flow for unsupported provider."""
        with pytest.raises(ValueError, match="OAuth not supported for provider"):
            create_oauth_flow("unsupported", "client_id", "client_secret")
