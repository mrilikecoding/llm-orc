"""Additional tests for OAuth flows to improve coverage."""

from unittest.mock import Mock, patch

import pytest
import requests

from llm_orc.core.auth.oauth_flows import (
    AnthropicOAuthFlow,
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


class TestAnthropicOAuthFlowInteractive:
    """Test Anthropic OAuth flow interactive methods."""

    @patch("builtins.input")
    @patch("webbrowser.open")
    @patch("builtins.print")
    def test_create_with_guidance_with_browser(
        self, mock_print: Mock, mock_browser: Mock, mock_input: Mock
    ) -> None:
        """Test creating Anthropic OAuth with browser opening."""
        # Mock user inputs
        mock_input.side_effect = [
            "y",  # Open browser
            "test_client_id",  # Client ID
            "test_client_secret",  # Client secret
        ]

        flow = AnthropicOAuthFlow.create_with_guidance()

        # Verify browser was opened
        mock_browser.assert_called_once_with("https://console.anthropic.com")

        # Verify flow was created correctly
        assert flow.client_id == "test_client_id"
        assert flow.client_secret == "test_client_secret"
        assert flow.provider == "anthropic"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_create_with_guidance_without_browser(
        self, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test creating Anthropic OAuth without opening browser."""
        # Mock user inputs
        mock_input.side_effect = [
            "n",  # Don't open browser
            "test_client_id",  # Client ID
            "test_client_secret",  # Client secret
        ]

        flow = AnthropicOAuthFlow.create_with_guidance()

        # Verify flow was created correctly
        assert flow.client_id == "test_client_id"
        assert flow.client_secret == "test_client_secret"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_create_with_guidance_empty_client_id(
        self, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test error when client ID is empty."""
        # Mock user inputs
        mock_input.side_effect = [
            "n",  # Don't open browser
            "",  # Empty client ID
        ]

        with pytest.raises(ValueError, match="Client ID is required"):
            AnthropicOAuthFlow.create_with_guidance()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_create_with_guidance_empty_client_secret(
        self, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test error when client secret is empty."""
        # Mock user inputs
        mock_input.side_effect = [
            "n",  # Don't open browser
            "test_client_id",  # Client ID
            "",  # Empty client secret
        ]

        with pytest.raises(ValueError, match="Client Secret is required"):
            AnthropicOAuthFlow.create_with_guidance()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_start_manual_callback_flow(
        self, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test manual callback flow."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock user input
        mock_input.return_value = "test_auth_code_12345"

        auth_code = flow.start_manual_callback_flow()

        assert auth_code == "test_auth_code_12345"
        assert mock_print.call_count > 0  # Should print instructions

    @patch("builtins.input")
    @patch("builtins.print")
    def test_start_manual_callback_flow_empty_input(
        self, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test manual callback flow with empty input retry."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock user inputs: empty first, then valid
        mock_input.side_effect = ["", "valid_auth_code"]

        auth_code = flow.start_manual_callback_flow()

        assert auth_code == "valid_auth_code"
        # Should have prompted twice
        assert mock_input.call_count == 2

    @patch("requests.post")
    @patch("builtins.print")
    def test_exchange_code_for_tokens_success(
        self, mock_print: Mock, mock_post: Mock
    ) -> None:
        """Test successful token exchange."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_response

        tokens = flow.exchange_code_for_tokens("test_auth_code")

        assert tokens["access_token"] == "test_access_token"
        assert tokens["refresh_token"] == "test_refresh_token"

    @patch("requests.post")
    @patch("builtins.print")
    def test_exchange_code_for_tokens_http_error(
        self, mock_print: Mock, mock_post: Mock
    ) -> None:
        """Test token exchange with HTTP error."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        tokens = flow.exchange_code_for_tokens("test_auth_code")

        # Should return fallback response
        assert tokens["requires_manual_extraction"] is True
        assert tokens["auth_code"] == "test_auth_code"

    @patch("requests.post")
    @patch("builtins.print")
    def test_exchange_code_for_tokens_network_error(
        self, mock_print: Mock, mock_post: Mock
    ) -> None:
        """Test token exchange with network error."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock network error
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        tokens = flow.exchange_code_for_tokens("test_auth_code")

        # Should return fallback response
        assert tokens["requires_manual_extraction"] is True

    @patch("requests.post")
    @patch("builtins.print")
    def test_exchange_code_for_tokens_unexpected_error(
        self, mock_print: Mock, mock_post: Mock
    ) -> None:
        """Test token exchange with unexpected error."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock unexpected error
        mock_post.side_effect = Exception("Unexpected error")

        tokens = flow.exchange_code_for_tokens("test_auth_code")

        # Should return fallback response
        assert tokens["requires_manual_extraction"] is True

    @patch("builtins.print")
    def test_fallback_to_browser_instructions(self, mock_print: Mock) -> None:
        """Test fallback instructions method."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        result = flow._fallback_to_browser_instructions("test_auth_code_12345")

        assert result["requires_manual_extraction"] is True
        assert result["auth_code"] == "test_auth_code_12345"
        assert "Manual token extraction required" in result["instructions"]
        assert "console.anthropic.com/settings/keys" in result["alternative_url"]

    @patch("urllib.request.urlopen")
    def test_validate_credentials_success(self, mock_urlopen: Mock) -> None:
        """Test successful credential validation."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock successful urlopen
        mock_urlopen.return_value = Mock()

        result = flow.validate_credentials()

        assert result is True

    @patch("urllib.request.urlopen")
    @patch("builtins.print")
    def test_validate_credentials_forbidden(
        self, mock_print: Mock, mock_urlopen: Mock
    ) -> None:
        """Test credential validation with 403 Forbidden (still valid)."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock 403 error
        import urllib.error
        from email.message import EmailMessage

        headers = EmailMessage()
        mock_error = urllib.error.HTTPError(
            "http://example.com",
            403,
            "Forbidden",
            headers,
            None,
        )
        mock_urlopen.side_effect = mock_error

        result = flow.validate_credentials()

        assert result is True  # 403 is considered valid

    @patch("urllib.request.urlopen")
    @patch("builtins.print")
    def test_validate_credentials_http_error(
        self, mock_print: Mock, mock_urlopen: Mock
    ) -> None:
        """Test credential validation with other HTTP error."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock 404 error
        import urllib.error
        from email.message import EmailMessage

        headers = EmailMessage()
        mock_error = urllib.error.HTTPError(
            "http://example.com",
            404,
            "Not Found",
            headers,
            None,
        )
        mock_urlopen.side_effect = mock_error

        result = flow.validate_credentials()

        assert result is False

    @patch("urllib.request.urlopen")
    @patch("builtins.print")
    def test_validate_credentials_general_exception(
        self, mock_print: Mock, mock_urlopen: Mock
    ) -> None:
        """Test credential validation with general exception."""
        flow = AnthropicOAuthFlow("test_client", "test_secret")

        # Mock general exception
        mock_urlopen.side_effect = Exception("Network error")

        result = flow.validate_credentials()

        assert result is False


class TestCreateOAuthFlow:
    """Test OAuth flow factory function."""

    def test_create_google_oauth_flow(self) -> None:
        """Test creating Google OAuth flow."""
        flow = create_oauth_flow("google", "client_id", "client_secret")

        assert isinstance(flow, GoogleGeminiOAuthFlow)
        assert flow.provider == "google"

    def test_create_anthropic_oauth_flow(self) -> None:
        """Test creating Anthropic OAuth flow."""
        flow = create_oauth_flow("anthropic", "client_id", "client_secret")

        assert isinstance(flow, AnthropicOAuthFlow)
        assert flow.provider == "anthropic"

    def test_create_unsupported_oauth_flow(self) -> None:
        """Test creating OAuth flow for unsupported provider."""
        with pytest.raises(ValueError, match="OAuth not supported for provider"):
            create_oauth_flow("unsupported", "client_id", "client_secret")
