"""Tests for authentication system including credential storage."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.auth.authentication import AuthenticationManager, CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager


class TestCredentialStorage:
    """Test credential storage functionality."""

    @pytest.fixture
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def credential_storage(self, temp_config_dir: Path) -> CredentialStorage:
        """Create CredentialStorage instance with temp directory."""
        # Create a mock config manager with the temp directory
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        return CredentialStorage(config_manager)

    def test_store_api_key_creates_encrypted_file(
        self, credential_storage: CredentialStorage, temp_config_dir: Path
    ) -> None:
        """Test that storing an API key creates an encrypted credentials file."""
        # Given
        provider = "anthropic"
        api_key = "test_api_key_123"

        # When
        credential_storage.store_api_key(provider, api_key)

        # Then
        credentials_file = temp_config_dir / "credentials.yaml"
        assert credentials_file.exists()

        # File should be encrypted (not readable as plain text)
        with open(credentials_file) as f:
            content = f.read()
            assert api_key not in content  # Should be encrypted

    def test_retrieve_api_key_returns_stored_key(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test retrieving a stored API key."""
        # Given
        provider = "anthropic"
        api_key = "test_api_key_123"
        credential_storage.store_api_key(provider, api_key)

        # When
        retrieved_key = credential_storage.get_api_key(provider)

        # Then
        assert retrieved_key == api_key

    def test_get_api_key_returns_none_for_nonexistent_provider(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that getting API key for non-existent provider returns None."""
        # Given
        provider = "nonexistent_provider"

        # When
        retrieved_key = credential_storage.get_api_key(provider)

        # Then
        assert retrieved_key is None

    def test_list_providers_returns_stored_providers(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test listing all configured providers."""
        # Given
        providers = ["anthropic", "google", "openai"]
        for provider in providers:
            credential_storage.store_api_key(provider, f"key_for_{provider}")

        # When
        stored_providers = credential_storage.list_providers()

        # Then
        assert set(stored_providers) == set(providers)

    def test_remove_provider_deletes_credentials(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test removing a provider's credentials."""
        # Given
        provider = "anthropic"
        api_key = "test_api_key_123"
        credential_storage.store_api_key(provider, api_key)

        # When
        credential_storage.remove_provider(provider)

        # Then
        assert credential_storage.get_api_key(provider) is None
        assert provider not in credential_storage.list_providers()


class TestCredentialStorageAdvanced:
    """Test advanced credential storage functionality for better coverage."""

    @pytest.fixture
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def credential_storage(self, temp_config_dir: Path) -> CredentialStorage:
        """Create CredentialStorage instance with temp directory."""
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        return CredentialStorage(config_manager)

    def test_encryption_key_creation_new_file(self, temp_config_dir: Path) -> None:
        """Test encryption key creation when no existing key file (lines 54-58)."""
        # Given - key file doesn't exist initially
        key_file = temp_config_dir / ".encryption_key"
        assert not key_file.exists()

        # When - creating CredentialStorage triggers key creation
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        credential_storage = CredentialStorage(config_manager)

        # Then - key file should be created with proper permissions
        assert key_file.exists()
        assert oct(key_file.stat().st_mode)[-3:] == "600"  # Check file permissions
        assert credential_storage._encryption_key is not None

    def test_load_credentials_empty_file(
        self, credential_storage: CredentialStorage, temp_config_dir: Path
    ) -> None:
        """Test loading credentials from empty file (lines 71-72)."""
        # Given - create empty credentials file
        credentials_file = temp_config_dir / "credentials.yaml"
        with open(credentials_file, "w") as f:
            f.write("")

        # When
        result = credential_storage._load_credentials()

        # Then
        assert result == {}

    def test_load_credentials_file_not_exists(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test loading credentials when file doesn't exist (lines 64-65)."""
        # Given - no credentials file exists

        # When
        result = credential_storage._load_credentials()

        # Then
        assert result == {}

    def test_load_credentials_decryption_error(
        self, credential_storage: CredentialStorage, temp_config_dir: Path
    ) -> None:
        """Test loading credentials with decryption error (lines 77-78)."""
        # Given - create invalid encrypted file
        credentials_file = temp_config_dir / "credentials.yaml"
        with open(credentials_file, "w") as f:
            f.write("invalid_encrypted_data")

        # When
        result = credential_storage._load_credentials()

        # Then - should return empty dict on decryption error
        assert result == {}

    def test_save_credentials_sets_file_permissions(
        self, credential_storage: CredentialStorage, temp_config_dir: Path
    ) -> None:
        """Test that saving credentials sets proper file permissions (lines 88-89)."""
        # Given
        test_credentials = {"test_provider": {"api_key": "test_key"}}

        # When
        credential_storage._save_credentials(test_credentials)

        # Then
        credentials_file = temp_config_dir / "credentials.yaml"
        assert credentials_file.exists()
        # Check file permissions
        assert oct(credentials_file.stat().st_mode)[-3:] == "600"

    def test_store_oauth_token_with_all_parameters(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test storing OAuth token with all optional parameters (lines 125-139)."""
        # Given
        provider = "test_provider"
        access_token = "access_123"
        refresh_token = "refresh_456"
        expires_at = 1234567890
        client_id = "client_789"

        # When
        credential_storage.store_oauth_token(
            provider, access_token, refresh_token, expires_at, client_id
        )

        # Then
        stored_token = credential_storage.get_oauth_token(provider)
        assert stored_token is not None
        assert stored_token["access_token"] == access_token
        assert stored_token["refresh_token"] == refresh_token
        assert stored_token["expires_at"] == expires_at
        assert stored_token["client_id"] == client_id

        # Verify auth method is set correctly
        assert credential_storage.get_auth_method(provider) == "oauth"

    def test_store_oauth_token_minimal_parameters(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test storing OAuth token with minimal parameters."""
        # Given
        provider = "test_provider"
        access_token = "access_123"

        # When
        credential_storage.store_oauth_token(provider, access_token)

        # Then
        stored_token = credential_storage.get_oauth_token(provider)
        assert stored_token is not None
        assert stored_token["access_token"] == access_token
        assert "refresh_token" not in stored_token
        assert "expires_at" not in stored_token
        assert "client_id" not in stored_token

    def test_get_api_key_none_value(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test getting API key that has None value (lines 154-155)."""
        # Given - manually store a provider with None api_key
        credentials = {"test_provider": {"auth_method": "api_key", "api_key": None}}
        credential_storage._save_credentials(credentials)

        # When
        result = credential_storage.get_api_key("test_provider")

        # Then
        assert result is None

    def test_get_oauth_token_partial_data(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test getting OAuth token with partial data (lines 173-182)."""
        # Given - store OAuth provider with only access token
        credentials = {
            "test_provider": {
                "auth_method": "oauth",
                "access_token": "test_access",
                # Missing other OAuth fields
            }
        }
        credential_storage._save_credentials(credentials)

        # When
        result = credential_storage.get_oauth_token("test_provider")

        # Then
        assert result is not None
        assert result["access_token"] == "test_access"
        assert "refresh_token" not in result
        assert "expires_at" not in result
        assert "client_id" not in result

    def test_get_oauth_token_non_oauth_provider(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test getting OAuth token for non-OAuth provider."""
        # Given - store API key provider
        credential_storage.store_api_key("test_provider", "test_key")

        # When
        result = credential_storage.get_oauth_token("test_provider")

        # Then
        assert result is None

    def test_get_auth_method_none_value(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test getting auth method with None value (lines 198-199)."""
        # Given - manually store provider with None auth_method
        credentials = {"test_provider": {"auth_method": None}}
        credential_storage._save_credentials(credentials)

        # When
        result = credential_storage.get_auth_method("test_provider")

        # Then
        assert result is None


class TestAuthenticationManager:
    """Test authentication manager functionality."""

    @pytest.fixture
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def auth_manager(self, temp_config_dir: Path) -> AuthenticationManager:
        """Create AuthenticationManager instance with temp directory."""
        # Create a mock config manager with the temp directory
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        storage = CredentialStorage(config_manager)
        return AuthenticationManager(storage)

    def test_authenticate_with_api_key_success(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test successful authentication with API key."""
        # Given
        provider = "anthropic"
        api_key = "test_api_key_123"

        # When
        result = auth_manager.authenticate(provider, api_key=api_key)

        # Then
        assert result is True
        assert auth_manager.is_authenticated(provider)

    def test_authenticate_with_invalid_api_key_fails(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test that authentication fails with invalid API key."""
        # Given
        provider = "anthropic"
        api_key = "invalid_key"

        # When
        result = auth_manager.authenticate(provider, api_key=api_key)

        # Then
        assert result is False
        assert not auth_manager.is_authenticated(provider)

    def test_get_authenticated_client_returns_configured_client(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test getting an authenticated client for a provider."""
        # Given
        provider = "anthropic"
        api_key = "test_api_key_123"
        auth_manager.authenticate(provider, api_key=api_key)

        # When
        client = auth_manager.get_authenticated_client(provider)

        # Then
        assert client is not None
        # Client should be configured with the API key
        assert hasattr(client, "api_key") or hasattr(client, "_api_key")

    def test_get_authenticated_client_returns_none_for_unauthenticated(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test that getting client for unauthenticated provider returns None."""
        # Given
        provider = "anthropic"

        # When
        client = auth_manager.get_authenticated_client(provider)

        # Then
        assert client is None


class TestOAuthProviderIntegration:
    """Test OAuth provider-specific functionality."""

    @pytest.fixture
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def credential_storage(self, temp_config_dir: Path) -> CredentialStorage:
        """Create CredentialStorage instance with temp directory."""
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        return CredentialStorage(config_manager)

    def test_google_gemini_oauth_authorization_url_generation(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that Google Gemini OAuth generates correct authorization URL."""
        # Given
        from llm_orc.core.auth.oauth_flows import GoogleGeminiOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        oauth_flow = GoogleGeminiOAuthFlow(client_id, client_secret)

        # When
        auth_url = oauth_flow.get_authorization_url()

        # Then
        assert "accounts.google.com/o/oauth2/v2/auth" in auth_url
        # Check for the scope parameter (URL encoded)
        expected_scope = (
            "scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2F"
            "generative-language.retriever"
        )
        assert expected_scope in auth_url
        assert f"client_id={client_id}" in auth_url
        assert "response_type=code" in auth_url

    def test_google_gemini_oauth_token_exchange(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that Google Gemini OAuth can exchange code for tokens."""
        # Given
        from llm_orc.core.auth.oauth_flows import GoogleGeminiOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        oauth_flow = GoogleGeminiOAuthFlow(client_id, client_secret)
        auth_code = "test_authorization_code"

        # When
        tokens = oauth_flow.exchange_code_for_tokens(auth_code)

        # Then
        assert "access_token" in tokens
        assert "token_type" in tokens
        assert tokens["token_type"] == "Bearer"

    def test_anthropic_oauth_authorization_url_generation(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that Anthropic OAuth generates correct authorization URL."""
        # Given
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        oauth_flow = AnthropicOAuthFlow(client_id, client_secret)

        # When
        auth_url = oauth_flow.get_authorization_url()

        # Then
        assert "anthropic.com" in auth_url or "console.anthropic.com" in auth_url
        assert f"client_id={client_id}" in auth_url
        assert "response_type=code" in auth_url

    def test_anthropic_oauth_token_exchange(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that Anthropic OAuth can exchange code for tokens."""
        # Given
        from unittest.mock import Mock, patch

        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        oauth_flow = AnthropicOAuthFlow(client_id, client_secret)
        auth_code = "test_authorization_code"

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "sk-ant-oat01-test-token",
            "refresh_token": "sk-ant-ort01-test-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        # When
        with patch("requests.post", return_value=mock_response):
            tokens = oauth_flow.exchange_code_for_tokens(auth_code)

        # Then
        assert "access_token" in tokens
        assert "token_type" in tokens
        assert tokens["token_type"] == "Bearer"

    def test_oauth_flow_factory_creates_correct_provider(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that OAuth flow factory creates the correct provider-specific flow."""
        # Given
        from llm_orc.core.auth.oauth_flows import create_oauth_flow

        # When & Then - Google
        google_flow = create_oauth_flow("google", "client_id", "client_secret")
        assert google_flow.__class__.__name__ == "GoogleGeminiOAuthFlow"

        # When & Then - Anthropic
        anthropic_flow = create_oauth_flow("anthropic", "client_id", "client_secret")
        assert anthropic_flow.__class__.__name__ == "AnthropicOAuthFlow"

    def test_oauth_flow_factory_raises_for_unsupported_provider(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that OAuth flow factory raises error for unsupported provider."""
        # Given
        from llm_orc.core.auth.oauth_flows import create_oauth_flow

        # When & Then
        with pytest.raises(ValueError, match="OAuth not supported for provider"):
            create_oauth_flow("unsupported_provider", "client_id", "client_secret")


class TestAnthropicOAuthFlow:
    """Test improved Anthropic OAuth flow functionality."""

    def test_uses_validated_oauth_parameters(self) -> None:
        """Test AnthropicOAuthFlow uses validated OAuth parameters from issue #32."""
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        # Create flow with the shared client ID discovered in testing
        client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        flow = AnthropicOAuthFlow(client_id, "")  # No client secret needed for PKCE

        # Test that authorization URL includes validated parameters
        auth_url = flow.get_authorization_url()

        # Should include the shared client ID
        assert "client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e" in auth_url

        # Should include all required scopes (URL-encoded - Python uses + for spaces)
        assert "scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference" in auth_url

        # Should use Anthropic's callback endpoint (working implementation)
        expected_redirect = (
            "redirect_uri=https%3A%2F%2Fconsole.anthropic.com%2Foauth%2Fcode%2Fcallback"
        )
        assert expected_redirect in auth_url

        # Should include PKCE parameters
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
        assert "response_type=code" in auth_url

    def test_token_exchange_uses_real_endpoint(self) -> None:
        """Test that token exchange makes real API call to Anthropic OAuth endpoint."""
        from unittest.mock import Mock, patch

        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        flow = AnthropicOAuthFlow(client_id, "")
        auth_code = "test_auth_code_12345"

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "sk-ant-oat01-test-token",
            "refresh_token": "sk-ant-ort01-test-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            tokens = flow.exchange_code_for_tokens(auth_code)

            # Should call the real Anthropic OAuth endpoint
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args

            # Verify endpoint URL matches working implementation
            assert args[0] == "https://console.anthropic.com/v1/oauth/token"

            # Verify request data includes all required fields
            assert kwargs["json"]["grant_type"] == "authorization_code"
            assert kwargs["json"]["client_id"] == client_id
            assert kwargs["json"]["code"] == auth_code
            assert kwargs["json"]["code_verifier"] == flow.code_verifier
            assert kwargs["json"]["redirect_uri"] == flow.redirect_uri

            # Verify headers
            assert kwargs["headers"]["Content-Type"] == "application/json"

            # Should return the tokens
            assert tokens["access_token"] == "sk-ant-oat01-test-token"
            assert tokens["refresh_token"] == "sk-ant-ort01-test-refresh"

    def test_anthropic_oauth_flow_initialization(self) -> None:
        """Test AnthropicOAuthFlow can be initialized correctly."""
        # Given
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"

        # When
        flow = AnthropicOAuthFlow(client_id, client_secret)

        # Then
        assert flow.client_id == client_id
        assert flow.client_secret == client_secret
        assert flow.provider == "anthropic"
        assert flow.redirect_uri == "https://console.anthropic.com/oauth/code/callback"

    def test_get_authorization_url_contains_required_parameters(self) -> None:
        """Test that authorization URL contains all required OAuth parameters."""
        # Given
        from urllib.parse import parse_qs, urlparse

        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        flow = AnthropicOAuthFlow(client_id, client_secret)

        # When
        auth_url = flow.get_authorization_url()

        # Then
        parsed_url = urlparse(auth_url)
        query_params = parse_qs(parsed_url.query)

        assert parsed_url.netloc == "claude.ai"
        assert parsed_url.path == "/oauth/authorize"
        assert query_params["client_id"][0] == client_id
        assert query_params["response_type"][0] == "code"
        assert query_params["redirect_uri"][0] == flow.redirect_uri
        assert "state" in query_params

    def test_validate_credentials_with_accessible_endpoint(self) -> None:
        """Test credential validation when OAuth endpoint is accessible."""
        # Given
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        flow = AnthropicOAuthFlow(client_id, client_secret)

        # When & Then
        # This should work since we've confirmed the endpoint exists
        result = flow.validate_credentials()
        assert isinstance(result, bool)

    def test_exchange_code_for_tokens_returns_valid_structure(self) -> None:
        """Test that token exchange returns proper token structure."""
        # Given
        from unittest.mock import Mock, patch

        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        flow = AnthropicOAuthFlow(client_id, client_secret)
        auth_code = "test_auth_code_123"

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "sk-ant-oat01-test-token",
            "refresh_token": "sk-ant-ort01-test-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        # When
        with patch("requests.post", return_value=mock_response):
            tokens = flow.exchange_code_for_tokens(auth_code)

        # Then
        assert isinstance(tokens, dict)
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "expires_in" in tokens
        assert "token_type" in tokens

        # Verify token format
        assert tokens["access_token"] == "sk-ant-oat01-test-token"
        assert tokens["refresh_token"] == "sk-ant-ort01-test-refresh"
        assert tokens["expires_in"] == 3600
        assert tokens["token_type"] == "Bearer"

    def test_mock_create_with_guidance_method_exists(self) -> None:
        """Test that create_with_guidance method exists for future testing."""
        # Given
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        # When & Then
        assert hasattr(AnthropicOAuthFlow, "create_with_guidance")
        assert callable(AnthropicOAuthFlow.create_with_guidance)

    def test_uses_manual_callback_flow(self) -> None:
        """Test AnthropicOAuthFlow uses manual callback flow instead of local server."""
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        flow = AnthropicOAuthFlow(client_id, "")

        # Should have manual callback flow method
        assert hasattr(flow, "start_manual_callback_flow")

        # Should use Anthropic's callback endpoint
        assert "console.anthropic.com" in flow.redirect_uri

    def test_callback_server_handles_authorization_code(self) -> None:
        """Test that callback server can handle OAuth authorization code."""
        import requests

        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        flow = AnthropicOAuthFlow(client_id, "")

        # Start callback server
        server, port = flow.start_callback_server()

        try:
            # Simulate OAuth callback request
            test_code = "test_auth_code_12345"
            test_state = "test_state_67890"
            callback_url = (
                f"http://localhost:{port}/callback?code={test_code}&state={test_state}"
            )

            # Make request to callback server
            response = requests.get(callback_url, timeout=5)

            # Should return success response
            assert response.status_code == 200
            assert "Authorization Successful" in response.text

            # Server should have captured the authorization code
            assert server.auth_code == test_code  # type: ignore
            assert server.auth_error is None  # type: ignore

        finally:
            # Clean up server
            server.server_close()


class TestImprovedAuthenticationManager:
    """Test enhanced authentication manager with better error handling."""

    @pytest.fixture
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def auth_manager(self, temp_config_dir: Path) -> AuthenticationManager:
        """Create AuthenticationManager instance with temp directory."""
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        storage = CredentialStorage(config_manager)
        return AuthenticationManager(storage)

    def test_oauth_validation_called_when_available(
        self, auth_manager: AuthenticationManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that OAuth validation is called when available."""
        # Given
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        validation_called = False

        def mock_validate(self: Any) -> bool:
            nonlocal validation_called
            validation_called = True
            return True

        monkeypatch.setattr(AnthropicOAuthFlow, "validate_credentials", mock_validate)

        # Mock the OAuth flow to avoid actual browser/manual input operations
        def mock_manual_callback_flow(self: Any) -> str:
            return "test_auth_code_from_manual_flow"

        def mock_open_browser(url: str) -> None:
            pass

        def mock_exchange_tokens(self: Any, auth_code: str) -> dict[str, Any]:
            return {
                "access_token": "sk-ant-oat01-test-token",
                "refresh_token": "sk-ant-ort01-test-refresh",
                "expires_in": 3600,
                "token_type": "Bearer",
            }

        monkeypatch.setattr(
            AnthropicOAuthFlow, "start_manual_callback_flow", mock_manual_callback_flow
        )
        monkeypatch.setattr("webbrowser.open", mock_open_browser)
        monkeypatch.setattr(
            AnthropicOAuthFlow, "exchange_code_for_tokens", mock_exchange_tokens
        )

        # When
        result = auth_manager.authenticate_oauth(
            "anthropic", "test_client", "test_secret"
        )

        # Then
        assert validation_called
        assert result is True  # Should succeed with mocked validation

    def test_oauth_error_handling_for_invalid_provider(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test proper error handling for unsupported OAuth provider."""
        # When
        result = auth_manager.authenticate_oauth(
            "unsupported_provider", "client_id", "client_secret"
        )

        # Then
        assert result is False

    def test_oauth_timeout_handling(
        self, auth_manager: AuthenticationManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that OAuth flow handles timeout correctly."""
        # Given
        from llm_orc.core.auth.oauth_flows import AnthropicOAuthFlow

        def mock_start_server(self: Any) -> tuple[Any, int]:
            # Return a server that never receives auth code (simulating timeout)
            server = type("MockServer", (), {"auth_code": None, "auth_error": None})()
            return server, 8080

        def mock_open_browser(url: str) -> None:
            pass

        # Mock time to simulate timeout quickly
        import time

        call_count = 0
        start_time = time.time()

        def mock_time() -> float:
            nonlocal call_count
            call_count += 1
            # First few calls return normal time, then jump to timeout
            if call_count > 5:
                return start_time + 150  # Beyond the 120 second timeout
            return start_time + (call_count * 0.1)  # Gradual increase initially

        def mock_sleep(duration: float) -> None:
            pass  # Don't actually sleep in tests

        monkeypatch.setattr(
            AnthropicOAuthFlow, "start_callback_server", mock_start_server
        )
        monkeypatch.setattr("webbrowser.open", mock_open_browser)
        monkeypatch.setattr("time.time", mock_time)
        monkeypatch.setattr("time.sleep", mock_sleep)

        # When
        result = auth_manager.authenticate_oauth(
            "anthropic", "test_client", "test_secret"
        )

        # Then
        assert result is False

    def test_logout_oauth_provider_revokes_tokens(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test logging out OAuth provider revokes tokens and removes credentials."""
        # Given - Store OAuth credentials first
        provider = "anthropic-claude-pro-max"
        access_token = "test_access_token"
        refresh_token = "test_refresh_token"
        client_id = "test_client_id"

        credential_storage = auth_manager.credential_storage
        credential_storage.store_oauth_token(provider, access_token, refresh_token)

        # Store client_id in OAuth token data (simulating full OAuth setup)
        credentials = credential_storage._load_credentials()
        credentials[provider]["client_id"] = client_id
        credential_storage._save_credentials(credentials)

        # Mock successful token revocation
        with patch("llm_orc.core.auth.oauth_client.requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            # When
            result = auth_manager.logout_oauth_provider(provider)

            # Then
            assert result is True

            # Verify tokens were revoked (two calls: access + refresh)
            assert mock_post.call_count == 2

            # Check the revocation calls
            calls = mock_post.call_args_list
            access_call = calls[0]
            refresh_call = calls[1]

            assert access_call[0][0] == "https://console.anthropic.com/v1/oauth/revoke"
            assert access_call[1]["json"]["token"] == access_token
            assert access_call[1]["json"]["token_type_hint"] == "access_token"

            assert refresh_call[0][0] == "https://console.anthropic.com/v1/oauth/revoke"
            assert refresh_call[1]["json"]["token"] == refresh_token
            assert refresh_call[1]["json"]["token_type_hint"] == "refresh_token"

            # Verify credentials were removed locally
            assert provider not in credential_storage.list_providers()

    def test_logout_oauth_provider_handles_missing_provider(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test that logging out non-existent OAuth provider returns False."""
        # When
        result = auth_manager.logout_oauth_provider("nonexistent-provider")

        # Then
        assert result is False

    def test_logout_oauth_provider_handles_non_oauth_provider(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test that logging out non-OAuth provider returns False."""
        # Given - Store API key credentials (not OAuth)
        provider = "anthropic-api"
        credential_storage = auth_manager.credential_storage
        credential_storage.store_api_key(provider, "test_api_key")

        # When
        result = auth_manager.logout_oauth_provider(provider)

        # Then
        assert result is False

    def test_logout_oauth_provider_continues_on_revocation_failure(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test that logout removes local credentials even if token revocation fails."""
        # Given - Store OAuth credentials
        provider = "anthropic-claude-pro-max"
        credential_storage = auth_manager.credential_storage
        credential_storage.store_oauth_token(provider, "test_token", "test_refresh")

        # Store client_id
        credentials = credential_storage._load_credentials()
        credentials[provider]["client_id"] = "test_client"
        credential_storage._save_credentials(credentials)

        # Mock failed token revocation
        with patch(
            "llm_orc.core.auth.oauth_client.requests.post",
            side_effect=Exception("Network error"),
        ):
            # When
            result = auth_manager.logout_oauth_provider(provider)

            # Then - Should still succeed in removing local credentials
            assert result is True
            assert provider not in credential_storage.list_providers()

    def test_logout_all_oauth_providers(
        self, auth_manager: AuthenticationManager
    ) -> None:
        """Test that logout_all_oauth_providers logs out all OAuth providers."""
        # Given - Store multiple OAuth providers
        providers = ["anthropic-claude-pro-max", "google-oauth"]
        credential_storage = auth_manager.credential_storage
        for provider in providers:
            credential_storage.store_oauth_token(
                provider, f"token_{provider}", f"refresh_{provider}"
            )
            # Store client_id for each
            credentials = credential_storage._load_credentials()
            credentials[provider]["client_id"] = f"client_{provider}"
            credential_storage._save_credentials(credentials)

        # Also store a non-OAuth provider (should not be affected)
        credential_storage.store_api_key("anthropic-api", "api_key")

        # Mock successful token revocations
        with patch("llm_orc.core.auth.oauth_client.requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            # When
            results = auth_manager.logout_all_oauth_providers()

            # Then
            assert len(results) == 2
            assert all(results.values())  # All should be True
            assert "anthropic-claude-pro-max" in results
            assert "google-oauth" in results

            # Verify all OAuth providers removed but API key provider remains
            remaining_providers = credential_storage.list_providers()
            assert "anthropic-api" in remaining_providers
            assert "anthropic-claude-pro-max" not in remaining_providers
            assert "google-oauth" not in remaining_providers

            # Verify revocation calls were made (2 per provider: access + refresh)
            assert mock_post.call_count == 4


class TestAuthenticateOAuthHelperMethods:
    """Test helper methods extracted from authenticate_oauth for complexity reduction."""

    @pytest.fixture
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def auth_manager(self, temp_config_dir: Path) -> AuthenticationManager:
        """Create an AuthenticationManager for testing."""
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = temp_config_dir
        storage = CredentialStorage(config_manager)
        return AuthenticationManager(storage)

    def test_setup_and_validate_oauth_flow_success(self, auth_manager: AuthenticationManager) -> None:
        """Test OAuth flow setup and validation success case."""
        from llm_orc.core.auth.authentication import _setup_and_validate_oauth_flow

        # Given
        provider = "anthropic"
        client_id = "test_client_id"
        client_secret = "test_client_secret"

        # When
        with patch("llm_orc.core.auth.authentication.create_oauth_flow") as mock_create_flow:
            mock_flow = Mock()
            mock_flow.validate_credentials.return_value = True
            mock_create_flow.return_value = mock_flow

            result = _setup_and_validate_oauth_flow(provider, client_id, client_secret)

            # Then
            assert result == mock_flow
            mock_create_flow.assert_called_once_with(provider, client_id, client_secret)
            mock_flow.validate_credentials.assert_called_once()

    def test_setup_and_validate_oauth_flow_validation_failure(self, auth_manager: AuthenticationManager) -> None:
        """Test OAuth flow setup when validation fails."""
        from llm_orc.core.auth.authentication import _setup_and_validate_oauth_flow

        # Given
        provider = "anthropic"
        client_id = "test_client_id"
        client_secret = "test_client_secret"

        # When
        with patch("llm_orc.core.auth.authentication.create_oauth_flow") as mock_create_flow:
            mock_flow = Mock()
            mock_flow.validate_credentials.return_value = False
            mock_create_flow.return_value = mock_flow

            result = _setup_and_validate_oauth_flow(provider, client_id, client_secret)

            # Then
            assert result is None

    def test_setup_and_validate_oauth_flow_no_validation_method(self, auth_manager: AuthenticationManager) -> None:
        """Test OAuth flow setup when flow doesn't have validation method."""
        from llm_orc.core.auth.authentication import _setup_and_validate_oauth_flow

        # Given
        provider = "anthropic"
        client_id = "test_client_id"
        client_secret = "test_client_secret"

        # When
        with patch("llm_orc.core.auth.authentication.create_oauth_flow") as mock_create_flow:
            mock_flow = Mock()
            del mock_flow.validate_credentials  # Remove the method
            mock_create_flow.return_value = mock_flow

            result = _setup_and_validate_oauth_flow(provider, client_id, client_secret)

            # Then
            assert result == mock_flow
            mock_create_flow.assert_called_once_with(provider, client_id, client_secret)

    def test_get_authorization_url_and_open_browser_success(self, auth_manager: AuthenticationManager) -> None:
        """Test getting authorization URL and opening browser successfully."""
        from llm_orc.core.auth.authentication import _get_authorization_url_and_open_browser

        # Given
        mock_flow = Mock()
        mock_flow.get_authorization_url.return_value = "https://example.com/auth"

        # When
        with patch("llm_orc.core.auth.authentication.webbrowser.open") as mock_open:
            result = _get_authorization_url_and_open_browser(mock_flow)

            # Then
            assert result is True
            mock_flow.get_authorization_url.assert_called_once()
            mock_open.assert_called_once_with("https://example.com/auth")

    def test_get_authorization_url_and_open_browser_failure(self, auth_manager: AuthenticationManager) -> None:
        """Test getting authorization URL when it fails."""
        from llm_orc.core.auth.authentication import _get_authorization_url_and_open_browser

        # Given
        mock_flow = Mock()
        mock_flow.get_authorization_url.side_effect = Exception("URL generation failed")

        # When
        result = _get_authorization_url_and_open_browser(mock_flow)

        # Then
        assert result is False

    def test_exchange_authorization_code_for_tokens_success(self, auth_manager: AuthenticationManager) -> None:
        """Test exchanging authorization code for tokens successfully."""
        from llm_orc.core.auth.authentication import _exchange_authorization_code_for_tokens

        # Given
        mock_flow = Mock()
        mock_flow.start_manual_callback_flow.return_value = "auth_code_123"
        mock_flow.exchange_code_for_tokens.return_value = {
            "access_token": "access_123",
            "refresh_token": "refresh_123",
            "expires_in": 3600
        }

        # When
        result = _exchange_authorization_code_for_tokens(mock_flow)

        # Then
        assert result == {
            "access_token": "access_123",
            "refresh_token": "refresh_123",
            "expires_in": 3600
        }
        mock_flow.start_manual_callback_flow.assert_called_once()
        mock_flow.exchange_code_for_tokens.assert_called_once_with("auth_code_123")

    def test_exchange_authorization_code_for_tokens_manual_extraction(self, auth_manager: AuthenticationManager) -> None:
        """Test handling manual token extraction requirement."""
        from llm_orc.core.auth.authentication import _exchange_authorization_code_for_tokens

        # Given
        mock_flow = Mock()
        mock_flow.start_manual_callback_flow.return_value = "auth_code_123"
        mock_flow.exchange_code_for_tokens.return_value = {
            "requires_manual_extraction": True
        }

        # When
        result = _exchange_authorization_code_for_tokens(mock_flow)

        # Then
        assert result is None

    def test_exchange_authorization_code_for_tokens_fallback_input(self, auth_manager: AuthenticationManager) -> None:
        """Test fallback to manual input when flow doesn't have callback method."""
        from llm_orc.core.auth.authentication import _exchange_authorization_code_for_tokens

        # Given
        mock_flow = Mock()
        del mock_flow.start_manual_callback_flow  # Remove the method
        mock_flow.exchange_code_for_tokens.return_value = {
            "access_token": "access_123",
            "expires_in": 3600
        }

        # When
        with patch("llm_orc.core.auth.authentication.input", return_value="fallback_code"):
            result = _exchange_authorization_code_for_tokens(mock_flow)

            # Then
            assert result == {
                "access_token": "access_123",
                "expires_in": 3600
            }
            mock_flow.exchange_code_for_tokens.assert_called_once_with("fallback_code")

    def test_exchange_authorization_code_for_tokens_invalid_tokens(self, auth_manager: AuthenticationManager) -> None:
        """Test handling invalid or empty tokens."""
        from llm_orc.core.auth.authentication import _exchange_authorization_code_for_tokens

        # Given
        mock_flow = Mock()
        mock_flow.start_manual_callback_flow.return_value = "auth_code_123"
        mock_flow.exchange_code_for_tokens.return_value = {}  # No access_token

        # When
        result = _exchange_authorization_code_for_tokens(mock_flow)

        # Then
        assert result is None

    def test_store_tokens_and_create_client_success(self, auth_manager: AuthenticationManager) -> None:
        """Test storing tokens and creating client successfully."""
        from llm_orc.core.auth.authentication import _store_tokens_and_create_client

        # Given
        provider = "anthropic"
        tokens = {
            "access_token": "access_123",
            "refresh_token": "refresh_123",
            "expires_in": 3600,
            "token_type": "Bearer"
        }

        # When
        with patch("llm_orc.core.auth.authentication.time.time", return_value=1000):
            result = _store_tokens_and_create_client(auth_manager, provider, tokens)

            # Then
            assert result is not None
            assert result.access_token == "access_123"
            assert result.token_type == "Bearer"
            
            # Verify tokens were stored
            stored_method = auth_manager.credential_storage.get_auth_method(provider)
            assert stored_method == "oauth"

    def test_store_tokens_and_create_client_storage_failure(self, auth_manager: AuthenticationManager) -> None:
        """Test handling token storage failure."""
        from llm_orc.core.auth.authentication import _store_tokens_and_create_client

        # Given
        provider = "anthropic"
        tokens = {
            "access_token": "access_123",
            "expires_in": 3600
        }

        # When
        with patch.object(auth_manager.credential_storage, 'store_oauth_token', side_effect=Exception("Storage failed")):
            result = _store_tokens_and_create_client(auth_manager, provider, tokens)

            # Then
            assert result is None
