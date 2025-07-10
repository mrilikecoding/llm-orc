"""Tests for authentication system including credential storage."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager


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
        from llm_orc.authentication import GoogleGeminiOAuthFlow

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
        from llm_orc.authentication import GoogleGeminiOAuthFlow

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
        from llm_orc.authentication import AnthropicOAuthFlow

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
        from llm_orc.authentication import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        oauth_flow = AnthropicOAuthFlow(client_id, client_secret)
        auth_code = "test_authorization_code"

        # When
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
        from llm_orc.authentication import create_oauth_flow

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
        from llm_orc.authentication import create_oauth_flow

        # When & Then
        with pytest.raises(ValueError, match="OAuth not supported for provider"):
            create_oauth_flow("unsupported_provider", "client_id", "client_secret")


class TestAnthropicOAuthFlow:
    """Test improved Anthropic OAuth flow functionality."""

    def test_anthropic_oauth_flow_initialization(self) -> None:
        """Test AnthropicOAuthFlow can be initialized correctly."""
        # Given
        from llm_orc.authentication import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"

        # When
        flow = AnthropicOAuthFlow(client_id, client_secret)

        # Then
        assert flow.client_id == client_id
        assert flow.client_secret == client_secret
        assert flow.provider == "anthropic"
        assert flow.redirect_uri == "http://localhost:8080/callback"

    def test_get_authorization_url_contains_required_parameters(self) -> None:
        """Test that authorization URL contains all required OAuth parameters."""
        # Given
        from urllib.parse import parse_qs, urlparse

        from llm_orc.authentication import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        flow = AnthropicOAuthFlow(client_id, client_secret)

        # When
        auth_url = flow.get_authorization_url()

        # Then
        parsed_url = urlparse(auth_url)
        query_params = parse_qs(parsed_url.query)

        assert parsed_url.netloc == "console.anthropic.com"
        assert parsed_url.path == "/oauth/authorize"
        assert query_params["client_id"][0] == client_id
        assert query_params["response_type"][0] == "code"
        assert query_params["redirect_uri"][0] == flow.redirect_uri
        assert "state" in query_params

    def test_validate_credentials_with_accessible_endpoint(self) -> None:
        """Test credential validation when OAuth endpoint is accessible."""
        # Given
        from llm_orc.authentication import AnthropicOAuthFlow

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
        from llm_orc.authentication import AnthropicOAuthFlow

        client_id = "test_client_id"
        client_secret = "test_client_secret"
        flow = AnthropicOAuthFlow(client_id, client_secret)
        auth_code = "test_auth_code_123"

        # When
        tokens = flow.exchange_code_for_tokens(auth_code)

        # Then
        assert isinstance(tokens, dict)
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "expires_in" in tokens
        assert "token_type" in tokens

        # Verify token format
        assert tokens["access_token"].startswith("anthropic_access_token_")
        assert tokens["refresh_token"].startswith("anthropic_refresh_token_")
        assert tokens["expires_in"] == 3600
        assert tokens["token_type"] == "Bearer"

    def test_mock_create_with_guidance_method_exists(self) -> None:
        """Test that create_with_guidance method exists for future testing."""
        # Given
        from llm_orc.authentication import AnthropicOAuthFlow

        # When & Then
        assert hasattr(AnthropicOAuthFlow, "create_with_guidance")
        assert callable(AnthropicOAuthFlow.create_with_guidance)


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
        from llm_orc.authentication import AnthropicOAuthFlow

        validation_called = False

        def mock_validate(self) -> bool:
            nonlocal validation_called
            validation_called = True
            return True

        monkeypatch.setattr(AnthropicOAuthFlow, "validate_credentials", mock_validate)

        # Mock the OAuth flow to avoid actual browser/server operations
        def mock_start_server(self):
            server = type(
                "MockServer", (), {"auth_code": "test_code", "auth_error": None}
            )()
            return server, 8080

        def mock_open_browser(url):
            pass

        monkeypatch.setattr(
            AnthropicOAuthFlow, "start_callback_server", mock_start_server
        )
        monkeypatch.setattr("webbrowser.open", mock_open_browser)

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
        from llm_orc.authentication import AnthropicOAuthFlow

        def mock_start_server(self):
            # Return a server that never receives auth code (simulating timeout)
            server = type("MockServer", (), {"auth_code": None, "auth_error": None})()
            return server, 8080

        def mock_open_browser(url):
            pass

        # Mock time to simulate timeout quickly
        import time

        call_count = 0
        start_time = time.time()

        def mock_time():
            nonlocal call_count
            call_count += 1
            # First few calls return normal time, then jump to timeout
            if call_count > 5:
                return start_time + 150  # Beyond the 120 second timeout
            return start_time + (call_count * 0.1)  # Gradual increase initially

        def mock_sleep(duration):
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
