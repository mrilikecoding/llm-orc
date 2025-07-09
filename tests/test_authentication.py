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
