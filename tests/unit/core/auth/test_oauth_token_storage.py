"""Tests for OAuth token storage enhancements including client_id support."""

from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.auth.authentication import CredentialStorage


@pytest.fixture(autouse=True)
def mock_expensive_dependencies() -> Generator[None, None, None]:
    """Mock expensive dependencies for all OAuth token storage tests."""
    with patch("llm_orc.core.auth.authentication.ConfigurationManager"):
        # Use surgical mocking - only mock expensive config I/O
        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_config"
        ):
            with patch(
                "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_ensembles"
            ):
                with patch(
                    "llm_orc.core.config.config_manager.ConfigurationManager._copy_profile_templates"
                ):
                    yield


class TestOAuthTokenStorage:
    """Test OAuth token storage with client_id support."""

    @pytest.fixture
    def credential_storage(self, tmp_path: Any) -> CredentialStorage:
        """Create credential storage for testing."""
        config_manager = Mock()
        config_manager._global_config_dir = tmp_path / ".test-llm-orc"
        config_manager._global_config_dir.mkdir(parents=True, exist_ok=True)
        config_manager.get_encryption_key_file.return_value = (
            config_manager._global_config_dir / ".encryption_key"
        )
        config_manager.get_credentials_file.return_value = (
            config_manager._global_config_dir / "credentials.yaml"
        )
        config_manager.ensure_global_config_dir.return_value = None

        storage = CredentialStorage(config_manager=config_manager)
        return storage

    def test_store_oauth_token_with_client_id(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that OAuth tokens can be stored with client_id."""
        # Store OAuth token with all parameters including client_id
        credential_storage.store_oauth_token(
            provider="google-oauth",
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_at=1234567890,
            client_id="test-client-id-12345",
        )

        # Retrieve and verify all fields are stored
        oauth_token = credential_storage.get_oauth_token("google-oauth")

        assert oauth_token is not None
        assert oauth_token["access_token"] == "test_access_token"
        assert oauth_token["refresh_token"] == "test_refresh_token"
        assert oauth_token["expires_at"] == 1234567890
        assert oauth_token["client_id"] == "test-client-id-12345"

    def test_store_oauth_token_without_client_id_backward_compatibility(
        self, credential_storage: CredentialStorage
    ) -> None:
        """
        Test that OAuth tokens can be stored without client_id (backward compatibility).
        """
        # Store OAuth token without client_id (legacy behavior)
        credential_storage.store_oauth_token(
            provider="legacy-provider",
            access_token="legacy_access_token",
            refresh_token="legacy_refresh_token",
            expires_at=1234567890,
            # No client_id provided
        )

        # Retrieve and verify basic fields are stored, client_id is not present
        oauth_token = credential_storage.get_oauth_token("legacy-provider")

        assert oauth_token is not None
        assert oauth_token["access_token"] == "legacy_access_token"
        assert oauth_token["refresh_token"] == "legacy_refresh_token"
        assert oauth_token["expires_at"] == 1234567890
        assert "client_id" not in oauth_token

    def test_auth_method_stored_as_oauth(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that auth_method is correctly stored as 'oauth'."""
        credential_storage.store_oauth_token(
            provider="test-oauth-provider",
            access_token="test_token",
            client_id="test_client_id",
        )

        # Verify auth method is stored correctly
        auth_method = credential_storage.get_auth_method("test-oauth-provider")
        assert auth_method == "oauth"

    def test_get_oauth_token_returns_none_for_non_oauth_provider(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that get_oauth_token returns None for non-OAuth providers."""
        # Store an API key provider
        credential_storage.store_api_key("api-key-provider", "test_api_key")

        # Should not return OAuth token for API key provider
        oauth_token = credential_storage.get_oauth_token("api-key-provider")
        assert oauth_token is None

    def test_get_oauth_token_returns_none_for_nonexistent_provider(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that get_oauth_token returns None for nonexistent providers."""
        oauth_token = credential_storage.get_oauth_token("nonexistent-provider")
        assert oauth_token is None

    def test_oauth_token_update_preserves_client_id(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that updating OAuth tokens preserves client_id."""
        # Initial storage with client_id
        credential_storage.store_oauth_token(
            provider="test-provider",
            access_token="initial_token",
            refresh_token="initial_refresh",
            expires_at=1000000000,
            client_id="preserved_client_id",
        )

        # Update with new tokens but same client_id
        credential_storage.store_oauth_token(
            provider="test-provider",
            access_token="updated_token",
            refresh_token="updated_refresh",
            expires_at=2000000000,
            client_id="preserved_client_id",
        )

        # Verify client_id is preserved and tokens are updated
        oauth_token = credential_storage.get_oauth_token("test-provider")
        assert oauth_token is not None
        assert oauth_token["access_token"] == "updated_token"
        assert oauth_token["refresh_token"] == "updated_refresh"
        assert oauth_token["expires_at"] == 2000000000
        assert oauth_token["client_id"] == "preserved_client_id"

    def test_multiple_oauth_providers_with_different_client_ids(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test storing multiple OAuth providers with different client_ids."""
        # Store multiple OAuth providers
        credential_storage.store_oauth_token(
            provider="provider-one",
            access_token="provider_one_token",
            client_id="client_id_one",
        )

        credential_storage.store_oauth_token(
            provider="provider-two",
            access_token="provider_two_token",
            client_id="client_id_two",
        )

        # Verify both are stored with correct client_ids
        token_one = credential_storage.get_oauth_token("provider-one")
        token_two = credential_storage.get_oauth_token("provider-two")

        assert token_one is not None
        assert token_two is not None
        assert token_one["client_id"] == "client_id_one"
        assert token_two["client_id"] == "client_id_two"
        assert token_one["access_token"] == "provider_one_token"
        assert token_two["access_token"] == "provider_two_token"
