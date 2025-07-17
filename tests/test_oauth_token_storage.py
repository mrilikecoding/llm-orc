"""Tests for OAuth token storage enhancements including client_id support."""

from typing import Any
from unittest.mock import Mock

import pytest

from llm_orc.authentication import CredentialStorage
from llm_orc.config import ConfigurationManager
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.models import OAuthClaudeModel


class TestOAuthTokenStorage:
    """Test OAuth token storage with client_id support."""

    @pytest.fixture
    def credential_storage(self, tmp_path: Any) -> CredentialStorage:
        """Create credential storage for testing."""
        config_manager = ConfigurationManager()
        config_manager._global_config_dir = tmp_path / ".test-llm-orc"
        config_manager._global_config_dir.mkdir(parents=True, exist_ok=True)

        storage = CredentialStorage(config_manager=config_manager)
        return storage

    def test_store_oauth_token_with_client_id(
        self, credential_storage: CredentialStorage
    ) -> None:
        """Test that OAuth tokens can be stored with client_id."""
        # Store OAuth token with all parameters including client_id
        credential_storage.store_oauth_token(
            provider="anthropic-claude-pro-max",
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_at=1234567890,
            client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        )

        # Retrieve and verify all fields are stored
        oauth_token = credential_storage.get_oauth_token("anthropic-claude-pro-max")

        assert oauth_token is not None
        assert oauth_token["access_token"] == "test_access_token"
        assert oauth_token["refresh_token"] == "test_refresh_token"
        assert oauth_token["expires_at"] == 1234567890
        assert oauth_token["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

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

    @pytest.mark.asyncio
    async def test_ensemble_execution_client_id_fallback(self) -> None:
        """
        Test ensemble execution uses client_id fallback for anthropic-claude-pro-max.
        """
        executor = EnsembleExecutor()

        # Mock the credential storage to return OAuth token without client_id
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            # No client_id in stored token (simulating legacy storage)
        }

        # Replace the executor's credential storage with mock
        executor._credential_storage = mock_storage

        # Mock the OAuthClaudeModel constructor to verify client_id is passed
        oauth_model_calls = []

        def mock_oauth_claude_model(*args: Any, **kwargs: Any) -> Mock:
            oauth_model_calls.append(kwargs)
            mock_model = Mock(spec=OAuthClaudeModel)
            return mock_model

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "llm_orc.ensemble_execution.OAuthClaudeModel", mock_oauth_claude_model
            )

            # Load the model
            await executor._load_model("anthropic-claude-pro-max")

            # Verify that the hardcoded client_id was used as fallback
            assert len(oauth_model_calls) == 1
            expected_client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
            assert oauth_model_calls[0]["client_id"] == expected_client_id
            assert oauth_model_calls[0]["access_token"] == "test_access_token"
            assert oauth_model_calls[0]["refresh_token"] == "test_refresh_token"

    @pytest.mark.asyncio
    async def test_ensemble_execution_uses_stored_client_id_when_available(
        self,
    ) -> None:
        """Test that ensemble execution uses stored client_id when available."""
        executor = EnsembleExecutor()

        # Mock the credential storage to return OAuth token WITH client_id
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"
        mock_storage.get_oauth_token.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "client_id": "custom_client_id_from_storage",  # Custom client_id stored
        }

        # Replace the executor's credential storage with mock
        executor._credential_storage = mock_storage

        # Mock the OAuthClaudeModel constructor to verify client_id is passed
        oauth_model_calls = []

        def mock_oauth_claude_model(*args: Any, **kwargs: Any) -> Mock:
            oauth_model_calls.append(kwargs)
            mock_model = Mock(spec=OAuthClaudeModel)
            return mock_model

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "llm_orc.ensemble_execution.OAuthClaudeModel", mock_oauth_claude_model
            )

            # Load the model
            await executor._load_model("anthropic-claude-pro-max")

            # Verify that the stored client_id was used (not the fallback)
            assert len(oauth_model_calls) == 1
            assert oauth_model_calls[0]["client_id"] == "custom_client_id_from_storage"

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
            provider="anthropic-claude-pro-max",
            access_token="anthropic_token",
            client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        )

        credential_storage.store_oauth_token(
            provider="google-gemini-oauth",
            access_token="google_token",
            client_id="google_client_id_123",
        )

        # Verify both are stored with correct client_ids
        anthropic_token = credential_storage.get_oauth_token("anthropic-claude-pro-max")
        google_token = credential_storage.get_oauth_token("google-gemini-oauth")

        assert anthropic_token is not None
        assert google_token is not None
        assert anthropic_token["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        assert google_token["client_id"] == "google_client_id_123"
        assert anthropic_token["access_token"] == "anthropic_token"
        assert google_token["access_token"] == "google_token"
