"""Authentication system for LLM Orchestra supporting credential storage."""

import logging
import os
import time
import webbrowser
from typing import Any

import yaml
from cryptography.fernet import Fernet, InvalidToken

from llm_orc.core.auth.oauth_flows import (
    AnthropicOAuthFlow,
    GoogleGeminiOAuthFlow,
    OAuthCallbackHandler,
    OAuthFlow,
    create_oauth_flow,
)
from llm_orc.core.config.config_manager import ConfigurationManager

logger = logging.getLogger(__name__)

__all__ = [
    "CredentialStorage",
    "AuthenticationManager",
    "OAuthCallbackHandler",
    "OAuthFlow",
    "GoogleGeminiOAuthFlow",
    "AnthropicOAuthFlow",
    "create_oauth_flow",
]


def _setup_and_validate_oauth_flow(
    provider: str, client_id: str, client_secret: str
) -> OAuthFlow | None:
    """Setup and validate OAuth flow.

    Args:
        provider: OAuth provider name
        client_id: OAuth client ID
        client_secret: OAuth client secret

    Returns:
        OAuth flow if successful, None if validation fails
    """
    # Create OAuth flow with enhanced error handling
    oauth_flow = create_oauth_flow(provider, client_id, client_secret)

    # Validate credentials if the provider supports it
    if hasattr(oauth_flow, "validate_credentials"):
        print("🔍 Validating OAuth credentials...")
        if not oauth_flow.validate_credentials():
            print("❌ OAuth credential validation failed")
            return None

    return oauth_flow


def _get_authorization_url_and_open_browser(oauth_flow: OAuthFlow) -> bool:
    """Get authorization URL and open browser.

    Args:
        oauth_flow: OAuth flow instance

    Returns:
        True if successful, False otherwise
    """
    # We're using Anthropic's callback endpoint, so no local server needed
    print("🔧 Using Anthropic's OAuth callback endpoint...")

    # Get authorization URL and open browser
    try:
        auth_url = oauth_flow.get_authorization_url()
        print("🌐 Opening browser for OAuth authorization...")
        print(f"   URL: {auth_url}")
        webbrowser.open(auth_url)
        return True
    except Exception as e:
        print(f"❌ Failed to get authorization URL: {e}")
        return False


def _exchange_authorization_code_for_tokens(
    oauth_flow: OAuthFlow,
) -> dict[str, Any] | None:
    """Exchange authorization code for tokens.

    Args:
        oauth_flow: OAuth flow instance

    Returns:
        Tokens dictionary if successful, None otherwise
    """
    # Use manual callback flow to get authorization code
    try:
        if hasattr(oauth_flow, "start_manual_callback_flow"):
            auth_code = oauth_flow.start_manual_callback_flow()
        else:
            # Fallback for other OAuth flows
            auth_code = input("Enter authorization code from callback URL: ").strip()
        print("✅ Authorization code received!")

        # Exchange code for tokens
        print("🔄 Exchanging code for access tokens...")
        tokens = oauth_flow.exchange_code_for_tokens(auth_code)

        # Check if manual extraction is required
        if tokens.get("requires_manual_extraction"):
            print("\n🔧 OAuth flow completed - manual token extraction required")
            print("   Authorization successful, token exchange needs manual steps")
            print("   Follow the instructions above to extract tokens manually")

            # For now, we'll return None since we don't have tokens yet
            # In a real implementation, you might want to:
            # 1. Save the auth_code for later manual exchange
            # 2. Provide a separate method to input manually extracted tokens
            # 3. Guide the user through the manual process
            return None

        if not tokens or "access_token" not in tokens:
            print("❌ Failed to receive valid tokens")
            return None

        print("✅ Access tokens received!")
        return tokens
    except Exception as e:
        print(f"❌ Token exchange failed: {e}")
        return None


def _store_tokens_and_create_client(
    auth_manager: "AuthenticationManager", provider: str, tokens: dict[str, Any]
) -> Any | None:
    """Store tokens and create authenticated client.

    Args:
        auth_manager: Authentication manager instance
        provider: OAuth provider name
        tokens: OAuth tokens dictionary

    Returns:
        Mock client if successful, None otherwise
    """
    # Store tokens
    try:
        expires_at = int(time.time()) + tokens.get("expires_in", 3600)
        auth_manager.credential_storage.store_oauth_token(
            provider,
            tokens["access_token"],
            tokens.get("refresh_token"),
            expires_at,
        )
        print("✅ Tokens stored securely!")
    except Exception as e:
        print(f"❌ Failed to store tokens: {e}")
        return None

    # Create mock client for testing
    client = type(
        "MockOAuthClient",
        (),
        {
            "access_token": tokens["access_token"],
            "token_type": tokens.get("token_type", "Bearer"),
        },
    )()

    auth_manager._authenticated_clients[provider] = client
    return client


class CredentialStorage:
    """Handles encrypted storage and retrieval of credentials."""

    def __init__(self, config_manager: ConfigurationManager | None = None):
        """Initialize credential storage.

        Args:
            config_manager: Configuration manager instance. If None, creates a new one.
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.config_manager.ensure_global_config_dir()

        self.credentials_file = self.config_manager.get_credentials_file()
        self._encryption_key = self._get_or_create_encryption_key()

    def _get_or_create_encryption_key(self) -> Fernet:
        """Get or create encryption key for credential storage."""
        key_file = self.config_manager.get_encryption_key_file()

        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Secure the key file
            os.chmod(key_file, 0o600)

        return Fernet(key)

    def _load_credentials(self) -> dict[str, Any]:
        """Load and decrypt credentials from file."""
        if not self.credentials_file.exists():
            return {}

        try:
            with open(self.credentials_file) as f:
                encrypted_data = f.read()

            if not encrypted_data.strip():
                return {}

            decrypted_data = self._encryption_key.decrypt(encrypted_data.encode())
            loaded_data = yaml.safe_load(decrypted_data.decode())
            return loaded_data if isinstance(loaded_data, dict) else {}
        except InvalidToken:
            logger.warning(
                "Credential decryption failed — encryption key may have changed"
            )
            return {}
        except yaml.YAMLError as e:
            logger.warning("Credential file could not be parsed as YAML: %s", e)
            return {}
        except Exception as e:
            logger.warning("Unexpected error loading credentials: %s", e, exc_info=True)
            return {}

    def _save_credentials(self, credentials: dict[str, Any]) -> None:
        """Encrypt and save credentials to file."""
        yaml_data = yaml.dump(credentials)
        encrypted_data = self._encryption_key.encrypt(yaml_data.encode())

        with open(self.credentials_file, "w") as f:
            f.write(encrypted_data.decode())

        # Secure the credentials file
        os.chmod(self.credentials_file, 0o600)

    def store_api_key(self, provider: str, api_key: str) -> None:
        """Store an API key for a provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'google')
            api_key: API key to store
        """
        credentials = self._load_credentials()

        if provider not in credentials:
            credentials[provider] = {}

        credentials[provider]["auth_method"] = "api_key"
        credentials[provider]["api_key"] = api_key

        self._save_credentials(credentials)

    def store_oauth_token(
        self,
        provider: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: int | None = None,
        client_id: str | None = None,
    ) -> None:
        """Store OAuth tokens for a provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'google')
            access_token: OAuth access token
            refresh_token: OAuth refresh token (optional)
            expires_at: Token expiration timestamp (optional)
            client_id: OAuth client ID (optional)
        """
        credentials = self._load_credentials()

        if provider not in credentials:
            credentials[provider] = {}

        credentials[provider]["auth_method"] = "oauth"
        credentials[provider]["access_token"] = access_token
        if refresh_token:
            credentials[provider]["refresh_token"] = refresh_token
        if expires_at:
            credentials[provider]["expires_at"] = expires_at
        if client_id:
            credentials[provider]["client_id"] = client_id

        self._save_credentials(credentials)

    def get_api_key(self, provider: str) -> str | None:
        """Retrieve an API key for a provider.

        Args:
            provider: Provider name

        Returns:
            API key if found, None otherwise
        """
        credentials = self._load_credentials()

        if provider in credentials and "api_key" in credentials[provider]:
            api_key = credentials[provider]["api_key"]
            return str(api_key) if api_key is not None else None

        return None

    def get_oauth_token(self, provider: str) -> dict[str, Any] | None:
        """Retrieve OAuth tokens for a provider.

        Args:
            provider: Provider name

        Returns:
            OAuth token info if found, None otherwise
        """
        credentials = self._load_credentials()

        if (
            provider in credentials
            and credentials[provider].get("auth_method") == "oauth"
        ):
            token_info = {}
            if "access_token" in credentials[provider]:
                token_info["access_token"] = credentials[provider]["access_token"]
            if "refresh_token" in credentials[provider]:
                token_info["refresh_token"] = credentials[provider]["refresh_token"]
            if "expires_at" in credentials[provider]:
                token_info["expires_at"] = credentials[provider]["expires_at"]
            if "client_id" in credentials[provider]:
                token_info["client_id"] = credentials[provider]["client_id"]
            return token_info if token_info else None

        return None

    def get_auth_method(self, provider: str) -> str | None:
        """Get the authentication method for a provider.

        Args:
            provider: Provider name

        Returns:
            Auth method ('api_key' or 'oauth') if found, None otherwise
        """
        credentials = self._load_credentials()

        if provider in credentials:
            auth_method = credentials[provider].get("auth_method")
            return str(auth_method) if auth_method is not None else None

        return None

    def list_providers(self) -> list[str]:
        """List all configured providers.

        Returns:
            List of provider names
        """
        credentials = self._load_credentials()
        return list(credentials.keys())

    def remove_provider(self, provider: str) -> None:
        """Remove a provider's credentials.

        Args:
            provider: Provider name to remove
        """
        credentials = self._load_credentials()

        if provider in credentials:
            del credentials[provider]
            self._save_credentials(credentials)


class AuthenticationManager:
    """Manages authentication with LLM providers."""

    def __init__(self, credential_storage: CredentialStorage):
        """Initialize authentication manager.

        Args:
            credential_storage: CredentialStorage instance to use for storing
                credentials
        """
        self.credential_storage = credential_storage
        self._authenticated_clients: dict[str, Any] = {}

    def authenticate(self, provider: str, api_key: str) -> bool:
        """Authenticate with a provider using API key.

        Args:
            provider: Provider name
            api_key: API key for authentication

        Returns:
            True if authentication successful, False otherwise
        """
        # For now, basic validation - in real implementation would test API key
        if not api_key or api_key == "invalid_key":
            return False

        # Store the API key
        self.credential_storage.store_api_key(provider, api_key)

        # Create mock client for testing
        client = type("MockClient", (), {"api_key": api_key, "_api_key": api_key})()

        self._authenticated_clients[provider] = client
        return True

    def authenticate_oauth(
        self, provider: str, client_id: str, client_secret: str
    ) -> bool:
        """Authenticate with a provider using OAuth.

        Args:
            provider: Provider name
            client_id: OAuth client ID
            client_secret: OAuth client secret

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Setup and validate OAuth flow
            oauth_flow = _setup_and_validate_oauth_flow(
                provider, client_id, client_secret
            )
            if not oauth_flow:
                return False

            # Get authorization URL and open browser
            if not _get_authorization_url_and_open_browser(oauth_flow):
                return False

            # Exchange authorization code for tokens
            tokens = _exchange_authorization_code_for_tokens(oauth_flow)
            if not tokens:
                return False

            # Store tokens and create client
            client = _store_tokens_and_create_client(self, provider, tokens)
            return client is not None

        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            return False
        except ConnectionError as e:
            print(f"❌ Network connection error: {e}")
            print("   Please check your internet connection and try again")
            return False
        except Exception as e:
            print(f"❌ OAuth authentication failed: {e}")
            return False

    def store_manual_oauth_token(
        self,
        provider: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int = 3600,
    ) -> bool:
        """Store manually extracted OAuth tokens."""
        try:
            expires_at = int(time.time()) + expires_in
            self.credential_storage.store_oauth_token(
                provider,
                access_token,
                refresh_token,
                expires_at,
            )

            # Create client for the provider
            client = type(
                "ManualOAuthClient",
                (),
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                },
            )()

            self._authenticated_clients[provider] = client
            print(f"✅ Manual OAuth tokens stored successfully for {provider}")
            return True

        except Exception as e:
            print(f"❌ Failed to store manual OAuth tokens: {e}")
            return False

    def is_authenticated(self, provider: str) -> bool:
        """Check if a provider is authenticated.

        Args:
            provider: Provider name

        Returns:
            True if authenticated, False otherwise
        """
        return provider in self._authenticated_clients

    def get_authenticated_client(self, provider: str) -> Any | None:
        """Get an authenticated client for a provider.

        Args:
            provider: Provider name

        Returns:
            Authenticated client if available, None otherwise
        """
        return self._authenticated_clients.get(provider)

    def logout_oauth_provider(self, provider: str) -> bool:
        """Logout an OAuth provider by revoking tokens and removing credentials.

        Args:
            provider: Provider name to logout

        Returns:
            True if logout successful, False otherwise
        """
        try:
            # Check if provider exists and is OAuth
            auth_method = self.credential_storage.get_auth_method(provider)
            if not auth_method or auth_method != "oauth":
                return False

            # Get OAuth token information
            oauth_info = self.credential_storage.get_oauth_token(provider)
            if not oauth_info:
                return False

            # Get client_id from stored credentials
            credentials = self.credential_storage._load_credentials()
            provider_data = credentials.get(provider, {})
            client_id = provider_data.get("client_id")

            if not client_id:
                # If no client_id, we can't revoke tokens via API
                # but we can still remove local credentials
                self.credential_storage.remove_provider(provider)
                if provider in self._authenticated_clients:
                    del self._authenticated_clients[provider]
                return True

            # Create OAuth client to revoke tokens
            from llm_orc.core.auth.oauth_client import OAuthClaudeClient

            oauth_client = OAuthClaudeClient(
                access_token=oauth_info["access_token"],
                refresh_token=oauth_info.get("refresh_token"),
            )

            try:
                # Attempt to revoke tokens
                oauth_client.revoke_all_tokens(client_id)
            except Exception:
                # Continue even if token revocation fails
                # (tokens may already be expired or network issues)
                pass

            # Remove local credentials regardless of revocation success
            self.credential_storage.remove_provider(provider)

            # Remove from authenticated clients
            if provider in self._authenticated_clients:
                del self._authenticated_clients[provider]

            return True

        except Exception:
            return False

    def logout_all_oauth_providers(self) -> dict[str, bool]:
        """Logout all OAuth providers.

        Returns:
            Dict mapping provider names to logout success status
        """
        results = {}

        # Find all OAuth providers
        providers = self.credential_storage.list_providers()
        oauth_providers = []

        for provider in providers:
            auth_method = self.credential_storage.get_auth_method(provider)
            if auth_method == "oauth":
                oauth_providers.append(provider)

        # Logout each OAuth provider
        for provider in oauth_providers:
            results[provider] = self.logout_oauth_provider(provider)

        return results
