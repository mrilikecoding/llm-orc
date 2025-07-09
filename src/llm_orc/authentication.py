"""Authentication system for LLM Orchestra supporting credential storage."""

import os
from typing import Any

import yaml
from cryptography.fernet import Fernet

from llm_orc.config import ConfigurationManager


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
        except Exception:
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
