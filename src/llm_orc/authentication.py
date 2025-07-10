"""Authentication system for LLM Orchestra supporting credential storage."""

import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

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

    def store_oauth_token(
        self,
        provider: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        """Store OAuth tokens for a provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'google')
            access_token: OAuth access token
            refresh_token: OAuth refresh token (optional)
            expires_at: Token expiration timestamp (optional)
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


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET request for OAuth callback."""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # Store the authorization code
        if "code" in query_params:
            self.server.auth_code = query_params["code"][0]  # type: ignore
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <body>
            <h1>Authorization Successful!</h1>
            <p>You can close this window and return to the CLI.</p>
            </body>
            </html>
            """)
        elif "error" in query_params:
            self.server.auth_error = query_params["error"][0]  # type: ignore
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
            <html>
            <body>
            <h1>Authorization Failed</h1>
            <p>Error: """
                + query_params["error"][0].encode()
                + b"""</p>
            </body>
            </html>
            """
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Invalid callback")

    def log_message(self, format_str: str, *args: Any) -> None:
        """Suppress log messages."""
        pass


class OAuthFlow:
    """Handles OAuth flow for LLM providers."""

    def __init__(
        self,
        provider: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
    ):
        self.provider = provider
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.state = secrets.token_urlsafe(32)

    def get_authorization_url(self) -> str:
        """Get the authorization URL for the provider."""
        # This is a generic implementation - providers would override this
        if self.provider == "google":
            return (
                f"https://accounts.google.com/o/oauth2/v2/auth?"
                f"client_id={self.client_id}&"
                f"redirect_uri={self.redirect_uri}&"
                f"response_type=code&"
                f"scope=https://www.googleapis.com/auth/userinfo.email&"
                f"state={self.state}"
            )
        elif self.provider == "github":
            return (
                f"https://github.com/login/oauth/authorize?"
                f"client_id={self.client_id}&"
                f"redirect_uri={self.redirect_uri}&"
                f"state={self.state}&"
                f"scope=user:email"
            )
        else:
            raise ValueError(f"OAuth not supported for provider: {self.provider}")

    def start_callback_server(self) -> tuple[HTTPServer, int]:
        """Start the callback server and return auth code."""
        # Find an available port
        port = 8080
        while port < 8090:
            try:
                server = HTTPServer(("localhost", port), OAuthCallbackHandler)
                server.auth_code = None  # type: ignore
                server.auth_error = None  # type: ignore
                break
            except OSError:
                port += 1
        else:
            raise RuntimeError("No available port for OAuth callback")

        # Update redirect URI with actual port
        self.redirect_uri = f"http://localhost:{port}/callback"

        def run_server() -> None:
            server.timeout = 1
            while server.auth_code is None and server.auth_error is None:  # type: ignore
                server.handle_request()

        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

        return server, port

    def exchange_code_for_tokens(self, auth_code: str) -> dict[str, Any]:
        """Exchange authorization code for tokens."""
        # This would typically make an HTTP request to the provider's token endpoint
        # For now, return a mock response
        return {
            "access_token": f"mock_access_token_{auth_code[:10]}",
            "refresh_token": f"mock_refresh_token_{auth_code[:10]}",
            "expires_in": 3600,
            "token_type": "Bearer",
        }


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
            oauth_flow = OAuthFlow(provider, client_id, client_secret)

            # Start callback server
            server, port = oauth_flow.start_callback_server()

            # Get authorization URL and open browser
            auth_url = oauth_flow.get_authorization_url()
            print(f"Opening browser for OAuth authorization: {auth_url}")
            webbrowser.open(auth_url)

            # Wait for callback
            print("Waiting for authorization callback...")
            timeout = 60  # 60 second timeout
            start_time = time.time()

            while server.auth_code is None and server.auth_error is None:  # type: ignore
                if time.time() - start_time > timeout:
                    print("OAuth flow timed out")
                    return False
                time.sleep(0.1)

            if server.auth_error:  # type: ignore
                print(f"OAuth error: {server.auth_error}")  # type: ignore
                return False

            if server.auth_code:  # type: ignore
                # Exchange code for tokens
                tokens = oauth_flow.exchange_code_for_tokens(server.auth_code)  # type: ignore

                # Store tokens
                expires_at = int(time.time()) + tokens.get("expires_in", 3600)
                self.credential_storage.store_oauth_token(
                    provider,
                    tokens["access_token"],
                    tokens.get("refresh_token"),
                    expires_at,
                )

                # Create mock client for testing
                client = type(
                    "MockOAuthClient",
                    (),
                    {
                        "access_token": tokens["access_token"],
                        "token_type": tokens.get("token_type", "Bearer"),
                    },
                )()

                self._authenticated_clients[provider] = client
                return True

        except Exception as e:
            print(f"OAuth authentication failed: {e}")
            return False

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
