"""OAuth client for Claude API authentication."""

import time
from typing import Any

import requests

from . import __version__


class OAuthClaudeClient:
    """OAuth-enabled Claude client that bypasses anthropic client authentication."""

    def __init__(self, access_token: str, refresh_token: str | None = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.base_url = "https://api.anthropic.com/v1"

    def _get_headers(self) -> dict[str, str]:
        """Get headers with OAuth authentication and LLM-Orchestra identification."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "User-Agent": f"LLM-Orchestra/Python {__version__}",
            "X-Stainless-Lang": "python",
            "X-Stainless-Package-Version": __version__,
        }

    def is_token_expired(self, expires_at: int | None = None) -> bool:
        """Check if the access token is expired or about to expire.

        Args:
            expires_at: Unix timestamp of token expiration (optional)

        Returns:
            True if token is expired or expires within 5 minutes
        """
        if expires_at is None:
            # If we don't have expiration info, assume it might be expired
            return False

        # Consider token expired if it expires within 5 minutes (300 seconds)
        return time.time() >= (expires_at - 300)

    def refresh_access_token(self, client_id: str) -> bool:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            raise ValueError("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": client_id,
        }

        try:
            response = requests.post(
                "https://console.anthropic.com/v1/oauth/token",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens["access_token"]
                if "refresh_token" in tokens:
                    self.refresh_token = tokens["refresh_token"]
                return True
            else:
                # Log detailed error information for debugging
                error_msg = (
                    f"Token refresh failed with status {response.status_code}: "
                    f"{response.text}"
                )
                print(f"🔄 Token refresh error: {error_msg}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"🔄 Token refresh network error: {str(e)}")
            return False
        except Exception as e:
            print(f"🔄 Token refresh unexpected error: {str(e)}")
            return False

    def revoke_token(self, client_id: str, token_type: str = "access_token") -> bool:
        """Revoke the access or refresh token.

        Args:
            client_id: OAuth client ID
            token_type: Type of token to revoke ("access_token" or "refresh_token")

        Returns:
            True if revocation successful, False otherwise
        """
        token = (
            self.access_token if token_type == "access_token" else self.refresh_token
        )

        if not token:
            return False

        data = {
            "token": token,
            "token_type_hint": token_type,
            "client_id": client_id,
        }

        try:
            response = requests.post(
                "https://console.anthropic.com/v1/oauth/revoke",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            # Token revocation typically returns 200 even for invalid tokens
            # per OAuth 2.0 RFC 7009 for security reasons
            return response.status_code == 200

        except Exception:
            return False

    def revoke_all_tokens(self, client_id: str) -> bool:
        """Revoke both access and refresh tokens.

        Args:
            client_id: OAuth client ID

        Returns:
            True if both revocations successful, False otherwise
        """
        access_revoked = self.revoke_token(client_id, "access_token")
        refresh_revoked = True  # Default to True if no refresh token

        if self.refresh_token:
            refresh_revoked = self.revoke_token(client_id, "refresh_token")

        return access_revoked and refresh_revoked

    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        system: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a message using the Claude API with OAuth authentication."""
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }

        if system:
            data["system"] = system

        response = requests.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=data,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()  # type: ignore[no-any-return]
        elif response.status_code == 401:
            raise Exception("Token expired - refresh needed")
        else:
            raise Exception(
                f"API request failed: {response.status_code} - {response.text}"
            )
