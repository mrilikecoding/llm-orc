"""OAuth client for Claude API authentication."""

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
                return False

        except Exception:
            return False

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
