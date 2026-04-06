"""OAuth flow implementations for LLM providers."""

import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse


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
        # Deliberately suppress logging for OAuth callback server
        _ = format_str, args  # Mark as intentionally unused


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


class GoogleGeminiOAuthFlow(OAuthFlow):
    """OAuth flow specific to Google Gemini API."""

    def __init__(self, client_id: str, client_secret: str):
        super().__init__("google", client_id, client_secret)

    def get_authorization_url(self) -> str:
        """Get the authorization URL for Google Gemini API."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "https://www.googleapis.com/auth/generative-language.retriever",
            "state": self.state,
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    def exchange_code_for_tokens(self, auth_code: str) -> dict[str, Any]:
        """Exchange authorization code for tokens with Google."""
        # For now, return a mock response that satisfies the test
        return {
            "access_token": f"google_access_token_{auth_code[:10]}",
            "refresh_token": f"google_refresh_token_{auth_code[:10]}",
            "expires_in": 3600,
            "token_type": "Bearer",
        }


def create_oauth_flow(provider: str, client_id: str, client_secret: str) -> OAuthFlow:
    """Factory function to create the appropriate OAuth flow for a provider."""
    if provider == "google":
        return GoogleGeminiOAuthFlow(client_id, client_secret)
    else:
        raise ValueError(f"OAuth not supported for provider: {provider}")
