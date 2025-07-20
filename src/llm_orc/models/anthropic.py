"""Anthropic Claude model implementations."""

import asyncio
import subprocess
import time
from typing import Any

from anthropic import AsyncAnthropic

from llm_orc.models.base import HTTPConnectionPool, ModelInterface
from llm_orc.core.auth.oauth_client import OAuthClaudeClient


class ClaudeModel(ModelInterface):
    """Claude model implementation."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model

        # Use shared HTTP client with connection pooling
        shared_client = HTTPConnectionPool.get_httpx_client()
        self.client = AsyncAnthropic(api_key=api_key, http_client=shared_client)

    @property
    def name(self) -> str:
        return f"claude-{self.model}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Claude API."""
        start_time = time.time()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=role_prompt,
            messages=[{"role": "user", "content": message}],
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Record usage metrics
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Estimate cost (simplified pricing for Claude)
        cost_per_input_token = 0.000003  # $3 per million input tokens
        cost_per_output_token = 0.000015  # $15 per million output tokens
        cost_usd = (input_tokens * cost_per_input_token) + (
            output_tokens * cost_per_output_token
        )

        self._record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            model_name=self.model,
        )

        # Handle different content block types
        content_block = response.content[0]
        if hasattr(content_block, "text"):
            return content_block.text
        else:
            return str(content_block)


class OAuthClaudeModel(ModelInterface):
    """OAuth-enabled Claude model implementation."""

    def __init__(
        self,
        access_token: str,
        refresh_token: str | None = None,
        client_id: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        credential_storage: Any = None,
        provider_key: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        super().__init__()
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.model = model
        self.expires_at = expires_at
        self.client = OAuthClaudeClient(access_token, refresh_token)
        self._credential_storage = credential_storage
        self._provider_key = provider_key
        self._role_established = False
        self._current_role: str | None = None

    @property
    def name(self) -> str:
        return f"oauth-claude-{self.model}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using OAuth-authenticated Claude API."""
        start_time = time.time()

        # Proactive token refresh if token is about to expire
        if (
            self.expires_at
            and self.refresh_token
            and self.client_id
            and self.client.is_token_expired(self.expires_at)
        ):
            print("🔄 Proactively refreshing token (expires soon)")
            if self.client.refresh_access_token(self.client_id):
                # Update stored credentials and local references
                if self._credential_storage and self._provider_key:
                    new_expires_at = int(time.time()) + 3600  # Default 1 hour expiry
                    self._credential_storage.store_oauth_token(
                        self._provider_key,
                        self.client.access_token,
                        self.client.refresh_token,
                        expires_at=new_expires_at,
                        client_id=self.client_id,
                    )
                    self.expires_at = new_expires_at

                self.access_token = self.client.access_token
                self.refresh_token = self.client.refresh_token
                print("✅ Token refreshed proactively")
            else:
                print("❌ Proactive token refresh failed")

        try:
            # OAuth tokens require specific Claude Code system prompt for authorization
            oauth_system_prompt = (
                "You are Claude Code, Anthropic's official CLI for Claude."
            )

            # Handle role injection if needed
            self._inject_role_if_needed(role_prompt)

            # Add current user message to conversation history
            self.add_to_conversation("user", message)

            # Prepare messages with conversation history
            messages = self.get_conversation_history()

            # Run in thread pool since our OAuth client is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.create_message(
                    model=self.model,
                    max_tokens=1000,
                    system=oauth_system_prompt,
                    messages=messages,
                ),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Record usage metrics
            usage = response.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # OAuth Claude uses existing subscription (no additional API costs)
            cost_usd = 0.0

            self._record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
                model_name=self.model,
            )

            # Extract response content
            content = response.get("content", [])
            if content and len(content) > 0:
                assistant_response = str(content[0].get("text", ""))
                # Add assistant response to conversation history
                self.add_to_conversation("assistant", assistant_response)
                return assistant_response
            else:
                # Add empty response to conversation history
                self.add_to_conversation("assistant", "")
                return ""

        except Exception as e:
            if "Token expired" in str(e) and self.refresh_token and self.client_id:
                # Attempt token refresh
                if self.client.refresh_access_token(self.client_id):
                    # Update stored credentials if credential storage is available
                    if self._credential_storage and self._provider_key:
                        expires_at = int(time.time()) + 3600  # Default 1 hour expiry
                        self._credential_storage.store_oauth_token(
                            self._provider_key,
                            self.client.access_token,
                            self.client.refresh_token,
                            expires_at=expires_at,
                            client_id=self.client_id,
                        )

                    # Update local token references
                    self.access_token = self.client.access_token
                    self.refresh_token = self.client.refresh_token

                    # Retry request - remove last user message since we'll add it again
                    if (
                        self._conversation_history
                        and self._conversation_history[-1]["role"] == "user"
                    ):
                        self._conversation_history.pop()
                    return await self.generate_response(message, role_prompt)
            raise e

    def _inject_role_if_needed(self, role_prompt: str) -> None:
        """Inject role establishment into conversation if needed."""
        oauth_system_prompt = (
            "You are Claude Code, Anthropic's official CLI for Claude."
        )

        # Don't inject role if it's the OAuth system prompt itself
        if role_prompt == oauth_system_prompt:
            return

        # Don't inject role if already established with the same role
        if self._role_established and self._current_role == role_prompt:
            return

        # Inject role establishment
        role_message = f"For this conversation, please act as: {role_prompt}"
        self.add_to_conversation("user", role_message)

        # Add assistant acknowledgment
        acknowledgment = "Understood. I'll act in that role for our conversation."
        self.add_to_conversation("assistant", acknowledgment)

        # Mark role as established
        self._role_established = True
        self._current_role = role_prompt


class ClaudeCLIModel(ModelInterface):
    """Claude CLI model implementation that uses local claude command."""

    def __init__(
        self,
        claude_path: str,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        super().__init__()
        self.claude_path = claude_path
        self.model = model

    @property
    def name(self) -> str:
        return f"claude-cli-{self.model}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using local Claude CLI."""
        start_time = time.time()

        # Prepare input for Claude CLI
        cli_input = f"{role_prompt}\n\nUser: {message}\nAssistant:"

        try:
            # Run claude command with --no-api-key flag to use local auth
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [self.claude_path, "--no-api-key"],
                    input=cli_input,
                    text=True,
                    capture_output=True,
                    timeout=30,
                ),
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                raise Exception(
                    f"Claude CLI error (code {result.returncode}): {error_msg}"
                )

            duration_ms = int((time.time() - start_time) * 1000)

            # Estimate token usage (rough approximation)
            estimated_input_tokens = len(cli_input) // 4
            estimated_output_tokens = len(result.stdout) // 4

            self._record_usage(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                duration_ms=duration_ms,
                cost_usd=0.0,  # Claude CLI uses user's existing auth
                model_name=self.model,
            )

            return result.stdout.strip()

        except subprocess.TimeoutExpired as e:
            raise Exception(
                "Claude CLI timeout - command took too long to respond"
            ) from e
        except Exception as e:
            if "Claude CLI error" in str(e):
                raise e
            raise Exception(f"Claude CLI error: {str(e)}") from e
