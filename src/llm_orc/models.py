"""Multi-model support for LLM agents."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import ollama
from anthropic import AsyncAnthropic
from google import genai

from .oauth_client import OAuthClaudeClient


class HTTPConnectionPool:
    """Manages shared HTTP connections for better performance."""

    _instance: "HTTPConnectionPool | None" = None
    _httpx_client: Any = None  # httpx.AsyncClient
    _performance_config: dict[str, Any] | None = None

    def __new__(cls) -> "HTTPConnectionPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_httpx_client(cls) -> Any:
        """Get or create a shared httpx client with connection pooling."""
        # Ensure singleton instance exists
        if cls._instance is None:
            cls._instance = cls()

        if cls._httpx_client is None or cls._httpx_client.is_closed:
            import httpx

            # Load performance configuration
            if cls._performance_config is None:
                try:
                    from .config import ConfigurationManager

                    config_manager = ConfigurationManager()
                    cls._performance_config = config_manager.load_performance_config()
                except Exception:
                    # Fallback to defaults if configuration loading fails
                    cls._performance_config = {
                        "concurrency": {
                            "connection_pool": {
                                "max_connections": 100,
                                "max_keepalive": 20,
                                "keepalive_expiry": 30,
                            }
                        }
                    }

            # Get connection pool settings from configuration
            pool_config = cls._performance_config.get("concurrency", {}).get(
                "connection_pool", {}
            )

            # Configure connection pooling for better performance
            limits = httpx.Limits(
                max_connections=pool_config.get("max_connections", 100),
                max_keepalive_connections=pool_config.get("max_keepalive", 20),
                keepalive_expiry=pool_config.get("keepalive_expiry", 30.0),
            )

            timeout = httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,  # Read timeout
                write=10.0,  # Write timeout
                pool=5.0,  # Pool timeout
            )

            cls._httpx_client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                headers={
                    "User-Agent": "llm-orc/1.0",
                },
            )

        return cls._httpx_client

    @classmethod
    async def close(cls) -> None:
        """Close the shared HTTP client."""
        if cls._httpx_client is not None and not cls._httpx_client.is_closed:
            await cls._httpx_client.aclose()
            cls._httpx_client = None


class ModelInterface(ABC):
    """Abstract interface for LLM models."""

    def __init__(self) -> None:
        self._last_usage: dict[str, Any] | None = None
        self._conversation_history: list[dict[str, str]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass

    @abstractmethod
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate a response from the model."""
        pass

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the conversation history."""
        return self._conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()

    def add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self._conversation_history.append({"role": role, "content": content})

    def get_last_usage(self) -> dict[str, Any] | None:
        """Get usage metrics from the last API call."""
        return self._last_usage

    def _record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int,
        cost_usd: float = 0.0,
        model_name: str = "",
    ) -> None:
        """Record usage metrics for the last API call."""
        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "model": model_name or self.name,
        }


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
        import asyncio
        import subprocess

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


class GeminiModel(ModelInterface):
    """Gemini model implementation."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        super().__init__()
        self.api_key = api_key
        self.model_name = model
        self.client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return f"gemini-{self.model_name}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Gemini API."""
        start_time = time.time()
        prompt = f"{role_prompt}\n\nUser: {message}\nAssistant:"

        # Run in thread pool since Gemini doesn't have async support
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name, contents=prompt
            ),
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Estimate token usage for Gemini (rough approximation)
        estimated_input_tokens = len(prompt) // 4
        response_text = response.text or ""
        estimated_output_tokens = len(response_text) // 4

        # Estimate cost (simplified Gemini pricing)
        cost_per_input_token = 0.0000005  # $0.50 per million input tokens
        cost_per_output_token = 0.0000015  # $1.50 per million output tokens
        cost_usd = (estimated_input_tokens * cost_per_input_token) + (
            estimated_output_tokens * cost_per_output_token
        )

        self._record_usage(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            model_name=self.model_name,
        )

        return response_text


class OllamaModel(ModelInterface):
    """Ollama model implementation."""

    def __init__(
        self, model_name: str = "llama2", host: str = "http://localhost:11434"
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.host = host
        self.client = ollama.AsyncClient(host=host)

    @property
    def name(self) -> str:
        return f"ollama-{self.model_name}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Ollama API."""
        start_time = time.time()

        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": message},
            ],
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # For local models, estimate token usage (Ollama doesn't provide exact counts)
        # This is a rough approximation: ~4 characters per token
        content = response["message"]["content"]
        prompt_length = len(role_prompt) + len(message)

        estimated_input_tokens = prompt_length // 4
        estimated_output_tokens = len(content) // 4

        self._record_usage(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            duration_ms=duration_ms,
            cost_usd=0.0,  # Local models have no API cost
            model_name=self.model_name,
        )

        return str(content)


class ModelManager:
    """Manages model instances and selection."""

    def __init__(self) -> None:
        self.models: dict[str, ModelInterface] = {}

    def register_model(self, key: str, model: ModelInterface) -> None:
        """Register a model instance."""
        self.models[key] = model

    def register_oauth_claude_model(
        self,
        key: str,
        access_token: str,
        refresh_token: str | None = None,
        client_id: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        credential_storage: Any = None,
        provider_key: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        """Register an OAuth-authenticated Claude model."""
        oauth_model = OAuthClaudeModel(
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            model=model,
            credential_storage=credential_storage,
            provider_key=provider_key,
            expires_at=expires_at,
        )
        self.models[key] = oauth_model

    def register_claude_model(
        self, key: str, api_key: str, model: str = "claude-3-5-sonnet-20241022"
    ) -> None:
        """Register an API key-authenticated Claude model."""
        claude_model = ClaudeModel(api_key=api_key, model=model)
        self.models[key] = claude_model

    def register_claude_cli_model(
        self, key: str, claude_path: str, model: str = "claude-3-5-sonnet-20241022"
    ) -> None:
        """Register a Claude CLI model."""
        claude_cli_model = ClaudeCLIModel(claude_path=claude_path, model=model)
        self.models[key] = claude_cli_model

    def get_model(self, key: str) -> ModelInterface:
        """Retrieve a registered model."""
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found")
        return self.models[key]

    def list_models(self) -> dict[str, str]:
        """List all registered models."""
        return {key: model.name for key, model in self.models.items()}

    def get_oauth_models(self) -> dict[str, OAuthClaudeModel]:
        """Get all registered OAuth-authenticated models."""
        return {
            key: model
            for key, model in self.models.items()
            if isinstance(model, OAuthClaudeModel)
        }

    def get_api_key_models(self) -> dict[str, ClaudeModel]:
        """Get all registered API key-authenticated models."""
        return {
            key: model
            for key, model in self.models.items()
            if isinstance(model, ClaudeModel)
        }
