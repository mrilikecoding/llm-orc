"""Anthropic Claude model implementations."""

import asyncio
import logging
import subprocess
import time
from typing import Any

from anthropic import AsyncAnthropic

from llm_orc.models.base import HTTPConnectionPool, ModelInterface

logger = logging.getLogger(__name__)


class ClaudeModel(ModelInterface):
    """Claude model implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
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

        # Build API call with generation parameters
        effective_max_tokens = self.max_tokens if self.max_tokens is not None else 1000
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "system": role_prompt,
            "messages": [{"role": "user", "content": message}],
        }
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature

        response = await self.client.messages.create(**create_kwargs)

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
            return str(content_block.text)
        else:
            return str(content_block)


class ClaudeCLIModel(ModelInterface):
    """Claude CLI model implementation that uses local claude command."""

    def __init__(
        self,
        claude_path: str,
        model: str = "claude-3-5-sonnet-20241022",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
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
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(  # nosec B603
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

            estimated_input_tokens = self._estimate_tokens(cli_input)
            estimated_output_tokens = self._estimate_tokens(result.stdout)

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
                raise
            raise Exception(f"Claude CLI error: {str(e)}") from e
