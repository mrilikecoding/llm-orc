"""Multi-model support for LLM agents."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import google.generativeai as genai
import ollama
from anthropic import AsyncAnthropic


class ModelInterface(ABC):
    """Abstract interface for LLM models."""

    def __init__(self) -> None:
        self._last_usage: dict[str, Any] | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass

    @abstractmethod
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate a response from the model."""
        pass

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
        self.client = AsyncAnthropic(api_key=api_key)

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


class GeminiModel(ModelInterface):
    """Gemini model implementation."""

    def __init__(self, api_key: str, model: str = "gemini-pro") -> None:
        super().__init__()
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        self.client = genai.GenerativeModel(model)  # type: ignore[attr-defined]

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
            None, lambda: self.client.generate_content(prompt)
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Estimate token usage for Gemini (rough approximation)
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = len(response.text) // 4

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

        return str(response.text)


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

    def get_model(self, key: str) -> ModelInterface:
        """Retrieve a registered model."""
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found")
        return self.models[key]

    def list_models(self) -> dict[str, str]:
        """List all registered models."""
        return {key: model.name for key, model in self.models.items()}
