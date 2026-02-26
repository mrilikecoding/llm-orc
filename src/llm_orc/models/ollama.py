"""Ollama model implementation."""

import time
from typing import Any

import ollama

from llm_orc.models.base import ModelInterface


class OllamaModel(ModelInterface):
    """Ollama model implementation."""

    def __init__(
        self,
        model_name: str = "llama2",
        host: str = "http://localhost:11434",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model_name = model_name
        self.host = host
        self.client = ollama.AsyncClient(host=host)
        self._options = options

    @property
    def name(self) -> str:
        return f"ollama-{self.model_name}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Ollama API."""
        start_time = time.time()

        # Build options: generic options underlay, explicit fields overlay
        options: dict[str, float | int] = {}
        if self._options:
            options.update(self._options)
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": message},
            ],
            options=options if options else None,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        content = response["message"]["content"]

        estimated_input_tokens = self._estimate_tokens(role_prompt + message)
        estimated_output_tokens = self._estimate_tokens(content)

        self._record_usage(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            duration_ms=duration_ms,
            cost_usd=0.0,  # Local models have no API cost
            model_name=self.model_name,
        )

        return str(content)
