"""Ollama model implementation."""

import time

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
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model_name = model_name
        self.host = host
        self.client = ollama.AsyncClient(host=host)

    @property
    def name(self) -> str:
        return f"ollama-{self.model_name}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Ollama API."""
        start_time = time.time()

        # Build options dict for temperature and max_tokens
        options: dict[str, float | int] = {}
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
