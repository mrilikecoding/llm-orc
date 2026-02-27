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
        ollama_format: str | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model_name = model_name
        self.host = host
        self.client = ollama.AsyncClient(host=host)
        self._options = options
        self._format = ollama_format

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

        chat_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": message},
            ],
            "options": options if options else None,
        }
        if self._format is not None:
            chat_kwargs["format"] = self._format

        response = await self.client.chat(**chat_kwargs)

        content = response["message"]["content"]

        # Use real Ollama metrics when available, fall back to estimates
        input_tokens = response.get(
            "prompt_eval_count",
            self._estimate_tokens(role_prompt + message),
        )
        output_tokens = response.get(
            "eval_count",
            self._estimate_tokens(content),
        )

        total_duration_ns = response.get("total_duration")
        if total_duration_ns is not None:
            duration_ms = int(total_duration_ns / 1_000_000)
        else:
            duration_ms = int((time.time() - start_time) * 1000)

        self._record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=0.0,
            model_name=self.model_name,
        )

        # Attach Ollama timing breakdown when available
        if self._last_usage is not None and total_duration_ns is not None:
            for key in ("eval_duration", "prompt_eval_duration", "load_duration"):
                value = response.get(key)
                if value is not None:
                    self._last_usage[f"{key}_ns"] = value

        return str(content)
