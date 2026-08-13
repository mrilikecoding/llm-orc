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

    @staticmethod
    def _attach_raw_counts(usage: dict[str, Any], response: dict[str, Any]) -> None:
        """C2 (#145): the RAW prompt_eval_count/eval_count survive alongside
        ``usage``'s input_tokens/output_tokens — those fall back to a text-
        length estimate when Ollama omits the real counts, so a truncation
        detector needs the un-conflated raw fields (present only when
        Ollama actually returned them, never synthesized)."""
        for key in ("prompt_eval_count", "eval_count"):
            value = response.get(key)
            if value is not None:
                usage[key] = value

    @staticmethod
    def _attach_timing_breakdown(
        usage: dict[str, Any], response: dict[str, Any], total_duration_ns: Any
    ) -> None:
        """Ollama's detailed timing breakdown, attached only when the
        response carried a total_duration (unchanged pre-existing
        behavior, extracted alongside ``_attach_raw_counts`` to keep
        ``generate_response`` under the complexity budget)."""
        if total_duration_ns is None:
            return
        for key in ("eval_duration", "prompt_eval_duration", "load_duration"):
            value = response.get(key)
            if value is not None:
                usage[f"{key}_ns"] = value

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Ollama API."""
        start_time = time.time()

        # Build options: generic options underlay, explicit fields overlay.
        # `think` is Ollama's native thinking toggle: a top-level chat field,
        # not a sampling option, so lift it out of options. (The qwen3 /no_think
        # prompt switch is not honored through the chat API; the native `think`
        # param is, and it is a large interactive-latency lever.)
        source_options = dict(self._options) if self._options else {}
        think = source_options.pop("think", None)

        options: dict[str, Any] = source_options
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
        if think is not None:
            chat_kwargs["think"] = think
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

        # Attach Ollama timing breakdown and raw usage counts when available
        if self._last_usage is not None:
            self._attach_timing_breakdown(self._last_usage, response, total_duration_ns)
            self._attach_raw_counts(self._last_usage, response)

        return str(content)
