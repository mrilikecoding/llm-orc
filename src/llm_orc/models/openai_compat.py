"""OpenAI-compatible model implementation."""

import time
from typing import Any

from llm_orc.models.base import (
    HTTPConnectionPool,
    ModelInterface,
    ToolCall,
    ToolCallingResponse,
    ToolCallUsage,
)


class OpenAICompatibleModel(ModelInterface):
    """Model for any OpenAI-compatible API (vLLM, LM Studio, OpenRouter, etc.)."""

    supports_tool_calling: bool = True
    """Supports OpenAI's tool-calling format natively. Covers Ollama
    (``/v1/chat/completions`` endpoint with tool-calling models like
    ``llama3.1``, ``qwen2.5``), OpenAI proper, OpenRouter, LM Studio,
    vLLM, and any compatible provider."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @property
    def name(self) -> str:
        return f"openai-compat-{self.model_name}"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using an OpenAI-compatible chat completions API."""
        start_time = time.time()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": message},
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        client = HTTPConnectionPool.get_httpx_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI-compatible API error {response.status_code}: {response.text}"
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", self._estimate_tokens(message))
        output_tokens = usage.get("completion_tokens", self._estimate_tokens(content))

        duration_ms = int((time.time() - start_time) * 1000)

        self._record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=0.0,
            model_name=self.model_name,
        )

        return str(content)

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        """Call the OpenAI-compat ``/v1/chat/completions`` endpoint with tools.

        Non-streaming — the Serving Layer handles streaming separately
        on its SSE surface. The tool-calling endpoint returns a single
        response with ``message.content`` (may be null when only tool
        calls were emitted), ``message.tool_calls``, and ``finish_reason``.
        """
        start_time = time.time()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        client = HTTPConnectionPool.get_httpx_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI-compatible tool-calling API error "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        raw_content = message.get("content")
        content = str(raw_content) if raw_content is not None else ""

        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=str(tc["id"]),
                name=str(tc["function"]["name"]),
                arguments_json=str(tc["function"].get("arguments", "")),
            )
            for tc in raw_tool_calls
        ]

        finish_reason = choice.get("finish_reason", "stop")
        if finish_reason not in ("stop", "length", "tool_calls"):
            finish_reason = "stop"

        usage_data = data.get("usage", {})
        usage = ToolCallUsage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
            completion_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
        )

        duration_ms = int((time.time() - start_time) * 1000)
        self._record_usage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            duration_ms=duration_ms,
            cost_usd=0.0,
            model_name=self.model_name,
        )

        return ToolCallingResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )
