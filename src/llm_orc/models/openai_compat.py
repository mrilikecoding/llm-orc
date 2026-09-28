"""OpenAI-compatible model implementation."""

import time
from typing import Any
from urllib.parse import urlparse

from llm_orc.models.base import (
    HTTPConnectionPool,
    ModelInterface,
    ToolCall,
    ToolCallingNotSupportedError,
    ToolCallingResponse,
    ToolCallUsage,
)

_OPENCODE_GO_HOST = "opencode.ai"
_OPENCODE_GO_PATH_PREFIX = "/zen/go"


def _is_opencode_go(base_url: str) -> bool:
    """Whether ``base_url`` is an OpenCode Go endpoint.

    Go asks every client for a stable ``x-opencode-session`` header per
    conversation (docs: opencode.ai/docs/go/; measured 2026-09-22:
    omitting it 400s with ``MissingSessionID``). Detected by host + path
    rather than provider string, so a profile pointing straight at the
    endpoint under any provider label is still caught. The pay-as-you-go
    sibling (``opencode.ai/zen/v1``, no ``/go``) is a different product
    surface and is out of scope.
    """
    parsed = urlparse(base_url)
    return parsed.hostname == _OPENCODE_GO_HOST and parsed.path.startswith(
        _OPENCODE_GO_PATH_PREFIX
    )


class OpenAICompatibleModel(ModelInterface):
    """Model for any OpenAI-compatible API (vLLM, LM Studio, OpenRouter, etc.)."""

    supports_tool_calling: bool = True
    """Supports OpenAI's tool-calling format natively. Covers llama-server
    (``/v1/chat/completions`` with ``--jinja`` tool parsing), OpenAI
    proper, OpenRouter, LM Studio, vLLM, and any compatible provider."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._options = options
        self._response_format = response_format
        self._execution_id = execution_id

    def _apply_options(self, body: dict[str, Any]) -> None:
        """Fold provider options into the request body.

        ``think`` is a chat-template switch, not a sampling option:
        llama-server reads it as ``chat_template_kwargs.enable_thinking``
        (#90 spike 2026-09-16: 1.2 s / 28 tokens off vs 76 s / 1500 on).
        """
        options = dict(self._options) if self._options else {}
        think = options.pop("think", None)
        if think is not None:
            body["chat_template_kwargs"] = {"enable_thinking": think}
        # Remaining keys are sampling params; explicit fields already in
        # the body (temperature, max_tokens) keep precedence.
        for key, value in options.items():
            body.setdefault(key, value)
        if isinstance(self._response_format, dict):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": self._response_format},
            }
        elif self._response_format == "json":
            body["response_format"] = {"type": "json_object"}

    def _attach_timings(self, data: dict[str, Any]) -> None:
        """Surface llama-server's ``timings`` on the usage record under the
        raw-count keys the serve's truncation backstop reads (turn_trace,
        C2 #145 / #151): ``prompt_n + cache_n`` is the full prompt as
        processed, ``predicted_n`` the generation. Present only when the
        server returned them, never synthesized."""
        timings = data.get("timings")
        if not isinstance(timings, dict) or self._last_usage is None:
            return
        prompt_n = timings.get("prompt_n")
        cache_n = timings.get("cache_n", 0)
        if isinstance(prompt_n, int):
            self._last_usage["prompt_eval_count"] = prompt_n + int(cache_n or 0)
        predicted_n = timings.get("predicted_n")
        if isinstance(predicted_n, int):
            self._last_usage["eval_count"] = predicted_n
        for src, dst in (
            ("prompt_ms", "prompt_eval_duration_ns"),
            ("predicted_ms", "eval_duration_ns"),
        ):
            value = timings.get(src)
            if isinstance(value, int | float):
                self._last_usage[dst] = int(value * 1_000_000)

    @property
    def name(self) -> str:
        return f"openai-compat-{self.model_name}"

    def _build_headers(self) -> dict[str, str]:
        """Request headers common to every OpenAI-compat call."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._execution_id and _is_opencode_go(self.base_url):
            headers["x-opencode-session"] = self._execution_id
        return headers

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using an OpenAI-compatible chat completions API."""
        start_time = time.time()

        headers = self._build_headers()

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
        self._apply_options(body)

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
        self._attach_timings(data)

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

        headers = self._build_headers()

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
        self._apply_options(body)

        client = HTTPConnectionPool.get_httpx_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        )

        if response.status_code != 200:
            # Per-model tool-calling unsupported: distinguish from generic
            # transport failure so downstream code can treat this as a
            # configuration error rather than a transient/retryable fault.
            # Empirical signal (research log 2026-04-28, S0-CAP-8): Ollama
            # returns 400 with "does not support tools" when the configured
            # model has no tool-calling capability metadata; the
            # ``supports_tool_calling`` flag on this class is class-level
            # and does not catch that per-model variation.
            if (
                response.status_code == 400
                and "does not support tools" in response.text
            ):
                raise ToolCallingNotSupportedError(
                    f"Model '{self.name}' does not support tool calling on "
                    f"this provider. Provider returned: {response.text}"
                )
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
        self._attach_timings(data)

        return ToolCallingResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )
