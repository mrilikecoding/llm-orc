"""Tests for OllamaModel options, format, and usage metrics."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_orc.models.ollama import OllamaModel


def _ollama_response(
    content: str = "test response",
    *,
    prompt_eval_count: int = 42,
    eval_count: int = 18,
    total_duration: int = 1_500_000_000,
    prompt_eval_duration: int = 400_000_000,
    eval_duration: int = 900_000_000,
    load_duration: int = 200_000_000,
) -> dict[str, Any]:
    """Build a realistic Ollama chat response."""
    return {
        "message": {"content": content},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "total_duration": total_duration,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_duration": eval_duration,
        "load_duration": load_duration,
    }


class TestOllamaUsageMetrics:
    """Scenario: real token counts and timing from Ollama responses."""

    @pytest.mark.asyncio
    async def test_real_token_counts_used(self) -> None:
        """prompt_eval_count and eval_count used instead of estimates."""
        model = OllamaModel(model_name="qwen3:14b")
        model.client = AsyncMock()
        model.client.chat = AsyncMock(
            return_value=_ollama_response(prompt_eval_count=100, eval_count=50)
        )

        await model.generate_response("hello", "system prompt")

        usage = model.get_last_usage()
        assert usage is not None
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_real_duration_from_total_duration(self) -> None:
        """duration_ms derived from Ollama total_duration (nanoseconds)."""
        model = OllamaModel(model_name="qwen3:14b")
        model.client = AsyncMock()
        model.client.chat = AsyncMock(
            return_value=_ollama_response(
                total_duration=2_500_000_000  # 2500ms
            )
        )

        await model.generate_response("hello", "system prompt")

        usage = model.get_last_usage()
        assert usage is not None
        assert usage["duration_ms"] == 2500

    @pytest.mark.asyncio
    async def test_timing_breakdown_in_usage(self) -> None:
        """Detailed Ollama timing fields included in usage dict."""
        model = OllamaModel(model_name="qwen3:14b")
        model.client = AsyncMock()
        model.client.chat = AsyncMock(
            return_value=_ollama_response(
                eval_duration=900_000_000,
                prompt_eval_duration=400_000_000,
                load_duration=200_000_000,
            )
        )

        await model.generate_response("hello", "system prompt")

        usage = model.get_last_usage()
        assert usage is not None
        assert usage["eval_duration_ns"] == 900_000_000
        assert usage["prompt_eval_duration_ns"] == 400_000_000
        assert usage["load_duration_ns"] == 200_000_000

    @pytest.mark.asyncio
    async def test_fallback_to_estimates_when_fields_missing(self) -> None:
        """Falls back to estimation when Ollama fields are absent."""
        model = OllamaModel(model_name="qwen3:14b")
        model.client = AsyncMock()
        # Minimal response without Ollama metrics
        model.client.chat = AsyncMock(return_value={"message": {"content": "test"}})

        await model.generate_response("hello world", "be helpful")

        usage = model.get_last_usage()
        assert usage is not None
        # Falls back to len//4 estimation
        assert usage["input_tokens"] == len("be helpfulhello world") // 4
        assert usage["output_tokens"] == len("test") // 4
        # No timing breakdown keys when absent
        assert "eval_duration_ns" not in usage


class TestOllamaOptionsPassThrough:
    """Scenario: provider options forwarded to Ollama API."""

    def test_init_stores_options(self) -> None:
        model = OllamaModel(
            model_name="qwen3:8b",
            options={"num_ctx": 8192, "top_k": 20},
        )
        assert model._options == {"num_ctx": 8192, "top_k": 20}

    def test_init_default_options_is_none(self) -> None:
        model = OllamaModel(model_name="qwen3:8b")
        assert model._options is None

    @pytest.mark.asyncio
    async def test_options_merged_into_api_call(self) -> None:
        model = OllamaModel(
            model_name="qwen3:8b",
            temperature=0.6,
            max_tokens=2000,
            options={"num_ctx": 8192, "top_k": 20},
        )

        mock_response: dict[str, Any] = {
            "message": {"content": "test response"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert options["num_ctx"] == 8192
        assert options["top_k"] == 20
        assert options["temperature"] == 0.6
        assert options["num_predict"] == 2000

    @pytest.mark.asyncio
    async def test_explicit_temperature_wins_over_options(self) -> None:
        """Explicit temperature field overlays options temperature."""
        model = OllamaModel(
            model_name="qwen3:8b",
            temperature=0.3,
            options={"temperature": 0.9, "num_ctx": 4096},
        )

        mock_response: dict[str, Any] = {
            "message": {"content": "test"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert options["temperature"] == 0.3
        assert options["num_ctx"] == 4096

    @pytest.mark.asyncio
    async def test_format_schema_passed_to_chat(self) -> None:
        """format dict passed to ollama client.chat()."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        model = OllamaModel(model_name="qwen3:14b", ollama_format=schema)

        mock_response: dict[str, Any] = {
            "message": {"content": '{"name": "test"}'},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        assert call_kwargs.kwargs.get("format") == schema

    @pytest.mark.asyncio
    async def test_format_string_passed_to_chat(self) -> None:
        """format string 'json' passed to ollama client.chat()."""
        model = OllamaModel(model_name="qwen3:14b", ollama_format="json")

        mock_response: dict[str, Any] = {
            "message": {"content": '{"ok": true}'},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        assert call_kwargs.kwargs.get("format") == "json"

    @pytest.mark.asyncio
    async def test_no_format_omits_kwarg(self) -> None:
        """No format param means no format kwarg to chat()."""
        model = OllamaModel(model_name="qwen3:14b")

        mock_response: dict[str, Any] = {
            "message": {"content": "test"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        assert "format" not in call_kwargs.kwargs

    def test_init_stores_format(self) -> None:
        model = OllamaModel(model_name="qwen3:14b", ollama_format={"type": "object"})
        assert model._format == {"type": "object"}

    def test_init_default_format_is_none(self) -> None:
        model = OllamaModel(model_name="qwen3:14b")
        assert model._format is None

    @pytest.mark.asyncio
    async def test_no_options_works_unchanged(self) -> None:
        """Backward compat: no options behaves exactly as before."""
        model = OllamaModel(model_name="qwen3:8b")

        mock_response: dict[str, Any] = {
            "message": {"content": "test"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert options is None
