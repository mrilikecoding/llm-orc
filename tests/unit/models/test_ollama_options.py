"""Tests for OllamaModel options pass-through."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_orc.models.ollama import OllamaModel


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
