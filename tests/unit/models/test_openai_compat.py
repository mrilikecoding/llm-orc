"""Tests for OpenAI-compatible model implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orc.models.openai_compat import OpenAICompatibleModel


class TestOpenAICompatibleModelInit:
    """Test OpenAICompatibleModel initialization."""

    def test_defaults(self) -> None:
        """Should initialize with sensible defaults."""
        model = OpenAICompatibleModel(model_name="gpt-4o")

        assert model.model_name == "gpt-4o"
        assert model.base_url == "https://api.openai.com/v1"
        assert model.api_key is None
        assert model.temperature is None
        assert model.max_tokens is None

    def test_custom_base_url(self) -> None:
        """Should accept a custom base_url."""
        model = OpenAICompatibleModel(
            model_name="deepseek-coder-v2",
            base_url="http://localhost:8000/v1",
        )

        assert model.base_url == "http://localhost:8000/v1"

    def test_with_api_key(self) -> None:
        """Should accept an API key."""
        model = OpenAICompatibleModel(
            model_name="gpt-4o",
            api_key="sk-test-key",
        )

        assert model.api_key == "sk-test-key"

    def test_accepts_temperature_and_max_tokens(self) -> None:
        """Should accept temperature and max_tokens kwargs."""
        model = OpenAICompatibleModel(
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=2000,
        )

        assert model.temperature == 0.7
        assert model.max_tokens == 2000

    def test_name_property(self) -> None:
        """Should return a descriptive name."""
        model = OpenAICompatibleModel(model_name="gpt-4o")
        assert model.name == "openai-compat-gpt-4o"


class TestOpenAICompatibleModelGenerate:
    """Test generate_response behavior."""

    @pytest.fixture
    def model(self) -> OpenAICompatibleModel:
        """Create a model with API key for testing."""
        return OpenAICompatibleModel(
            model_name="gpt-4o",
            api_key="sk-test-key",
        )

    @pytest.fixture
    def model_no_key(self) -> OpenAICompatibleModel:
        """Create a model without API key (local endpoint)."""
        return OpenAICompatibleModel(
            model_name="deepseek-coder-v2",
            base_url="http://localhost:8000/v1",
        )

    @pytest.fixture
    def mock_success_response(self) -> MagicMock:
        """Create a mock successful HTTP response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from OpenAI!",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 5,
                "total_tokens": 20,
            },
        }
        return response

    @pytest.mark.asyncio
    async def test_generates_response(
        self, model: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should generate response and return content."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            result = await model.generate_response(
                "Hello", role_prompt="You are helpful."
            )

        assert result == "Hello from OpenAI!"

    @pytest.mark.asyncio
    async def test_request_url(
        self, model: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should POST to {base_url}/chat/completions."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_response("Hello", role_prompt="You are helpful.")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_request_url_custom_base(
        self, model_no_key: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should use custom base_url for the endpoint."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model_no_key.generate_response(
                "Hello", role_prompt="You are helpful."
            )

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:8000/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_authorization_header_with_key(
        self, model: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should include Authorization header when API key is provided."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_response("Hello", role_prompt="You are helpful.")

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

    @pytest.mark.asyncio
    async def test_no_authorization_header_without_key(
        self, model_no_key: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should not include Authorization header when no API key."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model_no_key.generate_response(
                "Hello", role_prompt="You are helpful."
            )

        call_kwargs = mock_client.post.call_args[1]
        assert "Authorization" not in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_request_body_structure(
        self, model: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should send correct request body with model and messages."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_response("Hello", role_prompt="You are helpful.")

        call_kwargs = mock_client.post.call_args[1]
        body = call_kwargs["json"]
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_pass_through(
        self, mock_success_response: MagicMock
    ) -> None:
        """Should include temperature and max_tokens in request body."""
        model = OpenAICompatibleModel(
            model_name="gpt-4o",
            api_key="sk-test",
            temperature=0.3,
            max_tokens=500,
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = mock_client.post.call_args[1]
        body = call_kwargs["json"]
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_omits_temperature_and_max_tokens_when_none(
        self, model: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should not include temperature/max_tokens when not set."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = mock_client.post.call_args[1]
        body = call_kwargs["json"]
        assert "temperature" not in body
        assert "max_tokens" not in body

    @pytest.mark.asyncio
    async def test_records_real_token_usage(
        self, model: OpenAICompatibleModel, mock_success_response: MagicMock
    ) -> None:
        """Should record real token usage from response, not estimates."""
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_success_response

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_response("Hello", role_prompt="You are helpful.")

        usage = model.get_last_usage()
        assert usage is not None
        assert usage["input_tokens"] == 15
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 20
        assert usage["cost_usd"] == 0.0
        assert usage["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_error_on_non_200_response(
        self, model: OpenAICompatibleModel
    ) -> None:
        """Should raise on non-200 HTTP status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with (
            patch(
                "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
                return_value=mock_client,
            ),
            pytest.raises(
                RuntimeError,
                match="OpenAI-compatible API error.*500",
            ),
        ):
            await model.generate_response("Hello", role_prompt="You are helpful.")
