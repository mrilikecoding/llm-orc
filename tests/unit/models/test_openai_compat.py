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


class TestOpenAICompatibleModelToolCalling:
    """Test generate_with_tools behavior.

    OpenAI-compat is the first provider to implement tool calling under
    the extended ModelInterface — covers Ollama, OpenAI, OpenRouter,
    LM Studio, etc. The request/response shape is OpenAI's standard
    tool-calling format.
    """

    @pytest.fixture
    def model(self) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(model_name="gpt-4o", api_key="sk-test-key")

    def test_supports_tool_calling_is_true(self) -> None:
        assert OpenAICompatibleModel.supports_tool_calling is True

    def _tool_calling_response(
        self,
        *,
        content: str | None = "Hello!",
        tool_calls: list[dict[str, object]] | None = None,
        finish_reason: str = "stop",
        usage: dict[str, int] | None = None,
    ) -> MagicMock:
        mock = MagicMock()
        mock.status_code = 200
        message: dict[str, object] = {"content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        mock.json.return_value = {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage
            or {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
        }
        return mock

    @pytest.mark.asyncio
    async def test_generate_with_tools_parses_content_only_response(
        self, model: OpenAICompatibleModel
    ) -> None:
        """LLM emits text only — no tool_calls, finish_reason stop."""
        mock_client = AsyncMock()
        mock_client.post.return_value = self._tool_calling_response(
            content="I don't need any tools."
        )

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            result = await model.generate_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )

        assert result.content == "I don't need any tools."
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 20

    @pytest.mark.asyncio
    async def test_generate_with_tools_parses_tool_calls(
        self, model: OpenAICompatibleModel
    ) -> None:
        """LLM emits tool calls — content may be null, tool_calls populated."""
        mock_client = AsyncMock()
        mock_client.post.return_value = self._tool_calling_response(
            content=None,  # common when only tool calls emitted
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "list_ensembles",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "invoke_ensemble",
                        "arguments": '{"name":"analysis","input":"x"}',
                    },
                },
            ],
            finish_reason="tool_calls",
        )

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            result = await model.generate_with_tools(
                messages=[{"role": "user", "content": "check"}],
                tools=[{"type": "function", "function": {"name": "list_ensembles"}}],
            )

        assert result.content == ""  # null content normalized to empty string
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "list_ensembles"
        assert result.tool_calls[0].arguments_json == "{}"
        assert result.tool_calls[1].id == "call_456"
        assert result.tool_calls[1].name == "invoke_ensemble"
        assert result.tool_calls[1].arguments_json == '{"name":"analysis","input":"x"}'

    @pytest.mark.asyncio
    async def test_generate_with_tools_sends_messages_and_tools_in_body(
        self, model: OpenAICompatibleModel
    ) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = self._tool_calling_response()

        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_with_tools(messages=messages, tools=tools)

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
        body = call_args[1]["json"]
        assert body["model"] == "gpt-4o"
        assert body["messages"] == messages
        assert body["tools"] == tools
        # The tool-calling endpoint is non-streaming — streaming is a
        # Serving-Layer concern, the LLM adapter returns a single response.
        assert body.get("stream") is not True

    @pytest.mark.asyncio
    async def test_generate_with_tools_uses_bearer_token_when_provided(
        self, model: OpenAICompatibleModel
    ) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = self._tool_calling_response()

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_with_tools(messages=[], tools=[])

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test-key"

    @pytest.mark.asyncio
    async def test_generate_with_tools_omits_authorization_when_no_key(
        self,
    ) -> None:
        """Ollama and local endpoints typically accept no API key."""
        model = OpenAICompatibleModel(
            model_name="llama3.1", base_url="http://localhost:11434/v1"
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = self._tool_calling_response()

        with patch(
            "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
            return_value=mock_client,
        ):
            await model.generate_with_tools(messages=[], tools=[])

        headers = mock_client.post.call_args[1]["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_generate_with_tools_raises_on_non_200(
        self, model: OpenAICompatibleModel
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with (
            patch(
                "llm_orc.models.openai_compat.HTTPConnectionPool.get_httpx_client",
                return_value=mock_client,
            ),
            pytest.raises(
                RuntimeError,
                match="OpenAI-compatible tool-calling API error.*500",
            ),
        ):
            await model.generate_with_tools(messages=[], tools=[])
