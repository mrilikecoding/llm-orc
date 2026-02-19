"""Test suite for multi-model support."""

from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.models.anthropic import ClaudeCLIModel, ClaudeModel, OAuthClaudeModel
from llm_orc.models.base import ModelInterface
from llm_orc.models.google import GeminiModel
from llm_orc.models.ollama import OllamaModel


class TestModelInterface:
    """Test the abstract model interface."""

    def test_model_interface_is_abstract(self) -> None:
        """Should not be able to instantiate ModelInterface directly."""
        with pytest.raises(TypeError):
            ModelInterface()  # type: ignore[abstract]


class TestModelParameterDefaults:
    """Test that model implementations inherit temperature/max_tokens defaults."""

    def test_ollama_model_defaults_to_none(self) -> None:
        """OllamaModel should have None temperature and max_tokens by default."""
        model = OllamaModel(model_name="llama2")
        assert model.temperature is None
        assert model.max_tokens is None

    def test_ollama_model_accepts_temperature(self) -> None:
        """OllamaModel should accept temperature kwarg."""
        model = OllamaModel(model_name="llama2", temperature=0.7)
        assert model.temperature == 0.7

    def test_ollama_model_accepts_max_tokens(self) -> None:
        """OllamaModel should accept max_tokens kwarg."""
        model = OllamaModel(model_name="llama2", max_tokens=500)
        assert model.max_tokens == 500

    def test_claude_model_defaults_to_none(self) -> None:
        """ClaudeModel should have None temperature and max_tokens by default."""
        model = ClaudeModel(api_key="test-key")
        assert model.temperature is None
        assert model.max_tokens is None

    def test_claude_model_accepts_params(self) -> None:
        """ClaudeModel should accept temperature and max_tokens kwargs."""
        model = ClaudeModel(api_key="test-key", temperature=0.5, max_tokens=2000)
        assert model.temperature == 0.5
        assert model.max_tokens == 2000

    def test_gemini_model_defaults_to_none(self) -> None:
        """GeminiModel should have None temperature and max_tokens by default."""
        model = GeminiModel(api_key="test-key")
        assert model.temperature is None
        assert model.max_tokens is None

    def test_gemini_model_accepts_params(self) -> None:
        """GeminiModel should accept temperature and max_tokens kwargs."""
        model = GeminiModel(api_key="test-key", temperature=0.3, max_tokens=1000)
        assert model.temperature == 0.3
        assert model.max_tokens == 1000

    def test_claude_cli_model_defaults_to_none(self) -> None:
        """ClaudeCLIModel should have None temperature and max_tokens."""
        model = ClaudeCLIModel(claude_path="/usr/bin/claude")
        assert model.temperature is None
        assert model.max_tokens is None

    def test_oauth_claude_model_defaults_to_none(self) -> None:
        """OAuthClaudeModel should have None temperature and max_tokens."""
        model = OAuthClaudeModel(access_token="test-token")
        assert model.temperature is None
        assert model.max_tokens is None

    def test_oauth_claude_model_accepts_params(self) -> None:
        """OAuthClaudeModel should accept temperature and max_tokens."""
        model = OAuthClaudeModel(
            access_token="test-token",
            temperature=0.8,
            max_tokens=1500,
        )
        assert model.temperature == 0.8
        assert model.max_tokens == 1500


class TestClaudeModel:
    """Test Claude model implementation."""

    @pytest.mark.asyncio
    async def test_claude_model_generate_response(self) -> None:
        """Should generate response using Claude API."""
        model = ClaudeModel(api_key="test-key")

        # Mock the anthropic client
        model.client = AsyncMock()
        model.client.messages.create.return_value = Mock(
            content=[Mock(text="Hello from Claude!")],
            usage=Mock(input_tokens=10, output_tokens=5),
        )

        response = await model.generate_response(
            "Hello", role_prompt="You are helpful."
        )

        assert response == "Hello from Claude!"
        model.client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_claude_model_passes_temperature(self) -> None:
        """Should pass temperature to Anthropic API when set."""
        model = ClaudeModel(api_key="test-key", temperature=0.7, max_tokens=2000)
        model.client = AsyncMock()
        model.client.messages.create.return_value = Mock(
            content=[Mock(text="response")],
            usage=Mock(input_tokens=10, output_tokens=5),
        )

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2000

    @pytest.mark.asyncio
    async def test_claude_model_defaults_max_tokens_to_1000(self) -> None:
        """Should default max_tokens to 1000 when not set."""
        model = ClaudeModel(api_key="test-key")
        model.client = AsyncMock()
        model.client.messages.create.return_value = Mock(
            content=[Mock(text="response")],
            usage=Mock(input_tokens=10, output_tokens=5),
        )

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1000
        assert "temperature" not in call_kwargs


class TestGeminiModel:
    """Test Gemini model implementation."""

    @pytest.mark.asyncio
    async def test_gemini_model_generate_response(self) -> None:
        """Should generate response using Gemini API."""
        model = GeminiModel(api_key="test-key")

        # Mock the genai client with proper async handling
        mock_response = Mock()
        mock_response.text = "Hello from Gemini!"
        model.client = Mock()
        model.client.models = Mock()
        model.client.models.generate_content = Mock(return_value=mock_response)

        response = await model.generate_response(
            "Hello", role_prompt="You are helpful."
        )

        assert response == "Hello from Gemini!"
        model.client.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_gemini_passes_config_with_params(self) -> None:
        """Should pass GenerateContentConfig when params are set."""
        model = GeminiModel(api_key="test-key", temperature=0.3, max_tokens=500)
        mock_response = Mock()
        mock_response.text = "response"
        model.client = Mock()
        model.client.models.generate_content = Mock(return_value=mock_response)

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config is not None
        assert config.temperature == 0.3
        assert config.max_output_tokens == 500

    @pytest.mark.asyncio
    async def test_gemini_no_config_when_params_none(self) -> None:
        """Should pass config=None when no params set."""
        model = GeminiModel(api_key="test-key")
        mock_response = Mock()
        mock_response.text = "response"
        model.client = Mock()
        model.client.models.generate_content = Mock(return_value=mock_response)

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.models.generate_content.call_args
        assert call_kwargs.kwargs["config"] is None


class TestOllamaModel:
    """Test Ollama model implementation."""

    @pytest.mark.asyncio
    async def test_ollama_model_generate_response(self) -> None:
        """Should generate response using Ollama API."""
        model = OllamaModel(model_name="llama2")

        # Mock the ollama client
        model.client = AsyncMock()
        model.client.chat.return_value = {"message": {"content": "Hello from Ollama!"}}

        response = await model.generate_response(
            "Hello", role_prompt="You are helpful."
        )

        assert response == "Hello from Ollama!"
        model.client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_passes_temperature_and_max_tokens(self) -> None:
        """Should pass temperature and max_tokens as options to Ollama API."""
        model = OllamaModel(model_name="llama2", temperature=0.5, max_tokens=200)
        model.client = AsyncMock()
        model.client.chat.return_value = {"message": {"content": "response"}}

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.chat.call_args
        assert call_kwargs.kwargs["options"] == {
            "temperature": 0.5,
            "num_predict": 200,
        }

    @pytest.mark.asyncio
    async def test_ollama_no_options_when_params_none(self) -> None:
        """Should not pass options when temperature and max_tokens are None."""
        model = OllamaModel(model_name="llama2")
        model.client = AsyncMock()
        model.client.chat.return_value = {"message": {"content": "response"}}

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.chat.call_args
        assert call_kwargs.kwargs["options"] is None


class TestClaudeCLIModel:
    """Test cases for ClaudeCLIModel."""

    def test_initialization(self) -> None:
        """Test ClaudeCLIModel initialization."""
        from llm_orc.models.anthropic import ClaudeCLIModel

        model = ClaudeCLIModel(
            claude_path="/usr/local/bin/claude", model="claude-3-5-sonnet-20241022"
        )

        assert model.claude_path == "/usr/local/bin/claude"
        assert model.model == "claude-3-5-sonnet-20241022"
        assert model.name == "claude-cli-claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_generate_response_success(self) -> None:
        """Test successful response generation using Claude CLI."""
        from unittest.mock import Mock, patch

        from llm_orc.models.anthropic import ClaudeCLIModel

        model = ClaudeCLIModel(claude_path="/usr/local/bin/claude")

        # Mock subprocess call
        mock_result = Mock()
        mock_result.stdout = "Hello! How can I help you today?"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_subprocess:
            result = await model.generate_response(
                "Hello", "You are a helpful assistant"
            )

            assert result == "Hello! How can I help you today?"

            # Verify subprocess was called correctly
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args

            # Should call claude with proper arguments
            assert call_args[0][0] == ["/usr/local/bin/claude", "--no-api-key"]
            assert "You are a helpful assistant" in call_args[1]["input"]
            assert "Hello" in call_args[1]["input"]

    @pytest.mark.asyncio
    async def test_generate_response_claude_cli_error(self) -> None:
        """Test response generation when Claude CLI returns error."""
        from unittest.mock import Mock, patch

        from llm_orc.models.anthropic import ClaudeCLIModel

        model = ClaudeCLIModel(claude_path="/usr/local/bin/claude")

        # Mock subprocess error
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Authentication error: Please run 'claude auth login'"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(Exception, match="Claude CLI error"):
                await model.generate_response("Hello", "You are a helpful assistant")
