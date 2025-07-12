"""Test suite for multi-model support."""

from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.models import (
    ClaudeModel,
    GeminiModel,
    ModelInterface,
    ModelManager,
    OllamaModel,
)


class TestModelInterface:
    """Test the abstract model interface."""

    def test_model_interface_is_abstract(self) -> None:
        """Should not be able to instantiate ModelInterface directly."""
        with pytest.raises(TypeError):
            ModelInterface()  # type: ignore[abstract]


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
        model.client.generate_content = Mock(return_value=mock_response)

        response = await model.generate_response(
            "Hello", role_prompt="You are helpful."
        )

        assert response == "Hello from Gemini!"
        model.client.generate_content.assert_called_once()


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


class TestModelManager:
    """Test model management and selection."""

    def test_register_model(self) -> None:
        """Should register a new model."""
        manager = ModelManager()
        mock_model = Mock(spec=ModelInterface)
        mock_model.name = "test-model"

        manager.register_model("test", mock_model)

        assert "test" in manager.models
        assert manager.models["test"] == mock_model

    def test_get_model(self) -> None:
        """Should retrieve registered model."""
        manager = ModelManager()
        mock_model = Mock(spec=ModelInterface)
        mock_model.name = "test-model"

        manager.register_model("test", mock_model)
        retrieved = manager.get_model("test")

        assert retrieved == mock_model

    def test_get_nonexistent_model_raises_error(self) -> None:
        """Should raise error for non-existent model."""
        manager = ModelManager()

        with pytest.raises(KeyError):
            manager.get_model("nonexistent")


class TestClaudeCLIModel:
    """Test cases for ClaudeCLIModel."""

    def test_initialization(self) -> None:
        """Test ClaudeCLIModel initialization."""
        from llm_orc.models import ClaudeCLIModel

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

        from llm_orc.models import ClaudeCLIModel

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

        from llm_orc.models import ClaudeCLIModel

        model = ClaudeCLIModel(claude_path="/usr/local/bin/claude")

        # Mock subprocess error
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Authentication error: Please run 'claude auth login'"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(Exception, match="Claude CLI error"):
                await model.generate_response("Hello", "You are a helpful assistant")
