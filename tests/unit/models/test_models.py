"""Test suite for multi-model support."""

from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.models.anthropic import ClaudeCLIModel, ClaudeModel, OAuthClaudeModel
from llm_orc.models.base import ModelInterface
from llm_orc.models.google import GeminiModel
from llm_orc.models.manager import ModelManager
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
        model = ClaudeModel(
            api_key="test-key", temperature=0.5, max_tokens=2000
        )
        assert model.temperature == 0.5
        assert model.max_tokens == 2000

    def test_gemini_model_defaults_to_none(self) -> None:
        """GeminiModel should have None temperature and max_tokens by default."""
        model = GeminiModel(api_key="test-key")
        assert model.temperature is None
        assert model.max_tokens is None

    def test_gemini_model_accepts_params(self) -> None:
        """GeminiModel should accept temperature and max_tokens kwargs."""
        model = GeminiModel(
            api_key="test-key", temperature=0.3, max_tokens=1000
        )
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
        model = ClaudeModel(
            api_key="test-key", temperature=0.7, max_tokens=2000
        )
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
        model = GeminiModel(
            api_key="test-key", temperature=0.3, max_tokens=500
        )
        mock_response = Mock()
        mock_response.text = "response"
        model.client = Mock()
        model.client.models.generate_content = Mock(
            return_value=mock_response
        )

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
        model.client.models.generate_content = Mock(
            return_value=mock_response
        )

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
        model = OllamaModel(
            model_name="llama2", temperature=0.5, max_tokens=200
        )
        model.client = AsyncMock()
        model.client.chat.return_value = {
            "message": {"content": "response"}
        }

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
        model.client.chat.return_value = {
            "message": {"content": "response"}
        }

        await model.generate_response("Hi", role_prompt="Be helpful.")

        call_kwargs = model.client.chat.call_args
        assert call_kwargs.kwargs["options"] is None


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

        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            manager.get_model("nonexistent")

    def test_register_oauth_claude_model_basic(self) -> None:
        """Should register OAuth Claude model with basic parameters."""
        manager = ModelManager()

        manager.register_oauth_claude_model(
            key="oauth_claude",
            access_token="test_access_token",
        )

        assert "oauth_claude" in manager.models
        model = manager.models["oauth_claude"]
        assert hasattr(model, "access_token")
        assert model.access_token == "test_access_token"

    def test_register_oauth_claude_model_with_all_parameters(self) -> None:
        """Should register OAuth Claude model with all parameters."""
        manager = ModelManager()
        mock_credential_storage = Mock()

        manager.register_oauth_claude_model(
            key="oauth_claude_full",
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            client_id="test_client_id",
            model="claude-3-opus-20240229",
            credential_storage=mock_credential_storage,
            provider_key="test_provider",
            expires_at=1234567890,
        )

        assert "oauth_claude_full" in manager.models
        model = cast(OAuthClaudeModel, manager.models["oauth_claude_full"])
        assert model.access_token == "test_access_token"
        assert model.refresh_token == "test_refresh_token"
        assert model.client_id == "test_client_id"
        assert model.model == "claude-3-opus-20240229"
        assert model._credential_storage == mock_credential_storage
        assert model._provider_key == "test_provider"
        assert model.expires_at == 1234567890

    def test_register_claude_model_basic(self) -> None:
        """Should register Claude model with basic parameters."""
        manager = ModelManager()

        manager.register_claude_model(
            key="claude_api",
            api_key="test_api_key",
        )

        assert "claude_api" in manager.models
        model = cast(ClaudeModel, manager.models["claude_api"])
        assert model.api_key == "test_api_key"
        assert model.model == "claude-3-5-sonnet-20241022"  # Default model

    def test_register_claude_model_with_custom_model(self) -> None:
        """Should register Claude model with custom model name."""
        manager = ModelManager()

        manager.register_claude_model(
            key="claude_api_custom",
            api_key="test_api_key",
            model="claude-3-opus-20240229",
        )

        assert "claude_api_custom" in manager.models
        model = cast(ClaudeModel, manager.models["claude_api_custom"])
        assert model.api_key == "test_api_key"
        assert model.model == "claude-3-opus-20240229"

    def test_register_claude_cli_model_basic(self) -> None:
        """Should register Claude CLI model with basic parameters."""
        manager = ModelManager()

        manager.register_claude_cli_model(
            key="claude_cli",
            claude_path="/usr/local/bin/claude",
        )

        assert "claude_cli" in manager.models
        model = cast(ClaudeCLIModel, manager.models["claude_cli"])
        assert model.claude_path == "/usr/local/bin/claude"
        assert model.model == "claude-3-5-sonnet-20241022"  # Default model

    def test_register_claude_cli_model_with_custom_model(self) -> None:
        """Should register Claude CLI model with custom model name."""
        manager = ModelManager()

        manager.register_claude_cli_model(
            key="claude_cli_custom",
            claude_path="/opt/claude/bin/claude",
            model="claude-3-opus-20240229",
        )

        assert "claude_cli_custom" in manager.models
        model = cast(ClaudeCLIModel, manager.models["claude_cli_custom"])
        assert model.claude_path == "/opt/claude/bin/claude"
        assert model.model == "claude-3-opus-20240229"

    def test_list_models_empty(self) -> None:
        """Should return empty dict when no models registered."""
        manager = ModelManager()

        result = manager.list_models()

        assert result == {}

    def test_list_models_with_registered_models(self) -> None:
        """Should return all registered models with their names."""
        manager = ModelManager()

        # Register different types of models
        manager.register_claude_model("claude1", "api_key1")
        manager.register_oauth_claude_model("oauth1", "access_token1")
        manager.register_claude_cli_model("cli1", "/usr/bin/claude")

        result = manager.list_models()

        assert len(result) == 3
        assert "claude1" in result
        assert "oauth1" in result
        assert "cli1" in result
        # Verify the values are model names
        assert all(isinstance(name, str) for name in result.values())

    def test_get_oauth_models_empty(self) -> None:
        """Should return empty dict when no OAuth models registered."""
        manager = ModelManager()

        # Register non-OAuth models
        manager.register_claude_model("claude1", "api_key1")
        manager.register_claude_cli_model("cli1", "/usr/bin/claude")

        result = manager.get_oauth_models()

        assert result == {}

    def test_get_oauth_models_with_oauth_models(self) -> None:
        """Should return only OAuth models."""
        manager = ModelManager()

        # Register different types of models
        manager.register_claude_model("claude1", "api_key1")
        manager.register_oauth_claude_model("oauth1", "access_token1")
        manager.register_oauth_claude_model("oauth2", "access_token2")
        manager.register_claude_cli_model("cli1", "/usr/bin/claude")

        result = manager.get_oauth_models()

        assert len(result) == 2
        assert "oauth1" in result
        assert "oauth2" in result
        assert "claude1" not in result
        assert "cli1" not in result
        # Verify the values are OAuthClaudeModel instances
        from llm_orc.models.anthropic import OAuthClaudeModel

        assert all(isinstance(model, OAuthClaudeModel) for model in result.values())

    def test_get_api_key_models_empty(self) -> None:
        """Should return empty dict when no API key models registered."""
        manager = ModelManager()

        # Register non-API key models
        manager.register_oauth_claude_model("oauth1", "access_token1")
        manager.register_claude_cli_model("cli1", "/usr/bin/claude")

        result = manager.get_api_key_models()

        assert result == {}

    def test_get_api_key_models_with_api_key_models(self) -> None:
        """Should return only API key models."""
        manager = ModelManager()

        # Register different types of models
        manager.register_claude_model("claude1", "api_key1")
        manager.register_claude_model("claude2", "api_key2")
        manager.register_oauth_claude_model("oauth1", "access_token1")
        manager.register_claude_cli_model("cli1", "/usr/bin/claude")

        result = manager.get_api_key_models()

        assert len(result) == 2
        assert "claude1" in result
        assert "claude2" in result
        assert "oauth1" not in result
        assert "cli1" not in result
        # Verify the values are ClaudeModel instances
        from llm_orc.models.anthropic import ClaudeModel

        assert all(isinstance(model, ClaudeModel) for model in result.values())

    def test_mixed_model_registration_and_retrieval(self) -> None:
        """Should handle mixed model types correctly."""
        manager = ModelManager()

        # Register all types of models
        manager.register_claude_model("claude_api", "api_key")
        manager.register_oauth_claude_model("claude_oauth", "access_token")
        manager.register_claude_cli_model("claude_cli", "/usr/bin/claude")

        # Test retrieval
        claude_api = manager.get_model("claude_api")
        claude_oauth = manager.get_model("claude_oauth")
        claude_cli = manager.get_model("claude_cli")

        # Verify types
        from llm_orc.models.anthropic import (
            ClaudeCLIModel,
            ClaudeModel,
            OAuthClaudeModel,
        )

        assert isinstance(claude_api, ClaudeModel)
        assert isinstance(claude_oauth, OAuthClaudeModel)
        assert isinstance(claude_cli, ClaudeCLIModel)

        # Test filtering methods
        oauth_models = manager.get_oauth_models()
        api_key_models = manager.get_api_key_models()

        assert len(oauth_models) == 1
        assert len(api_key_models) == 1
        assert "claude_oauth" in oauth_models
        assert "claude_api" in api_key_models


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
