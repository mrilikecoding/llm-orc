"""Test suite for multi-model support."""

import pytest
from unittest.mock import Mock, AsyncMock
from llm_orc.models import ModelInterface, ClaudeModel, GeminiModel, OllamaModel, ModelManager


class TestModelInterface:
    """Test the abstract model interface."""
    
    def test_model_interface_is_abstract(self):
        """Should not be able to instantiate ModelInterface directly."""
        with pytest.raises(TypeError):
            ModelInterface()


class TestClaudeModel:
    """Test Claude model implementation."""
    
    @pytest.mark.asyncio
    async def test_claude_model_generate_response(self):
        """Should generate response using Claude API."""
        model = ClaudeModel(api_key="test-key")
        
        # Mock the anthropic client
        model.client = AsyncMock()
        model.client.messages.create.return_value = Mock(
            content=[Mock(text="Hello from Claude!")]
        )
        
        response = await model.generate_response("Hello", role_prompt="You are helpful.")
        
        assert response == "Hello from Claude!"
        model.client.messages.create.assert_called_once()


class TestGeminiModel:
    """Test Gemini model implementation."""
    
    @pytest.mark.asyncio
    async def test_gemini_model_generate_response(self):
        """Should generate response using Gemini API."""
        model = GeminiModel(api_key="test-key")
        
        # Mock the genai client with proper async handling
        mock_response = Mock()
        mock_response.text = "Hello from Gemini!"
        model.client.generate_content = Mock(return_value=mock_response)
        
        response = await model.generate_response("Hello", role_prompt="You are helpful.")
        
        assert response == "Hello from Gemini!"
        model.client.generate_content.assert_called_once()


class TestOllamaModel:
    """Test Ollama model implementation."""
    
    @pytest.mark.asyncio
    async def test_ollama_model_generate_response(self):
        """Should generate response using Ollama API."""
        model = OllamaModel(model_name="llama2")
        
        # Mock the ollama client
        model.client = AsyncMock()
        model.client.chat.return_value = {
            "message": {"content": "Hello from Ollama!"}
        }
        
        response = await model.generate_response("Hello", role_prompt="You are helpful.")
        
        assert response == "Hello from Ollama!"
        model.client.chat.assert_called_once()


class TestModelManager:
    """Test model management and selection."""
    
    def test_register_model(self):
        """Should register a new model."""
        manager = ModelManager()
        mock_model = Mock(spec=ModelInterface)
        mock_model.name = "test-model"
        
        manager.register_model("test", mock_model)
        
        assert "test" in manager.models
        assert manager.models["test"] == mock_model
    
    def test_get_model(self):
        """Should retrieve registered model."""
        manager = ModelManager()
        mock_model = Mock(spec=ModelInterface)
        mock_model.name = "test-model"
        
        manager.register_model("test", mock_model)
        retrieved = manager.get_model("test")
        
        assert retrieved == mock_model
    
    def test_get_nonexistent_model_raises_error(self):
        """Should raise error for non-existent model."""
        manager = ModelManager()
        
        with pytest.raises(KeyError):
            manager.get_model("nonexistent")