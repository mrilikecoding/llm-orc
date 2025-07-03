"""Multi-model support for LLM agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
from anthropic import AsyncAnthropic
import google.generativeai as genai
import ollama


class ModelInterface(ABC):
    """Abstract interface for LLM models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass
    
    @abstractmethod
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate a response from the model."""
        pass


class ClaudeModel(ModelInterface):
    """Claude model implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)
    
    @property
    def name(self) -> str:
        return f"claude-{self.model}"
    
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Claude API."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=role_prompt,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text


class GeminiModel(ModelInterface):
    """Gemini model implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    @property
    def name(self) -> str:
        return f"gemini-{self.model_name}"
    
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Gemini API."""
        prompt = f"{role_prompt}\n\nUser: {message}\nAssistant:"
        
        # Run in thread pool since Gemini doesn't have async support
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.client.generate_content(prompt)
        )
        return response.text


class OllamaModel(ModelInterface):
    """Ollama model implementation."""
    
    def __init__(self, model_name: str = "llama2", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.AsyncClient(host=host)
    
    @property
    def name(self) -> str:
        return f"ollama-{self.model_name}"
    
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Ollama API."""
        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": message}
            ]
        )
        return response["message"]["content"]


class ModelManager:
    """Manages model instances and selection."""
    
    def __init__(self):
        self.models: Dict[str, ModelInterface] = {}
    
    def register_model(self, key: str, model: ModelInterface) -> None:
        """Register a model instance."""
        self.models[key] = model
    
    def get_model(self, key: str) -> ModelInterface:
        """Retrieve a registered model."""
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found")
        return self.models[key]
    
    def list_models(self) -> Dict[str, str]:
        """List all registered models."""
        return {key: model.name for key, model in self.models.items()}