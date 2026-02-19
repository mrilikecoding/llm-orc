"""Multi-model support for LLM agents."""

# Import all classes for backward compatibility
from llm_orc.models.anthropic import ClaudeCLIModel, ClaudeModel, OAuthClaudeModel
from llm_orc.models.base import HTTPConnectionPool, ModelInterface
from llm_orc.models.google import GeminiModel
from llm_orc.models.ollama import OllamaModel

__all__ = [
    "HTTPConnectionPool",
    "ModelInterface",
    "ClaudeModel",
    "OAuthClaudeModel",
    "ClaudeCLIModel",
    "GeminiModel",
    "OllamaModel",
]
