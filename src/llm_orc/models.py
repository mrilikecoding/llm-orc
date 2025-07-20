"""Multi-model support for LLM agents.

This module provides backward compatibility by importing all model classes
from their new organized structure. The actual implementations have been
moved to focused, provider-specific modules.
"""

# For backward compatibility, import all model classes from provider modules

# Explicit re-exports for clarity
from llm_orc.models.anthropic import (  # noqa: F401
    ClaudeCLIModel,
    ClaudeModel,
    OAuthClaudeModel,
)
from llm_orc.models.base import HTTPConnectionPool, ModelInterface  # noqa: F401
from llm_orc.models.google import GeminiModel  # noqa: F401
from llm_orc.models.manager import ModelManager  # noqa: F401
from llm_orc.models.ollama import OllamaModel  # noqa: F401
