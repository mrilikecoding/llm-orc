"""Core schema definitions for llm-orc."""

from .script_agent import (
    AgentRequest,
    FileOperationOutput,
    FileOperationRequest,
    ScriptAgentInput,
    ScriptAgentOutput,
    UserInputOutput,
    UserInputRequest,
)

__all__ = [
    # Universal script agent schemas (ADR-001)
    "ScriptAgentInput",
    "ScriptAgentOutput",
    "AgentRequest",
    "UserInputRequest",
    "UserInputOutput",
    "FileOperationRequest",
    "FileOperationOutput",
]
