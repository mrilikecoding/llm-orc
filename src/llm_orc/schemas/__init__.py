"""Core schema definitions for llm-orc."""

from .primitive_categories import (
    APIIntegrationInput,
    APIIntegrationOutput,
    ComputationInput,
    ComputationOutput,
    ControlFlowInput,
    ControlFlowOutput,
    DataTransformInput,
    DataTransformOutput,
    ExternalExecutionInput,
    ExternalExecutionOutput,
    FileOperationInput,
    UserInteractionInput,
    UserInteractionOutput,
)
from .primitive_categories import (
    FileOperationOutput as PrimitiveFileOperationOutput,
)
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
    # Category-specific primitive schemas (ADR-002)
    "UserInteractionInput",
    "UserInteractionOutput",
    "DataTransformInput",
    "DataTransformOutput",
    "FileOperationInput",
    "PrimitiveFileOperationOutput",
    "APIIntegrationInput",
    "APIIntegrationOutput",
    "ComputationInput",
    "ComputationOutput",
    "ControlFlowInput",
    "ControlFlowOutput",
    "ExternalExecutionInput",
    "ExternalExecutionOutput",
]
