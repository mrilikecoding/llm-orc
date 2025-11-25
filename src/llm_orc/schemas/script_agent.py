"""Pydantic schemas for script agent interfaces (ADR-001)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScriptAgentInput(BaseModel):
    """Base input schema for all script agents (ADR-001)."""

    agent_name: str
    input_data: str
    context: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """Request for another agent to perform an action (ADR-001)."""

    target_agent_type: str
    parameters: dict[str, Any]
    priority: int = 0


class ScriptAgentOutput(BaseModel):
    """Base output schema for all script agents (ADR-001)."""

    success: bool
    data: Any = None
    error: str | None = None
    agent_requests: list[AgentRequest] = Field(default_factory=list)


class UserInputRequest(BaseModel):
    """Schema for requesting user input."""

    prompt: str
    multiline: bool = False
    validation_pattern: str | None = None
    retry_message: str | None = None
    max_attempts: int = 3


class UserInputOutput(ScriptAgentOutput):
    """Output from user input collection."""

    user_input: str
    attempts_used: int
    validation_passed: bool


class FileOperationRequest(BaseModel):
    """Schema for file operations."""

    operation: Literal["read", "write", "append", "delete"]
    path: str
    content: str | None = None
    encoding: str = "utf-8"


class FileOperationOutput(ScriptAgentOutput):
    """Output from file operations."""

    path: str
    size: int
    bytes_processed: int
    operation_performed: str
