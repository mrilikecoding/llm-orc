"""Category-specific schemas for primitive scripts (ADR-002).

These schemas define the input/output contracts for different categories
of primitive scripts, enabling type-safe composition and validation.
"""

from typing import Any

from pydantic import BaseModel, Field


# User Interaction Primitives
class UserInteractionInput(BaseModel):
    """Base input schema for user interaction primitives."""

    agent_name: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)
    multiline: bool = False
    default_value: str | None = None


class UserInteractionOutput(BaseModel):
    """Base output schema for user interaction primitives."""

    success: bool
    user_response: str | None = None
    error: str | None = None
    cancelled: bool = False


# Data Transformation Primitives
class DataTransformInput(BaseModel):
    """Base input schema for data transformation primitives."""

    source_data: Any
    transform_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class DataTransformOutput(BaseModel):
    """Base output schema for data transformation primitives."""

    success: bool
    transformed_data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# File Operations Primitives
class FileOperationInput(BaseModel):
    """Base input schema for file operation primitives."""

    operation: str  # read, write, list, delete, etc.
    path: str
    content: str | None = None
    encoding: str = "utf-8"
    parameters: dict[str, Any] = Field(default_factory=dict)


class FileOperationOutput(BaseModel):
    """Base output schema for file operation primitives."""

    success: bool
    content: str | None = None
    files: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# API Integration Primitives
class APIIntegrationInput(BaseModel):
    """Base input schema for API integration primitives."""

    url: str
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class APIIntegrationOutput(BaseModel):
    """Base output schema for API integration primitives."""

    success: bool
    status_code: int | None = None
    response_data: Any = None
    error: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


# Computation Primitives
class ComputationInput(BaseModel):
    """Base input schema for computation primitives."""

    operation: str
    operands: list[Any]
    parameters: dict[str, Any] = Field(default_factory=dict)


class ComputationOutput(BaseModel):
    """Base output schema for computation primitives."""

    success: bool
    result: Any = None
    error: str | None = None
    computation_time: float | None = None


# Control Flow Primitives
class ControlFlowInput(BaseModel):
    """Base input schema for control flow primitives."""

    control_type: str  # loop, condition, branch, etc.
    condition: str | None = None
    max_iterations: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ControlFlowOutput(BaseModel):
    """Base output schema for control flow primitives."""

    success: bool
    iterations: int = 0
    continue_execution: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# External Execution Primitives
class ExternalExecutionInput(BaseModel):
    """Base input schema for external execution primitives."""

    command: str
    arguments: list[str] = Field(default_factory=list)
    stdin_data: str | None = None
    timeout: int | None = None
    environment: dict[str, str] = Field(default_factory=dict)


class ExternalExecutionOutput(BaseModel):
    """Base output schema for external execution primitives."""

    success: bool
    stdout: str | None = None
    stderr: str | None = None
    return_code: int | None = None
    error: str | None = None
