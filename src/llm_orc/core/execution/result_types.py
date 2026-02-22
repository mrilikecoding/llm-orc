"""Typed result models for the execution pipeline.

Replaces bare dict[str, Any] with named dataclasses for
AgentResult and ExecutionResult, providing type safety
at construction and clear documentation of expected shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentResult:
    """Result from a single agent execution.

    Constructed by AgentDispatcher, consumed by PhaseResultProcessor.
    The model_instance field is stripped before serialization.
    """

    status: Literal["success", "failed"]
    response: str | None = None
    error: str | None = None
    model_substituted: bool = False
    model_instance: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, excluding model_instance."""
        result: dict[str, Any] = {
            "response": self.response,
            "status": self.status,
            "model_substituted": self.model_substituted,
        }
        if self.status == "failed" and self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class ExecutionMetadata:
    """Metadata accumulated during ensemble execution."""

    agents_used: int
    started_at: float
    duration: str | None = None
    completed_at: float | None = None
    usage: dict[str, Any] | None = None
    adaptive_resource_management: dict[str, Any] | None = None
    fan_out: dict[str, dict[str, int]] | None = None
    interactive_mode: bool | None = None
    user_inputs_collected: int | None = None
    processed_agent_requests: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None optional fields."""
        result: dict[str, Any] = {
            "agents_used": self.agents_used,
            "started_at": self.started_at,
        }
        if self.duration is not None:
            result["duration"] = self.duration
        if self.completed_at is not None:
            result["completed_at"] = self.completed_at
        if self.usage is not None:
            result["usage"] = self.usage
        if self.adaptive_resource_management is not None:
            result["adaptive_resource_management"] = self.adaptive_resource_management
        if self.fan_out is not None:
            result["fan_out"] = self.fan_out
        if self.interactive_mode is not None:
            result["interactive_mode"] = self.interactive_mode
        if self.user_inputs_collected is not None:
            result["user_inputs_collected"] = self.user_inputs_collected
        if self.processed_agent_requests is not None:
            result["processed_agent_requests"] = self.processed_agent_requests
        return result


@dataclass
class ExecutionResult:
    """Top-level result from ensemble execution.

    Created by ResultsProcessor.create_initial_result(),
    finalized by ResultsProcessor.finalize_result().
    """

    ensemble: str
    status: str  # "running", "completed", "completed_with_errors"
    input: dict[str, str]
    results: dict[str, Any]
    metadata: ExecutionMetadata
    synthesis: Any = None
    execution_order: list[str] = field(default_factory=list)
    validation_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for artifact saving and API responses."""
        result: dict[str, Any] = {
            "ensemble": self.ensemble,
            "status": self.status,
            "input": self.input,
            "results": {
                name: (
                    agent_result.to_dict()
                    if isinstance(agent_result, AgentResult)
                    else agent_result
                )
                for name, agent_result in self.results.items()
            },
            "metadata": self.metadata.to_dict(),
            "synthesis": self.synthesis,
        }
        if self.execution_order:
            result["execution_order"] = self.execution_order
        if self.validation_result is not None:
            result["validation_result"] = self.validation_result
        return result
