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

    ``payload`` (fail-closed-composition B2, addendum 2026-09-23) carries
    every field alongside a failed script agent's own ``error``/
    ``success`` keys — ``stderr`` (turn_trace's ``_engine_failure_fields``
    reads ``payload.stderr``) and producer-specific fields like
    web_searcher's ``backend`` — so they survive on the failed record
    instead of being dropped when only ``error`` was kept. Nested under
    its own ``payload`` key by ``to_dict``/``PhaseResultProcessor``, NEVER
    merged onto the record: engine-owned keys (``status``, ``response``,
    ``error``, ``outcome``, ``has_errors``, ...) are reserved, so a
    script's own JSON claiming e.g. ``"status": "success"`` cannot
    overwrite the record's real status (the blocker this addendum closes
    — a script printing ``{"success": false, "error": "x", "status":
    "success", "response": "FABRICATED"}`` used to report success).
    """

    status: Literal["success", "failed"]
    response: str | None = None
    error: str | None = None
    model_substituted: bool = False
    model_instance: Any = field(default=None, repr=False)
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, excluding model_instance."""
        result: dict[str, Any] = {
            "response": self.response,
            "status": self.status,
            "model_substituted": self.model_substituted,
        }
        if self.status == "failed" and self.error is not None:
            result["error"] = self.error
        if self.status == "failed" and self.payload:
            result["payload"] = self.payload
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
    deliverable: str | None = None
    execution_order: list[str] = field(default_factory=list)
    validation_result: Any = None
    has_errors: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for artifact saving and API responses.

        ``has_errors`` (addendum 2026-09-23) is the same bool
        ``finalize_result`` used to derive ``status``
        (``completed``/``completed_with_errors``) — surfaced directly so
        a caller (a parent ``ensemble:``/``dispatch:``/``loop:`` node's
        ``outcome.child_has_errors``, or an external reader) does not
        need to re-derive it from the status string.
        """
        result: dict[str, Any] = {
            "ensemble": self.ensemble,
            "status": self.status,
            "has_errors": self.has_errors,
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
        }
        if self.deliverable is not None:
            result["deliverable"] = self.deliverable
        if self.execution_order:
            result["execution_order"] = self.execution_order
        if self.validation_result is not None:
            result["validation_result"] = self.validation_result
        return result
