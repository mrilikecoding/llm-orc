"""Event system for ensemble execution monitoring."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutionEventType(Enum):
    """Types of execution events that can occur during ensemble execution."""

    # Ensemble lifecycle events
    ENSEMBLE_STARTED = "ensemble_started"
    ENSEMBLE_COMPLETED = "ensemble_completed"
    ENSEMBLE_FAILED = "ensemble_failed"

    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    # Dependency events
    DEPENDENCY_WAITING = "dependency_waiting"
    DEPENDENCY_SATISFIED = "dependency_satisfied"

    # Resource events
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_RELEASED = "resource_released"

    # Performance events
    PERFORMANCE_METRIC = "performance_metric"
    COST_UPDATE = "cost_update"

    # Debug events
    DEBUG_BREAKPOINT = "debug_breakpoint"
    DEBUG_STEP = "debug_step"


@dataclass
class ExecutionEvent:
    """Represents a single execution event."""

    event_type: ExecutionEventType
    timestamp: datetime
    data: dict[str, Any]

    # Optional fields for context
    agent_name: str | None = None
    ensemble_name: str | None = None
    execution_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "agent_name": self.agent_name,
            "ensemble_name": self.ensemble_name,
            "execution_id": self.execution_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionEvent":
        """Create event from dictionary."""
        return cls(
            event_type=ExecutionEventType(data["type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            agent_name=data.get("agent_name"),
            ensemble_name=data.get("ensemble_name"),
            execution_id=data.get("execution_id"),
        )


class EventFactory:
    """Factory for creating common execution events."""

    @staticmethod
    def ensemble_started(
        ensemble_name: str,
        execution_id: str,
        total_agents: int,
        estimated_duration: float | None = None,
    ) -> ExecutionEvent:
        """Create ensemble started event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "total_agents": total_agents,
                "estimated_duration": estimated_duration,
            },
        )

    @staticmethod
    def agent_started(
        agent_name: str,
        ensemble_name: str,
        execution_id: str,
        model: str,
        dependencies: list[str] | None = None,
    ) -> ExecutionEvent:
        """Create agent started event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            agent_name=agent_name,
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "model": model,
                "dependencies": dependencies or [],
                "status": "running",
            },
        )

    @staticmethod
    def agent_progress(
        agent_name: str,
        ensemble_name: str,
        execution_id: str,
        progress_percentage: float,
        intermediate_result: str | None = None,
    ) -> ExecutionEvent:
        """Create agent progress event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.AGENT_PROGRESS,
            timestamp=datetime.now(),
            agent_name=agent_name,
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "progress_percentage": progress_percentage,
                "intermediate_result": intermediate_result,
            },
        )

    @staticmethod
    def agent_completed(
        agent_name: str,
        ensemble_name: str,
        execution_id: str,
        result: str,
        duration_ms: int,
        cost_usd: float | None = None,
        tokens_used: int | None = None,
    ) -> ExecutionEvent:
        """Create agent completed event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            agent_name=agent_name,
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "result": result,
                "duration_ms": duration_ms,
                "cost_usd": cost_usd,
                "tokens_used": tokens_used,
                "status": "completed",
            },
        )

    @staticmethod
    def agent_failed(
        agent_name: str,
        ensemble_name: str,
        execution_id: str,
        error: str,
        duration_ms: int,
    ) -> ExecutionEvent:
        """Create agent failed event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.now(),
            agent_name=agent_name,
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "error": error,
                "duration_ms": duration_ms,
                "status": "failed",
            },
        )

    @staticmethod
    def dependency_waiting(
        agent_name: str,
        ensemble_name: str,
        execution_id: str,
        waiting_for: list[str],
    ) -> ExecutionEvent:
        """Create dependency waiting event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.DEPENDENCY_WAITING,
            timestamp=datetime.now(),
            agent_name=agent_name,
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "waiting_for": waiting_for,
                "status": "waiting",
            },
        )

    @staticmethod
    def performance_metric(
        ensemble_name: str,
        execution_id: str,
        metric_name: str,
        metric_value: float,
        agent_name: str | None = None,
    ) -> ExecutionEvent:
        """Create performance metric event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            agent_name=agent_name,
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "metric_name": metric_name,
                "metric_value": metric_value,
            },
        )

    @staticmethod
    def ensemble_completed(
        ensemble_name: str,
        execution_id: str,
        result: dict[str, Any],
        duration_ms: int,
    ) -> ExecutionEvent:
        """Create ensemble completed event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "result": result,
                "duration_ms": duration_ms,
                "status": "completed",
            },
        )

    @staticmethod
    def ensemble_failed(
        ensemble_name: str,
        execution_id: str,
        error: str,
    ) -> ExecutionEvent:
        """Create ensemble failed event."""
        return ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_FAILED,
            timestamp=datetime.now(),
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "error": error,
                "status": "failed",
            },
        )
