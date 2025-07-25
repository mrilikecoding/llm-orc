"""Tests for visualization events module."""

from datetime import datetime

from llm_orc.visualization.events import (
    EventFactory,
    ExecutionEvent,
    ExecutionEventType,
)


class TestExecutionEvent:
    """Test ExecutionEvent class."""

    def test_to_dict_basic(self) -> None:
        """Test converting event to dictionary."""
        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            data={"test": "data"},
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
        )

        result = event.to_dict()

        assert result == {
            "type": "agent_started",
            "timestamp": "2023-01-01T12:00:00",
            "data": {"test": "data"},
            "agent_name": "test_agent",
            "ensemble_name": "test_ensemble",
            "execution_id": "test_exec_id",
        }

    def test_to_dict_with_none_values(self) -> None:
        """Test converting event with None values to dictionary (line 55)."""
        event = ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            data={"metric": "latency", "value": 100},
            agent_name=None,
            ensemble_name=None,
            execution_id=None,
        )

        result = event.to_dict()

        assert result == {
            "type": "performance_metric",
            "timestamp": "2023-01-01T12:00:00",
            "data": {"metric": "latency", "value": 100},
            "agent_name": None,
            "ensemble_name": None,
            "execution_id": None,
        }

    def test_from_dict_basic(self) -> None:
        """Test creating event from dictionary (line 67)."""
        data = {
            "type": "agent_completed",
            "timestamp": "2023-01-01T12:00:00",
            "data": {"result": "success"},
            "agent_name": "test_agent",
            "ensemble_name": "test_ensemble",
            "execution_id": "test_exec_id",
        }

        event = ExecutionEvent.from_dict(data)

        assert event.event_type == ExecutionEventType.AGENT_COMPLETED
        assert event.timestamp == datetime(2023, 1, 1, 12, 0, 0)
        assert event.data == {"result": "success"}
        assert event.agent_name == "test_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"


class TestEventFactory:
    """Test EventFactory class."""

    def test_agent_progress_basic(self) -> None:
        """Test agent progress event creation (line 132)."""
        event = EventFactory.agent_progress(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            progress_percentage=50.0,
            intermediate_result="partial result",
        )

        assert event.event_type == ExecutionEventType.AGENT_PROGRESS
        assert event.agent_name == "test_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["progress_percentage"] == 50.0
        assert event.data["intermediate_result"] == "partial result"

    def test_agent_failed_basic(self) -> None:
        """Test agent failed event creation (line 179)."""
        event = EventFactory.agent_failed(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            error="Something went wrong",
            duration_ms=1500,
        )

        assert event.event_type == ExecutionEventType.AGENT_FAILED
        assert event.agent_name == "test_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["error"] == "Something went wrong"
        assert event.data["duration_ms"] == 1500
        assert event.data["status"] == "failed"

    def test_dependency_waiting_basic(self) -> None:
        """Test dependency waiting event creation (line 200)."""
        event = EventFactory.dependency_waiting(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            waiting_for=["agent1", "agent2"],
        )

        assert event.event_type == ExecutionEventType.DEPENDENCY_WAITING
        assert event.agent_name == "test_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["waiting_for"] == ["agent1", "agent2"]
        assert event.data["status"] == "waiting"

    def test_performance_metric_basic(self) -> None:
        """Test performance metric event creation (line 221)."""
        event = EventFactory.performance_metric(
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            metric_name="latency",
            metric_value=123.45,
            agent_name="test_agent",
        )

        assert event.event_type == ExecutionEventType.PERFORMANCE_METRIC
        assert event.agent_name == "test_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["metric_name"] == "latency"
        assert event.data["metric_value"] == 123.45

    def test_performance_metric_without_agent(self) -> None:
        """Test performance metric event creation without agent."""
        event = EventFactory.performance_metric(
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            metric_name="throughput",
            metric_value=456.78,
        )

        assert event.event_type == ExecutionEventType.PERFORMANCE_METRIC
        assert event.agent_name is None
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["metric_name"] == "throughput"
        assert event.data["metric_value"] == 456.78

    def test_ensemble_completed_basic(self) -> None:
        """Test ensemble completed event creation (line 241)."""
        result_data = {"output": "final result", "summary": "completed successfully"}

        event = EventFactory.ensemble_completed(
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            result=result_data,
            duration_ms=5000,
        )

        assert event.event_type == ExecutionEventType.ENSEMBLE_COMPLETED
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["result"] == result_data
        assert event.data["duration_ms"] == 5000
        assert event.data["status"] == "completed"

    def test_ensemble_failed_basic(self) -> None:
        """Test ensemble failed event creation (line 260)."""
        event = EventFactory.ensemble_failed(
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            error="Critical error occurred",
        )

        assert event.event_type == ExecutionEventType.ENSEMBLE_FAILED
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_exec_id"
        assert event.data["error"] == "Critical error occurred"
        assert event.data["status"] == "failed"

    def test_event_timestamps_are_set(self) -> None:
        """Test that all factory methods set timestamps."""
        before = datetime.now()

        event = EventFactory.ensemble_started(
            ensemble_name="test",
            execution_id="test_id",
            total_agents=3,
        )

        after = datetime.now()

        assert before <= event.timestamp <= after

    def test_ensemble_started_with_optional_parameters(self) -> None:
        """Test ensemble started with all optional parameters."""
        agents_config = [
            {"name": "agent1", "model": "claude-3-sonnet"},
            {"name": "agent2", "model": "claude-3-sonnet"},
        ]

        event = EventFactory.ensemble_started(
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            total_agents=2,
            estimated_duration=10.5,
            agents_config=agents_config,
        )

        assert event.event_type == ExecutionEventType.ENSEMBLE_STARTED
        assert event.data["total_agents"] == 2
        assert event.data["estimated_duration"] == 10.5
        assert event.data["agents_config"] == agents_config

    def test_ensemble_started_without_optional_parameters(self) -> None:
        """Test ensemble started without optional parameters."""
        event = EventFactory.ensemble_started(
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            total_agents=1,
        )

        assert event.event_type == ExecutionEventType.ENSEMBLE_STARTED
        assert event.data["total_agents"] == 1
        assert event.data["estimated_duration"] is None
        assert event.data["agents_config"] == []

    def test_agent_started_with_dependencies(self) -> None:
        """Test agent started with dependencies."""
        event = EventFactory.agent_started(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            model="claude-3-sonnet",
            depends_on=["agent1", "agent2"],
        )

        assert event.data["depends_on"] == ["agent1", "agent2"]

    def test_agent_started_without_dependencies(self) -> None:
        """Test agent started without dependencies."""
        event = EventFactory.agent_started(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            model="claude-3-sonnet",
        )

        assert event.data["depends_on"] == []

    def test_agent_completed_with_optional_parameters(self) -> None:
        """Test agent completed with all optional parameters."""
        event = EventFactory.agent_completed(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            result="Success result",
            duration_ms=2500,
            cost_usd=0.05,
            tokens_used=1500,
        )

        assert event.data["cost_usd"] == 0.05
        assert event.data["tokens_used"] == 1500

    def test_agent_completed_without_optional_parameters(self) -> None:
        """Test agent completed without optional parameters."""
        event = EventFactory.agent_completed(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            result="Success result",
            duration_ms=2500,
        )

        assert event.data["cost_usd"] is None
        assert event.data["tokens_used"] is None

    def test_agent_progress_with_intermediate_result(self) -> None:
        """Test agent progress with intermediate result."""
        event = EventFactory.agent_progress(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            progress_percentage=75.0,
            intermediate_result="Intermediate output",
        )

        assert event.data["intermediate_result"] == "Intermediate output"

    def test_agent_progress_without_intermediate_result(self) -> None:
        """Test agent progress without intermediate result."""
        event = EventFactory.agent_progress(
            agent_name="test_agent",
            ensemble_name="test_ensemble",
            execution_id="test_exec_id",
            progress_percentage=25.0,
        )

        assert event.data["intermediate_result"] is None
