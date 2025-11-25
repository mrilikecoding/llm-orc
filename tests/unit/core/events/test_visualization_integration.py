"""Tests for integrating Pydantic events with visualization system using TDD."""

from llm_orc.visualization.events import EventFactory, ExecutionEventType


class TestUserInputVisualizationIntegration:
    """Test suite for user input event integration with visualization system."""

    def test_user_input_required_event_type_exists(self) -> None:
        """Test that USER_INPUT_REQUIRED event type exists in ExecutionEventType."""
        # When
        event_type = ExecutionEventType("user_input_required")

        # Then
        assert event_type == ExecutionEventType.USER_INPUT_REQUIRED

    def test_user_input_received_event_type_exists(self) -> None:
        """Test that USER_INPUT_RECEIVED event type exists in ExecutionEventType."""
        # When
        event_type = ExecutionEventType("user_input_received")

        # Then
        assert event_type == ExecutionEventType.USER_INPUT_RECEIVED

    def test_streaming_paused_event_type_exists(self) -> None:
        """Test that STREAMING_PAUSED event type exists in ExecutionEventType."""
        # When
        event_type = ExecutionEventType("streaming_paused")

        # Then
        assert event_type == ExecutionEventType.STREAMING_PAUSED

    def test_streaming_resumed_event_type_exists(self) -> None:
        """Test that STREAMING_RESUMED event type exists in ExecutionEventType."""
        # When
        event_type = ExecutionEventType("streaming_resumed")

        # Then
        assert event_type == ExecutionEventType.STREAMING_RESUMED

    def test_event_factory_user_input_required(self) -> None:
        """Test EventFactory can create user_input_required events."""
        # When
        event = EventFactory.user_input_required(
            agent_name="script_agent",
            ensemble_name="test_ensemble",
            execution_id="exec_123",
            prompt="Enter your name:",
            script_path="/path/to/script.py",
        )

        # Then
        assert event.event_type == ExecutionEventType.USER_INPUT_REQUIRED
        assert event.agent_name == "script_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "exec_123"
        assert event.data["prompt"] == "Enter your name:"
        assert event.data["script_path"] == "/path/to/script.py"
        assert event.data["status"] == "waiting_for_input"

    def test_event_factory_user_input_received(self) -> None:
        """Test EventFactory can create user_input_received events."""
        # When
        event = EventFactory.user_input_received(
            agent_name="script_agent",
            ensemble_name="test_ensemble",
            execution_id="exec_123",
            user_input="John Doe",
            script_path="/path/to/script.py",
        )

        # Then
        assert event.event_type == ExecutionEventType.USER_INPUT_RECEIVED
        assert event.agent_name == "script_agent"
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "exec_123"
        assert event.data["user_input"] == "John Doe"
        assert event.data["script_path"] == "/path/to/script.py"
        assert event.data["status"] == "input_received"

    def test_event_factory_streaming_paused(self) -> None:
        """Test EventFactory can create streaming_paused events."""
        # When
        event = EventFactory.streaming_paused(
            ensemble_name="test_ensemble",
            execution_id="exec_123",
            reason="waiting_for_user_input",
        )

        # Then
        assert event.event_type == ExecutionEventType.STREAMING_PAUSED
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "exec_123"
        assert event.data["reason"] == "waiting_for_user_input"
        assert event.data["status"] == "paused"

    def test_event_factory_streaming_resumed(self) -> None:
        """Test EventFactory can create streaming_resumed events."""
        # When
        event = EventFactory.streaming_resumed(
            ensemble_name="test_ensemble",
            execution_id="exec_123",
            reason="user_input_received",
        )

        # Then
        assert event.event_type == ExecutionEventType.STREAMING_RESUMED
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "exec_123"
        assert event.data["reason"] == "user_input_received"
        assert event.data["status"] == "resumed"
