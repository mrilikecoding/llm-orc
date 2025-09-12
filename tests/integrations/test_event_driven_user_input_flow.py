"""End-to-end integration tests for event-driven user input flow using TDD."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler
from llm_orc.visualization.events import ExecutionEventType
from llm_orc.visualization.stream import EventStream


class TestEventDrivenUserInputFlow:
    """Test suite for end-to-end event-driven user input flow."""

    @pytest.mark.asyncio
    async def test_event_stream_receives_user_input_events(self) -> None:
        """Test that EventStream receives and processes user input events."""
        # Given
        execution_id = "test_exec_123"
        event_stream = EventStream(execution_id)

        # We'll collect events from the stream's history after execution

        # Create handler with event emitter that feeds into the stream
        async def emit_to_stream(event: Any) -> None:
            await event_stream.emit(event)

        handler = ScriptUserInputHandler(event_emitter=emit_to_stream)

        # Mock CLI input collector
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "integration test input"

        # Create input request
        input_request = {
            "prompt": "Enter test value:",
            "agent_name": "integration_script_agent",
            "script_path": "/integration/test/script.py",
        }

        # When
        # Handle input request (this should emit events)
        await handler.handle_input_request(
            input_request=input_request,
            protocol=Mock(),
            conversation_id="integration_conv",
            cli_input_collector=cli_input_collector,
            ensemble_name="integration_ensemble",
            execution_id=execution_id,
        )

        # Collect events from history
        collected_events = event_stream.get_event_history(
            [
                ExecutionEventType.USER_INPUT_REQUIRED,
                ExecutionEventType.USER_INPUT_RECEIVED,
            ]
        )

        # Then
        assert len(collected_events) == 2

        # Check USER_INPUT_REQUIRED event
        required_event = collected_events[0]
        assert required_event.event_type == ExecutionEventType.USER_INPUT_REQUIRED
        assert required_event.data["prompt"] == "Enter test value:"
        assert required_event.data["script_path"] == "/integration/test/script.py"
        assert required_event.agent_name == "integration_script_agent"

        # Check USER_INPUT_RECEIVED event
        received_event = collected_events[1]
        assert received_event.event_type == ExecutionEventType.USER_INPUT_RECEIVED
        assert received_event.data["user_input"] == "integration test input"

    @pytest.mark.asyncio
    async def test_streaming_pause_resume_coordination(self) -> None:
        """Test coordination between streaming pause/resume and user input events."""
        # Given
        execution_id = "streaming_test_456"
        event_stream = EventStream(execution_id)

        # Event stream will track all streaming control events

        # Mock streaming progress tracker that responds to events
        streaming_tracker = Mock()
        streaming_tracker.is_paused = False

        async def handle_streaming_event(event: Any) -> None:
            if event.event_type == ExecutionEventType.STREAMING_PAUSED:
                streaming_tracker.is_paused = True
            elif event.event_type == ExecutionEventType.STREAMING_RESUMED:
                streaming_tracker.is_paused = False

        # Create handler with event emitter
        events_emitted = []

        async def emit_and_handle(event: Any) -> None:
            events_emitted.append(event)
            await event_stream.emit(event)
            await handle_streaming_event(event)

        handler = ScriptUserInputHandler(event_emitter=emit_and_handle)

        # Mock CLI input collector with delay to simulate user thinking
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "delayed user input"

        # Create input request
        input_request = {
            "prompt": "Enter something important:",
            "agent_name": "streaming_test_agent",
            "script_path": "/streaming/test.py",
        }

        # When
        await handler.handle_input_request(
            input_request=input_request,
            protocol=Mock(),
            conversation_id="streaming_conv",
            cli_input_collector=cli_input_collector,
            ensemble_name="streaming_ensemble",
            execution_id=execution_id,
        )

        # Then
        # Verify 4 events were emitted in correct order
        assert len(events_emitted) == 4

        # Check event sequence
        assert events_emitted[0].event_type == ExecutionEventType.STREAMING_PAUSED
        assert events_emitted[1].event_type == ExecutionEventType.USER_INPUT_REQUIRED
        assert events_emitted[2].event_type == ExecutionEventType.USER_INPUT_RECEIVED
        assert events_emitted[3].event_type == ExecutionEventType.STREAMING_RESUMED

        # Verify streaming was properly paused and resumed
        # (streaming_tracker would be paused during input collection)
        assert not streaming_tracker.is_paused  # Should be resumed after completion

        # Verify reasons are correct
        assert events_emitted[0].data["reason"] == "waiting_for_user_input"
        assert events_emitted[3].data["reason"] == "user_input_received"

    @pytest.mark.asyncio
    async def test_event_driven_flow_without_streaming_interference(self) -> None:
        """Test that event-driven flow prevents streaming interference with terminal."""
        # Given
        execution_id = "terminal_test_789"

        # Mock terminal state tracker
        terminal_state = {
            "streaming_active": True,
            "user_input_blocked": False,
            "events_received": [],
        }

        # Event handler that manages terminal state
        async def terminal_event_handler(event: Any) -> None:
            events_received = terminal_state["events_received"]
            if isinstance(events_received, list):
                events_received.append(event.event_type.value)

            if event.event_type == ExecutionEventType.STREAMING_PAUSED:
                terminal_state["streaming_active"] = False
                terminal_state["user_input_blocked"] = False

            elif event.event_type == ExecutionEventType.USER_INPUT_REQUIRED:
                # Terminal is now ready for user input
                assert not terminal_state["streaming_active"]

            elif event.event_type == ExecutionEventType.USER_INPUT_RECEIVED:
                # User input was successfully collected
                assert not terminal_state["user_input_blocked"]

            elif event.event_type == ExecutionEventType.STREAMING_RESUMED:
                terminal_state["streaming_active"] = True
                terminal_state["user_input_blocked"] = True

        handler = ScriptUserInputHandler(event_emitter=terminal_event_handler)

        # Mock successful user input collection
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "terminal safe input"

        # When
        await handler.handle_input_request(
            input_request={
                "prompt": "Terminal test input:",
                "agent_name": "terminal_agent",
                "script_path": "/terminal/test.py",
            },
            protocol=Mock(),
            conversation_id="terminal_conv",
            cli_input_collector=cli_input_collector,
            ensemble_name="terminal_ensemble",
            execution_id=execution_id,
        )

        # Then
        # Verify complete event flow occurred
        expected_events = [
            "streaming_paused",
            "user_input_required",
            "user_input_received",
            "streaming_resumed",
        ]
        assert terminal_state["events_received"] == expected_events

        # Verify terminal state returned to streaming
        assert terminal_state["streaming_active"]
        assert terminal_state["user_input_blocked"]

        # Verify user input was collected successfully
        cli_input_collector.collect_input.assert_called_once_with(
            "Terminal test input:"
        )

    def test_integration_with_existing_script_detection(self) -> None:
        """Test integration with existing script user input detection methods."""
        # Given
        handler = ScriptUserInputHandler()

        # When/Then - Test existing detection methods still work
        assert handler.requires_user_input("get_user_input.py")
        assert handler.requires_user_input("some_script_with_input().py")
        assert handler.requires_user_input(
            'print("hello"); user_name = input("Name: ")'
        )
        assert not handler.requires_user_input("print_only_script.py")

        # Test ensemble detection
        mock_ensemble = Mock()
        mock_ensemble.agents = [
            {"type": "script", "script": "get_user_input.py"},
            {"type": "llm", "model": "gpt-4"},
        ]

        assert handler.ensemble_requires_user_input(mock_ensemble)

        # Test ensemble without user input
        mock_ensemble_no_input = Mock()
        mock_ensemble_no_input.agents = [
            {"type": "script", "script": "data_processor.py"},
            {"type": "llm", "model": "gpt-4"},
        ]

        assert not handler.ensemble_requires_user_input(mock_ensemble_no_input)
