"""Tests for ScriptUserInputHandler with typed event emission using TDD."""

from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler
from llm_orc.visualization.events import ExecutionEventType


class TestScriptUserInputHandlerEvents:
    """Test suite for ScriptUserInputHandler event emission."""

    def test_script_user_input_handler_has_event_emitter(self) -> None:
        """Test that ScriptUserInputHandler can accept an event emitter."""
        # Given
        event_emitter = Mock()

        # When
        handler = ScriptUserInputHandler(event_emitter=event_emitter)

        # Then
        assert handler.event_emitter == event_emitter

    @pytest.mark.asyncio
    async def test_emits_user_input_required_event(self) -> None:
        """Test that handler emits USER_INPUT_REQUIRED event when input is needed."""
        # Given
        event_emitter = AsyncMock()
        handler = ScriptUserInputHandler(event_emitter=event_emitter)

        # Mock the CLI input collector
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "test input"

        # Create input request
        input_request = {
            "prompt": "Enter your name:",
            "agent_name": "script_agent",
            "script_path": "/path/to/script.py",
        }

        # When
        await handler.handle_input_request(
            input_request=input_request,
            _protocol=Mock(),
            conversation_id="conv_123",
            cli_input_collector=cli_input_collector,
            ensemble_name="test_ensemble",
            execution_id="exec_123",
        )

        # Then
        # Verify event was emitted
        event_emitter.assert_called()

        # Find the USER_INPUT_REQUIRED event
        user_input_required_event = None
        for call in event_emitter.call_args_list:
            event = call[0][0]
            if event.event_type == ExecutionEventType.USER_INPUT_REQUIRED:
                user_input_required_event = event
                break

        assert user_input_required_event is not None
        assert user_input_required_event.data["prompt"] == "Enter your name:"
        assert user_input_required_event.data["script_path"] == "/path/to/script.py"

    @pytest.mark.asyncio
    async def test_emits_user_input_received_event(self) -> None:
        """Test that handler emits USER_INPUT_RECEIVED event after getting input."""
        # Given
        event_emitter = AsyncMock()
        handler = ScriptUserInputHandler(event_emitter=event_emitter)

        # Mock the CLI input collector
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "John Doe"

        # Create input request
        input_request = {
            "prompt": "Enter your name:",
            "agent_name": "script_agent",
            "script_path": "/path/to/script.py",
        }

        # When
        await handler.handle_input_request(
            input_request=input_request,
            _protocol=Mock(),
            conversation_id="conv_123",
            cli_input_collector=cli_input_collector,
            ensemble_name="test_ensemble",
            execution_id="exec_123",
        )

        # Then
        # Verify USER_INPUT_RECEIVED event was emitted
        user_input_received_event = None
        for call in event_emitter.call_args_list:
            event = call[0][0]
            if event.event_type == ExecutionEventType.USER_INPUT_RECEIVED:
                user_input_received_event = event
                break

        assert user_input_received_event is not None
        assert user_input_received_event.data["user_input"] == "John Doe"

    @pytest.mark.asyncio
    async def test_emits_streaming_paused_and_resumed_events(self) -> None:
        """Test that handler emits STREAMING_PAUSED and STREAMING_RESUMED events."""
        # Given
        event_emitter = AsyncMock()
        handler = ScriptUserInputHandler(event_emitter=event_emitter)

        # Mock the CLI input collector
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "test input"

        # Create input request
        input_request = {
            "prompt": "Enter value:",
            "agent_name": "script_agent",
            "script_path": "/path/to/script.py",
        }

        # When
        await handler.handle_input_request(
            input_request=input_request,
            _protocol=Mock(),
            conversation_id="conv_123",
            cli_input_collector=cli_input_collector,
            ensemble_name="test_ensemble",
            execution_id="exec_123",
        )

        # Then
        # Verify streaming events were emitted
        assert (
            event_emitter.call_count == 4
        )  # PAUSED, INPUT_REQUIRED, INPUT_RECEIVED, RESUMED

        # Find STREAMING_PAUSED event (should be first)
        paused_event = None
        resumed_event = None

        for call in event_emitter.call_args_list:
            event = call[0][0]
            if event.event_type == ExecutionEventType.STREAMING_PAUSED:
                paused_event = event
            elif event.event_type == ExecutionEventType.STREAMING_RESUMED:
                resumed_event = event

        assert paused_event is not None
        assert resumed_event is not None
        assert paused_event.data["reason"] == "waiting_for_user_input"
        assert resumed_event.data["reason"] == "user_input_received"

    @pytest.mark.asyncio
    async def test_handles_input_without_event_emitter(self) -> None:
        """Test that handler works without an event emitter."""
        # Given
        handler = ScriptUserInputHandler()  # No event emitter

        # Mock the CLI input collector
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "test input"

        # Create input request
        input_request = {
            "prompt": "Enter value:",
        }

        # When
        await handler.handle_input_request(
            input_request=input_request,
            _protocol=Mock(),
            conversation_id="conv_123",
            cli_input_collector=cli_input_collector,
        )

        # Then
        cli_input_collector.collect_input.assert_called_once_with("Enter value:")
