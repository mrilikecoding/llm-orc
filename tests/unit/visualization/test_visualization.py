"""Tests for visualization system."""

import asyncio
from collections.abc import Generator
from datetime import datetime
from unittest.mock import patch

import pytest

from llm_orc.visualization.config import (
    TerminalVisualizationConfig,
    VisualizationConfig,
)
from llm_orc.visualization.events import (
    EventFactory,
    ExecutionEvent,
    ExecutionEventType,
)
from llm_orc.visualization.stream import EventStream, EventStreamManager


@pytest.fixture(autouse=True)
def mock_expensive_dependencies() -> Generator[None, None, None]:
    """Mock expensive dependencies for all visualization tests."""
    with patch("llm_orc.core.execution.ensemble_execution.ConfigurationManager"):
        with patch("llm_orc.core.execution.ensemble_execution.CredentialStorage"):
            with patch("llm_orc.core.execution.ensemble_execution.ModelFactory"):
                yield


class TestExecutionEvent:
    """Test ExecutionEvent class."""

    def test_event_creation(self) -> None:
        """Test basic event creation."""
        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test_ensemble",
            execution_id="test_id",
            data={"key": "value"},
        )

        assert event.event_type == ExecutionEventType.ENSEMBLE_STARTED
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_id"
        assert event.data == {"key": "value"}

    def test_event_factory_ensemble_started(self) -> None:
        """Test EventFactory.ensemble_started method."""
        event = EventFactory.ensemble_started(
            ensemble_name="test_ensemble", execution_id="test_id", total_agents=3
        )

        assert event.event_type == ExecutionEventType.ENSEMBLE_STARTED
        assert event.ensemble_name == "test_ensemble"
        assert event.execution_id == "test_id"
        assert event.data["total_agents"] == 3


class TestEventStream:
    """Test EventStream class."""

    def test_event_stream_creation(self) -> None:
        """Test basic event stream creation."""
        stream = EventStream("test_execution")
        assert stream.execution_id == "test_execution"
        assert len(stream._event_history) == 0

    @pytest.mark.asyncio
    async def test_event_emission(self) -> None:
        """Test event emission."""
        stream = EventStream("test_execution")
        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test_ensemble",
            execution_id="test_execution",
            data={},
        )

        await stream.emit(event)

        history = stream.get_event_history()
        assert len(history) == 1
        assert history[0] == event

    @pytest.mark.asyncio
    async def test_event_subscription(self) -> None:
        """Test event subscription."""
        stream = EventStream("test_execution")
        events_received = []

        async def collect_events() -> None:
            async for event in stream.subscribe([ExecutionEventType.ENSEMBLE_STARTED]):
                events_received.append(event)
                if len(events_received) >= 1:
                    break

        # Start collecting events
        collection_task = asyncio.create_task(collect_events())

        # Give the subscription time to set up
        await asyncio.sleep(0.1)

        # Emit an event
        test_event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test_ensemble",
            execution_id="test_execution",
            data={},
        )
        await stream.emit(test_event)

        # Wait for collection to complete
        await collection_task

        assert len(events_received) == 1
        assert events_received[0] == test_event


class TestEventStreamManager:
    """Test EventStreamManager class."""

    def test_stream_manager_creation(self) -> None:
        """Test stream manager creation."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        assert len(manager._streams) == 0

    @pytest.mark.asyncio
    async def test_create_stream(self) -> None:
        """Test stream creation."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        stream = manager.create_stream("test_execution")

        assert stream.execution_id == "test_execution"
        assert manager.get_stream("test_execution") == stream

    @pytest.mark.asyncio
    async def test_create_stream_with_existing_id_raises_error(self) -> None:
        """Test that creating a stream with existing ID raises error."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        manager.create_stream("test_execution")

        with pytest.raises(
            ValueError, match="Stream for execution test_execution already exists"
        ):
            manager.create_stream("test_execution")

    @pytest.mark.asyncio
    async def test_remove_stream(self) -> None:
        """Test stream removal."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        manager.create_stream("test_execution")

        manager.remove_stream("test_execution")

        assert manager.get_stream("test_execution") is None


class TestVisualizationConfig:
    """Test VisualizationConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = VisualizationConfig()

        assert config.enabled is True
        assert config.default_mode == "simple"
        assert isinstance(config.terminal, TerminalVisualizationConfig)

    def test_config_to_dict(self) -> None:
        """Test configuration serialization."""
        config = VisualizationConfig(enabled=False, default_mode="web")
        result = config.to_dict()

        assert result["enabled"] is False
        assert result["default_mode"] == "web"
        assert "terminal" in result
