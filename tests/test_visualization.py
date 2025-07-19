"""Tests for visualization system."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.visualization.config import (
    TerminalVisualizationConfig,
    VisualizationConfig,
)
from llm_orc.visualization.events import (
    EventFactory,
    ExecutionEvent,
    ExecutionEventType,
)
from llm_orc.visualization.integration import VisualizationIntegratedExecutor
from llm_orc.visualization.stream import EventStream, EventStreamManager


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
        manager = EventStreamManager()
        assert len(manager._streams) == 0

    @pytest.mark.asyncio
    async def test_create_stream(self) -> None:
        """Test stream creation."""
        manager = EventStreamManager()
        stream = manager.create_stream("test_execution")

        assert stream.execution_id == "test_execution"
        assert manager.get_stream("test_execution") == stream

    @pytest.mark.asyncio
    async def test_create_stream_with_existing_id_raises_error(self) -> None:
        """Test that creating a stream with existing ID raises error."""
        manager = EventStreamManager()
        manager.create_stream("test_execution")

        with pytest.raises(
            ValueError, match="Stream for execution test_execution already exists"
        ):
            manager.create_stream("test_execution")

    @pytest.mark.asyncio
    async def test_remove_stream(self) -> None:
        """Test stream removal."""
        manager = EventStreamManager()
        manager.create_stream("test_execution")

        manager.remove_stream("test_execution")

        assert manager.get_stream("test_execution") is None


class TestVisualizationConfig:
    """Test VisualizationConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = VisualizationConfig()

        assert config.enabled is True
        assert config.default_mode == "terminal"
        assert isinstance(config.terminal, TerminalVisualizationConfig)

    def test_config_to_dict(self) -> None:
        """Test configuration serialization."""
        config = VisualizationConfig(enabled=False, default_mode="web")
        result = config.to_dict()

        assert result["enabled"] is False
        assert result["default_mode"] == "web"
        assert "terminal" in result


class TestVisualizationIntegratedExecutor:
    """Test VisualizationIntegratedExecutor class."""

    def test_executor_creation(self) -> None:
        """Test executor creation."""
        executor = VisualizationIntegratedExecutor()
        assert executor.viz_config is not None
        assert executor.current_stream is None

    @pytest.mark.asyncio
    async def test_execute_with_visualization_disabled(self) -> None:
        """Test execution with visualization disabled."""
        config = VisualizationConfig(enabled=False)
        executor = VisualizationIntegratedExecutor(config)

        # Mock the parent execute method
        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"result": "test"}

            ensemble_config = Mock(spec=EnsembleConfig)
            result = await executor.execute_with_visualization(
                ensemble_config, "test_input"
            )

            assert result == {"result": "test"}
            mock_execute.assert_called_once_with(ensemble_config, "test_input")

    @pytest.mark.asyncio
    async def test_execute_with_terminal_visualization(self) -> None:
        """Test execution with terminal visualization."""
        config = VisualizationConfig(enabled=True, default_mode="terminal")
        executor = VisualizationIntegratedExecutor(config)

        # Mock the parent execute method
        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"result": "test"}

            # Mock the terminal visualizer
            with patch(
                "llm_orc.visualization.integration.TerminalVisualizer"
            ) as mock_visualizer_class:
                mock_visualizer = Mock()
                mock_visualizer.visualize_execution = AsyncMock()
                mock_visualizer.print_summary = Mock()
                mock_visualizer_class.return_value = mock_visualizer

                ensemble_config = Mock(spec=EnsembleConfig)
                ensemble_config.name = "test_ensemble"
                ensemble_config.agents = [{"name": "test_agent", "role": "test_role"}]

                result = await executor.execute_with_visualization(
                    ensemble_config, "test_input", "terminal"
                )

                assert result == {"result": "test"}
                mock_execute.assert_called_once()
                mock_visualizer.print_summary.assert_called_once()

    def test_convert_performance_event_agent_started(self) -> None:
        """Test performance event conversion for agent started."""
        executor = VisualizationIntegratedExecutor()
        executor.current_execution_id = "test_id"

        event = executor._convert_performance_event(
            "agent_started",
            {
                "agent_name": "test_agent",
                "model": "test_model",
                "depends_on": ["dep1", "dep2"],
            },
        )

        assert event is not None
        assert event.event_type == ExecutionEventType.AGENT_STARTED
        assert event.agent_name == "test_agent"
        assert event.execution_id == "test_id"
        assert event.data["model"] == "test_model"
        assert event.data["depends_on"] == ["dep1", "dep2"]

    def test_convert_performance_event_agent_completed(self) -> None:
        """Test performance event conversion for agent completed."""
        executor = VisualizationIntegratedExecutor()
        executor.current_execution_id = "test_id"

        event = executor._convert_performance_event(
            "agent_completed",
            {
                "agent_name": "test_agent",
                "result": "test_result",
                "duration_ms": 1000,
                "cost_usd": 0.05,
            },
        )

        assert event is not None
        assert event.event_type == ExecutionEventType.AGENT_COMPLETED
        assert event.agent_name == "test_agent"
        assert event.execution_id == "test_id"
        assert event.data["result"] == "test_result"
        assert event.data["duration_ms"] == 1000
        assert event.data["cost_usd"] == 0.05

    def test_convert_performance_event_unknown_type(self) -> None:
        """Test performance event conversion for unknown event type."""
        executor = VisualizationIntegratedExecutor()
        executor.current_execution_id = "test_id"

        event = executor._convert_performance_event("unknown_event", {"data": "test"})

        assert event is None
