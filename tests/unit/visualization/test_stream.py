"""Comprehensive tests for visualization event streaming system."""

import asyncio
from collections import defaultdict
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from llm_orc.visualization.events import ExecutionEvent, ExecutionEventType
from llm_orc.visualization.stream import (
    EventStream,
    EventStreamManager,
    PerformanceEventCollector,
    get_stream_manager,
)


class TestEventStream:
    """Test EventStream class comprehensively."""

    def test_init(self) -> None:
        """Test EventStream initialization."""
        execution_id = "test-execution-123"
        stream = EventStream(execution_id)

        assert stream.execution_id == execution_id
        assert stream._subscribers == defaultdict(list)
        assert stream._event_history == []
        assert stream._is_closed is False

    @pytest.mark.asyncio
    async def test_emit_event_stores_in_history(self) -> None:
        """Test that emitting event stores it in history."""
        stream = EventStream("test-id")
        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            data={"test": "data"},
        )

        await stream.emit(event)

        assert len(stream._event_history) == 1
        assert stream._event_history[0] == event

    @pytest.mark.asyncio
    async def test_emit_multiple_events_preserves_order(self) -> None:
        """Test that multiple events are stored in emission order."""
        stream = EventStream("test-id")
        events = []

        for i in range(3):
            event = ExecutionEvent(
                event_type=ExecutionEventType.AGENT_STARTED,
                timestamp=datetime.now(),
                ensemble_name="test",
                execution_id="test-id",
                agent_name=f"agent-{i}",
                data={"index": i},
            )
            events.append(event)
            await stream.emit(event)

        assert len(stream._event_history) == 3
        for i, stored_event in enumerate(stream._event_history):
            assert stored_event.agent_name == f"agent-{i}"
            assert stored_event.data["index"] == i

    @pytest.mark.asyncio
    async def test_emit_to_closed_stream_does_nothing(self) -> None:
        """Test that emitting to closed stream does nothing."""
        stream = EventStream("test-id")
        stream._is_closed = True

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            data={},
        )

        await stream.emit(event)

        assert len(stream._event_history) == 0

    @pytest.mark.asyncio
    async def test_emit_sends_to_specific_event_type_subscribers(self) -> None:
        """Test that events are sent to subscribers of specific event types."""
        stream = EventStream("test-id")
        queue = asyncio.Queue[ExecutionEvent](maxsize=10)

        # Register subscriber for specific event type
        stream._subscribers[ExecutionEventType.AGENT_STARTED.value].append(queue)

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )

        await stream.emit(event)

        # Queue should have received the event
        assert queue.qsize() == 1
        received_event = await queue.get()
        assert received_event == event

    @pytest.mark.asyncio
    async def test_emit_sends_to_wildcard_subscribers(self) -> None:
        """Test that events are sent to wildcard subscribers."""
        stream = EventStream("test-id")
        queue = asyncio.Queue[ExecutionEvent](maxsize=10)

        # Register wildcard subscriber
        stream._subscribers["*"].append(queue)

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )

        await stream.emit(event)

        # Queue should have received the event
        assert queue.qsize() == 1
        received_event = await queue.get()
        assert received_event == event

    @pytest.mark.asyncio
    async def test_emit_sends_to_both_specific_and_wildcard_subscribers(self) -> None:
        """Test that events are sent to both specific and wildcard subscribers."""
        stream = EventStream("test-id")
        specific_queue = asyncio.Queue[ExecutionEvent](maxsize=10)
        wildcard_queue = asyncio.Queue[ExecutionEvent](maxsize=10)

        # Register both types of subscribers
        stream._subscribers[ExecutionEventType.AGENT_STARTED.value].append(
            specific_queue
        )
        stream._subscribers["*"].append(wildcard_queue)

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )

        await stream.emit(event)

        # Both queues should have received the event
        assert specific_queue.qsize() == 1
        assert wildcard_queue.qsize() == 1

        specific_event = await specific_queue.get()
        wildcard_event = await wildcard_queue.get()
        assert specific_event == event
        assert wildcard_event == event

    @pytest.mark.asyncio
    async def test_emit_handles_full_queue_gracefully(self) -> None:
        """Test that full queues are skipped gracefully."""
        stream = EventStream("test-id")
        queue = asyncio.Queue[ExecutionEvent](maxsize=1)

        # Fill the queue completely
        fill_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="initial",
            data={},
        )
        queue.put_nowait(fill_event)  # Use put_nowait to avoid blocking

        stream._subscribers[ExecutionEventType.AGENT_STARTED.value].append(queue)

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )

        # Should not raise an exception despite full queue
        await stream.emit(event)

        # Event should still be in history
        assert len(stream._event_history) == 1
        assert stream._event_history[0] == event

    def test_subscribe_to_closed_stream_raises_error(self) -> None:
        """Test that subscribing to closed stream raises error."""
        stream = EventStream("test-id")
        stream._is_closed = True

        with pytest.raises(RuntimeError, match="EventStream is closed"):
            stream.subscribe([ExecutionEventType.AGENT_STARTED])

    def test_subscribe_with_none_event_types_subscribes_to_all(self) -> None:
        """Test that subscribing with None event types subscribes to all."""
        stream = EventStream("test-id")

        # This should not raise an exception and should create wildcard subscription
        subscription = stream.subscribe(None)

        # Verify the subscription generator was created
        assert subscription is not None

    @pytest.mark.asyncio
    async def test_subscribe_specific_event_types(self) -> None:
        """Test subscribing to specific event types."""
        stream = EventStream("test-id")
        received_events = []

        async def collect_events() -> None:
            async for event in stream.subscribe([ExecutionEventType.AGENT_COMPLETED]):
                received_events.append(event)
                if len(received_events) >= 1:
                    break

        collection_task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.01)  # Give subscription time to set up

        # Emit matching event
        matching_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )
        await stream.emit(matching_event)

        # Emit non-matching event
        non_matching_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="other-agent",
            data={},
        )
        await stream.emit(non_matching_event)

        await collection_task

        # Should only receive the matching event
        assert len(received_events) == 1
        assert received_events[0] == matching_event

    @pytest.mark.asyncio
    async def test_subscribe_with_custom_queue_size(self) -> None:
        """Test subscription with custom queue size."""
        stream = EventStream("test-id")

        # Create subscription with small queue size
        subscription = stream.subscribe(
            [ExecutionEventType.AGENT_STARTED], queue_size=2
        )

        # Verify we can create the subscription
        assert subscription is not None

    @pytest.mark.asyncio
    async def test_subscription_cleanup_on_cancellation(self) -> None:
        """Test that subscription cleans up when cancelled."""
        stream = EventStream("test-id")

        async def cancelled_subscription() -> None:
            async for _event in stream.subscribe([ExecutionEventType.AGENT_STARTED]):
                pass  # This will be cancelled

        subscription_task = asyncio.create_task(cancelled_subscription())
        await asyncio.sleep(0.01)  # Give subscription time to set up

        # Verify subscription was registered
        assert len(stream._subscribers[ExecutionEventType.AGENT_STARTED.value]) == 1

        # Cancel the subscription
        subscription_task.cancel()

        try:
            await subscription_task
        except asyncio.CancelledError:
            pass

        # Give cleanup time to run
        await asyncio.sleep(0.01)

        # Subscription should be cleaned up
        assert len(stream._subscribers[ExecutionEventType.AGENT_STARTED.value]) == 0

    @pytest.mark.asyncio
    async def test_subscription_continues_on_timeout(self) -> None:
        """Test that subscription continues when timeout occurs."""
        stream = EventStream("test-id")
        timeout_count = 0
        received_events = []

        async def subscription_with_timeout() -> None:
            nonlocal timeout_count
            try:
                async for event in stream.subscribe([ExecutionEventType.AGENT_STARTED]):
                    received_events.append(event)
                    break  # Exit after first event
            except TimeoutError:
                timeout_count += 1

        subscription_task = asyncio.create_task(subscription_with_timeout())
        await asyncio.sleep(0.01)  # Give subscription time to set up

        # Wait a bit to let timeout occur, then emit event
        await asyncio.sleep(0.01)
        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )
        await stream.emit(event)

        await subscription_task

        # Should have received the event despite timeouts
        assert len(received_events) == 1
        assert received_events[0] == event

    def test_get_event_history_all_events(self) -> None:
        """Test getting all events from history."""
        stream = EventStream("test-id")
        events = []

        for i in range(3):
            event = ExecutionEvent(
                event_type=ExecutionEventType.AGENT_STARTED,
                timestamp=datetime.now(),
                ensemble_name="test",
                execution_id="test-id",
                agent_name=f"agent-{i}",
                data={"index": i},
            )
            events.append(event)
            stream._event_history.append(event)

        history = stream.get_event_history()

        assert len(history) == 3
        assert history == events

    def test_get_event_history_filtered_by_event_types(self) -> None:
        """Test getting filtered event history by event types."""
        stream = EventStream("test-id")

        # Add different types of events
        started_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="agent-1",
            data={},
        )
        completed_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="agent-1",
            data={},
        )
        failed_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="agent-2",
            data={},
        )

        stream._event_history.extend([started_event, completed_event, failed_event])

        # Filter for only started and completed events
        filtered_history = stream.get_event_history(
            event_types=[
                ExecutionEventType.AGENT_STARTED,
                ExecutionEventType.AGENT_COMPLETED,
            ]
        )

        assert len(filtered_history) == 2
        assert started_event in filtered_history
        assert completed_event in filtered_history
        assert failed_event not in filtered_history

    def test_get_event_history_with_limit(self) -> None:
        """Test getting limited event history."""
        stream = EventStream("test-id")

        # Add 5 events
        for i in range(5):
            event = ExecutionEvent(
                event_type=ExecutionEventType.AGENT_STARTED,
                timestamp=datetime.now(),
                ensemble_name="test",
                execution_id="test-id",
                agent_name=f"agent-{i}",
                data={"index": i},
            )
            stream._event_history.append(event)

        # Get last 3 events
        limited_history = stream.get_event_history(limit=3)

        assert len(limited_history) == 3
        # Should get the last 3 events (indices 2, 3, 4)
        for i, event in enumerate(limited_history):
            assert event.data["index"] == i + 2

    def test_get_event_history_filtered_and_limited(self) -> None:
        """Test getting event history with both filtering and limit."""
        stream = EventStream("test-id")

        # Add alternating event types
        for i in range(6):
            event_type = (
                ExecutionEventType.AGENT_STARTED
                if i % 2 == 0
                else ExecutionEventType.AGENT_COMPLETED
            )
            event = ExecutionEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                ensemble_name="test",
                execution_id="test-id",
                agent_name=f"agent-{i}",
                data={"index": i},
            )
            stream._event_history.append(event)

        # Get last 2 AGENT_STARTED events
        filtered_limited_history = stream.get_event_history(
            event_types=[ExecutionEventType.AGENT_STARTED], limit=2
        )

        assert len(filtered_limited_history) == 2
        # Should get the last 2 AGENT_STARTED events (indices 4 and 2, in that order)
        assert filtered_limited_history[0].data["index"] == 2
        assert filtered_limited_history[1].data["index"] == 4

    def test_close_stream(self) -> None:
        """Test closing the stream."""
        stream = EventStream("test-id")

        # Add some subscribers
        queue1 = asyncio.Queue[ExecutionEvent](maxsize=10)
        queue2 = asyncio.Queue[ExecutionEvent](maxsize=10)
        stream._subscribers["agent_started"].append(queue1)
        stream._subscribers["*"].append(queue2)

        stream.close()

        assert stream._is_closed is True
        assert len(stream._subscribers) == 0


class TestEventStreamManager:
    """Test EventStreamManager class comprehensively."""

    def test_init(self) -> None:
        """Test EventStreamManager initialization."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        assert manager._streams == {}
        assert manager._cleanup_tasks == set()

    def test_create_stream_with_explicit_id(self) -> None:
        """Test creating stream with explicit execution ID."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        execution_id = "explicit-test-id"

        stream = manager.create_stream(execution_id)

        assert stream.execution_id == execution_id
        assert manager._streams[execution_id] == stream

    def test_create_stream_with_none_id_generates_uuid(self) -> None:
        """Test creating stream with None ID generates UUID."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        with patch("uuid.uuid4", return_value="generated-uuid"):
            stream = manager.create_stream(None)

        assert stream.execution_id == "generated-uuid"
        assert manager._streams["generated-uuid"] == stream

    def test_create_stream_with_existing_id_raises_error(self) -> None:
        """Test that creating stream with existing ID raises ValueError."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        execution_id = "duplicate-id"

        manager.create_stream(execution_id)

        with pytest.raises(
            ValueError, match=f"Stream for execution {execution_id} already exists"
        ):
            manager.create_stream(execution_id)

    @pytest.mark.asyncio
    async def test_create_stream_schedules_cleanup_task(self) -> None:
        """Test that creating stream schedules a cleanup task."""
        manager = EventStreamManager(enable_cleanup_tasks=True)  # Enable for this test

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            # Mock the cleanup method to prevent warnings
            with patch.object(manager, "_cleanup_stream_after_delay", new=Mock()):
                manager.create_stream("test-id")

                mock_create_task.assert_called_once()
                assert mock_task in manager._cleanup_tasks

    def test_get_existing_stream(self) -> None:
        """Test getting an existing stream."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        execution_id = "existing-id"

        created_stream = manager.create_stream(execution_id)
        retrieved_stream = manager.get_stream(execution_id)

        assert retrieved_stream == created_stream

    def test_get_nonexistent_stream_returns_none(self) -> None:
        """Test getting a non-existent stream returns None."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        retrieved_stream = manager.get_stream("nonexistent-id")

        assert retrieved_stream is None

    def test_remove_existing_stream(self) -> None:
        """Test removing an existing stream."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        execution_id = "to-remove"

        stream = manager.create_stream(execution_id)
        manager.remove_stream(execution_id)

        assert manager.get_stream(execution_id) is None
        assert stream._is_closed is True

    def test_remove_nonexistent_stream_does_nothing(self) -> None:
        """Test removing a non-existent stream does nothing."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        # Should not raise an error
        manager.remove_stream("nonexistent-id")

    @pytest.mark.asyncio
    async def test_cleanup_stream_after_delay(self) -> None:
        """Test the cleanup mechanism after delay."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        execution_id = "cleanup-test"

        # Create stream
        stream = manager.create_stream(execution_id)
        assert manager.get_stream(execution_id) == stream

        # Test cleanup with very short delay
        await manager._cleanup_stream_after_delay(execution_id, delay=1)

        # Stream should be removed
        assert manager.get_stream(execution_id) is None

    def test_list_active_streams_empty(self) -> None:
        """Test listing active streams when empty."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        active_streams = manager.list_active_streams()

        assert active_streams == []

    def test_list_active_streams_multiple(self) -> None:
        """Test listing multiple active streams."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        stream_ids = ["stream-1", "stream-2", "stream-3"]
        for stream_id in stream_ids:
            manager.create_stream(stream_id)

        active_streams = manager.list_active_streams()

        assert set(active_streams) == set(stream_ids)

    def test_close_all_streams(self) -> None:
        """Test closing all streams."""
        manager = EventStreamManager(enable_cleanup_tasks=False)

        # Create multiple streams
        stream_ids = ["stream-1", "stream-2", "stream-3"]
        streams = []
        for stream_id in stream_ids:
            stream = manager.create_stream(stream_id)
            streams.append(stream)

        # Add mock cleanup tasks
        mock_tasks = [Mock(), Mock()]
        manager._cleanup_tasks.update(mock_tasks)

        manager.close_all()

        # All streams should be closed and removed
        assert len(manager._streams) == 0
        for stream in streams:
            assert stream._is_closed is True

        # All cleanup tasks should be cancelled
        for task in mock_tasks:
            task.cancel.assert_called_once()

        assert len(manager._cleanup_tasks) == 0


class TestPerformanceEventCollector:
    """Test PerformanceEventCollector class comprehensively."""

    def test_init(self) -> None:
        """Test PerformanceEventCollector initialization."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        assert collector.stream == stream
        assert collector._metrics == defaultdict(list)
        assert collector._costs == defaultdict(float)
        assert collector._durations == {}
        assert collector._start_times == {}

    @pytest.mark.asyncio
    async def test_process_agent_started_event(self) -> None:
        """Test processing AGENT_STARTED event."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.fromtimestamp(1000.0),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )

        await collector._process_performance_event(event)

        assert collector._start_times["test-agent"] == 1000.0

    @pytest.mark.asyncio
    async def test_process_agent_completed_event(self) -> None:
        """Test processing AGENT_COMPLETED event."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Set up start time
        collector._start_times["test-agent"] = 1000.0

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.fromtimestamp(1005.0),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={"duration_ms": 5000, "cost_usd": 0.05},
        )

        await collector._process_performance_event(event)

        assert collector._durations["test-agent"] == 5000
        assert collector._costs["test-agent"] == 0.05

    @pytest.mark.asyncio
    async def test_process_agent_failed_event(self) -> None:
        """Test processing AGENT_FAILED event."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Set up start time
        collector._start_times["failed-agent"] = 1000.0

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.fromtimestamp(1003.0),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="failed-agent",
            data={"duration_ms": 3000, "cost_usd": 0.02, "error": "Test error"},
        )

        await collector._process_performance_event(event)

        assert collector._durations["failed-agent"] == 3000
        assert collector._costs["failed-agent"] == 0.02

    @pytest.mark.asyncio
    async def test_process_agent_completed_without_start_time(self) -> None:
        """Test processing AGENT_COMPLETED without corresponding start time."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.fromtimestamp(1005.0),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="unknown-agent",
            data={"duration_ms": 5000, "cost_usd": 0.05},
        )

        await collector._process_performance_event(event)

        # Should not record duration/cost for unknown agent
        assert "unknown-agent" not in collector._durations
        assert "unknown-agent" not in collector._costs

    @pytest.mark.asyncio
    async def test_process_agent_completed_with_zero_cost(self) -> None:
        """Test processing AGENT_COMPLETED with zero cost."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Set up start time
        collector._start_times["free-agent"] = 1000.0

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.fromtimestamp(1005.0),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="free-agent",
            data={"duration_ms": 5000, "cost_usd": 0.0},
        )

        await collector._process_performance_event(event)

        assert collector._durations["free-agent"] == 5000
        # Zero cost should not be recorded
        assert "free-agent" not in collector._costs

    @pytest.mark.asyncio
    async def test_process_agent_completed_without_cost(self) -> None:
        """Test processing AGENT_COMPLETED without cost data."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Set up start time
        collector._start_times["no-cost-agent"] = 1000.0

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.fromtimestamp(1005.0),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="no-cost-agent",
            data={"duration_ms": 5000},
        )

        await collector._process_performance_event(event)

        assert collector._durations["no-cost-agent"] == 5000
        assert "no-cost-agent" not in collector._costs

    @pytest.mark.asyncio
    async def test_process_performance_metric_event(self) -> None:
        """Test processing PERFORMANCE_METRIC event."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        event = ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            data={"metric_name": "response_time", "metric_value": 1.5},
        )

        await collector._process_performance_event(event)

        assert len(collector._metrics["response_time"]) == 1
        assert collector._metrics["response_time"][0] == 1.5

    @pytest.mark.asyncio
    async def test_process_performance_metric_event_multiple_values(self) -> None:
        """Test processing multiple PERFORMANCE_METRIC events for same metric."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        values = [1.5, 2.3, 0.8, 3.1]
        for value in values:
            event = ExecutionEvent(
                event_type=ExecutionEventType.PERFORMANCE_METRIC,
                timestamp=datetime.now(),
                ensemble_name="test",
                execution_id="test-id",
                data={"metric_name": "latency", "metric_value": value},
            )
            await collector._process_performance_event(event)

        assert len(collector._metrics["latency"]) == 4
        assert collector._metrics["latency"] == values

    @pytest.mark.asyncio
    async def test_process_performance_metric_event_missing_fields(self) -> None:
        """Test processing PERFORMANCE_METRIC event with missing fields."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Missing metric_value
        event1 = ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            data={"metric_name": "incomplete_metric"},
        )

        # Missing metric_name
        event2 = ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            data={"metric_value": 2.5},
        )

        await collector._process_performance_event(event1)
        await collector._process_performance_event(event2)

        # Neither should be recorded
        assert len(collector._metrics) == 0

    @pytest.mark.asyncio
    async def test_collect_performance_events_integration(self) -> None:
        """Test the full collect_performance_events integration."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Start collection task
        collection_task = asyncio.create_task(collector.collect_performance_events())
        await asyncio.sleep(0.01)  # Give time for subscription to set up

        # Emit various performance events
        events = [
            ExecutionEvent(
                event_type=ExecutionEventType.AGENT_STARTED,
                timestamp=datetime.fromtimestamp(1000.0),
                ensemble_name="test",
                execution_id="test-id",
                agent_name="agent-1",
                data={},
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.AGENT_COMPLETED,
                timestamp=datetime.fromtimestamp(1005.0),
                ensemble_name="test",
                execution_id="test-id",
                agent_name="agent-1",
                data={"duration_ms": 5000, "cost_usd": 0.1},
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.PERFORMANCE_METRIC,
                timestamp=datetime.now(),
                ensemble_name="test",
                execution_id="test-id",
                data={"metric_name": "throughput", "metric_value": 15.7},
            ),
        ]

        for event in events:
            await stream.emit(event)

        # Give time for events to be processed
        await asyncio.sleep(0.01)

        # Cancel collection task
        collection_task.cancel()
        try:
            await collection_task
        except asyncio.CancelledError:
            pass

        # Verify data was collected
        assert collector._start_times["agent-1"] == 1000.0
        assert collector._durations["agent-1"] == 5000
        assert collector._costs["agent-1"] == 0.1
        assert collector._metrics["throughput"][0] == 15.7

    def test_get_summary_empty(self) -> None:
        """Test getting summary when no data collected."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        summary = collector.get_summary()

        expected = {
            "total_duration_ms": 0,
            "total_cost_usd": 0.0,
            "agent_durations": {},
            "agent_costs": {},
            "metrics": {},
        }
        assert summary == expected

    def test_get_summary_with_data(self) -> None:
        """Test getting summary with collected data."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Set up test data
        collector._durations = {"agent-1": 5000, "agent-2": 3000}
        collector._costs = {"agent-1": 0.1, "agent-2": 0.05}
        collector._metrics = {"latency": [1.5, 2.0, 1.8], "throughput": [10.0, 12.5]}

        summary = collector.get_summary()

        expected = {
            "total_duration_ms": 8000,  # 5000 + 3000
            "total_cost_usd": 0.15,  # 0.1 + 0.05
            "agent_durations": {"agent-1": 5000, "agent-2": 3000},
            "agent_costs": {"agent-1": 0.1, "agent-2": 0.05},
            "metrics": {
                "latency": {
                    "count": 3,
                    "sum": 5.3,
                    "avg": 5.3 / 3,
                    "min": 1.5,
                    "max": 2.0,
                },
                "throughput": {
                    "count": 2,
                    "sum": 22.5,
                    "avg": 11.25,
                    "min": 10.0,
                    "max": 12.5,
                },
            },
        }
        # Check each field separately to handle floating point precision
        assert summary["total_duration_ms"] == expected["total_duration_ms"]
        assert abs(summary["total_cost_usd"] - expected["total_cost_usd"]) < 1e-10
        assert summary["agent_durations"] == expected["agent_durations"]
        assert summary["agent_costs"] == expected["agent_costs"]
        assert summary["metrics"] == expected["metrics"]

    def test_get_summary_metrics_empty_values(self) -> None:
        """Test getting summary with metrics that have empty values."""
        stream = EventStream("test-id")
        collector = PerformanceEventCollector(stream)

        # Set up test data with empty metrics
        collector._metrics = {"empty_metric": []}

        summary = collector.get_summary()

        expected_metric_summary = {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
        assert summary["metrics"]["empty_metric"] == expected_metric_summary


class TestGlobalStreamManager:
    """Test global stream manager function."""

    def test_get_stream_manager_returns_singleton(self) -> None:
        """Test that get_stream_manager returns the same instance."""
        manager1 = get_stream_manager()
        manager2 = get_stream_manager()

        assert manager1 is manager2
        assert isinstance(manager1, EventStreamManager)

    @pytest.mark.asyncio
    async def test_subscription_timeout_continue_logic(self) -> None:
        """Test that subscription timeout continues properly."""
        from llm_orc.visualization.stream import reset_global_stream_manager

        # Reset to ensure clean state
        reset_global_stream_manager()

        stream = EventStream("test-id")
        received_events = []
        continue_count = 0

        async def subscription_with_timeout_tracking() -> None:
            nonlocal continue_count
            async for event in stream.subscribe([ExecutionEventType.AGENT_STARTED]):
                received_events.append(event)
                if len(received_events) >= 1:
                    break
                continue_count += 1  # This tracks the continue execution

        subscription_task = asyncio.create_task(subscription_with_timeout_tracking())
        await asyncio.sleep(0.01)  # Give subscription time to set up

        # Send event after short delay to trigger timeout -> continue -> event cycle
        await asyncio.sleep(0.01)
        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test",
            execution_id="test-id",
            agent_name="test-agent",
            data={},
        )
        await stream.emit(event)

        await subscription_task

        # Should have received the event
        assert len(received_events) == 1
        assert received_events[0] == event

    def test_create_stream_no_event_loop_runtime_error(self) -> None:
        """Test creating stream when no event loop is running."""
        manager = EventStreamManager(enable_cleanup_tasks=True)

        # Mock get_running_loop to raise RuntimeError
        with patch(
            "asyncio.get_running_loop", side_effect=RuntimeError("No running loop")
        ):
            # Should not raise exception, just skip cleanup task scheduling
            stream = manager.create_stream("test-no-loop")

        assert stream is not None
        assert stream.execution_id == "test-no-loop"
        assert "test-no-loop" not in manager._stream_cleanup_tasks

    def test_remove_stream_with_cleanup_task(self) -> None:
        """Test removing stream that has an associated cleanup task."""
        manager = EventStreamManager(enable_cleanup_tasks=False)
        execution_id = "test-with-cleanup"

        # Create stream
        manager.create_stream(execution_id)

        # Manually add a mock cleanup task
        mock_task = Mock()
        manager._cleanup_tasks.add(mock_task)
        manager._stream_cleanup_tasks[execution_id] = mock_task

        # Remove the stream
        manager.remove_stream(execution_id)

        # Verify cleanup task was cancelled and removed
        mock_task.cancel.assert_called_once()
        assert mock_task not in manager._cleanup_tasks
        assert execution_id not in manager._stream_cleanup_tasks

    @pytest.mark.asyncio
    async def test_cleanup_task_done_callback(self) -> None:
        """Test the cleanup task done callback functionality."""
        manager = EventStreamManager(enable_cleanup_tasks=True)
        execution_id = "test-callback"

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            # Create stream (should add done callback)
            with patch.object(manager, "_cleanup_stream_after_delay", new=Mock()):
                manager.create_stream(execution_id)

            # Verify task was created and done callback was added
            mock_create_task.assert_called_once()
            mock_task.add_done_callback.assert_called_once()

            # Get the callback function
            callback_func = mock_task.add_done_callback.call_args[0][0]

            # Simulate task completion by calling the callback
            callback_func(mock_task)

            # Verify task was removed from cleanup sets
            assert mock_task not in manager._cleanup_tasks
            assert execution_id not in manager._stream_cleanup_tasks
