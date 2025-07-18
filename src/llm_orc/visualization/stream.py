"""Event streaming system for real-time visualization."""

import asyncio
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from .events import ExecutionEvent, ExecutionEventType


class EventStream:
    """Manages event streaming for a single execution."""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._event_history: list[ExecutionEvent] = []
        self._is_closed = False

    async def emit(self, event: ExecutionEvent) -> None:
        """Emit an event to all subscribers."""
        if self._is_closed:
            return

        # Store event in history
        self._event_history.append(event)

        # Send to all subscribers interested in this event type
        event_type = event.event_type.value
        queues_to_notify = []

        # Add specific event type subscribers
        if event_type in self._subscribers:
            queues_to_notify.extend(self._subscribers[event_type])

        # Add subscribers listening to all events
        if "*" in self._subscribers:
            queues_to_notify.extend(self._subscribers["*"])

        # Send event to all relevant queues
        for queue in queues_to_notify:
            try:
                await queue.put(event)
            except asyncio.QueueEmpty:
                # Queue is full, skip this subscriber
                pass

    def subscribe(
        self,
        event_types: list[ExecutionEventType] | None = None,
        queue_size: int = 100,
    ) -> AsyncIterator[ExecutionEvent]:
        """Subscribe to events of specified types."""
        if self._is_closed:
            raise RuntimeError("EventStream is closed")

        # Default to all event types
        if event_types is None:
            event_types = [ExecutionEventType.ENSEMBLE_STARTED]  # Subscribe to all with "*"
            subscription_keys = ["*"]
        else:
            subscription_keys = [event_type.value for event_type in event_types]

        # Create queue for this subscription
        queue = asyncio.Queue(maxsize=queue_size)

        # Register queue for each event type
        for key in subscription_keys:
            self._subscribers[key].append(queue)

        async def event_generator() -> AsyncIterator[ExecutionEvent]:
            try:
                while not self._is_closed:
                    try:
                        # Wait for next event with timeout
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                        yield event
                    except TimeoutError:
                        # Continue waiting
                        continue
                    except asyncio.CancelledError:
                        # Subscription cancelled
                        break
            finally:
                # Clean up subscription
                for key in subscription_keys:
                    if queue in self._subscribers[key]:
                        self._subscribers[key].remove(queue)

        return event_generator()

    def get_event_history(
        self,
        event_types: list[ExecutionEventType] | None = None,
        limit: int | None = None,
    ) -> list[ExecutionEvent]:
        """Get historical events."""
        events = self._event_history

        # Filter by event types if specified
        if event_types:
            event_type_values = {et.value for et in event_types}
            events = [e for e in events if e.event_type.value in event_type_values]

        # Apply limit if specified
        if limit:
            events = events[-limit:]

        return events

    def close(self) -> None:
        """Close the event stream."""
        self._is_closed = True
        # Clear all subscribers
        self._subscribers.clear()


class EventStreamManager:
    """Manages multiple event streams across different executions."""

    def __init__(self):
        self._streams: dict[str, EventStream] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()

    def create_stream(self, execution_id: str | None = None) -> EventStream:
        """Create a new event stream for an execution."""
        if execution_id is None:
            execution_id = str(uuid.uuid4())

        if execution_id in self._streams:
            raise ValueError(f"Stream for execution {execution_id} already exists")

        stream = EventStream(execution_id)
        self._streams[execution_id] = stream

        # Schedule cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_stream_after_delay(execution_id))
        self._cleanup_tasks.add(cleanup_task)

        return stream

    def get_stream(self, execution_id: str) -> EventStream | None:
        """Get an existing event stream."""
        return self._streams.get(execution_id)

    def remove_stream(self, execution_id: str) -> None:
        """Remove an event stream."""
        if execution_id in self._streams:
            stream = self._streams[execution_id]
            stream.close()
            del self._streams[execution_id]

    async def _cleanup_stream_after_delay(self, execution_id: str, delay: int = 3600) -> None:
        """Clean up a stream after a delay (default 1 hour)."""
        await asyncio.sleep(delay)
        self.remove_stream(execution_id)

    def list_active_streams(self) -> list[str]:
        """List all active execution IDs."""
        return list(self._streams.keys())

    def close_all(self) -> None:
        """Close all event streams."""
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()

        # Cancel all cleanup tasks
        for task in self._cleanup_tasks:
            task.cancel()
        self._cleanup_tasks.clear()


class PerformanceEventCollector:
    """Collects and aggregates performance events."""

    def __init__(self, stream: EventStream):
        self.stream = stream
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._costs: dict[str, float] = defaultdict(float)
        self._durations: dict[str, int] = {}
        self._start_times: dict[str, float] = {}

    async def collect_performance_events(self) -> None:
        """Collect performance events from the stream."""
        async for event in self.stream.subscribe([
            ExecutionEventType.AGENT_STARTED,
            ExecutionEventType.AGENT_COMPLETED,
            ExecutionEventType.AGENT_FAILED,
            ExecutionEventType.PERFORMANCE_METRIC,
            ExecutionEventType.COST_UPDATE,
        ]):
            await self._process_performance_event(event)

    async def _process_performance_event(self, event: ExecutionEvent) -> None:
        """Process a single performance event."""
        if event.event_type == ExecutionEventType.AGENT_STARTED:
            self._start_times[event.agent_name] = event.timestamp.timestamp()

        elif event.event_type in [ExecutionEventType.AGENT_COMPLETED, ExecutionEventType.AGENT_FAILED]:
            if event.agent_name and event.agent_name in self._start_times:
                duration = event.data.get("duration_ms", 0)
                self._durations[event.agent_name] = duration

                cost = event.data.get("cost_usd", 0.0)
                if cost:
                    self._costs[event.agent_name] = cost

        elif event.event_type == ExecutionEventType.PERFORMANCE_METRIC:
            metric_name = event.data.get("metric_name")
            metric_value = event.data.get("metric_value")
            if metric_name and metric_value is not None:
                self._metrics[metric_name].append(metric_value)

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        return {
            "total_duration_ms": sum(self._durations.values()),
            "total_cost_usd": sum(self._costs.values()),
            "agent_durations": dict(self._durations),
            "agent_costs": dict(self._costs),
            "metrics": {
                name: {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                }
                for name, values in self._metrics.items()
            },
        }


# Global event stream manager
_global_stream_manager = EventStreamManager()

def get_stream_manager() -> EventStreamManager:
    """Get the global event stream manager."""
    return _global_stream_manager
