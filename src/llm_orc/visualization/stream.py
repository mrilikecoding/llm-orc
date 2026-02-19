"""Event streaming system for real-time visualization."""

import asyncio
import os
import sys
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator

from .events import ExecutionEvent, ExecutionEventType


class EventStream:
    """Manages event streaming for a single execution."""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self._subscribers: dict[str, list[asyncio.Queue[ExecutionEvent]]] = defaultdict(
            list
        )
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
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue is full, skip this subscriber
                pass

    async def _event_generator(
        self,
        queue: asyncio.Queue[ExecutionEvent],
        subscription_keys: list[str],
    ) -> AsyncIterator[ExecutionEvent]:
        """Yield events from the queue until the stream closes."""
        try:
            while not self._is_closed:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
        finally:
            for key in subscription_keys:
                if queue in self._subscribers[key]:
                    self._subscribers[key].remove(queue)

    def subscribe(
        self,
        event_types: list[ExecutionEventType] | None = None,
        queue_size: int = 100,
    ) -> AsyncIterator[ExecutionEvent]:
        """Subscribe to events of specified types."""
        if self._is_closed:
            raise RuntimeError("EventStream is closed")

        if event_types is None:
            subscription_keys = ["*"]
        else:
            subscription_keys = [event_type.value for event_type in event_types]

        queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue(maxsize=queue_size)

        for key in subscription_keys:
            self._subscribers[key].append(queue)

        return self._event_generator(queue, subscription_keys)

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

    def __init__(self, enable_cleanup_tasks: bool = True) -> None:
        self._streams: dict[str, EventStream] = {}
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._stream_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._enable_cleanup_tasks = enable_cleanup_tasks

    def create_stream(self, execution_id: str | None = None) -> EventStream:
        """Create a new event stream for an execution."""
        if execution_id is None:
            execution_id = str(uuid.uuid4())

        if execution_id in self._streams:
            raise ValueError(f"Stream for execution {execution_id} already exists")

        stream = EventStream(execution_id)
        self._streams[execution_id] = stream

        # Schedule cleanup task if enabled and event loop is running
        if self._enable_cleanup_tasks:
            try:
                loop = asyncio.get_running_loop()
                if loop:
                    cleanup_task = asyncio.create_task(
                        self._cleanup_stream_after_delay(execution_id)
                    )
                    self._cleanup_tasks.add(cleanup_task)
                    self._stream_cleanup_tasks[execution_id] = cleanup_task

                    # Remove completed tasks from set to prevent memory leaks
                    def cleanup_task_done(task: asyncio.Task[None]) -> None:
                        self._cleanup_tasks.discard(task)
                        if execution_id in self._stream_cleanup_tasks:
                            del self._stream_cleanup_tasks[execution_id]

                    cleanup_task.add_done_callback(cleanup_task_done)
            except RuntimeError:
                # No event loop running, skip cleanup task scheduling
                pass

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

            # Cancel the cleanup task if it exists
            if execution_id in self._stream_cleanup_tasks:
                cleanup_task = self._stream_cleanup_tasks[execution_id]
                cleanup_task.cancel()
                self._cleanup_tasks.discard(cleanup_task)
                del self._stream_cleanup_tasks[execution_id]

    async def _cleanup_stream_after_delay(
        self, execution_id: str, delay: int = 3600
    ) -> None:
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
        self._stream_cleanup_tasks.clear()


# Global event stream manager
# Disable cleanup tasks during testing
_is_testing = "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None
_global_stream_manager = EventStreamManager(enable_cleanup_tasks=not _is_testing)


def get_stream_manager() -> EventStreamManager:
    """Get the global event stream manager."""
    return _global_stream_manager
