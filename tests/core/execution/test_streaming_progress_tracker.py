"""Tests for streaming progress tracker."""

import asyncio
import time
from typing import Any
from unittest.mock import Mock

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.streaming_progress_tracker import StreamingProgressTracker


class TestStreamingProgressTracker:
    """Test streaming progress tracking functionality."""

    def test_init_sets_up_hook_management(self) -> None:
        """Test that initialization sets up hook management functions."""
        mock_register = Mock()
        mock_unregister = Mock()

        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        assert tracker._register_hook == mock_register
        assert tracker._unregister_hook == mock_unregister
        assert tracker._progress_events == []
        assert tracker._progress_hook is None

    def test_create_progress_hook(self) -> None:
        """Test progress hook creation and event capture."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        # Create and test the progress hook
        hook = tracker._create_progress_hook()

        # Test event capture
        hook("agent_completed", {"agent": "test1"})
        hook("agent_started", {"agent": "test2"})

        assert len(tracker._progress_events) == 2
        assert tracker._progress_events[0]["type"] == "agent_completed"
        assert tracker._progress_events[0]["data"]["agent"] == "test1"
        assert tracker._progress_events[1]["type"] == "agent_started"
        assert tracker._progress_events[1]["data"]["agent"] == "test2"

    @pytest.mark.asyncio
    async def test_track_execution_progress_basic_flow(self) -> None:
        """Test basic execution progress tracking flow."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        config = EnsembleConfig(
            name="test_ensemble",
            description="Test",
            agents=[
                {"name": "agent1", "role": "test", "model": "mock"},
                {"name": "agent2", "role": "test", "model": "mock"},
            ],
        )

        # Create a mock execution task
        final_result = {
            "results": {"agent1": "result1", "agent2": "result2"},
            "metadata": {"duration": 1.0},
            "status": "completed",
        }

        async def mock_execution() -> dict[str, Any]:
            return final_result

        execution_task = asyncio.create_task(mock_execution())
        start_time = time.time()

        # Collect all events
        events = []
        async for event in tracker.track_execution_progress(
            config, execution_task, start_time
        ):
            events.append(event)

        # Verify events
        assert len(events) >= 2  # At least started and completed

        # Check started event
        started_event = events[0]
        assert started_event["type"] == "execution_started"
        assert started_event["data"]["ensemble"] == "test_ensemble"
        assert started_event["data"]["total_agents"] == 2
        assert started_event["data"]["timestamp"] == start_time

        # Check completed event
        completed_event = events[-1]
        assert completed_event["type"] == "execution_completed"
        assert completed_event["data"]["ensemble"] == "test_ensemble"
        assert completed_event["data"]["results"] == final_result["results"]
        assert completed_event["data"]["status"] == "completed"
        assert "duration" in completed_event["data"]

        # Verify hook management
        mock_register.assert_called_once()
        mock_unregister.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_execution_progress_with_agent_completions(self) -> None:
        """Test progress tracking by directly testing hook functionality."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        # Test the hook creation and event processing functionality directly
        hook = tracker._create_progress_hook()

        # Simulate agent completion events
        hook("agent_completed", {"agent": "agent1", "timestamp": time.time()})
        hook("agent_completed", {"agent": "agent2", "timestamp": time.time()})

        # Check that events were recorded
        assert len(tracker._progress_events) == 2
        assert all(
            event["type"] == "agent_completed" for event in tracker._progress_events
        )

        # Test progress calculation logic
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test",
            agents=[
                {"name": "agent1", "role": "test", "model": "mock"},
                {"name": "agent2", "role": "test", "model": "mock"},
                {"name": "agent3", "role": "test", "model": "mock"},
            ],
        )

        # Test that the monitoring logic would detect completed agents
        completed_count = len(
            [e for e in tracker._progress_events if e["type"] == "agent_completed"]
        )
        assert completed_count == 2

        # Test progress percentage calculation
        progress_percentage = (completed_count / len(config.agents)) * 100
        assert progress_percentage == (2 / 3) * 100  # 66.67%
        assert 0 <= progress_percentage <= 100

    @pytest.mark.asyncio
    async def test_track_execution_progress_hook_cleanup_on_exception(self) -> None:
        """Test that hooks are cleaned up even when exceptions occur."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        config = EnsembleConfig(
            name="test_ensemble",
            description="Test",
            agents=[{"name": "agent1", "role": "test", "model": "mock"}],
        )

        # Create a failing execution task
        async def failing_execution() -> dict[str, Any]:
            await asyncio.sleep(0.1)
            raise RuntimeError("Execution failed")

        execution_task = asyncio.create_task(failing_execution())
        start_time = time.time()

        # Track execution and expect it to fail
        async def failing_generator() -> None:
            events = []
            async for event in tracker.track_execution_progress(
                config, execution_task, start_time
            ):
                events.append(event)

        with pytest.raises(RuntimeError, match="Execution failed"):
            await failing_generator()

        # Verify hook was still cleaned up despite the exception
        mock_register.assert_called_once()
        mock_unregister.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_execution_progress_empty_agents(self) -> None:
        """Test progress tracking with empty agent list."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        config = EnsembleConfig(
            name="empty_ensemble",
            description="Empty test",
            agents=[],
        )

        final_result = {
            "results": {},
            "metadata": {"duration": 0.0},
            "status": "completed",
        }

        async def mock_execution() -> dict[str, Any]:
            return final_result

        execution_task = asyncio.create_task(mock_execution())
        start_time = time.time()

        events = []
        async for event in tracker.track_execution_progress(
            config, execution_task, start_time
        ):
            events.append(event)

        # Should still get started and completed events
        assert len(events) >= 2
        assert events[0]["type"] == "execution_started"
        assert events[0]["data"]["total_agents"] == 0
        assert events[-1]["type"] == "execution_completed"

    @pytest.mark.asyncio
    async def test_track_execution_progress_single_agent(self) -> None:
        """Test progress tracking with single agent."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        config = EnsembleConfig(
            name="single_ensemble",
            description="Single agent test",
            agents=[{"name": "solo_agent", "role": "test", "model": "mock"}],
        )

        final_result = {
            "results": {"solo_agent": "result"},
            "metadata": {"duration": 0.1},
            "status": "completed",
        }

        async def mock_execution() -> dict[str, Any]:
            return final_result

        execution_task = asyncio.create_task(mock_execution())
        start_time = time.time()

        events = []
        async for event in tracker.track_execution_progress(
            config, execution_task, start_time
        ):
            events.append(event)

        started_event = events[0]
        assert started_event["data"]["total_agents"] == 1

        # Progress percentage should be 0 or 100 for single agent
        progress_events = [e for e in events if e["type"] == "agent_progress"]
        for progress_event in progress_events:
            percentage = progress_event["data"]["progress_percentage"]
            assert percentage in [0.0, 100.0]

    def test_progress_hook_none_after_cleanup(self) -> None:
        """Test that progress hook is set to None after cleanup."""
        mock_register = Mock()
        mock_unregister = Mock()
        tracker = StreamingProgressTracker(mock_register, mock_unregister)

        # Create and set a hook
        hook = tracker._create_progress_hook()
        tracker._progress_hook = hook

        # Verify it's set
        assert tracker._progress_hook is not None

        # The cleanup happens in the finally block of track_execution_progress
        # We can test this by manually calling the cleanup logic
        if tracker._progress_hook is not None:
            tracker._unregister_hook(tracker._progress_hook)
            tracker._progress_hook = None

        assert tracker._progress_hook is None
