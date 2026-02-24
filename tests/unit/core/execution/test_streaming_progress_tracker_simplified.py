"""Tests for simplified streaming progress tracker (Phase 5)."""

import asyncio
import time
from typing import Any

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.monitoring.streaming_progress_tracker import (
    StreamingProgressTracker,
)
from llm_orc.schemas.agent_config import LlmAgentConfig


class TestStreamingProgressTrackerSimplified:
    """Test simplified streaming progress tracking functionality."""

    def test_init_simple(self) -> None:
        """Test that initialization requires no parameters."""
        StreamingProgressTracker()
        # No attributes to check - simplified design

    @pytest.mark.asyncio
    async def test_track_execution_progress_basic_flow(self) -> None:
        """Test basic execution progress tracking flow."""
        tracker = StreamingProgressTracker()

        config = EnsembleConfig(
            name="test_ensemble",
            description="Test",
            agents=[
                LlmAgentConfig(name="agent1", model_profile="mock"),
                LlmAgentConfig(name="agent2", model_profile="mock"),
            ],
        )

        # Create a mock execution task
        final_result = {
            "results": {"agent1": "result1", "agent2": "result2"},
            "metadata": {"duration": 1.0},
            "status": "completed",
        }

        async def mock_execution() -> dict[str, Any]:
            await asyncio.sleep(0.01)  # Small delay to make it realistic
            return final_result

        execution_task = asyncio.create_task(mock_execution())
        start_time = time.time()

        # Collect all events
        events = []
        async for event in tracker.track_execution_progress(
            config, execution_task, start_time
        ):
            events.append(event)

        # Verify events (now includes progress event)
        assert len(events) == 3  # Started, progress, and completed

        # Check started event
        started_event = events[0]
        assert started_event["type"] == "execution_started"
        assert started_event["data"]["ensemble"] == "test_ensemble"
        assert started_event["data"]["total_agents"] == 2
        assert started_event["data"]["timestamp"] == start_time

        # Check progress event
        progress_event = events[1]
        assert progress_event["type"] == "execution_progress"
        assert progress_event["data"]["ensemble"] == "test_ensemble"
        assert "elapsed" in progress_event["data"]

        # Check completed event
        completed_event = events[2]
        assert completed_event["type"] == "execution_completed"
        assert completed_event["data"]["ensemble"] == "test_ensemble"
        assert completed_event["data"]["results"] == final_result["results"]
        assert completed_event["data"]["status"] == "completed"
        assert "duration" in completed_event["data"]

    @pytest.mark.asyncio
    async def test_track_execution_progress_exception_handling(self) -> None:
        """Test that exceptions are properly handled with execution_failed event."""
        tracker = StreamingProgressTracker()

        config = EnsembleConfig(
            name="test_ensemble",
            description="Test",
            agents=[LlmAgentConfig(name="agent1", model_profile="mock")],
        )

        # Create a failing execution task
        async def failing_execution() -> dict[str, Any]:
            await asyncio.sleep(0.01)
            raise RuntimeError("Execution failed")

        execution_task = asyncio.create_task(failing_execution())
        start_time = time.time()

        # Collect events
        events = []
        async for event in tracker.track_execution_progress(
            config, execution_task, start_time
        ):
            events.append(event)

        # Should get started, progress, and failed events
        assert len(events) == 3

        # Check started event
        started_event = events[0]
        assert started_event["type"] == "execution_started"
        assert started_event["data"]["ensemble"] == "test_ensemble"

        # Check progress event
        progress_event = events[1]
        assert progress_event["type"] == "execution_progress"
        assert progress_event["data"]["ensemble"] == "test_ensemble"

        # Check failed event
        failed_event = events[2]
        assert failed_event["type"] == "execution_failed"
        assert failed_event["data"]["ensemble"] == "test_ensemble"
        assert "error" in failed_event["data"]
        assert "Execution failed" in failed_event["data"]["error"]
        assert "duration" in failed_event["data"]

    @pytest.mark.asyncio
    async def test_track_execution_progress_empty_agents(self) -> None:
        """Test progress tracking with empty agent list."""
        tracker = StreamingProgressTracker()

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

        # Should get started, progress, and completed events
        assert len(events) == 3
        assert events[0]["type"] == "execution_started"
        assert events[0]["data"]["total_agents"] == 0
        assert events[1]["type"] == "execution_progress"
        assert events[2]["type"] == "execution_completed"

    @pytest.mark.asyncio
    async def test_track_execution_progress_single_agent(self) -> None:
        """Test progress tracking with single agent."""
        tracker = StreamingProgressTracker()

        config = EnsembleConfig(
            name="single_ensemble",
            description="Single agent test",
            agents=[LlmAgentConfig(name="solo_agent", model_profile="mock")],
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

        # Should get started, progress, and completed events
        assert len(events) == 3
        assert events[0]["type"] == "execution_started"
        assert events[0]["data"]["total_agents"] == 1
        assert events[1]["type"] == "execution_progress"
        assert events[2]["type"] == "execution_completed"
        assert events[2]["data"]["results"] == {"solo_agent": "result"}
