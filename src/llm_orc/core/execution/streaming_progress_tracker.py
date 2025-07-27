"""Streaming progress tracking for ensemble execution."""

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleConfig


class StreamingProgressTracker:
    """Tracks and yields progress events during ensemble execution."""

    def __init__(
        self,
        hook_registrar: Callable[[Callable[[str, dict[str, Any]], None]], None],
        hook_unregistrar: Callable[[Callable[[str, dict[str, Any]], None]], None],
    ) -> None:
        """Initialize progress tracker with hook management functions."""
        self._register_hook = hook_registrar
        self._unregister_hook = hook_unregistrar
        self._progress_events: list[dict[str, Any]] = []
        self._progress_hook: Callable[[str, dict[str, Any]], None] | None = None

    def _create_progress_hook(self) -> Callable[[str, dict[str, Any]], None]:
        """Create progress hook to capture streaming events."""

        def progress_hook(event_type: str, data: dict[str, Any]) -> None:
            self._progress_events.append({"type": event_type, "data": data})

        return progress_hook

    async def track_execution_progress(
        self,
        config: EnsembleConfig,
        execution_task: asyncio.Task[dict[str, Any]],
        start_time: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Track progress and yield events during execution.

        Yields progress events: execution_started, agent_progress, execution_completed.
        """
        # Emit execution started event
        yield {
            "type": "execution_started",
            "data": {
                "ensemble": config.name,
                "timestamp": start_time,
                "total_agents": len(config.agents),
            },
        }

        # Set up progress tracking
        self._progress_events = []
        self._progress_hook = self._create_progress_hook()
        self._register_hook(self._progress_hook)

        try:
            # Monitor progress while execution runs
            last_progress_count = 0
            last_started_count = 0
            while not execution_task.done():
                # Check for new progress events
                completed_count = len(
                    [e for e in self._progress_events if e["type"] == "agent_completed"]
                )
                started_count = len(
                    [e for e in self._progress_events if e["type"] == "agent_started"]
                )

                # Emit progress update if we have new completions or new starts
                if completed_count > last_progress_count or started_count > last_started_count:
                    # Get which agents have started and completed
                    started_agents = [
                        e["data"]["agent_name"] for e in self._progress_events 
                        if e["type"] == "agent_started"
                    ]
                    completed_agents = [
                        e["data"]["agent_name"] for e in self._progress_events 
                        if e["type"] == "agent_completed"
                    ]
                    
                    yield {
                        "type": "agent_progress",
                        "data": {
                            "completed_agents": completed_count,
                            "started_agents": started_count,
                            "total_agents": len(config.agents),
                            "progress_percentage": (
                                completed_count / len(config.agents) * 100
                            ),
                            "timestamp": time.time(),
                            "started_agent_names": started_agents,
                            "completed_agent_names": completed_agents,
                        },
                    }
                    last_progress_count = completed_count
                    last_started_count = started_count

                # Small delay to avoid busy waiting
                await asyncio.sleep(0.05)

            # Get final results
            final_result = await execution_task

            # Emit execution completed event with full results
            yield {
                "type": "execution_completed",
                "data": {
                    "ensemble": config.name,
                    "timestamp": time.time(),
                    "duration": time.time() - start_time,
                    "results": final_result["results"],
                    "metadata": final_result["metadata"],
                    "status": final_result["status"],
                },
            }

        finally:
            # Clean up the progress hook
            if self._progress_hook is not None:
                self._unregister_hook(self._progress_hook)
                self._progress_hook = None
