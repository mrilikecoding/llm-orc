"""Streaming progress tracking for ensemble execution."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleConfig


class StreamingProgressTracker:
    """Tracks and yields progress events during ensemble execution.

    Phase 5: Simplified to focus only on execution progress tracking.
    Performance events now flow through the unified event queue.
    """

    def __init__(self) -> None:
        """Initialize progress tracker."""
        pass

    async def track_execution_progress(
        self,
        config: EnsembleConfig,
        execution_task: asyncio.Task[dict[str, Any]],
        start_time: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Track progress and yield events during execution.

        Phase 5: Simplified progress tracking - no hooks, no event forwarding.
        Only yields execution lifecycle events: started, progress, completed.
        All other events (fallbacks, performance) flow through unified queue.
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

        # Monitor execution task for completion with periodic progress checks
        try:
            # Poll execution task completion with small intervals
            # This ensures frequent queue checks in _merge_streaming_events
            while not execution_task.done():
                await asyncio.sleep(0.1)  # Check every 100ms

                # Yield periodic progress event to trigger queue processing
                yield {
                    "type": "execution_progress",
                    "data": {
                        "ensemble": config.name,
                        "timestamp": time.time(),
                        "elapsed": time.time() - start_time,
                    },
                }

            # Get final result
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

        except Exception as e:
            # Handle execution failures
            yield {
                "type": "execution_failed",
                "data": {
                    "ensemble": config.name,
                    "timestamp": time.time(),
                    "duration": time.time() - start_time,
                    "error": str(e),
                },
            }
