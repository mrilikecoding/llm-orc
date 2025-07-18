"""Integration between ensemble execution and visualization system."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

from ..ensemble_config import EnsembleConfig
from ..ensemble_execution import EnsembleExecutor
from .config import VisualizationConfig, load_visualization_config
from .events import EventFactory, ExecutionEventType
from .stream import EventStream, get_stream_manager
from .terminal import TerminalVisualizer


class VisualizationIntegratedExecutor(EnsembleExecutor):
    """Ensemble executor with integrated visualization support."""

    def __init__(self, visualization_config: VisualizationConfig | None = None):
        super().__init__()
        self.viz_config = visualization_config or load_visualization_config()
        self.stream_manager = get_stream_manager()
        self.current_stream: EventStream | None = None
        self.current_execution_id: str | None = None

    async def execute_with_visualization(
        self,
        config: EnsembleConfig,
        input_data: str,
        visualization_mode: str | None = None
    ) -> dict[str, Any]:
        """Execute ensemble with integrated visualization."""
        if not self.viz_config.enabled:
            # Fall back to standard execution if visualization is disabled
            return await self.execute(config, input_data)

        # Determine visualization mode
        mode = visualization_mode or self.viz_config.default_mode

        # Create execution ID and event stream
        execution_id = str(uuid.uuid4())
        self.current_execution_id = execution_id
        self.current_stream = self.stream_manager.create_stream(execution_id)

        try:
            # Set up visualization hook
            self.register_performance_hook(self._visualization_hook)

            # Start visualization based on mode
            if mode == "terminal":
                return await self._execute_with_terminal_visualization(config, input_data)
            elif mode == "web":
                return await self._execute_with_web_visualization(config, input_data)
            elif mode == "debug":
                return await self._execute_with_debug_visualization(config, input_data)
            elif mode == "minimal":
                return await self._execute_with_minimal_visualization(config, input_data)
            else:
                # Default to terminal
                return await self._execute_with_terminal_visualization(config, input_data)

        finally:
            # Clean up
            self.current_stream = None
            self.current_execution_id = None

    async def _execute_with_terminal_visualization(
        self, config: EnsembleConfig, input_data: str
    ) -> dict[str, Any]:
        """Execute with terminal visualization."""
        visualizer = TerminalVisualizer(self.viz_config)

        # Start execution and visualization concurrently
        execution_task = asyncio.create_task(
            self._execute_with_events(config, input_data)
        )

        visualization_task = asyncio.create_task(
            visualizer.visualize_execution(self.current_stream)
        )

        try:
            # Wait for execution to complete
            result = await execution_task

            # Cancel visualization task
            visualization_task.cancel()
            try:
                await visualization_task
            except asyncio.CancelledError:
                pass

            # Print summary
            visualizer.print_summary()

            return result

        except Exception as e:
            # Cancel visualization task
            visualization_task.cancel()
            raise e

    async def _execute_with_web_visualization(
        self, config: EnsembleConfig, input_data: str
    ) -> dict[str, Any]:
        """Execute with web dashboard visualization."""
        # TODO: Implement web visualization
        # For now, fall back to terminal
        return await self._execute_with_terminal_visualization(config, input_data)

    async def _execute_with_debug_visualization(
        self, config: EnsembleConfig, input_data: str
    ) -> dict[str, Any]:
        """Execute with debug visualization."""
        # TODO: Implement debug visualization
        # For now, fall back to terminal
        return await self._execute_with_terminal_visualization(config, input_data)

    async def _execute_with_minimal_visualization(
        self, config: EnsembleConfig, input_data: str
    ) -> dict[str, Any]:
        """Execute with minimal visualization."""
        visualizer = TerminalVisualizer(self.viz_config)

        # Execute with simple progress reporting
        result = await self._execute_with_events(config, input_data)

        # Print simple summary
        visualizer.print_summary()

        return result

    async def _execute_with_events(
        self, config: EnsembleConfig, input_data: str
    ) -> dict[str, Any]:
        """Execute ensemble while emitting events."""
        if not self.current_stream:
            raise RuntimeError("No event stream available")

        # Emit ensemble started event
        await self.current_stream.emit(
            EventFactory.ensemble_started(
                ensemble_name=config.name,
                execution_id=self.current_execution_id,
                total_agents=len(config.agents),
            )
        )

        try:
            # Execute using parent method
            result = await self.execute(config, input_data)

            # Emit ensemble completed event
            await self.current_stream.emit(
                EventFactory.ensemble_completed(
                    ensemble_name=config.name,
                    execution_id=self.current_execution_id,
                    result=result,
                    duration_ms=int(time.time() * 1000),
                )
            )

            return result

        except Exception as e:
            # Emit ensemble failed event
            await self.current_stream.emit(
                EventFactory.ensemble_failed(
                    ensemble_name=config.name,
                    execution_id=self.current_execution_id,
                    error=str(e),
                )
            )
            raise e

    def _visualization_hook(self, event_type: str, data: dict[str, Any]) -> None:
        """Hook to convert performance events to visualization events."""
        if not self.current_stream or not self.current_execution_id:
            return

        try:
            # Convert performance events to visualization events
            event = self._convert_performance_event(event_type, data)
            if event:
                # Schedule event emission in a thread-safe way
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.current_stream.emit(event))
                else:
                    # If no loop is running, emit synchronously
                    asyncio.run(self.current_stream.emit(event))
        except Exception:
            # Silently ignore conversion errors
            pass

    def _convert_performance_event(self, event_type: str, data: dict[str, Any]):
        """Convert performance event to visualization event."""
        if event_type == "agent_started":
            return EventFactory.agent_started(
                agent_name=data.get("agent_name", "unknown"),
                ensemble_name="current_ensemble",  # Use a placeholder for now
                execution_id=self.current_execution_id,
                model=data.get("model", "unknown"),
                dependencies=data.get("dependencies", []),
            )

        elif event_type == "agent_completed":
            return EventFactory.agent_completed(
                agent_name=data.get("agent_name", "unknown"),
                ensemble_name="current_ensemble",  # Use a placeholder for now
                execution_id=self.current_execution_id,
                result=data.get("result", ""),
                duration_ms=data.get("duration_ms", 0),
                cost_usd=data.get("cost_usd"),
                tokens_used=data.get("tokens_used"),
            )

        elif event_type == "agent_failed":
            return EventFactory.agent_failed(
                agent_name=data.get("agent_name", "unknown"),
                ensemble_name="current_ensemble",  # Use a placeholder for now
                execution_id=self.current_execution_id,
                error=data.get("error", "Unknown error"),
                duration_ms=data.get("duration_ms", 0),
            )

        return None


# Add missing EventFactory methods
def _add_missing_event_factory_methods():
    """Add missing methods to EventFactory."""

    @staticmethod
    def ensemble_completed(
        ensemble_name: str,
        execution_id: str,
        result: dict[str, Any],
        duration_ms: int,
    ):
        """Create ensemble completed event."""
        from .events import ExecutionEvent

        return ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "result": result,
                "duration_ms": duration_ms,
                "status": "completed",
            },
        )

    @staticmethod
    def ensemble_failed(
        ensemble_name: str,
        execution_id: str,
        error: str,
    ):
        """Create ensemble failed event."""
        from .events import ExecutionEvent

        return ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_FAILED,
            timestamp=datetime.now(),
            ensemble_name=ensemble_name,
            execution_id=execution_id,
            data={
                "error": error,
                "status": "failed",
            },
        )

    # Add methods to EventFactory
    EventFactory.ensemble_completed = ensemble_completed
    EventFactory.ensemble_failed = ensemble_failed

# Apply the missing methods
_add_missing_event_factory_methods()
