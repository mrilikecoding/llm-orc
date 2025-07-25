"""Comprehensive tests for terminal visualizer."""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress

from llm_orc.visualization.config import VisualizationConfig
from llm_orc.visualization.events import ExecutionEvent, ExecutionEventType
from llm_orc.visualization.stream import EventStream
from llm_orc.visualization.terminal.visualizer import TerminalVisualizer


class TestTerminalVisualizer:
    """Test TerminalVisualizer class comprehensively."""

    def test_init_with_config(self) -> None:
        """Test visualizer initialization with config."""
        config = VisualizationConfig()
        visualizer = TerminalVisualizer(config)

        assert visualizer.config == config
        assert isinstance(visualizer.console, Console)
        assert visualizer.live_display is None
        assert isinstance(visualizer.progress, Progress)

    def test_init_with_default_config(self) -> None:
        """Test visualizer initialization with default config."""
        visualizer = TerminalVisualizer()

        assert isinstance(visualizer.config, VisualizationConfig)
        assert isinstance(visualizer.console, Console)
        assert visualizer.live_display is None

    def test_init_state_structure(self) -> None:
        """Test initial execution state structure."""
        visualizer = TerminalVisualizer()

        expected_state: dict[str, Any] = {
            "ensemble_name": "",
            "execution_id": "",
            "status": "starting",
            "start_time": None,
            "total_agents": 0,
            "completed_agents": 0,
            "failed_agents": 0,
            "agents": {},
            "performance": {
                "total_cost": 0.0,
                "total_duration": 0,
                "memory_usage": 0,
            },
            "live_results": [],
        }

        assert visualizer.execution_state == expected_state
        assert visualizer.overall_task_id is None
        assert visualizer.agent_task_ids == {}

    @pytest.mark.asyncio
    @patch("llm_orc.visualization.terminal.visualizer.Live")
    async def test_visualize_execution_success(self, mock_live_class: Mock) -> None:
        """Test successful execution visualization."""
        visualizer = TerminalVisualizer()
        stream = EventStream("test-execution")

        # Mock Live context manager
        mock_live = Mock()
        mock_live_class.return_value.__enter__.return_value = mock_live
        mock_live_class.return_value.__exit__.return_value = None

        # Mock event stream subscription
        events = [
            ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_STARTED,
                timestamp=datetime.now(),
                ensemble_name="test-ensemble",
                execution_id="test-execution",
                data={"total_agents": 2},
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
                timestamp=datetime.now(),
                ensemble_name="test-ensemble",
                execution_id="test-execution",
                data={},
            ),
        ]

        async def mock_subscribe() -> AsyncIterator[ExecutionEvent]:
            for event in events:
                yield event

        with patch.object(stream, "subscribe", return_value=mock_subscribe()):
            await visualizer.visualize_execution(stream)

        # Verify Live was used correctly
        mock_live_class.assert_called_once()
        assert mock_live.update.call_count >= len(events)

    @pytest.mark.asyncio
    @patch("llm_orc.visualization.terminal.visualizer.Live")
    async def test_visualize_execution_exception_handling(
        self, mock_live_class: Mock
    ) -> None:
        """Test visualization with exception handling."""
        visualizer = TerminalVisualizer()
        stream = EventStream("test-execution")

        # Mock Live to raise exception
        mock_live_class.side_effect = Exception("Live display error")

        # Mock console.print
        with patch.object(visualizer.console, "print") as mock_print:
            await visualizer.visualize_execution(stream)

            mock_print.assert_called_once_with(
                "❌ Visualization error: Live display error"
            )

    @pytest.mark.asyncio
    async def test_process_event_ensemble_started(self) -> None:
        """Test processing ensemble started event."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={"total_agents": 3},
        )

        await visualizer._process_event(event)

        assert visualizer.execution_state["ensemble_name"] == "test-ensemble"
        assert visualizer.execution_state["execution_id"] == "test-execution"
        assert visualizer.execution_state["status"] == "running"
        assert visualizer.execution_state["total_agents"] == 3

    @pytest.mark.asyncio
    async def test_process_event_agent_started(self) -> None:
        """Test processing agent started event."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={"model": "claude-3-sonnet", "depends_on": ["agent1"]},
        )

        await visualizer._process_event(event)

        agent_info = visualizer.execution_state["agents"]["test-agent"]
        assert agent_info["name"] == "test-agent"
        assert agent_info["status"] == "running"
        assert agent_info["model"] == "claude-3-sonnet"
        assert agent_info["dependencies"] == ["agent1"]
        assert agent_info["progress"] == 0.0

    @pytest.mark.asyncio
    async def test_process_event_agent_progress(self) -> None:
        """Test processing agent progress event."""
        visualizer = TerminalVisualizer()

        # First start an agent
        start_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={},
        )
        await visualizer._process_event(start_event)

        # Then send progress event
        progress_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_PROGRESS,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={"progress_percentage": 0.75, "message": "Processing..."},
        )

        await visualizer._process_event(progress_event)

        agent_info = visualizer.execution_state["agents"]["test-agent"]
        assert agent_info["progress"] == 0.75

    @pytest.mark.asyncio
    async def test_process_event_agent_completed(self) -> None:
        """Test processing agent completed event."""
        visualizer = TerminalVisualizer()

        # First start an agent
        start_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={},
        )
        await visualizer._process_event(start_event)

        # Then complete the agent
        completed_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={
                "result": "Agent completed successfully",
                "duration_ms": 5000,
                "cost_usd": 0.05,
            },
        )

        await visualizer._process_event(completed_event)

        agent_info = visualizer.execution_state["agents"]["test-agent"]
        assert agent_info["status"] == "completed"
        assert agent_info["result"] == "Agent completed successfully"
        assert agent_info["duration"] == 5000
        assert agent_info["cost"] == 0.05
        assert visualizer.execution_state["completed_agents"] == 1

    @pytest.mark.asyncio
    async def test_process_event_agent_failed(self) -> None:
        """Test processing agent failed event."""
        visualizer = TerminalVisualizer()

        # First start an agent
        start_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={},
        )
        await visualizer._process_event(start_event)

        # Then fail the agent
        failed_event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={
                "error": "Agent failed with error",
                "duration_ms": 2000,
                "cost_usd": 0.01,
            },
        )

        await visualizer._process_event(failed_event)

        agent_info = visualizer.execution_state["agents"]["test-agent"]
        assert agent_info["status"] == "failed"
        assert agent_info["error"] == "Agent failed with error"
        assert agent_info["duration"] == 2000
        assert visualizer.execution_state["failed_agents"] == 1

    @pytest.mark.asyncio
    async def test_process_event_ensemble_completed(self) -> None:
        """Test processing ensemble completed event."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={"total_duration": 10000, "total_cost": 0.15},
        )

        await visualizer._process_event(event)

        assert visualizer.execution_state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_process_event_performance_metric(self) -> None:
        """Test processing performance metric event."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={"metric_name": "memory_usage", "metric_value": 1024},
        )

        await visualizer._process_event(event)

        assert visualizer.execution_state["performance"]["memory_usage"] == 1024

    def test_create_layout(self) -> None:
        """Test layout creation."""
        visualizer = TerminalVisualizer()

        layout = visualizer.create_layout()

        assert isinstance(layout, Layout)
        # Layout contains different sections based on compact_mode
        # Just verify it's a valid Layout object

    def test_create_header(self) -> None:
        """Test header creation."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state.update(
            {
                "ensemble_name": "test-ensemble",
                "execution_id": "test-123",
                "status": "running",
            }
        )

        header = visualizer.create_header()

        assert isinstance(header, Panel)

    def test_create_dependency_graph_empty(self) -> None:
        """Test dependency graph creation with no agents."""
        visualizer = TerminalVisualizer()

        graph = visualizer.create_dependency_graph()

        assert isinstance(graph, Panel)

    def test_create_dependency_graph_with_agents(self) -> None:
        """Test dependency graph creation with agents."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["agents"] = {
            "agent1": {
                "name": "agent1",
                "dependencies": [],
                "status": "completed",
                "model": "claude-3-sonnet",
            },
            "agent2": {
                "name": "agent2",
                "dependencies": ["agent1"],
                "status": "running",
                "model": "claude-3-haiku",
            },
        }

        graph = visualizer.create_dependency_graph()

        assert isinstance(graph, Panel)

    def test_group_agents_by_level(self) -> None:
        """Test agent grouping by dependency level."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["agents"] = {
            "agent1": {"name": "agent1", "dependencies": []},
            "agent2": {"name": "agent2", "dependencies": ["agent1"]},
            "agent3": {"name": "agent3", "dependencies": ["agent1"]},
            "agent4": {"name": "agent4", "dependencies": ["agent2", "agent3"]},
        }

        levels = visualizer._group_agents_by_level()

        assert 0 in levels
        assert len(levels[0]) == 1  # agent1
        assert levels[0][0]["name"] == "agent1"

        assert 1 in levels
        assert len(levels[1]) == 2  # agent2, agent3

        assert 2 in levels
        assert len(levels[2]) == 1  # agent4
        assert levels[2][0]["name"] == "agent4"

    def test_calculate_agent_level_no_dependencies(self) -> None:
        """Test agent level calculation with no dependencies."""
        visualizer = TerminalVisualizer()

        level = visualizer._calculate_agent_level("agent1", [])

        assert level == 0

    def test_calculate_agent_level_with_dependencies(self) -> None:
        """Test agent level calculation with dependencies."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["agents"] = {
            "agent1": {"dependencies": []},
            "agent2": {"dependencies": ["agent1"]},
        }

        level = visualizer._calculate_agent_level("agent3", ["agent1", "agent2"])

        assert level == 2  # max(0, 1) + 1

    def test_create_progress_section(self) -> None:
        """Test progress section creation."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state.update(
            {
                "total_agents": 5,
                "completed_agents": 3,
                "failed_agents": 1,
            }
        )

        progress = visualizer.create_progress_section()

        assert isinstance(progress, Panel)

    def test_create_agents_section_empty(self) -> None:
        """Test agents section creation with no agents."""
        visualizer = TerminalVisualizer()

        agents = visualizer.create_agents_section()

        assert isinstance(agents, Panel)

    def test_create_agents_section_with_agents(self) -> None:
        """Test agents section creation with agents."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["agents"] = {
            "agent1": {
                "name": "agent1",
                "status": "completed",
                "model": "claude-3-sonnet",
                "progress": 1.0,
                "duration": 5000,
                "cost": 0.05,
                "result": "Success",
                "error": None,
            },
            "agent2": {
                "name": "agent2",
                "status": "failed",
                "model": "claude-3-haiku",
                "progress": 0.5,
                "duration": 2000,
                "cost": 0.01,
                "result": None,
                "error": "Timeout error",
            },
        }

        agents = visualizer.create_agents_section()

        assert isinstance(agents, Panel)

    def test_create_results_section_empty(self) -> None:
        """Test results section creation with no results."""
        visualizer = TerminalVisualizer()

        results = visualizer.create_results_section()

        assert isinstance(results, Panel)

    def test_create_results_section_with_results(self) -> None:
        """Test results section creation with results."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["live_results"] = [
            "Agent1 completed: Success",
            "Agent2 failed: Error occurred",
        ]

        results = visualizer.create_results_section()

        assert isinstance(results, Panel)

    def test_create_metrics_section(self) -> None:
        """Test metrics section creation."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["performance"].update(
            {
                "total_cost": 0.15,
                "total_duration": 10000,
                "memory_usage": 1024,
            }
        )

        metrics = visualizer.create_metrics_section()

        assert isinstance(metrics, Panel)

    def test_is_execution_complete_ensemble_completed(self) -> None:
        """Test execution completion detection with ensemble completed."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={},
        )

        assert visualizer._is_execution_complete(event) is True

    def test_is_execution_complete_not_complete(self) -> None:
        """Test execution completion detection with non-completion event."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={},
        )

        assert visualizer._is_execution_complete(event) is False

    @patch("builtins.print")
    def test_print_summary(self, mock_print: Mock) -> None:
        """Test summary printing."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state.update(
            {
                "ensemble_name": "test-ensemble",
                "execution_id": "test-123",
                "status": "completed",
                "total_agents": 3,
                "completed_agents": 2,
                "failed_agents": 1,
                "performance": {
                    "total_cost": 0.15,
                    "total_duration": 10000,
                },
            }
        )

        with patch.object(visualizer.console, "print") as mock_console_print:
            visualizer.print_summary()

            # Verify console.print was called multiple times
            assert mock_console_print.call_count > 0

    def test_print_simple_progress_ensemble_started(self) -> None:
        """Test simple progress printing for ensemble started."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={"total_agents": 3},
        )

        with patch.object(visualizer.console, "print") as mock_print:
            visualizer.print_simple_progress(event)
            mock_print.assert_called_once()

    def test_print_simple_progress_agent_started(self) -> None:
        """Test simple progress printing for agent started."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={"model": "claude-3-sonnet"},
        )

        with patch.object(visualizer.console, "print") as mock_print:
            visualizer.print_simple_progress(event)
            mock_print.assert_called_once()

    def test_print_simple_progress_agent_completed(self) -> None:
        """Test simple progress printing for agent completed."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={"duration_ms": 5000, "cost_usd": 0.05},
        )

        with patch.object(visualizer.console, "print") as mock_print:
            visualizer.print_simple_progress(event)
            mock_print.assert_called_once()

    def test_print_simple_progress_agent_failed(self) -> None:
        """Test simple progress printing for agent failed."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="test-agent",
            data={"error": "Test error"},
        )

        with patch.object(visualizer.console, "print") as mock_print:
            visualizer.print_simple_progress(event)
            mock_print.assert_called_once()

    def test_print_simple_progress_ensemble_completed(self) -> None:
        """Test simple progress printing for ensemble completed."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={"total_duration": 10000, "total_cost": 0.15},
        )

        with patch.object(visualizer.console, "print") as mock_print:
            visualizer.print_simple_progress(event)
            mock_print.assert_called_once()

    def test_print_simple_progress_unknown_event(self) -> None:
        """Test simple progress printing for unknown event type."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            data={"metric_name": "test", "metric_value": 1.0},
        )

        with patch.object(visualizer.console, "print") as mock_print:
            visualizer.print_simple_progress(event)
            # Should not print anything for unknown event types
            mock_print.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_agent_started_without_agent_name(self) -> None:
        """Test agent started handling without agent name."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name=None,  # No agent name
            data={},
        )

        await visualizer._handle_agent_started(event)

        # Should not add any agent to the state
        assert len(visualizer.execution_state["agents"]) == 0

    @pytest.mark.asyncio
    async def test_handle_agent_progress_unknown_agent(self) -> None:
        """Test agent progress handling for unknown agent."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_PROGRESS,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="unknown-agent",
            data={"progress": 0.5},
        )

        await visualizer._handle_agent_progress(event)

        # Should not crash, but also should not add agent
        assert "unknown-agent" not in visualizer.execution_state["agents"]

    @pytest.mark.asyncio
    async def test_handle_agent_completed_unknown_agent(self) -> None:
        """Test agent completed handling for unknown agent."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="unknown-agent",
            data={"result": "Success"},
        )

        await visualizer._handle_agent_completed(event)

        # Should not crash or affect state
        assert visualizer.execution_state["completed_agents"] == 0

    @pytest.mark.asyncio
    async def test_handle_agent_failed_unknown_agent(self) -> None:
        """Test agent failed handling for unknown agent."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.now(),
            ensemble_name="test-ensemble",
            execution_id="test-execution",
            agent_name="unknown-agent",
            data={"error": "Test error"},
        )

        await visualizer._handle_agent_failed(event)

        # Should not crash or affect state
        assert visualizer.execution_state["failed_agents"] == 0

    @pytest.mark.asyncio
    async def test_handle_ensemble_started_with_none_values(self) -> None:
        """Test ensemble started handling with None values."""
        visualizer = TerminalVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=datetime.now(),
            ensemble_name=None,
            execution_id=None,
            data={},
        )

        await visualizer._handle_ensemble_started(event)

        assert visualizer.execution_state["ensemble_name"] == "Unknown"
        assert visualizer.execution_state["execution_id"] == "Unknown"
        assert visualizer.execution_state["total_agents"] == 0

    def test_calculate_agent_level_unknown_dependency(self) -> None:
        """Test agent level calculation with unknown dependency."""
        visualizer = TerminalVisualizer()
        visualizer.execution_state["agents"] = {
            "agent1": {"dependencies": []},
        }

        # Agent with dependency on unknown agent
        level = visualizer._calculate_agent_level("agent2", ["unknown-agent"])

        # Should default to level 0 for unknown dependencies
        assert level == 1

    def test_dependency_calculation_circular_avoidance(self) -> None:
        """Test that circular dependencies don't cause infinite loops."""
        visualizer = TerminalVisualizer()

        # This tests the safety of the level calculation
        # Even with complex dependencies, it should not hang
        # Self-reference
        level = visualizer._calculate_agent_level("agent1", ["agent1"])

        # Should handle self-reference gracefully
        assert level == 1
