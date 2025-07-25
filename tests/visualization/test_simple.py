"""Tests for simple horizontal dependency graph visualization."""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.visualization.config import SimpleVisualizationConfig
from llm_orc.visualization.events import ExecutionEvent, ExecutionEventType
from llm_orc.visualization.simple import SimpleVisualizer


class TestSimpleVisualizer:
    """Test the simple visualizer."""

    def test_init_default_config(self) -> None:
        """Test initialization with default configuration."""
        visualizer = SimpleVisualizer()

        assert isinstance(visualizer.config, SimpleVisualizationConfig)
        assert visualizer.console is not None
        assert visualizer.live_display is None

        # Check initial state
        expected_state: dict[str, Any] = {
            "ensemble_name": "",
            "execution_id": "",
            "status": "starting",
            "start_time": None,
            "total_agents": 0,
            "completed_agents": 0,
            "failed_agents": 0,
            "agents": {},
        }
        assert visualizer.execution_state == expected_state

    def test_init_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = SimpleVisualizationConfig(
            show_dependency_graph=False,
            use_colors=False,
            refresh_rate_ms=100,
        )
        visualizer = SimpleVisualizer(config)

        assert visualizer.config is config
        assert not visualizer.config.show_dependency_graph
        assert not visualizer.config.use_colors
        assert visualizer.config.refresh_rate_ms == 100

    @pytest.mark.asyncio
    async def test_handle_ensemble_started(self) -> None:
        """Test handling ensemble started event."""
        visualizer = SimpleVisualizer()
        timestamp = datetime.now()

        # Create event with agents configuration
        agents_config = [
            {"name": "agent1", "depends_on": [], "model_profile": "claude"},
            {"name": "agent2", "depends_on": ["agent1"], "model_profile": "gpt4"},
        ]

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_STARTED,
            timestamp=timestamp,
            data={
                "total_agents": 2,
                "agents_config": agents_config,
            },
            ensemble_name="test_ensemble",
            execution_id="exec_123",
        )

        await visualizer._handle_ensemble_started(event)

        # Verify state updates
        assert visualizer.execution_state["ensemble_name"] == "test_ensemble"
        assert visualizer.execution_state["execution_id"] == "exec_123"
        assert visualizer.execution_state["status"] == "running"
        assert visualizer.execution_state["start_time"] == timestamp
        assert visualizer.execution_state["total_agents"] == 2

        # Verify agents are populated
        assert len(visualizer.execution_state["agents"]) == 2

        agent1 = visualizer.execution_state["agents"]["agent1"]
        assert agent1["name"] == "agent1"
        assert agent1["status"] == "pending"
        assert agent1["dependencies"] == []
        assert agent1["model_profile"] == "claude"

        agent2 = visualizer.execution_state["agents"]["agent2"]
        assert agent2["name"] == "agent2"
        assert agent2["status"] == "pending"
        assert agent2["dependencies"] == ["agent1"]
        assert agent2["model_profile"] == "gpt4"

    @pytest.mark.asyncio
    async def test_handle_agent_started(self) -> None:
        """Test handling agent started event."""
        visualizer = SimpleVisualizer()
        timestamp = datetime.now()

        # Set up initial state with an agent
        visualizer.execution_state["agents"]["agent1"] = {
            "name": "agent1",
            "status": "pending",
            "dependencies": [],
        }

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=timestamp,
            data={"model": "claude-3-sonnet"},
            agent_name="agent1",
        )

        await visualizer._handle_agent_started(event)

        # Verify agent state updated
        agent = visualizer.execution_state["agents"]["agent1"]
        assert agent["status"] == "running"
        assert agent["model"] == "claude-3-sonnet"
        assert agent["start_time"] == timestamp

    @pytest.mark.asyncio
    async def test_handle_agent_started_unknown_agent(self) -> None:
        """Test handling agent started event for unknown agent."""
        visualizer = SimpleVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            data={},
            agent_name="unknown_agent",
        )

        # Should not raise exception, should handle gracefully
        await visualizer._handle_agent_started(event)

        # Agent should not be added to state
        assert "unknown_agent" not in visualizer.execution_state["agents"]

    @pytest.mark.asyncio
    async def test_handle_agent_completed(self) -> None:
        """Test handling agent completed event."""
        visualizer = SimpleVisualizer()

        # Set up initial state
        visualizer.execution_state["agents"]["agent1"] = {
            "name": "agent1",
            "status": "running",
        }
        visualizer.execution_state["completed_agents"] = 0

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            data={},
            agent_name="agent1",
        )

        await visualizer._handle_agent_completed(event)

        # Verify state updates
        assert visualizer.execution_state["agents"]["agent1"]["status"] == "completed"
        assert visualizer.execution_state["completed_agents"] == 1

    @pytest.mark.asyncio
    async def test_handle_agent_failed(self) -> None:
        """Test handling agent failed event."""
        visualizer = SimpleVisualizer()

        # Set up initial state
        visualizer.execution_state["agents"]["agent1"] = {
            "name": "agent1",
            "status": "running",
        }
        visualizer.execution_state["failed_agents"] = 0

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_FAILED,
            timestamp=datetime.now(),
            data={},
            agent_name="agent1",
        )

        await visualizer._handle_agent_failed(event)

        # Verify state updates
        assert visualizer.execution_state["agents"]["agent1"]["status"] == "failed"
        assert visualizer.execution_state["failed_agents"] == 1

    @pytest.mark.asyncio
    async def test_handle_ensemble_completed(self) -> None:
        """Test handling ensemble completed event."""
        visualizer = SimpleVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            data={},
        )

        await visualizer._handle_ensemble_completed(event)

        assert visualizer.execution_state["status"] == "completed"

    def test_calculate_agent_level_no_dependencies(self) -> None:
        """Test calculating agent level with no dependencies."""
        visualizer = SimpleVisualizer()

        level = visualizer._calculate_agent_level("agent1", [])

        assert level == 0

    def test_calculate_agent_level_with_dependencies(self) -> None:
        """Test calculating agent level with dependencies."""
        visualizer = SimpleVisualizer()

        # Set up agent dependencies
        visualizer.execution_state["agents"] = {
            "agent1": {"dependencies": []},
            "agent2": {"dependencies": ["agent1"]},
            "agent3": {"dependencies": ["agent2"]},
        }

        # Test different levels
        assert visualizer._calculate_agent_level("agent1", []) == 0
        assert visualizer._calculate_agent_level("agent2", ["agent1"]) == 1
        assert visualizer._calculate_agent_level("agent3", ["agent2"]) == 2

    def test_calculate_agent_level_multiple_dependencies(self) -> None:
        """Test calculating agent level with multiple dependencies."""
        visualizer = SimpleVisualizer()

        # Set up complex dependency tree
        visualizer.execution_state["agents"] = {
            "agent1": {"dependencies": []},
            "agent2": {"dependencies": []},
            "agent3": {"dependencies": ["agent1"]},
            "agent4": {"dependencies": ["agent1", "agent2"]},
            "agent5": {"dependencies": ["agent3", "agent4"]},
        }

        # agent5 depends on agent3 (level 1) and agent4 (level 1), so should be level 2
        assert visualizer._calculate_agent_level("agent5", ["agent3", "agent4"]) == 2

    def test_group_agents_by_level(self) -> None:
        """Test grouping agents by dependency level."""
        visualizer = SimpleVisualizer()

        # Set up agents with different dependency levels
        visualizer.execution_state["agents"] = {
            "agent1": {"name": "agent1", "dependencies": []},
            "agent2": {"name": "agent2", "dependencies": []},
            "agent3": {"name": "agent3", "dependencies": ["agent1"]},
            "agent4": {"name": "agent4", "dependencies": ["agent2", "agent3"]},
        }

        grouped = visualizer._group_agents_by_level()

        # Level 0: agent1, agent2
        assert len(grouped[0]) == 2
        level_0_names = {agent["name"] for agent in grouped[0]}
        assert level_0_names == {"agent1", "agent2"}

        # Level 1: agent3
        assert len(grouped[1]) == 1
        assert grouped[1][0]["name"] == "agent3"

        # Level 2: agent4
        assert len(grouped[2]) == 1
        assert grouped[2][0]["name"] == "agent4"

    def test_create_dependency_graph_no_agents(self) -> None:
        """Test creating dependency graph with no agents."""
        visualizer = SimpleVisualizer()

        graph = visualizer._create_dependency_graph()

        assert graph == "Agents loading..."

    def test_create_dependency_graph_with_agents(self) -> None:
        """Test creating dependency graph with agents."""
        visualizer = SimpleVisualizer()

        # Set up agents with different statuses
        visualizer.execution_state["agents"] = {
            "agent1": {"name": "agent1", "dependencies": [], "status": "completed"},
            "agent2": {"name": "agent2", "dependencies": [], "status": "running"},
            "agent3": {
                "name": "agent3",
                "dependencies": ["agent1", "agent2"],
                "status": "pending",
            },
            "agent4": {"name": "agent4", "dependencies": [], "status": "failed"},
        }

        graph = visualizer._create_dependency_graph()

        # Should show agents grouped by level with status indicators
        # Level 0: agent1 (completed ✓), agent2 (running ⠋), agent4 (failed ✗)
        # Level 1: agent3 (pending, dimmed)
        assert "✓" in graph  # completed
        assert "⠋" in graph  # running
        assert "✗" in graph  # failed
        assert "agent1" in graph
        assert "agent2" in graph
        assert "agent3" in graph
        assert "agent4" in graph
        assert "→" in graph  # arrow between levels

    def test_create_complete_dependency_tree(self) -> None:
        """Test creating complete dependency tree."""
        visualizer = SimpleVisualizer()

        # Set up agents
        visualizer.execution_state["agents"] = {
            "agent1": {"name": "agent1", "dependencies": []},
            "agent2": {"name": "agent2", "dependencies": ["agent1"]},
        }

        tree = visualizer._create_complete_dependency_tree()

        # Should show all agents with dim styling (static)
        assert "agent1" in tree
        assert "agent2" in tree
        assert "→" in tree

    def test_create_complete_dependency_tree_no_agents(self) -> None:
        """Test creating complete dependency tree with no agents."""
        visualizer = SimpleVisualizer()

        tree = visualizer._create_complete_dependency_tree()

        assert tree == "No agents configured"

    def test_is_execution_complete_ensemble_completed(self) -> None:
        """Test execution completion detection for ensemble completed."""
        visualizer = SimpleVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
            timestamp=datetime.now(),
            data={},
        )

        assert visualizer._is_execution_complete(event)

    def test_is_execution_complete_ensemble_failed(self) -> None:
        """Test execution completion detection for ensemble failed."""
        visualizer = SimpleVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.ENSEMBLE_FAILED,
            timestamp=datetime.now(),
            data={},
        )

        assert visualizer._is_execution_complete(event)

    def test_is_execution_complete_other_events(self) -> None:
        """Test execution completion detection for other events."""
        visualizer = SimpleVisualizer()

        event = ExecutionEvent(
            event_type=ExecutionEventType.AGENT_STARTED,
            timestamp=datetime.now(),
            data={},
        )

        assert not visualizer._is_execution_complete(event)

    @patch("llm_orc.visualization.simple.Console")
    def test_print_summary_no_ensemble(self, mock_console_class: Mock) -> None:
        """Test print summary with no ensemble name."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        visualizer = SimpleVisualizer()
        visualizer.print_summary()

        # Should not print anything if no ensemble name
        mock_console.print.assert_not_called()

    @patch("llm_orc.visualization.simple.Console")
    def test_print_summary_with_ensemble(self, mock_console_class: Mock) -> None:
        """Test print summary with ensemble."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        visualizer = SimpleVisualizer()
        visualizer.execution_state["ensemble_name"] = "test_ensemble"
        visualizer.execution_state["agents"] = {
            "agent1": {"name": "agent1", "dependencies": [], "status": "completed"},
        }

        visualizer.print_summary()

        # Should print final graph
        mock_console.print.assert_called_once()
        args = mock_console.print.call_args[0]
        assert "Final:" in args[0]

    @pytest.mark.asyncio
    async def test_process_event_routing(self) -> None:
        """Test that events are routed to correct handlers."""
        visualizer = SimpleVisualizer()

        # Mock all handler methods
        with (
            patch.object(
                visualizer, "_handle_ensemble_started"
            ) as mock_ensemble_started,
            patch.object(visualizer, "_handle_agent_started") as mock_agent_started,
            patch.object(visualizer, "_handle_agent_completed") as mock_agent_completed,
            patch.object(visualizer, "_handle_agent_failed") as mock_agent_failed,
            patch.object(
                visualizer, "_handle_ensemble_completed"
            ) as mock_ensemble_completed,
        ):
            # Test each event type
            events = [
                (ExecutionEventType.ENSEMBLE_STARTED, mock_ensemble_started),
                (ExecutionEventType.AGENT_STARTED, mock_agent_started),
                (ExecutionEventType.AGENT_COMPLETED, mock_agent_completed),
                (ExecutionEventType.AGENT_FAILED, mock_agent_failed),
                (ExecutionEventType.ENSEMBLE_COMPLETED, mock_ensemble_completed),
            ]

            for event_type, expected_handler in events:
                event = ExecutionEvent(
                    event_type=event_type,
                    timestamp=datetime.now(),
                    data={},
                )

                await visualizer._process_event(event)
                expected_handler.assert_called_once_with(event)
                expected_handler.reset_mock()

    @pytest.mark.asyncio
    async def test_visualize_execution_complete_flow(self) -> None:
        """Test complete visualization flow."""
        visualizer = SimpleVisualizer()

        # Create a mock stream with events
        mock_events = [
            ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_STARTED,
                timestamp=datetime.now(),
                data={
                    "total_agents": 1,
                    "agents_config": [{"name": "agent1", "depends_on": []}],
                },
                ensemble_name="test",
                execution_id="exec1",
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.AGENT_STARTED,
                timestamp=datetime.now(),
                data={},
                agent_name="agent1",
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.AGENT_COMPLETED,
                timestamp=datetime.now(),
                data={},
                agent_name="agent1",
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
                timestamp=datetime.now(),
                data={},
            ),
        ]

        # Create mock stream
        async def mock_subscribe() -> AsyncIterator[ExecutionEvent]:
            for event in mock_events:
                yield event

        mock_stream = Mock()
        mock_stream.subscribe = mock_subscribe

        # Mock console to avoid actual output
        with patch.object(visualizer, "console"):
            await visualizer.visualize_execution(mock_stream)

        # Verify final state
        assert visualizer.execution_state["status"] == "completed"
        assert visualizer.execution_state["completed_agents"] == 1
        assert visualizer.execution_state["agents"]["agent1"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_visualize_execution_with_exception(self) -> None:
        """Test visualization handles exceptions gracefully."""
        visualizer = SimpleVisualizer()

        # Create mock stream that raises exception
        async def mock_subscribe() -> AsyncIterator[ExecutionEvent]:
            yield ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_STARTED,
                timestamp=datetime.now(),
                data={},
            )
            raise ValueError("Test exception")

        mock_stream = Mock()
        mock_stream.subscribe = mock_subscribe

        # Mock console to capture error message
        with patch.object(visualizer, "console") as mock_console:
            await visualizer.visualize_execution(mock_stream)

            # Should print error message
            mock_console.print.assert_called()
            error_call = mock_console.print.call_args[0][0]
            assert "Visualization error:" in error_call
            assert "Test exception" in error_call

    @pytest.mark.asyncio
    async def test_visualize_execution_shows_dependency_tree(self) -> None:
        """Test that visualization shows complete dependency tree once."""
        visualizer = SimpleVisualizer()

        # Create events that will populate agents
        mock_events = [
            ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_STARTED,
                timestamp=datetime.now(),
                data={
                    "total_agents": 2,
                    "agents_config": [
                        {"name": "agent1", "depends_on": []},
                        {"name": "agent2", "depends_on": ["agent1"]},
                    ],
                },
            ),
            ExecutionEvent(
                event_type=ExecutionEventType.ENSEMBLE_COMPLETED,
                timestamp=datetime.now(),
                data={},
            ),
        ]

        async def mock_subscribe() -> AsyncIterator[ExecutionEvent]:
            for event in mock_events:
                yield event

        mock_stream = Mock()
        mock_stream.subscribe = mock_subscribe

        # Mock console to capture dependency tree output
        with patch.object(visualizer, "console") as mock_console:
            await visualizer.visualize_execution(mock_stream)

            # Should have printed dependency flow
            print_calls = mock_console.print.call_args_list
            dependency_call = next(
                (call for call in print_calls if "Dependency flow:" in str(call)), None
            )
            assert dependency_call is not None
