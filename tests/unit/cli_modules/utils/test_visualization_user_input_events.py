"""Tests for user input event handling in visualization using TDD methodology."""

# Import directly from the specific file, not the package
import sys
from pathlib import Path
from unittest.mock import Mock

# Import from the actual module file to get the right function
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "visualization",
    str(
        Path(__file__).parent.parent.parent.parent.parent
        / "src"
        / "llm_orc"
        / "cli_modules"
        / "utils"
        / "visualization.py"
    ),
)
if spec is not None and spec.loader is not None:
    viz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_module)
    from llm_orc.core.config.ensemble_config import EnsembleConfig
else:
    raise RuntimeError("Could not load visualization module")


class TestVisualizationUserInputEvents:
    """Test suite for user input event handling in streaming visualization."""

    def test_user_input_required_event_stops_status_and_clears_display(self) -> None:
        """Test user_input_required event stops status display and clears terminal."""
        # Given
        mock_console = Mock()
        mock_status = Mock()
        agent_statuses: dict[str, str] = {}
        agents = [{"name": "test-agent", "script": "get_user_input.py"}]
        ensemble_config = EnsembleConfig(name="test", description="test", agents=agents)

        event = {
            "type": "user_input_required",
            "data": {
                "agent_name": "test-agent",
                "message": "Waiting for user input...",
                "script": "get_user_input.py",
            },
        }

        # When
        should_continue = viz_module._handle_streaming_event(
            "user_input_required",
            event,
            agent_statuses,
            ensemble_config,
            mock_status,
            mock_console,
            "rich",
            False,
        )

        # Then
        assert should_continue is True
        mock_status.stop.assert_called_once()
        mock_console.clear.assert_called_once()

    def test_user_input_completed_event_restarts_status_display(self) -> None:
        """Test that user_input_completed event restarts status display."""
        # Given
        mock_console = Mock()
        mock_status = Mock()
        agent_statuses = {"test-agent": "running"}
        agents = [{"name": "test-agent", "script": "get_user_input.py"}]
        ensemble_config = EnsembleConfig(name="test", description="test", agents=agents)

        event = {
            "type": "user_input_completed",
            "data": {
                "agent_name": "test-agent",
                "message": "User input completed, continuing...",
            },
        }

        # When
        should_continue = viz_module._handle_streaming_event(
            "user_input_completed",
            event,
            agent_statuses,
            ensemble_config,
            mock_status,
            mock_console,
            "rich",
            False,
        )

        # Then
        assert should_continue is True
        mock_status.start.assert_called_once()

    def test_user_input_events_do_not_break_execution_flow(self) -> None:
        """Test that user input events don't interrupt execution flow."""
        # Given
        mock_console = Mock()
        mock_status = Mock()
        agent_statuses: dict[str, str] = {}
        agents = [{"name": "test-agent"}]
        ensemble_config = EnsembleConfig(name="test", description="test", agents=agents)

        # Test both event types
        events = [
            {
                "type": "user_input_required",
                "data": {"agent_name": "test-agent", "message": "Input needed"},
            },
            {
                "type": "user_input_completed",
                "data": {"agent_name": "test-agent", "message": "Input done"},
            },
        ]

        # When/Then
        for event in events:
            should_continue = viz_module._handle_streaming_event(
                event["type"],
                event,
                agent_statuses,
                ensemble_config,
                mock_status,
                mock_console,
                "rich",
                False,
            )
            assert should_continue is True  # Should continue execution

    def test_user_input_events_update_agent_status_to_waiting(self) -> None:
        """Test that user_input_required event marks agent as waiting."""
        # Given
        mock_console = Mock()
        mock_status = Mock()
        agent_statuses = {"test-agent": "running"}
        agents = [{"name": "test-agent", "script": "get_user_input.py"}]
        ensemble_config = EnsembleConfig(name="test", description="test", agents=agents)

        event = {
            "type": "user_input_required",
            "data": {
                "agent_name": "test-agent",
                "message": "Waiting for user input...",
            },
        }

        # When
        viz_module._handle_streaming_event(
            "user_input_required",
            event,
            agent_statuses,
            ensemble_config,
            mock_status,
            mock_console,
            "rich",
            False,
        )

        # Then
        assert agent_statuses["test-agent"] == "waiting_input"
