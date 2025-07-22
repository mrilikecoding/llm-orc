"""Comprehensive tests for CLI visualization utilities."""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console
from rich.tree import Tree

from llm_orc.cli_modules.utils.visualization import (
    _calculate_agent_level,
    _group_agents_by_dependency_level,
    create_dependency_graph,
    create_dependency_graph_with_status,
    create_dependency_tree,
    display_results,
    display_simplified_results,
    find_final_agent,
    run_standard_execution,
    run_streaming_execution,
)
from llm_orc.core.config.ensemble_config import EnsembleConfig


class TestDependencyVisualization:
    """Test dependency graph and tree visualization functions."""

    def test_create_dependency_graph_simple(self) -> None:
        """Test creating dependency graph with simple agents."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # When
        result = create_dependency_graph(agents)

        # Then
        assert result == (
            "[dim]○[/dim] [dim]agent_a[/dim] → [dim]○[/dim] [dim]agent_b[/dim]"
        )

    def test_create_dependency_graph_complex(self) -> None:
        """Test creating dependency graph with complex dependencies."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
            {"name": "agent_d", "depends_on": ["agent_c"]},
        ]

        # When
        result = create_dependency_graph(agents)

        # Then
        expected = (
            "[dim]○[/dim] [dim]agent_a[/dim], [dim]○[/dim] [dim]agent_b[/dim] → "
            "[dim]○[/dim] [dim]agent_c[/dim] → [dim]○[/dim] [dim]agent_d[/dim]"
        )
        assert result == expected

    def test_create_dependency_graph_with_status(self) -> None:
        """Test creating dependency graph with status indicators."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        statuses = {"agent_a": "completed", "agent_b": "running"}

        # When
        result = create_dependency_graph_with_status(agents, statuses)

        # Then
        expected = (
            "[green]✓[/green] [green]agent_a[/green] → "
            "[yellow]◐[/yellow] [yellow]agent_b[/yellow]"
        )
        assert result == expected

    def test_create_dependency_graph_with_status_all_states(self) -> None:
        """Test dependency graph with all status states."""
        # Given
        agents = [
            {"name": "completed_agent", "depends_on": []},
            {"name": "running_agent", "depends_on": []},
            {"name": "failed_agent", "depends_on": []},
            {"name": "pending_agent", "depends_on": []},
        ]
        statuses = {
            "completed_agent": "completed",
            "running_agent": "running",
            "failed_agent": "failed",
            "pending_agent": "pending",
        }

        # When
        result = create_dependency_graph_with_status(agents, statuses)

        # Then
        assert "[green]✓[/green] [green]completed_agent[/green]" in result
        assert "[yellow]◐[/yellow] [yellow]running_agent[/yellow]" in result
        assert "[red]✗[/red] [red]failed_agent[/red]" in result
        assert "[dim]○[/dim] [dim]pending_agent[/dim]" in result

    def test_create_dependency_tree_simple(self) -> None:
        """Test creating dependency tree with simple agents."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # When
        result = create_dependency_tree(agents)

        # Then
        assert isinstance(result, Tree)
        assert result.label == "[bold blue]Orchestrating Agent Responses[/bold blue]"

    def test_create_dependency_tree_with_status(self) -> None:
        """Test creating dependency tree with status indicators."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        statuses = {"agent_a": "completed", "agent_b": "running"}

        # When
        result = create_dependency_tree(agents, statuses)

        # Then
        assert isinstance(result, Tree)
        assert result.label == "[bold blue]Orchestrating Agent Responses[/bold blue]"
        # Verify that the tree was created with statuses passed in
        # We can't easily test the tree structure without diving into
        # implementation details

    def test_create_dependency_tree_empty_agents(self) -> None:
        """Test creating dependency tree with empty agent list."""
        # Given
        agents: list[dict[str, Any]] = []

        # When
        result = create_dependency_tree(agents)

        # Then
        assert isinstance(result, Tree)
        assert result.label == "[bold blue]Orchestrating Agent Responses[/bold blue]"

    def test_group_agents_by_dependency_level(self) -> None:
        """Test grouping agents by dependency level."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a"]},
            {"name": "agent_d", "depends_on": ["agent_b", "agent_c"]},
        ]

        # When
        result = _group_agents_by_dependency_level(agents)

        # Then
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert len(result[0]) == 2  # agent_a, agent_b
        assert len(result[1]) == 1  # agent_c
        assert len(result[2]) == 1  # agent_d

    def test_calculate_agent_level_no_dependencies(self) -> None:
        """Test calculating agent level with no dependencies."""
        # Given
        agent_name = "agent_a"
        dependencies: list[str] = []
        all_agents = [{"name": "agent_a", "depends_on": []}]

        # When
        result = _calculate_agent_level(agent_name, dependencies, all_agents)

        # Then
        assert result == 0

    def test_calculate_agent_level_with_dependencies(self) -> None:
        """Test calculating agent level with dependencies."""
        # Given
        agent_name = "agent_c"
        dependencies = ["agent_a", "agent_b"]
        all_agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]

        # When
        result = _calculate_agent_level(agent_name, dependencies, all_agents)

        # Then
        assert result == 1

    def test_calculate_agent_level_nested_dependencies(self) -> None:
        """Test calculating agent level with nested dependencies."""
        # Given
        agent_name = "agent_d"
        dependencies = ["agent_c"]
        all_agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
            {"name": "agent_c", "depends_on": ["agent_b"]},
            {"name": "agent_d", "depends_on": ["agent_c"]},
        ]

        # When
        result = _calculate_agent_level(agent_name, dependencies, all_agents)

        # Then
        assert result == 3


class TestResultsDisplay:
    """Test results display and formatting functions."""

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    def test_display_results_detailed(self, mock_console_class: Mock) -> None:
        """Test displaying detailed results."""
        # Given
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console

        results = {
            "agent_a": {"status": "success", "response": "Test response from agent A"}
        }
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 1000,
                    "total_cost_usd": 0.05,
                    "agents_count": 1,
                },
                "agents": {
                    "agent_a": {
                        "total_tokens": 1000,
                        "cost_usd": 0.05,
                        "duration_ms": 2500,
                        "model": "gpt-4",
                    }
                },
            },
        }

        # When
        display_results(results, metadata, detailed=True)

        # Then
        mock_console_class.assert_called_once_with(
            soft_wrap=True, width=None, force_terminal=True
        )
        mock_console.print.assert_called_once()

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    def test_display_results_with_error(self, mock_console_class: Mock) -> None:
        """Test displaying results with error."""
        # Given
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console

        results = {
            "agent_a": {"status": "error", "error": "Something went wrong"}
        }
        metadata = {"duration": "1.0s"}

        # When
        display_results(results, metadata, detailed=True)

        # Then
        mock_console.print.assert_called_once()

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    def test_display_results_with_code_response(self, mock_console_class: Mock) -> None:
        """Test displaying results with code-like response."""
        # Given
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console

        results = {
            "agent_a": {
                "status": "success",
                "response": "def hello():\n    print('Hello world')",
            }
        }
        metadata = {"duration": "1.5s"}

        # When
        display_results(results, metadata, detailed=True)

        # Then
        mock_console.print.assert_called_once()

    @patch("llm_orc.cli_modules.utils.visualization.display_simplified_results")
    def test_display_results_simplified(self, mock_simplified: Mock) -> None:
        """Test displaying simplified results."""
        # Given
        results = {"agent_a": {"status": "success", "response": "Test response"}}
        metadata = {"duration": "1.0s"}

        # When
        display_results(results, metadata, detailed=False)

        # Then
        mock_simplified.assert_called_once_with(results, metadata)

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    @patch("llm_orc.cli_modules.utils.visualization.find_final_agent")
    def test_display_simplified_results_with_final_agent(
        self, mock_find_final: Mock, mock_console_class: Mock
    ) -> None:
        """Test displaying simplified results with final agent."""
        # Given
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console
        mock_find_final.return_value = "agent_final"

        results = {
            "agent_final": {"status": "success", "response": "Final result"}
        }
        metadata = {
            "duration": "3.0s",
            "usage": {"totals": {"agents_count": 2}},
        }

        # When
        display_simplified_results(results, metadata)

        # Then
        mock_console.print.assert_called_once()

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    @patch("llm_orc.cli_modules.utils.visualization.find_final_agent")
    def test_display_simplified_results_fallback(
        self, mock_find_final: Mock, mock_console_class: Mock
    ) -> None:
        """Test displaying simplified results with fallback logic."""
        # Given
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console
        mock_find_final.return_value = None

        results = {
            "agent_a": {"status": "success", "response": "Fallback result"}
        }
        metadata = {"duration": "2.0s"}

        # When
        display_simplified_results(results, metadata)

        # Then
        mock_console.print.assert_called_once()

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    @patch("llm_orc.cli_modules.utils.visualization.find_final_agent")
    def test_display_simplified_results_no_success(
        self, mock_find_final: Mock, mock_console_class: Mock
    ) -> None:
        """Test displaying simplified results with no successful agents."""
        # Given
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console
        mock_find_final.return_value = None

        results = {"agent_a": {"status": "error", "error": "Failed"}}
        metadata = {"duration": "1.0s"}

        # When
        display_simplified_results(results, metadata)

        # Then
        mock_console.print.assert_called_once()

    def test_find_final_agent_with_success(self) -> None:
        """Test finding final agent with successful results."""
        # Given
        results = {
            "agent_a": {"status": "success", "response": "First"},
            "agent_b": {"status": "success", "response": "Second"},
            "agent_c": {"status": "error", "error": "Failed"},
        }

        # When
        result = find_final_agent(results)

        # Then
        assert result == "agent_b"  # Last successful agent

    def test_find_final_agent_no_success(self) -> None:
        """Test finding final agent with no successful results."""
        # Given
        results = {
            "agent_a": {"status": "error", "error": "Failed"},
            "agent_b": {"status": "error", "error": "Also failed"},
        }

        # When
        result = find_final_agent(results)

        # Then
        assert result is None

    def test_find_final_agent_empty_results(self) -> None:
        """Test finding final agent with empty results."""
        # Given
        results: dict[str, Any] = {}

        # When
        result = find_final_agent(results)

        # Then
        assert result is None


class TestStreamingExecution:
    """Test streaming execution visualization functions."""

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    @patch("click.echo")
    async def test_run_streaming_execution_json_output(
        self, mock_click_echo: Mock, mock_console_class: Mock
    ) -> None:
        """Test streaming execution with JSON output format."""
        # Given
        mock_console = Mock(spec=Console)
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_console.status = Mock(return_value=mock_context_manager)
        mock_console_class.return_value = mock_console

        # Create an async iterator for the streaming response
        async def async_iter():
            yield {"type": "agent_progress", "data": {"completed_agents": 1}}
            yield {
                "type": "execution_completed",
                "data": {
                    "duration": 2.5,
                    "results": {"agent_a": {"status": "success"}},
                    "metadata": {"usage": {}},
                },
            }

        mock_executor = Mock()
        mock_executor.execute_streaming = Mock(return_value=async_iter())

        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a"}]

        # When
        await run_streaming_execution(
            mock_executor, ensemble_config, "test input", "json", False
        )

        # Then
        assert mock_click_echo.call_count == 2  # Two events

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    @patch("llm_orc.cli_modules.utils.visualization.display_results")
    async def test_run_streaming_execution_text_output(
        self, mock_display_results: Mock, mock_console_class: Mock
    ) -> None:
        """Test streaming execution with text output format."""
        # Given
        mock_console = Mock(spec=Console)
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_console.status = Mock(return_value=mock_context_manager)
        mock_console_class.return_value = mock_console

        # Create an async iterator for the streaming response
        async def async_iter():
            yield {
                "type": "agent_progress",
                "data": {"completed_agents": 1, "total_agents": 2},
            }
            yield {
                "type": "execution_completed",
                "data": {
                    "duration": 2.5,
                    "results": {"agent_a": {"status": "success"}},
                    "metadata": {"usage": {}},
                },
            }

        mock_executor = Mock()
        mock_executor.execute_streaming = Mock(return_value=async_iter())

        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a"}, {"name": "agent_b"}]

        # When
        await run_streaming_execution(
            mock_executor, ensemble_config, "test input", "text", True
        )

        # Then
        mock_display_results.assert_called_once()


class TestStandardExecution:
    """Test standard execution functions."""

    @patch("click.echo")
    async def test_run_standard_execution_json_output(
        self, mock_click_echo: Mock
    ) -> None:
        """Test standard execution with JSON output format."""
        # Given
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent_a": {"status": "success"}},
                "metadata": {"usage": {}},
            }
        )

        ensemble_config = Mock(spec=EnsembleConfig)

        # When
        await run_standard_execution(
            mock_executor, ensemble_config, "test input", "json", False
        )

        # Then
        mock_click_echo.assert_called_once()
        call_args = mock_click_echo.call_args[0][0]
        # Verify it's valid JSON
        json.loads(call_args)

    @patch("llm_orc.cli_modules.utils.visualization.display_results")
    async def test_run_standard_execution_text_output(
        self, mock_display_results: Mock
    ) -> None:
        """Test standard execution with text output format."""
        # Given
        mock_executor = Mock()
        result_data = {
            "results": {"agent_a": {"status": "success"}},
            "metadata": {"usage": {}},
        }
        mock_executor.execute = AsyncMock(return_value=result_data)

        ensemble_config = Mock(spec=EnsembleConfig)

        # When
        await run_standard_execution(
            mock_executor, ensemble_config, "test input", "text", True
        )

        # Then
        mock_display_results.assert_called_once_with(
            result_data["results"], result_data["metadata"], True
        )
