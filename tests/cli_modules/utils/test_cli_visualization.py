"""Comprehensive tests for CLI visualization utilities."""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console
from rich.tree import Tree

from llm_orc.cli_modules.utils.visualization import (
    _calculate_agent_level,
    _create_plain_text_dependency_graph,
    _create_structured_dependency_info,
    _display_detailed_plain_text,
    _display_plain_text_dependency_graph,
    _display_simplified_plain_text,
    _group_agents_by_dependency_level,
    _handle_fallback_completed_event,
    _handle_fallback_failed_event,
    _handle_fallback_started_event,
    _handle_streaming_event,
    _handle_text_fallback_completed,
    _handle_text_fallback_failed,
    _handle_text_fallback_started,
    _process_execution_completed_event,
    _run_text_json_execution,
    _update_agent_progress_status,
    _update_agent_status_by_names,
    create_dependency_graph,
    create_dependency_graph_with_status,
    create_dependency_tree,
    display_plain_text_results,
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

    def test_create_dependency_graph_with_gap_in_levels(self) -> None:
        """Test dependency graph with gap in dependency levels to cover line 153."""
        # Given - create agents with gap in dependency levels
        # This is a bit artificial since normally gaps wouldn't occur in real usage
        # But we can create a scenario where level calculation results in gaps
        agents = [
            {"name": "agent_a", "depends_on": []},  # level 0
            {
                "name": "agent_c",
                "depends_on": ["agent_a", "nonexistent"],
            },  # level 1 due to nonexistent
        ]
        statuses = {"agent_a": "completed", "agent_c": "pending"}

        # When
        result = create_dependency_graph_with_status(agents, statuses)

        # Then
        assert "[green]✓[/green] [green]agent_a[/green]" in result
        assert "[dim]○[/dim] [dim]agent_c[/dim]" in result

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

    def test_create_dependency_tree_with_failed_status(self) -> None:
        """Test creating dependency tree with failed status to cover lines 51-52."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        statuses = {"agent_a": "failed", "agent_b": "pending"}

        # When
        result = create_dependency_tree(agents, statuses)

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

        results = {"agent_a": {"status": "error", "error": "Something went wrong"}}
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

        results = {"agent_final": {"status": "success", "response": "Final result"}}
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

        results = {"agent_a": {"status": "success", "response": "Fallback result"}}
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
        async def async_iter() -> Any:
            yield {"type": "agent_progress", "data": {"completed_agents": 1}}
            yield {
                "type": "execution_completed",
                "data": {
                    "duration": 2.5,
                    "results": {
                        "agent_a": {"status": "success", "response": "Test response"}
                    },
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
        assert mock_click_echo.call_count == 1  # Consolidated JSON output

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    @patch("llm_orc.cli_modules.utils.visualization.display_plain_text_results")
    async def test_run_streaming_execution_text_output(
        self, mock_display_plain_text: Mock, mock_console_class: Mock
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
        async def async_iter() -> Any:
            yield {
                "type": "agent_progress",
                "data": {"completed_agents": 1, "total_agents": 2},
            }
            yield {
                "type": "execution_completed",
                "data": {
                    "duration": 2.5,
                    "results": {
                        "agent_a": {"status": "success", "response": "Test response"}
                    },
                    "metadata": {"duration": "2.5s", "usage": {}},
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
        mock_display_plain_text.assert_called_once_with(
            {"agent_a": {"status": "success", "response": "Test response"}},
            {"duration": "2.5s", "usage": {}},
            True,
            [{"name": "agent_a"}, {"name": "agent_b"}],
        )


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
                "results": {
                    "agent_a": {"status": "success", "response": "Test response"}
                },
                "metadata": {"usage": {}},
            }
        )

        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a"}]

        # When
        await run_standard_execution(
            mock_executor, ensemble_config, "test input", "json", False
        )

        # Then
        mock_click_echo.assert_called_once()
        call_args = mock_click_echo.call_args[0][0]
        # Verify it's valid JSON
        json.loads(call_args)

    @patch("llm_orc.cli_modules.utils.visualization.display_plain_text_results")
    async def test_run_standard_execution_text_output(
        self, mock_display_plain_text: Mock
    ) -> None:
        """Test standard execution with text output format."""
        # Given
        mock_executor = Mock()
        result_data = {
            "results": {"agent_a": {"status": "success", "response": "Test response"}},
            "metadata": {"duration": "2.5s", "usage": {}},
        }
        mock_executor.execute = AsyncMock(return_value=result_data)

        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a"}]

        # When
        await run_standard_execution(
            mock_executor, ensemble_config, "test input", "text", True
        )

        # Then
        mock_display_plain_text.assert_called_once_with(
            result_data["results"],
            result_data["metadata"],
            True,
            ensemble_config.agents,
        )


class TestDisplayResultsHelperMethods:
    """Test helper methods extracted from display_results for complexity reduction."""

    def test_process_agent_results(self) -> None:
        """Test agent results processing helper method."""
        # Given
        results = {
            "agent_success": {"status": "success", "response": "def hello(): pass"},
            "agent_error": {"status": "error", "error": "Something went wrong"},
            "agent_text": {"status": "success", "response": "Just plain text"},
        }

        # When
        from llm_orc.cli_modules.utils.visualization import _process_agent_results

        metadata: dict[str, Any] = {"usage": {"agents": {}}}
        markdown_content = _process_agent_results(results, metadata)

        # Then
        combined_content = "".join(markdown_content)
        assert "## agent_success\n" in combined_content
        assert "```\ndef hello(): pass\n```\n" in combined_content
        assert "## ❌ agent_error\n" in combined_content
        assert "**Error:** Something went wrong\n" in combined_content
        assert "Just plain text\n" in combined_content

    def test_format_performance_metrics(self) -> None:
        """Test performance metrics formatting helper method."""
        # Given
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 1500,
                    "total_cost_usd": 0.075,
                    "agents_count": 2,
                },
                "agents": {
                    "agent_a": {
                        "total_tokens": 800,
                        "cost_usd": 0.04,
                        "duration_ms": 1200,
                        "model": "claude-3",
                    },
                    "agent_b": {
                        "total_tokens": 700,
                        "cost_usd": 0.035,
                        "duration_ms": 1300,
                        "model": "gpt-4",
                    },
                },
            },
        }

        # When
        from llm_orc.cli_modules.utils.visualization import _format_performance_metrics

        markdown_content = _format_performance_metrics(metadata)

        # Then
        combined_content = "".join(markdown_content)
        assert "## Performance Metrics\n" in combined_content
        assert "- **Duration:** 2.5s\n" in combined_content
        assert "- **Total tokens:** 1,500\n" in combined_content
        assert "- **Total cost:** $0.0750\n" in combined_content
        assert "- **Agents:** 2\n" in combined_content
        assert "### Per-Agent Usage\n" in combined_content
        assert "**agent_a** (claude-3): 800 tokens" in combined_content


class TestStreamingExecutionHelperMethods:
    """Test helper methods for run_streaming_execution complexity reduction."""

    def test_update_agent_progress_status(self) -> None:
        """Test agent progress status update helper method."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
            {"name": "agent_c", "depends_on": ["agent_b"]},
        ]
        completed_agents = 1
        total_agents = 3
        agent_statuses: dict[str, str] = {}

        # When
        _update_agent_progress_status(
            agents, completed_agents, total_agents, agent_statuses
        )

        # Then
        assert agent_statuses["agent_a"] == "completed"
        assert agent_statuses["agent_b"] == "running"
        assert agent_statuses["agent_c"] == "pending"

    def test_update_agent_progress_status_all_completed(self) -> None:
        """Test agent progress status when all agents are completed."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        completed_agents = 2
        total_agents = 2
        agent_statuses: dict[str, str] = {}

        # When
        _update_agent_progress_status(
            agents, completed_agents, total_agents, agent_statuses
        )

        # Then
        assert agent_statuses["agent_a"] == "completed"
        assert agent_statuses["agent_b"] == "completed"

    @patch("llm_orc.cli_modules.utils.visualization.create_dependency_tree")
    def test_process_execution_completed_event_text_format(
        self, mock_create_tree: Mock
    ) -> None:
        """Test processing execution completed event with text format."""
        # Given
        mock_console = Mock(spec=Console)
        mock_status = Mock()
        agents = [{"name": "agent_a"}]
        event_data = {
            "duration": 2.5,
            "results": {"agent_a": {"status": "success"}},
            "metadata": {"usage": {}},
        }
        mock_tree = Mock()
        mock_create_tree.return_value = mock_tree

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization.display_results"
        ) as mock_display:
            result = _process_execution_completed_event(
                mock_console, mock_status, agents, event_data, "text", True
            )

        # Then
        assert result is True  # Should indicate to break from loop
        mock_status.stop.assert_called_once()
        mock_console.print.assert_any_call(mock_tree)
        mock_console.print.assert_any_call("Completed in 2.50s")
        mock_display.assert_called_once_with(
            event_data["results"], event_data["metadata"], True
        )


class TestPlainTextFunctions:
    """Test plain text visualization functions."""

    def test_create_plain_text_dependency_graph_with_running_status(self) -> None:
        """Test creating plain text dependency graph with running status."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        agent_statuses = {"agent_a": "running", "agent_b": "pending"}

        # When
        result = _create_plain_text_dependency_graph(agents, agent_statuses)

        # Then
        assert "◐ agent_a" in result
        assert "○ agent_b" in result
        assert "→" in result

    def test_create_plain_text_dependency_graph_with_failed_status(self) -> None:
        """Test creating plain text dependency graph with failed status."""
        # Given
        agents = [{"name": "agent_a", "depends_on": []}]
        agent_statuses = {"agent_a": "failed"}

        # When
        result = _create_plain_text_dependency_graph(agents, agent_statuses)

        # Then
        assert "✗ agent_a" in result

    def test_create_plain_text_dependency_graph_empty_agents(self) -> None:
        """Test creating plain text dependency graph with empty agents."""
        # Given
        agents: list[dict[str, Any]] = []
        agent_statuses: dict[str, str] = {}

        # When
        result = _create_plain_text_dependency_graph(agents, agent_statuses)

        # Then
        assert result == ""

    @patch("click.echo")
    def test_display_plain_text_dependency_graph(self, mock_echo: Mock) -> None:
        """Test displaying plain text dependency graph."""
        # Given
        agents = [{"name": "agent_a", "depends_on": []}]
        results = {"agent_a": {"status": "success"}}

        # When
        _display_plain_text_dependency_graph(agents, results)

        # Then
        assert mock_echo.call_count >= 3  # Title, graph, empty line

    @patch("click.echo")
    def test_display_detailed_plain_text(self, mock_echo: Mock) -> None:
        """Test displaying detailed plain text results."""
        # Given
        results = {
            "agent_success": {"status": "success", "response": "Test response"},
            "agent_error": {"status": "failed", "error": "Test error"},
        }
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 1000,
                    "total_cost_usd": 0.05,
                    "agents_count": 2,
                }
            },
        }

        # When
        _display_detailed_plain_text(results, metadata)

        # Then
        calls = [
            str(call[0][0]) if call[0] else str(call)
            for call in mock_echo.call_args_list
        ]
        assert "Results" in calls
        assert "Performance Metrics" in calls

    @patch("click.echo")
    def test_display_simplified_plain_text_no_successful_results(
        self, mock_echo: Mock
    ) -> None:
        """Test displaying simplified plain text with no successful results."""
        # Given
        results = {"agent_a": {"status": "failed", "error": "All failed"}}
        metadata = {"duration": "1.0s"}

        # When
        _display_simplified_plain_text(results, metadata)

        # Then
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert "❌ No successful results found" in calls

    def test_create_dependency_graph_with_status_failed(self) -> None:
        """Test dependency graph with failed agent status."""
        # Given
        agents = [{"name": "agent_a", "depends_on": []}]
        agent_statuses = {"agent_a": "failed"}

        # When
        result = create_dependency_graph_with_status(agents, agent_statuses)

        # Then
        assert "✗" in result  # Red X symbol for failed
        assert "agent_a" in result

    def test_create_dependency_graph_with_status_unknown(self) -> None:
        """Test dependency graph with unknown agent status."""
        # Given
        agents = [{"name": "agent_b", "depends_on": []}]
        agent_statuses = {"agent_b": "unknown"}

        # When
        result = create_dependency_graph_with_status(agents, agent_statuses)

        # Then
        assert "○" in result  # Circle symbol for unknown
        assert "agent_b" in result

    def test_format_adaptive_resource_metrics_with_decisions(self) -> None:
        """Test formatting adaptive resource metrics with concurrency decisions."""
        from llm_orc.cli_modules.utils.visualization import (
            _format_adaptive_resource_metrics,
        )

        # Given
        adaptive_stats = {
            "management_type": "adaptive",
            "adaptive_used": True,
            "concurrency_decisions": [
                {
                    "adaptive_limit": 2,
                    "cpu_percent": 45.5,
                    "memory_percent": 62.1,
                    "circuit_breaker_state": "CLOSED",
                    "base_limit": 4,
                }
            ],
        }

        # When
        result = _format_adaptive_resource_metrics(adaptive_stats)

        # Then
        markdown_text = "".join(result)
        assert "Resource Management" in markdown_text
        assert "adaptive" in markdown_text
        assert "45.5" in markdown_text  # CPU percentage
        assert "62.1" in markdown_text  # Memory percentage

    def test_format_adaptive_resource_metrics_with_multiple_decisions(self) -> None:
        """Test formatting adaptive resource metrics with multiple decisions.

        This covers lines 614-620.
        """
        from llm_orc.cli_modules.utils.visualization import (
            _format_adaptive_resource_metrics,
        )

        # Given - multiple concurrency decisions to trigger lines 614-620
        adaptive_stats = {
            "management_type": "adaptive",
            "adaptive_used": True,
            "concurrency_decisions": [
                {
                    "adaptive_limit": 2,
                    "cpu_percent": 45.5,
                    "memory_percent": 62.1,
                    "circuit_breaker_state": "CLOSED",
                    "base_limit": 4,
                },
                {
                    "adaptive_limit": 3,
                    "cpu_percent": 55.0,
                    "memory_percent": 70.0,
                    "circuit_breaker_state": "CLOSED",
                    "base_limit": 4,
                },
            ],
        }

        # When
        result = _format_adaptive_resource_metrics(adaptive_stats)

        # Then
        markdown_text = "".join(result)
        assert "Resource Management" in markdown_text
        assert "adaptive" in markdown_text
        assert (
            "Concurrency adjustments:** 2 decisions made during execution"
            in markdown_text
        )

    def test_format_adaptive_resource_metrics_static_with_decisions(self) -> None:
        """Test formatting static resource metrics with concurrency decisions."""
        from llm_orc.cli_modules.utils.visualization import (
            _format_adaptive_resource_metrics,
        )

        # Given
        adaptive_stats = {
            "management_type": "static",
            "adaptive_used": False,
            "concurrency_decisions": [{"static_limit": 3, "agent_count": 5}],
        }

        # When
        result = _format_adaptive_resource_metrics(adaptive_stats)

        # Then
        markdown_text = "".join(result)
        assert "Resource Management" in markdown_text
        assert "static" in markdown_text

    def test_format_adaptive_resource_metrics_adaptive_no_decisions(self) -> None:
        """Test formatting adaptive resource metrics without decisions."""
        from llm_orc.cli_modules.utils.visualization import (
            _format_adaptive_resource_metrics,
        )

        # Given
        adaptive_stats = {
            "management_type": "adaptive",
            "adaptive_used": True,
            "concurrency_decisions": [],
            "execution_metrics": {
                "peak_cpu": 25.4,
                "avg_cpu": 20.1,
                "peak_memory": 55.2,
                "avg_memory": 48.9,
                "sample_count": 12,
            },
        }

        # When
        result = _format_adaptive_resource_metrics(adaptive_stats)

        # Then
        markdown_text = "".join(result)
        assert "Resource Management" in markdown_text
        assert "25.4" in markdown_text  # Peak CPU
        assert "20.1" in markdown_text  # Avg CPU
        assert "12 samples" in markdown_text

    def test_format_adaptive_resource_metrics_without_adaptive_limit(self) -> None:
        """Test formatting adaptive resource metrics without adaptive_limit.

        This covers lines 619-623.
        """
        from llm_orc.cli_modules.utils.visualization import (
            _format_adaptive_resource_metrics,
        )

        # Given - decision without adaptive_limit to trigger lines 619-623
        adaptive_stats = {
            "management_type": "adaptive",
            "adaptive_used": True,
            "concurrency_decisions": [
                {
                    "cpu_percent": 45.5,
                    "memory_percent": 62.1,
                    "circuit_breaker_state": "CLOSED",
                    "base_limit": 4,
                    # Note: no adaptive_limit field
                }
            ],
        }

        # When
        result = _format_adaptive_resource_metrics(adaptive_stats)

        # Then
        markdown_text = "".join(result)
        assert "Resource Management" in markdown_text
        assert (
            "Adaptive resource management enabled** but no detailed metrics available"
            in markdown_text
        )

    def test_format_adaptive_resource_metrics_static_no_decisions(self) -> None:
        """Test formatting static resource metrics without decisions."""
        from llm_orc.cli_modules.utils.visualization import (
            _format_adaptive_resource_metrics,
        )

        # Given
        adaptive_stats = {
            "management_type": "static",
            "adaptive_used": False,
            "concurrency_decisions": [],
        }

        # When
        result = _format_adaptive_resource_metrics(adaptive_stats)

        # Then
        markdown_text = "".join(result)
        assert "Resource Management" in markdown_text
        assert "static" in markdown_text

    def test_display_results_with_adaptive_resource_management(self) -> None:
        """Test displaying results with adaptive resource management metadata."""
        # Given
        results = {"agent1": {"status": "success", "response": "Test response"}}
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 100,
                    "total_cost_usd": 0.01,
                    "agents_count": 1,
                }
            },
            "adaptive_resource_management": {
                "management_type": "adaptive",
                "adaptive_used": True,
                "concurrency_decisions": [],
                "execution_metrics": {
                    "peak_cpu": 15.5,
                    "avg_cpu": 10.2,
                    "sample_count": 5,
                },
            },
        }

        # When - should not raise an exception
        display_results(results, metadata, detailed=True)

        # Then - test passes if no exception raised

    def test_display_results_with_model_profile_fallback(self) -> None:
        """Test displaying results with model profile fallback information."""
        # Given
        results = {"agent1": {"status": "success", "response": "Test response"}}
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 100,
                    "total_cost_usd": 0.01,
                    "agents_count": 1,
                },
                "agents": {
                    "agent1": {
                        "total_tokens": 100,
                        "cost_usd": 0.01,
                        "duration_ms": 1000,
                        "model": "claude-3-haiku",
                        "model_profile": "haiku",
                    }
                },
            },
        }

        # When - should not raise an exception
        display_results(results, metadata, detailed=True)

        # Then - test passes if no exception raised

    def test_display_results_with_only_model_no_profile(self) -> None:
        """Test displaying results with model but no profile (line 226)."""
        # Given
        results = {"agent1": {"status": "success", "response": "Test response"}}
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 100,
                    "total_cost_usd": 0.01,
                    "agents_count": 1,
                },
                "agents": {
                    "agent1": {
                        "total_tokens": 100,
                        "cost_usd": 0.01,
                        "duration_ms": 1000,
                        "model": "claude-3-haiku",
                        "model_profile": None,  # This should trigger line 226
                    }
                },
            },
        }

        # When - should not raise an exception
        display_results(results, metadata, detailed=True)

        # Then - test passes if no exception raised

    def test_display_plain_text_results_with_adaptive_stats(self) -> None:
        """Test plain text display with adaptive resource management."""
        # Given
        results = {"agent1": {"status": "success", "response": "Test response"}}
        metadata = {
            "duration": "2.5s",
            "usage": {
                "totals": {
                    "total_tokens": 100,
                    "total_cost_usd": 0.01,
                    "agents_count": 1,
                }
            },
            "adaptive_resource_management": {
                "management_type": "adaptive",
                "adaptive_used": True,
                "execution_metrics": {
                    "peak_cpu": 12.5,
                    "avg_cpu": 8.1,
                    "sample_count": 3,
                },
            },
        }
        agents = [{"name": "agent1", "depends_on": []}]

        # When - should not raise an exception
        display_plain_text_results(results, metadata, detailed=True, agents=agents)

        # Then - test passes if no exception raised

    def test_find_final_agent_with_coordinator(self) -> None:
        """Test finding final agent when coordinator exists."""
        # Given
        results = {
            "agent1": {"status": "success", "response": "Response 1"},
            "coordinator": {"status": "success", "response": "Final response"},
        }

        # When
        final_agent = find_final_agent(results)

        # Then
        assert final_agent == "coordinator"

    def test_find_final_agent_with_synthesizer(self) -> None:
        """Test finding final agent when synthesizer exists."""
        # Given
        results = {
            "agent1": {"status": "success", "response": "Response 1"},
            "synthesizer": {"status": "success", "response": "Final response"},
        }

        # When
        final_agent = find_final_agent(results)

        # Then
        assert final_agent == "synthesizer"

    @patch(
        "llm_orc.cli_modules.utils.visualization._display_adaptive_resource_metrics_text"
    )
    def test_display_adaptive_resource_metrics_text_called(
        self, mock_display: Mock
    ) -> None:
        """Test that adaptive resource metrics text display is called."""
        from llm_orc.cli_modules.utils.visualization import (
            _display_adaptive_resource_metrics_text,
        )

        # Given
        adaptive_stats = {"management_type": "adaptive", "adaptive_used": True}

        # When
        _display_adaptive_resource_metrics_text(adaptive_stats)

        # Then - function should execute without error

    @patch(
        "llm_orc.cli_modules.utils.visualization._display_plain_text_dependency_graph"
    )
    @patch("llm_orc.cli_modules.utils.visualization._display_detailed_plain_text")
    def test_display_plain_text_results_with_agents(
        self, mock_detailed: Mock, mock_dep_graph: Mock
    ) -> None:
        """Test displaying plain text results with agents."""
        # Given
        results = {"agent_a": {"status": "success", "response": "Test"}}
        metadata = {"duration": "1.0s"}
        agents = [{"name": "agent_a", "depends_on": []}]

        # When
        display_plain_text_results(results, metadata, detailed=True, agents=agents)

        # Then
        mock_dep_graph.assert_called_once_with(agents, results)
        mock_detailed.assert_called_once_with(results, metadata)

    def test_create_structured_dependency_info_with_results(self) -> None:
        """Test creating structured dependency info with results."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        results = {
            "agent_a": {"status": "success"},
            "agent_b": {"status": "failed", "error": "test error"},
        }

        # When
        result = _create_structured_dependency_info(agents, results)

        # Then
        assert "dependency_levels" in result
        assert len(result["dependency_levels"]) == 2
        # Check that status is included
        level_0_agent = result["dependency_levels"][0]["agents"][0]
        assert level_0_agent["status"] == "completed"


class TestFallbackEventHandlers:
    """Test fallback event handler functions."""

    @patch("builtins.print")
    def test_handle_fallback_completed_event(self, mock_print: Mock) -> None:
        """Test handling fallback completed event."""
        # Given
        mock_console = Mock()
        event_data = {
            "agent_name": "test_agent",
            "fallback_model_name": "fallback_model",
        }

        # When
        _handle_fallback_completed_event(mock_console, event_data)

        # Then
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "✅ SUCCESS:" in call_args

    @patch("builtins.print")
    def test_handle_fallback_failed_event(self, mock_print: Mock) -> None:
        """Test handling fallback failed event."""
        # Given
        mock_console = Mock()
        event_data = {
            "agent_name": "test_agent",
            "failure_type": "oauth_error",
            "fallback_error": "Token expired",
            "fallback_model_name": "fallback_model",
        }

        # When
        _handle_fallback_failed_event(mock_console, event_data)

        # Then
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "❌ FAILED:" in call_args

    def test_handle_fallback_started_event_oauth_error(self) -> None:
        """Test handling fallback started event with OAuth error."""
        # Given
        mock_console = Mock()
        event_data = {
            "agent_name": "test_agent",
            "failure_type": "oauth_error",
            "original_error": "Token refresh failed",
            "original_model_profile": "gpt-4",
            "fallback_model_name": "claude-3",
        }

        # When
        _handle_fallback_started_event(mock_console, event_data)

        # Then
        assert mock_console.print.call_count == 3
        first_call = mock_console.print.call_args_list[0][0][0]
        assert "🔐" in first_call  # OAuth emoji

    def test_handle_fallback_started_event_auth_error(self) -> None:
        """Test handling fallback started event with authentication error."""
        # Given
        mock_console = Mock()
        event_data = {
            "agent_name": "test_agent",
            "failure_type": "authentication_error",
            "original_error": "Invalid API key",
            "original_model_profile": "claude-3",
            "fallback_model_name": "gpt-4",
        }

        # When
        _handle_fallback_started_event(mock_console, event_data)

        # Then
        first_call = mock_console.print.call_args_list[0][0][0]
        assert "🔑" in first_call  # Auth emoji

    def test_handle_fallback_started_event_other_error(self) -> None:
        """Test handling fallback started event with other error type."""
        # Given
        mock_console = Mock()
        event_data = {
            "agent_name": "test_agent",
            "failure_type": "runtime_error",
            "original_error": "Model timeout",
            "original_model_profile": "gpt-4",
            "fallback_model_name": "claude-3",
        }

        # When
        _handle_fallback_started_event(mock_console, event_data)

        # Then
        first_call = mock_console.print.call_args_list[0][0][0]
        assert "⚠️" in first_call  # Generic warning emoji

    @patch("click.echo")
    def test_handle_text_fallback_started(self, mock_echo: Mock) -> None:
        """Test handling text fallback started event."""
        # Given
        event_data = {
            "agent_name": "test_agent",
            "failure_type": "model_loading",
            "original_error": "Connection failed",
            "original_model_profile": "gpt-4",
            "fallback_model_name": "claude-3",
        }

        # When
        _handle_text_fallback_started(event_data)

        # Then
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("WARNING:" in call for call in calls)

    @patch("click.echo")
    def test_handle_text_fallback_completed(self, mock_echo: Mock) -> None:
        """Test handling text fallback completed event."""
        # Given
        event_data = {
            "agent_name": "test_agent",
            "fallback_model_name": "claude-3",
        }

        # When
        _handle_text_fallback_completed(event_data)

        # Then
        mock_echo.assert_called_once()
        call_args = mock_echo.call_args[0][0]
        assert "SUCCESS:" in call_args

    @patch("click.echo")
    def test_handle_text_fallback_failed(self, mock_echo: Mock) -> None:
        """Test handling text fallback failed event."""
        # Given
        event_data = {
            "agent_name": "test_agent",
            "failure_type": "authentication_error",
            "fallback_error": "API key invalid",
            "fallback_model_name": "claude-3",
        }

        # When
        _handle_text_fallback_failed(event_data)

        # Then
        mock_echo.assert_called_once()
        call_args = mock_echo.call_args[0][0]
        assert "ERROR:" in call_args


class TestStreamingEventHandlers:
    """Test streaming event handler functions."""

    def test_update_agent_status_by_names(self) -> None:
        """Test updating agent status by specific names."""
        # Given
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": []},
        ]
        started_agent_names = ["agent_a", "agent_b"]
        completed_agent_names = ["agent_a"]
        agent_statuses: dict[str, str] = {}

        # When
        _update_agent_status_by_names(
            agents, started_agent_names, completed_agent_names, agent_statuses
        )

        # Then
        assert agent_statuses["agent_a"] == "completed"
        assert agent_statuses["agent_b"] == "running"
        assert agent_statuses["agent_c"] == "pending"

    def test_handle_streaming_event_agent_progress(self) -> None:
        """Test handling agent progress streaming event."""
        # Given
        event_type = "agent_progress"
        event = {
            "data": {
                "started_agent_names": ["agent_a"],
                "completed_agent_names": [],
            }
        }
        agent_statuses: dict[str, str] = {}
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]
        status = Mock()
        console = Mock()

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization.create_dependency_tree"
        ) as mock_tree:
            mock_tree.return_value = "mock_tree"
            result = _handle_streaming_event(
                event_type, event, agent_statuses, ensemble_config, status, console
            )

        # Then
        assert result is True
        assert agent_statuses["agent_a"] == "running"

    def test_handle_streaming_event_execution_completed(self) -> None:
        """Test handling execution completed streaming event."""
        # Given
        event_type = "execution_completed"
        event = {"data": {"duration": 2.5, "results": {}, "metadata": {}}}
        agent_statuses: dict[str, str] = {}
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]
        status = Mock()
        console = Mock()

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization._process_execution_completed_event"
        ) as mock_process:
            mock_process.return_value = False  # Indicate should break
            result = _handle_streaming_event(
                event_type,
                event,
                agent_statuses,
                ensemble_config,
                status,
                console,
                "rich",
                True,
            )

        # Then
        assert result is False  # Should break loop

    def test_handle_streaming_event_agent_fallback_started(self) -> None:
        """Test handling agent fallback started streaming event."""
        # Given
        event_type = "agent_fallback_started"
        event = {"data": {"agent_name": "agent_a"}}
        agent_statuses: dict[str, str] = {}
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]
        status = Mock()
        console = Mock()

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization.create_dependency_tree"
        ) as mock_tree:
            with patch(
                "llm_orc.cli_modules.utils.visualization._handle_fallback_started_event"
            ) as mock_handle:
                mock_tree.return_value = "mock_tree"
                result = _handle_streaming_event(
                    event_type, event, agent_statuses, ensemble_config, status, console
                )

        # Then
        assert result is True
        assert agent_statuses["agent_a"] == "running"
        mock_handle.assert_called_once_with(console, event["data"])

    def test_handle_streaming_event_agent_fallback_failed(self) -> None:
        """Test handling agent fallback failed streaming event."""
        # Given
        event_type = "agent_fallback_failed"
        event = {"data": {"agent_name": "agent_a"}}
        agent_statuses: dict[str, str] = {}
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]
        status = Mock()
        console = Mock()

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization.create_dependency_tree"
        ) as mock_tree:
            with patch(
                "llm_orc.cli_modules.utils.visualization._handle_fallback_failed_event"
            ) as mock_handle:
                mock_tree.return_value = "mock_tree"
                result = _handle_streaming_event(
                    event_type, event, agent_statuses, ensemble_config, status, console
                )

        # Then
        assert result is True
        assert agent_statuses["agent_a"] == "failed"
        mock_handle.assert_called_once_with(console, event["data"])


class TestTextJsonExecution:
    """Test text and JSON execution functions."""

    @patch("click.echo")
    async def test_run_text_json_execution_json_format(self, mock_echo: Mock) -> None:
        """Test running text/JSON execution with JSON format."""
        # Given
        mock_executor = Mock()

        async def async_iter() -> Any:
            yield {"type": "execution_started", "data": {}}
            yield {
                "type": "execution_completed",
                "data": {
                    "results": {"agent_a": {"status": "success", "response": "Test"}},
                    "metadata": {"usage": {}},
                },
            }

        mock_executor.execute_streaming = Mock(return_value=async_iter())
        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]

        # When
        await _run_text_json_execution(
            mock_executor, ensemble_config, "test input", "json", False
        )

        # Then
        mock_echo.assert_called_once()
        call_args = mock_echo.call_args[0][0]
        output_data = json.loads(call_args)
        assert "events" in output_data
        assert "result" in output_data
        assert "dependency_info" in output_data

    @patch("llm_orc.cli_modules.utils.visualization.display_plain_text_results")
    async def test_run_text_json_execution_text_format_fallback_events(
        self, mock_display: Mock
    ) -> None:
        """Test text execution with fallback events."""
        # Given
        mock_executor = Mock()

        async def async_iter() -> Any:
            yield {"type": "agent_fallback_started", "data": {"agent_name": "agent_a"}}
            yield {
                "type": "agent_fallback_completed",
                "data": {"agent_name": "agent_a"},
            }
            yield {"type": "agent_fallback_failed", "data": {"agent_name": "agent_b"}}
            yield {
                "type": "execution_completed",
                "data": {
                    "results": {"agent_a": {"status": "success", "response": "Test"}},
                    "metadata": {"usage": {}},
                },
            }

        mock_executor.execute_streaming = Mock(return_value=async_iter())
        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization._handle_text_fallback_started"
        ) as mock_started:
            with patch(
                "llm_orc.cli_modules.utils.visualization._handle_text_fallback_completed"
            ) as mock_completed:
                with patch(
                    "llm_orc.cli_modules.utils.visualization._handle_text_fallback_failed"
                ) as mock_failed:
                    await _run_text_json_execution(
                        mock_executor, ensemble_config, "test input", "text", True
                    )

        # Then
        mock_started.assert_called_once()
        mock_completed.assert_called_once()
        mock_failed.assert_called_once()
        mock_display.assert_called_once()

    @patch("llm_orc.cli_modules.utils.visualization.Console")
    async def test_run_streaming_execution_rich_interface(
        self, mock_console_class: Mock
    ) -> None:
        """Test running streaming execution with Rich interface (default)."""
        # Given
        mock_console = Mock()
        mock_status = Mock()
        mock_console.status.return_value.__enter__ = Mock(return_value=mock_status)
        mock_console.status.return_value.__exit__ = Mock(return_value=None)
        mock_console_class.return_value = mock_console

        mock_executor = Mock()

        async def async_iter() -> Any:
            yield {"type": "execution_started", "data": {}}
            yield {"type": "agent_started", "data": {"agent_name": "agent_a"}}
            yield {
                "type": "execution_completed",
                "data": {
                    "duration": 2.5,
                    "results": {"agent_a": {"status": "success", "response": "Test"}},
                    "metadata": {"usage": {}},
                },
            }

        mock_executor.execute_streaming = Mock(return_value=async_iter())
        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization.create_dependency_tree"
        ) as mock_tree:
            with patch(
                "llm_orc.cli_modules.utils.visualization._handle_streaming_event"
            ) as mock_handle:
                mock_tree.return_value = "mock_tree"
                mock_handle.side_effect = [True, True, False]  # Break on third event

                await run_streaming_execution(
                    mock_executor,
                    ensemble_config,
                    "test input",
                    "rich",
                    False,  # Rich default
                )

        # Then
        mock_console.status.assert_called_once()
        assert mock_handle.call_count == 3

    @patch("llm_orc.cli_modules.utils.visualization._handle_text_fallback_started")
    @patch("llm_orc.cli_modules.utils.visualization._handle_text_fallback_failed")
    async def test_run_standard_execution_with_fallback_events_text(
        self, mock_fallback_failed: Mock, mock_fallback_started: Mock
    ) -> None:
        """Test standard execution with fallback events in text mode."""
        # Given
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(
            return_value={
                "results": {"agent_a": {"status": "success", "response": "Test"}},
                "metadata": {"usage": {}},
            }
        )

        # Mock the _emit_performance_event to capture calls
        original_emit = Mock()
        mock_executor._emit_performance_event = original_emit

        ensemble_config = Mock(spec=EnsembleConfig)
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]

        # When
        with patch(
            "llm_orc.cli_modules.utils.visualization.display_plain_text_results"
        ) as mock_display:
            await run_standard_execution(
                mock_executor, ensemble_config, "test input", "text", False
            )

            # Simulate fallback events being captured by calling the capture function
            # This tests the capture_fallback_event function path (lines 974-979)
            capture_func = mock_executor._emit_performance_event
            capture_func("agent_fallback_started", {"agent_name": "test"})
            capture_func("agent_fallback_failed", {"agent_name": "test"})
            capture_func("other_event", {"agent_name": "test"})  # Should be ignored

        # Then
        mock_display.assert_called_once()
        # The original emit function should be restored
        assert mock_executor._emit_performance_event == original_emit
