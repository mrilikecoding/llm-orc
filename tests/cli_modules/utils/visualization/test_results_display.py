"""Comprehensive tests for results display module."""

from typing import Any
from unittest.mock import Mock, patch

from llm_orc.cli_modules.utils.visualization.results_display import (
    _display_detailed_plain_text,
    _display_plain_text_dependency_graph,
    _display_simplified_plain_text,
    _format_performance_metrics,
    _has_code_content,
    _process_agent_results,
    display_plain_text_results,
    display_results,
    display_simplified_results,
)


class TestDisplayResults:
    """Test main results display functions."""

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display.find_final_agent')
    def test_display_results_simple_mode(
        self, mock_find_final: Mock, mock_console_class: Mock
    ) -> None:
        """Test display results in simple mode."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_find_final.return_value = "agent_a"

        results = {
            "agent_a": {"response": "Hello world"}
        }
        metadata: dict[str, Any] = {}
        agents = [{"name": "agent_a"}]

        display_results(results, metadata, agents, detailed=False)

        mock_console.print.assert_called()
        assert mock_console.print.call_count >= 1

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display._process_agent_results')
    @patch('llm_orc.cli_modules.utils.visualization.results_display._format_performance_metrics')
    def test_display_results_detailed_mode(
        self,
        mock_format_perf: Mock,
        mock_process_results: Mock,
        mock_console_class: Mock
    ) -> None:
        """Test display results in detailed mode."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_process_results.return_value = {
            "agent_a": {
                "status": "success",
                "response": "Hello",
                "error": "",
                "has_code": False
            }
        }
        mock_format_perf.return_value = ["Performance: Good"]

        results = {"agent_a": {"status": "success", "response": "Hello"}}
        metadata = {"usage": {"totals": {"tokens": 100}}}
        agents = [{"name": "agent_a"}]

        display_results(results, metadata, agents, detailed=True)

        mock_process_results.assert_called_once_with(results)
        mock_format_perf.assert_called_once_with(metadata)
        assert mock_console.print.call_count >= 1

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display.find_final_agent')
    def test_display_results_no_final_agent(
        self, mock_find_final: Mock, mock_console_class: Mock
    ) -> None:
        """Test display results when no final agent found."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_find_final.return_value = None

        results: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        agents: list[dict[str, Any]] = []

        display_results(results, metadata, agents, detailed=False)

        mock_console.print.assert_called_with("No results to display")

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display.find_final_agent')
    def test_display_results_final_agent_no_response(
        self, mock_find_final: Mock, mock_console_class: Mock
    ) -> None:
        """Test display results when final agent has no response."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_find_final.return_value = "agent_a"

        results = {"agent_a": {"status": "success"}}  # No response field
        metadata: dict[str, Any] = {}
        agents = [{"name": "agent_a"}]

        display_results(results, metadata, agents, detailed=False)

        mock_console.print.assert_called_with("No results to display")


class TestDisplayPlainTextResults:
    """Test plain text results display."""

    @patch('llm_orc.cli_modules.utils.visualization.results_display._display_detailed_plain_text')
    def test_display_plain_text_results_detailed(self, mock_detailed: Mock) -> None:
        """Test plain text display in detailed mode."""
        results = {"agent_a": {"status": "success"}}
        metadata: dict[str, Any] = {}
        agents = [{"name": "agent_a"}]

        display_plain_text_results(results, metadata, detailed=True, agents=agents)

        mock_detailed.assert_called_once_with(results, metadata, agents)

    @patch('llm_orc.cli_modules.utils.visualization.results_display._display_simplified_plain_text')
    def test_display_plain_text_results_simple(self, mock_simplified: Mock) -> None:
        """Test plain text display in simple mode."""
        results = {"agent_a": {"status": "success"}}
        metadata: dict[str, Any] = {}

        display_plain_text_results(results, metadata, detailed=False)

        mock_simplified.assert_called_once_with(results, metadata)

    @patch('llm_orc.cli_modules.utils.visualization.results_display._display_detailed_plain_text')
    def test_display_plain_text_results_no_agents(self, mock_detailed: Mock) -> None:
        """Test plain text display with no agents parameter."""
        results = {"agent_a": {"status": "success"}}
        metadata: dict[str, Any] = {}

        display_plain_text_results(results, metadata, detailed=True)

        mock_detailed.assert_called_once_with(results, metadata, [])


class TestDisplaySimplifiedResults:
    """Test simplified results display."""

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    def test_display_simplified_results_success(
        self, mock_echo: Mock, mock_console_class: Mock
    ) -> None:
        """Test simplified display with successful results."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        results = {
            "agent_a": {"status": "success", "response": "Hello"},
            "agent_b": {"status": "failed"},
        }
        metadata = {
            "usage": {"totals": {"agents_count": 2}},
            "duration": "5s"
        }

        display_simplified_results(results, metadata)

        mock_echo.assert_called()
        # Should show result from agent_a and performance summary
        calls = []
        for call in mock_echo.call_args_list:
            if call[0]:  # Check if call has positional arguments
                calls.append(call[0][0])
        assert any("Result from agent_a:" in call for call in calls)
        assert any("Hello" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    def test_display_simplified_results_no_success(
        self, mock_echo: Mock, mock_console_class: Mock
    ) -> None:
        """Test simplified display with no successful results."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        results = {
            "agent_a": {"status": "failed"},
            "agent_b": {"status": "error"},
        }
        metadata: dict[str, Any] = {}

        display_simplified_results(results, metadata)

        mock_echo.assert_called_with("❌ No successful results found")

    @patch('llm_orc.cli_modules.utils.visualization.results_display.Console')
    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    def test_display_simplified_results_with_performance(
        self, mock_echo: Mock, mock_console_class: Mock
    ) -> None:
        """Test simplified display includes performance info."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        results = {"agent_a": {"status": "success", "response": "Hello"}}
        metadata = {
            "usage": {"totals": {"agents_count": 1}},
            "duration": "2s"
        }

        display_simplified_results(results, metadata)

        calls = []
        for call in mock_echo.call_args_list:
            if call[0]:  # Check if call has positional arguments
                calls.append(call[0][0])
        assert any("⚡ 1 agents completed in 2s" in call for call in calls)


class TestProcessAgentResults:
    """Test processing agent results."""

    def test_process_agent_results_success(self) -> None:
        """Test processing successful agent results."""
        results = {
            "agent_a": {
                "status": "success",
                "response": "def hello(): pass",
                "error": ""
            }
        }

        processed = _process_agent_results(results)

        assert "agent_a" in processed
        assert processed["agent_a"]["status"] == "success"
        assert processed["agent_a"]["response"] == "def hello(): pass"
        assert processed["agent_a"]["error"] == ""
        assert processed["agent_a"]["has_code"]

    def test_process_agent_results_with_content_field(self) -> None:
        """Test processing results with content field instead of response."""
        results = {
            "agent_a": {
                "status": "success",
                "content": "Hello world",
            }
        }

        processed = _process_agent_results(results)

        assert processed["agent_a"]["response"] == "Hello world"
        assert not processed["agent_a"]["has_code"]

    def test_process_agent_results_with_error(self) -> None:
        """Test processing results with errors."""
        results = {
            "agent_a": {
                "status": "failed",
                "error": "Connection timeout"
            }
        }

        processed = _process_agent_results(results)

        assert processed["agent_a"]["status"] == "failed"
        assert processed["agent_a"]["error"] == "Connection timeout"
        assert processed["agent_a"]["response"] == ""

    def test_process_agent_results_defaults(self) -> None:
        """Test processing results with missing fields."""
        results: dict[str, Any] = {"agent_a": {}}

        processed = _process_agent_results(results)

        assert processed["agent_a"]["status"] == "unknown"
        assert processed["agent_a"]["response"] == ""
        assert processed["agent_a"]["error"] == ""
        assert not processed["agent_a"]["has_code"]


class TestFormatPerformanceMetrics:
    """Test formatting performance metrics."""

    def test_format_performance_metrics_with_usage(self) -> None:
        """Test formatting metrics with usage data."""
        metadata = {
            "usage": {
                "totals": {
                    "agents_count": 3,
                    "total_tokens": 1500,
                    "total_cost_usd": 0.0234
                }
            },
            "duration": "12.5s"
        }

        result = _format_performance_metrics(metadata)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "📊 Performance Summary" in result_str
        assert "Total agents: 3" in result_str
        assert "Total tokens: 1,500" in result_str
        assert "Total cost: $0.0234" in result_str
        assert "Duration: 12.5s" in result_str

    def test_format_performance_metrics_no_usage(self) -> None:
        """Test formatting metrics without usage data."""
        metadata = {"duration": "5s"}

        result = _format_performance_metrics(metadata)

        assert result == []

    def test_format_performance_metrics_empty(self) -> None:
        """Test formatting with empty metadata."""
        result = _format_performance_metrics({})

        assert result == []


class TestHasCodeContent:
    """Test code content detection."""

    def test_has_code_content_function_def(self) -> None:
        """Test detecting function definition."""
        text = "def hello_world(): return 'Hello'"

        assert _has_code_content(text)

    def test_has_code_content_class_def(self) -> None:
        """Test detecting class definition."""
        text = "class MyClass: pass"

        assert _has_code_content(text)

    def test_has_code_content_code_block(self) -> None:
        """Test detecting code block."""
        text = "Here's some code:\n```python\nprint('hello')\n```"

        assert _has_code_content(text)

    def test_has_code_content_import_statement(self) -> None:
        """Test detecting import statement."""
        text = "import os\nfrom typing import List"

        assert _has_code_content(text)

    def test_has_code_content_braces(self) -> None:
        """Test detecting braces."""
        text = "const obj = { key: 'value' };"

        assert _has_code_content(text)

    def test_has_code_content_regular_text(self) -> None:
        """Test regular text doesn't trigger detection."""
        text = "This is just regular text with no code indicators."

        assert not _has_code_content(text)

    def test_has_code_content_empty(self) -> None:
        """Test empty text."""
        assert not _has_code_content("")
        # Test empty string instead of None since function expects str
        assert not _has_code_content("")


class TestHelperDisplayFunctions:
    """Test helper display functions."""

    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    @patch('llm_orc.cli_modules.utils.visualization.results_display._display_plain_text_dependency_graph')
    @patch('llm_orc.cli_modules.utils.visualization.results_display._format_performance_metrics')
    def test_display_detailed_plain_text(
        self, mock_format_perf: Mock, mock_display_graph: Mock, mock_echo: Mock
    ) -> None:
        """Test detailed plain text display."""
        mock_format_perf.return_value = ["Performance: Good"]

        results = {
            "agent_a": {"status": "success", "response": "Hello"},
            "agent_b": {"status": "failed", "error": "Error occurred"}
        }
        metadata: dict[str, Any] = {}
        agents = [{"name": "agent_a"}, {"name": "agent_b"}]

        _display_detailed_plain_text(results, metadata, agents)

        mock_display_graph.assert_called_once_with(agents)
        mock_format_perf.assert_called_once_with(metadata)
        mock_echo.assert_called()

    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    def test_display_simplified_plain_text(self, mock_echo: Mock) -> None:
        """Test simplified plain text display."""
        results = {
            "agent_a": {"status": "success", "response": "Hello world"},
            "agent_b": {"status": "failed"}
        }
        metadata: dict[str, Any] = {}

        _display_simplified_plain_text(results, metadata)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Result from agent_a:" in call for call in calls)
        assert any("Hello world" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    def test_display_simplified_plain_text_no_success(self, mock_echo: Mock) -> None:
        """Test simplified plain text with no successful results."""
        results = {"agent_a": {"status": "failed"}}
        metadata: dict[str, Any] = {}

        _display_simplified_plain_text(results, metadata)

        mock_echo.assert_called_with("❌ No successful results found")

    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    @patch('llm_orc.cli_modules.utils.visualization.results_display._create_plain_text_dependency_graph')
    def test_display_plain_text_dependency_graph(
        self, mock_create_graph: Mock, mock_echo: Mock
    ) -> None:
        """Test plain text dependency graph display."""
        mock_create_graph.return_value = ["agent_a", "→", "agent_b"]
        agents = [{"name": "agent_a"}, {"name": "agent_b"}]

        _display_plain_text_dependency_graph(agents)

        mock_create_graph.assert_called_once_with(agents)
        mock_echo.assert_called()

    @patch('llm_orc.cli_modules.utils.visualization.results_display.click.echo')
    def test_display_plain_text_dependency_graph_empty(self, mock_echo: Mock) -> None:
        """Test plain text dependency graph with empty agents."""
        _display_plain_text_dependency_graph([])

        # Should not call echo since there are no agents
        assert mock_echo.call_count == 0  # Only "Agent Dependencies:" and empty line
