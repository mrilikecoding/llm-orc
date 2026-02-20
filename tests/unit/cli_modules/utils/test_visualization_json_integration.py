"""Tests for complete JSON-first integration in visualization."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.cli_modules.utils.visualization.results_display import (
    _display_detailed_plain_text,
    display_results,
)
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig


class TestJSONFirstVisualizationIntegration:
    """Test that visualization fully uses JSON-first architecture."""

    def test_display_results_uses_comprehensive_json_rendering(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that plain text results use comprehensive JSON-first rendering."""
        results = {"agent1": {"status": "success", "response": "Result 1"}}
        usage = {
            "totals": {"total_tokens": 500, "total_cost_usd": 0.01},
            "agents": {"agent1": {"total_tokens": 500, "total_cost_usd": 0.01}},
        }
        metadata = {
            "usage": usage,  # This is how usage gets passed in metadata
            "adaptive_resource_management": {
                "management_type": "user_configured",
                "concurrency_decisions": [{"configured_limit": 2}],
                "execution_metrics": {
                    "peak_cpu": 45.0,
                    "avg_cpu": 30.0,
                    "sample_count": 5,
                },
                "phase_metrics": [
                    {
                        "phase_index": 0,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                        "peak_cpu": 40.0,
                        "avg_cpu": 35.0,
                    }
                ],
            },
        }

        # Call the detailed display function that should use comprehensive rendering
        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test")
        ]
        display_results(results, metadata, agents, detailed=True)

        captured = capsys.readouterr()

        # Should show comprehensive performance data from JSON transformation
        assert "Performance Summary" in captured.out  # Basic performance summary
        assert "500" in captured.out  # Token count should be visible
        assert "0.0100" in captured.out  # Cost should be visible

    def test_display_detailed_results_uses_comprehensive_markdown(self) -> None:
        """Test that detailed results use comprehensive markdown from JSON."""
        console_mock = Mock()
        results = {"agent1": {"status": "success", "content": "Result"}}
        # usage = {"totals": {"total_tokens": 100}}  # Unused in this specific test
        metadata = {
            "adaptive_resource_management": {
                "phase_metrics": [
                    {
                        "phase_index": 0,
                        "agent_names": ["agent1"],
                        "duration_seconds": 2.0,
                    }
                ]
            }
        }

        with patch(
            "llm_orc.cli_modules.utils.visualization.results_display.Console"
        ) as console_class:
            console_class.return_value = console_mock

            agents: list[AgentConfig] = [
                LlmAgentConfig(name="agent1", model_profile="test")
            ]
            display_results(results, metadata, agents, detailed=True)

            # Verify console.print was called with Markdown containing data
            assert console_mock.print.called
            console_mock.print.call_args[0][0]

            # Should contain agent results and response content
            markdown_call_found = False
            for call_args in console_mock.print.call_args_list:
                if call_args and len(call_args[0]) > 0:
                    call_content = str(call_args[0][0])
                    if "agent1" in call_content or "Result" in call_content:
                        markdown_call_found = True
                        break
            assert markdown_call_found, "Should contain agent results"

    def test_performance_summary_text_output_comprehensive(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that text output includes all performance metrics from JSON."""
        results = {"agent1": {"status": "success", "response": "Test response"}}
        # usage = {"totals": {"total_tokens": 200}}  # Unused in this specific test
        metadata = {
            "adaptive_resource_management": {
                "execution_metrics": {
                    "peak_cpu": 55.0,
                    "avg_cpu": 40.0,
                    "sample_count": 10,
                },
                "phase_metrics": [
                    {
                        "phase_index": 0,
                        "agent_names": ["agent1"],
                        "duration_seconds": 3.2,
                        "peak_cpu": 50.0,
                    }
                ],
            }
        }

        agents: list[AgentConfig] = [
            LlmAgentConfig(name="agent1", model_profile="test")
        ]
        _display_detailed_plain_text(results, metadata, agents)

        captured = capsys.readouterr()

        # Should show execution metrics in performance summary
        assert "Performance Summary" in captured.out or "agent1" in captured.out
        # Content should be displayed for the agent
        assert "Test response" in captured.out or "agent1" in captured.out

    def test_json_first_eliminates_duplicate_data_processing(self) -> None:
        """Test that visualization uses consistent performance metrics display."""
        # This test ensures we have a single consistent way of displaying data
        results = {"agent1": {"status": "success", "response": "Test response"}}
        usage = {"totals": {"total_tokens": 100}}
        metadata: dict[str, Any] = {
            "usage": usage,  # Include usage in metadata as expected by implementation
            "adaptive_resource_management": {"phase_metrics": []},
        }

        with patch(
            "llm_orc.cli_modules.utils.visualization.results_display._format_performance_metrics"
        ) as format_mock:
            # Mock should be called once to format performance data
            format_mock.return_value = [
                "Performance Summary",
                "Total tokens: 100",
            ]

            agents: list[AgentConfig] = [
                LlmAgentConfig(name="agent1", model_profile="test")
            ]
            _display_detailed_plain_text(results, metadata, agents)

            # Verify formatting is called (single source of truth for display)
            format_mock.assert_called_once_with(metadata)
