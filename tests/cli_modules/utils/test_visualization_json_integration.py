"""Tests for complete JSON-first integration in visualization."""

from unittest.mock import Mock, patch

import pytest

from llm_orc.cli_modules.utils.visualization import (
    _display_detailed_plain_text,
    display_results,
)


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
            }
        }

        # Call the detailed display function that should use comprehensive rendering
        display_results(results, metadata, detailed=True)

        captured = capsys.readouterr()

        # Should show comprehensive performance data from JSON transformation
        assert "Phase 1" in captured.out  # 1-based numbering from JSON renderer
        assert "1.5 seconds" in captured.out  # Phase duration should be visible
        assert "Max concurrency limit used:" in captured.out
        assert "2" in captured.out

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

        with patch("llm_orc.cli_modules.utils.visualization.Console") as console_class:
            console_class.return_value = console_mock

            display_results(results, metadata, detailed=True)

            # Verify console.print was called with Markdown containing data
            assert console_mock.print.called
            markdown_call = console_mock.print.call_args[0][0]

            # Should contain comprehensive performance data from JSON transformation
            markdown_text = str(markdown_call.markup)
            assert "Phase 1" in markdown_text  # 1-based phase numbering
            assert "2.0 seconds" in markdown_text  # Phase duration
            assert "agent1" in markdown_text  # Agent names

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

        _display_detailed_plain_text(results, metadata)

        captured = capsys.readouterr()

        # Should show comprehensive execution metrics from JSON
        assert "Peak usage:" in captured.out or "55.0" in captured.out
        assert "3.2" in captured.out  # Phase duration should be shown

    def test_json_first_eliminates_duplicate_data_processing(self) -> None:
        """Test that visualization doesn't duplicate data processing logic."""
        # This test ensures we're not processing raw metadata in multiple places
        results = {"agent1": {"status": "success", "response": "Test response"}}
        usage = {"totals": {"total_tokens": 100}}
        metadata: dict[str, dict[str, list[dict[str, str]]]] = {
            "adaptive_resource_management": {"phase_metrics": []}
        }

        with patch(
            "llm_orc.cli_modules.utils.json_renderer.transform_to_execution_json"
        ) as transform_mock:
            # Mock should be called once to transform data
            transform_mock.return_value = {
                "execution_summary": {"total_agents": 1},
                "resource_management": {"concurrency_limit": 1, "phases": []},
                "usage_summary": {"total_tokens": 100},
                "agent_results": [],
            }

            _display_detailed_plain_text(results, metadata)

            # Verify transformation is called (single source of truth)
            transform_mock.assert_called_once_with(results, usage, metadata)

