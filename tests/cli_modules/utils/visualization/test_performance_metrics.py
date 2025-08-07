"""Comprehensive tests for performance metrics module."""

from unittest.mock import Mock, patch

from llm_orc.cli_modules.utils.visualization.performance_metrics import (
    _display_adaptive_resource_metrics_text,
    _display_execution_metrics,
    _display_performance_guidance,
    _display_phase_resource_usage,
    _display_phase_statistics,
    _display_phase_timing,
    _display_raw_samples,
    _display_simplified_metrics,
    _format_adaptive_decision_details,
    _format_adaptive_resource_metrics,
    _format_adaptive_with_decisions,
    _format_execution_metrics,
    _format_execution_summary,
    _format_per_phase_metrics,
)


class TestFormatAdaptiveResourceMetrics:
    """Test adaptive resource metrics formatting."""

    def test_format_adaptive_resource_metrics_with_decisions(self) -> None:
        """Test formatting adaptive metrics with decisions."""
        adaptive_stats = {
            "management_type": "adaptive",
            "adaptive_used": True,
            "concurrency_decisions": [
                {
                    "configured_limit": 5,
                    "recommendation": {"limit": 3}
                }
            ],
            "execution_metrics": {
                "peak_cpu": 80.5,
                "avg_cpu": 65.2
            }
        }

        result = _format_adaptive_resource_metrics(adaptive_stats)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "🎯 Adaptive Resource Management" in result_str
        assert "Configured limit: 5" in result_str
        assert "Recommended limit: 3" in result_str

    def test_format_adaptive_resource_metrics_static(self) -> None:
        """Test formatting static resource metrics."""
        adaptive_stats = {
            "management_type": "static",
            "concurrency_decisions": [
                {"static_limit": 4}
            ],
            "execution_metrics": {
                "peak_memory": 75.0,
                "avg_memory": 60.0
            }
        }

        result = _format_adaptive_resource_metrics(adaptive_stats)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "⚙️  Static Resource Management" in result_str
        assert "Static concurrency limit: 4" in result_str

    def test_format_adaptive_resource_metrics_no_decisions(self) -> None:
        """Test formatting when no decisions were made."""
        adaptive_stats = {
            "management_type": "unknown",
            "execution_metrics": {
                "sample_count": 15
            }
        }

        result = _format_adaptive_resource_metrics(adaptive_stats)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "📊 Resource Management" in result_str
        assert "Management type: unknown" in result_str

    def test_format_adaptive_resource_metrics_empty(self) -> None:
        """Test formatting with empty stats."""
        result = _format_adaptive_resource_metrics({})

        assert result == []

    def test_format_adaptive_resource_metrics_none(self) -> None:
        """Test formatting with None stats."""
        result = _format_adaptive_resource_metrics({})

        assert result == []


class TestFormatPerPhaseMetrics:
    """Test per-phase metrics formatting."""

    def test_format_per_phase_metrics_complete(self) -> None:
        """Test formatting complete phase metrics."""
        phase_metrics = [
            {
                "phase_index": 0,
                "agent_count": 2,
                "agent_names": ["agent_a", "agent_b"],
                "sample_count": 10,
                "peak_cpu": 85.5,
                "avg_cpu": 70.2,
                "peak_memory": 65.8,
                "avg_memory": 55.3,
                "duration_seconds": 12.5
            },
            {
                "phase_index": 1,
                "agent_count": 1,
                "agent_names": ["agent_c"],
                "sample_count": 5,
                "peak_cpu": 45.0,
                "avg_cpu": 35.0,
                "duration_seconds": 8.2
            }
        ]

        result = _format_per_phase_metrics(phase_metrics)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "#### Per-Phase Performance" in result_str
        assert "**Phase 0** (2 agents)" in result_str
        assert "**Phase 1** (1 agents)" in result_str
        assert "agent_a, agent_b" in result_str
        assert "agent_c" in result_str
        assert "Duration:** 12.5 seconds" in result_str
        assert "CPU Usage:** 70.2% avg, 85.5% peak" in result_str
        assert "Memory Usage:** 55.3% avg, 65.8% peak" in result_str
        assert "Samples:** 10 monitoring points" in result_str

    def test_format_per_phase_metrics_minimal(self) -> None:
        """Test formatting minimal phase metrics."""
        phase_metrics = [
            {
                "phase_index": 0,
                "agent_count": 1,
                "agent_names": ["agent_a"]
            }
        ]

        result = _format_per_phase_metrics(phase_metrics)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "**Phase 0** (1 agents)" in result_str
        assert "agent_a" in result_str

    def test_format_per_phase_metrics_empty(self) -> None:
        """Test formatting empty phase metrics."""
        result = _format_per_phase_metrics([])

        assert result == []

    def test_format_per_phase_metrics_none(self) -> None:
        """Test formatting None phase metrics."""
        result = _format_per_phase_metrics([])

        assert result == []


class TestFormatAdaptiveWithDecisions:
    """Test adaptive formatting with decisions."""

    def test_format_adaptive_with_decisions_complete(self) -> None:
        """Test complete adaptive formatting with decisions."""
        adaptive_stats = {
            "concurrency_decisions": [
                {
                    "configured_limit": 8,
                    "recommendation": {"limit": 6}
                }
            ],
            "execution_metrics": {
                "peak_cpu": 90.0,
                "avg_cpu": 75.5,
                "sample_count": 20
            }
        }

        result = _format_adaptive_with_decisions(adaptive_stats)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "🎯 Adaptive Resource Management" in result_str
        assert "Configured limit: 8" in result_str
        assert "Recommended limit: 6" in result_str
        assert "CPU usage: 75.5% avg, 90.0% peak" in result_str

    def test_format_adaptive_with_decisions_no_decisions(self) -> None:
        """Test adaptive formatting without decisions."""
        adaptive_stats = {
            "concurrency_decisions": [],
            "execution_metrics": {"peak_cpu": 50.0}
        }

        result = _format_adaptive_with_decisions(adaptive_stats)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "🎯 Adaptive Resource Management" in result_str


class TestFormatAdaptiveDecisionDetails:
    """Test formatting adaptive decision details."""

    def test_format_adaptive_decision_details_configured_limit(self) -> None:
        """Test formatting decision with configured limit."""
        decision = {
            "configured_limit": 10,
            "recommendation": {"limit": 8}
        }

        result = _format_adaptive_decision_details(decision)

        assert "Configured limit: 10" in result
        assert "Recommended limit: 8" in result

    def test_format_adaptive_decision_details_static_limit(self) -> None:
        """Test formatting decision with static limit."""
        decision = {
            "static_limit": 6,
            "recommendation": {"limit": 4}
        }

        result = _format_adaptive_decision_details(decision)

        assert "Static limit: 6" in result
        assert "Recommended limit: 4" in result

    def test_format_adaptive_decision_details_no_recommendation(self) -> None:
        """Test formatting decision without recommendation."""
        decision = {"configured_limit": 5}

        result = _format_adaptive_decision_details(decision)

        assert "Configured limit: 5" in result
        assert len(result) == 1

    def test_format_adaptive_decision_details_empty(self) -> None:
        """Test formatting empty decision."""
        result = _format_adaptive_decision_details({})

        assert result == []


class TestFormatExecutionMetrics:
    """Test execution metrics formatting."""

    def test_format_execution_metrics_complete(self) -> None:
        """Test formatting complete execution metrics."""
        execution_metrics = {
            "peak_cpu": 88.7,
            "avg_cpu": 72.3,
            "peak_memory": 91.2,
            "avg_memory": 78.5,
            "sample_count": 25
        }

        result = _format_execution_metrics(execution_metrics)

        assert len(result) == 3
        assert "CPU usage: 72.3% avg, 88.7% peak" in result
        assert "Memory usage: 78.5% avg, 91.2% peak" in result
        assert "Monitoring samples: 25" in result

    def test_format_execution_metrics_cpu_only(self) -> None:
        """Test formatting with CPU metrics only."""
        execution_metrics = {
            "peak_cpu": 65.0,
            "avg_cpu": 50.0
        }

        result = _format_execution_metrics(execution_metrics)

        assert len(result) == 1
        assert "CPU usage: 50.0% avg, 65.0% peak" in result

    def test_format_execution_metrics_memory_only(self) -> None:
        """Test formatting with memory metrics only."""
        execution_metrics = {
            "peak_memory": 45.0,
            "avg_memory": 38.0
        }

        result = _format_execution_metrics(execution_metrics)

        assert len(result) == 1
        assert "Memory usage: 38.0% avg, 45.0% peak" in result

    def test_format_execution_metrics_empty(self) -> None:
        """Test formatting empty execution metrics."""
        result = _format_execution_metrics({})

        assert result == []


class TestFormatExecutionSummary:
    """Test execution summary formatting."""

    def test_format_execution_summary_complete(self) -> None:
        """Test complete execution summary."""
        execution_metrics = {
            "peak_cpu": 95.5,
            "avg_cpu": 82.1,
            "peak_memory": 87.3,
            "avg_memory": 71.8
        }

        result = _format_execution_summary(execution_metrics)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "Execution Performance:" in result_str
        assert "Peak CPU: 95.5%" in result_str
        assert "Average CPU: 82.1%" in result_str
        assert "Peak Memory: 87.3%" in result_str
        assert "Average Memory: 71.8%" in result_str

    def test_format_execution_summary_partial(self) -> None:
        """Test partial execution summary."""
        execution_metrics = {
            "peak_cpu": 60.0,
            "avg_memory": 40.0
        }

        result = _format_execution_summary(execution_metrics)

        assert len(result) > 0
        result_str = "\n".join(result)
        assert "Peak CPU: 60.0%" in result_str
        assert "Average Memory: 40.0%" in result_str

    def test_format_execution_summary_empty(self) -> None:
        """Test empty execution summary."""
        result = _format_execution_summary({})

        assert result == []


class TestDisplayFunctions:
    """Test display functions."""

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    @patch(
        'llm_orc.cli_modules.utils.visualization.performance_metrics'
        '._format_adaptive_resource_metrics'
    )
    def test_display_adaptive_resource_metrics_text(
        self, mock_format: Mock, mock_echo: Mock
    ) -> None:
        """Test displaying adaptive resource metrics in text."""
        mock_format.return_value = ["Line 1", "Line 2", "Line 3"]
        adaptive_stats = {"management_type": "adaptive"}

        _display_adaptive_resource_metrics_text(adaptive_stats)

        mock_format.assert_called_once_with(adaptive_stats)
        assert mock_echo.call_count == 3
        mock_echo.assert_any_call("Line 1")
        mock_echo.assert_any_call("Line 2")
        mock_echo.assert_any_call("Line 3")

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_adaptive_resource_metrics_text_empty(
        self, mock_echo: Mock
    ) -> None:
        """Test displaying empty adaptive resource metrics."""
        _display_adaptive_resource_metrics_text({})

        mock_echo.assert_not_called()

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    @patch(
        'llm_orc.cli_modules.utils.visualization.performance_metrics'
        '._display_phase_resource_usage'
    )
    @patch(
        'llm_orc.cli_modules.utils.visualization.performance_metrics'
        '._display_phase_timing'
    )
    def test_display_phase_statistics(
        self, mock_timing: Mock, mock_usage: Mock, mock_echo: Mock
    ) -> None:
        """Test displaying phase statistics."""
        phase_metrics = [
            {
                "phase_index": 0,
                "agent_names": ["agent_a", "agent_b"],
                "duration_seconds": 10.5
            }
        ]

        _display_phase_statistics(phase_metrics)

        mock_echo.assert_called()
        mock_usage.assert_called_once()
        mock_timing.assert_called_once()

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_phase_statistics_empty(self, mock_echo: Mock) -> None:
        """Test displaying empty phase statistics."""
        _display_phase_statistics([])

        mock_echo.assert_not_called()

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_phase_resource_usage_complete(self, mock_echo: Mock) -> None:
        """Test displaying complete phase resource usage."""
        phase_data = {
            "peak_cpu": 85.5,
            "avg_cpu": 70.2,
            "peak_memory": 65.8,
            "avg_memory": 55.3,
            "sample_count": 15
        }

        _display_phase_resource_usage(phase_data)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("CPU 70.2% (peak 85.5%)" in call for call in calls)
        assert any("Memory 55.3% (peak 65.8%)" in call for call in calls)
        assert any("15 samples collected" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_phase_resource_usage_fallback(self, mock_echo: Mock) -> None:
        """Test displaying phase resource usage with fallback values."""
        phase_data = {
            "final_cpu_percent": 60.0,
            "final_memory_percent": 45.0
        }

        _display_phase_resource_usage(phase_data)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("CPU 60.0%, Memory 45.0%" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_phase_timing_with_percentage(self, mock_echo: Mock) -> None:
        """Test displaying phase timing with percentage calculation."""
        phase_data = {"duration_seconds": 10.0}
        phase_metrics = [
            {"duration_seconds": 10.0},
            {"duration_seconds": 20.0},
            {"duration_seconds": 20.0}
        ]

        _display_phase_timing(phase_data, 0, phase_metrics)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Duration: 10.0 seconds" in call for call in calls)
        assert any("(20.0% of total execution time)" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_performance_guidance_low_cpu(self, mock_echo: Mock) -> None:
        """Test performance guidance for low CPU usage."""
        adaptive_stats = {
            "execution_metrics": {
                "peak_cpu": 40.0,
                "avg_cpu": 25.0,
                "peak_memory": 30.0
            }
        }

        _display_performance_guidance(adaptive_stats)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("💡 CPU utilization is low" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_performance_guidance_high_resources(self, mock_echo: Mock) -> None:
        """Test performance guidance for high resource usage."""
        adaptive_stats = {
            "execution_metrics": {
                "peak_cpu": 95.0,
                "avg_cpu": 85.0,
                "peak_memory": 90.0
            }
        }

        _display_performance_guidance(adaptive_stats)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("⚠️  High CPU usage detected" in call for call in calls)
        assert any("⚠️  High memory usage" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics._display_simplified_metrics')
    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics._display_raw_samples')
    def test_display_execution_metrics(
        self, mock_raw: Mock, mock_simplified: Mock, mock_echo: Mock
    ) -> None:
        """Test displaying execution metrics."""
        execution_metrics = {
            "peak_cpu": 80.0,
            "sample_count": 10
        }

        _display_execution_metrics(execution_metrics)

        mock_simplified.assert_called_once_with(execution_metrics)
        mock_raw.assert_called_once_with(execution_metrics)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_execution_metrics_empty(self, mock_echo: Mock) -> None:
        """Test displaying empty execution metrics."""
        _display_execution_metrics({})

        mock_echo.assert_called_with("No execution metrics available")

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_simplified_metrics(self, mock_echo: Mock) -> None:
        """Test displaying simplified metrics."""
        execution_metrics = {
            "peak_cpu": 75.5,
            "avg_cpu": 60.2,
            "peak_memory": 85.1,
            "avg_memory": 70.3
        }

        _display_simplified_metrics(execution_metrics)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("CPU: 60.2% avg, 75.5% peak" in call for call in calls)
        assert any("Memory: 70.3% avg, 85.1% peak" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_raw_samples(self, mock_echo: Mock) -> None:
        """Test displaying raw sample data."""
        execution_metrics = {
            "sample_count": 8,
            "cpu_samples": [60.0, 65.0, 70.0],
            "memory_samples": [45.0, 50.0, 55.0]
        }

        _display_raw_samples(execution_metrics)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Based on 8 monitoring samples" in call for call in calls)
        assert any("CPU samples:" in call for call in calls)
        assert any("Memory samples:" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.performance_metrics.click.echo')
    def test_display_raw_samples_large_dataset(self, mock_echo: Mock) -> None:
        """Test displaying raw samples with large dataset (should skip output)."""
        execution_metrics = {
            "sample_count": 5,
            "cpu_samples": [i * 10.0 for i in range(15)],  # 15 samples > 10 limit
            "memory_samples": [i * 5.0 for i in range(15)]
        }

        _display_raw_samples(execution_metrics)

        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Based on 5 monitoring samples" in call for call in calls)
        # Should not show individual samples since > 10
        assert not any("CPU samples:" in call for call in calls)
