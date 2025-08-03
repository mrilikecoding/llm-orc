"""Tests for performance data normalization and consistent rendering."""

from llm_orc.cli_modules.utils.performance_data import normalize_performance_data


class TestPerformanceDataNormalization:
    """Test standardized performance data normalization."""

    def test_normalize_basic_adaptive_stats(self) -> None:
        """Test normalization of basic adaptive stats structure."""
        adaptive_stats = {
            "management_type": "user_configured",
            "concurrency_decisions": [
                {"configured_limit": 5, "agent_count": 3}
            ],
            "execution_metrics": {
                "peak_cpu": 45.2,
                "avg_cpu": 32.1,
                "peak_memory": 78.5,
                "avg_memory": 65.3,
                "sample_count": 12
            },
            "phase_metrics": [
                {
                    "phase_index": 0,
                    "agent_names": ["agent1", "agent2"],
                    "agent_count": 2,
                    "duration_seconds": 1.5,
                    "peak_cpu": 40.0,
                    "avg_cpu": 30.0,
                    "peak_memory": 70.0,
                    "avg_memory": 60.0,
                    "sample_count": 5
                },
                {
                    "phase_index": 1,
                    "agent_names": ["agent3"],
                    "agent_count": 1,
                    "duration_seconds": 0.8,
                    "final_cpu_percent": 25.0,
                    "final_memory_percent": 45.0
                }
            ]
        }

        result = normalize_performance_data(adaptive_stats)

        # Test resource management normalization
        assert result["resource_management"]["type"] == "user_configured"
        assert result["resource_management"]["concurrency_limit"] == 5
        assert result["resource_management"]["has_decisions"] is True

        # Test execution metrics normalization
        assert result["execution_metrics"]["peak_cpu"] == 45.2
        assert result["execution_metrics"]["avg_cpu"] == 32.1
        assert result["execution_metrics"]["sample_count"] == 12

        # Test phase normalization with 1-based numbering
        assert len(result["phases"]) == 2
        assert result["phases"][0]["phase_number"] == 1  # 1-based for users
        assert result["phases"][0]["phase_index"] == 0   # Keep original for internal
        assert result["phases"][1]["phase_number"] == 2
        assert result["phases"][1]["phase_index"] == 1

        # Test phase data preservation
        assert result["phases"][0]["agent_names"] == ["agent1", "agent2"]
        assert result["phases"][0]["peak_cpu"] == 40.0
        assert result["phases"][1]["agent_names"] == ["agent3"]
        assert result["phases"][1]["final_cpu_percent"] == 25.0

        # Test flags
        assert result["has_monitoring_data"] is True
        assert result["has_phase_data"] is True

    def test_normalize_handles_missing_data(self) -> None:
        """Test normalization handles missing or incomplete data gracefully."""
        adaptive_stats = {
            "management_type": "user_configured",
            # No concurrency_decisions
            # No execution_metrics
            # No phase_metrics
        }

        result = normalize_performance_data(adaptive_stats)

        # Should provide sensible defaults
        assert result["resource_management"]["concurrency_limit"] == 1
        assert result["resource_management"]["has_decisions"] is False
        assert result["execution_metrics"]["peak_cpu"] == 0.0
        assert result["phases"] == []
        assert result["has_monitoring_data"] is False
        assert result["has_phase_data"] is False

    def test_normalize_handles_static_limit(self) -> None:
        """Test normalization handles static_limit field from legacy data."""
        adaptive_stats = {
            "management_type": "user_configured",
            "concurrency_decisions": [
                {"static_limit": 3, "agent_count": 2}  # Legacy field name
            ]
        }

        result = normalize_performance_data(adaptive_stats)

        # Should extract static_limit as concurrency_limit
        assert result["resource_management"]["concurrency_limit"] == 3


class TestConsistentRendering:
    """Test that all rendering formats use the same normalized data."""

    def test_phase_numbering_consistency(self) -> None:
        """Test that phase numbering is 1-based in all output formats."""
        # This test will fail initially - that's the point of TDD
        from llm_orc.cli_modules.utils.performance_data import (
            render_performance_markdown,
            render_performance_text,
        )

        adaptive_stats = {
            "phase_metrics": [
                {"phase_index": 0, "agent_names": ["agent1"], "duration_seconds": 1.0}
            ]
        }

        markdown_output = render_performance_markdown(adaptive_stats)
        text_output = render_performance_text(adaptive_stats)

        # Both should show "Phase 1", not "Phase 0"
        assert "Phase 1:" in markdown_output
        assert "Phase 1:" in text_output

    def test_concurrency_limit_consistency(self) -> None:
        """Test that concurrency limit display is consistent across formats."""
        from llm_orc.cli_modules.utils.performance_data import (
            render_performance_markdown,
            render_performance_text,
        )

        adaptive_stats = {
            "management_type": "user_configured",
            "concurrency_decisions": [{"configured_limit": 5}]
        }

        markdown_output = render_performance_markdown(adaptive_stats)
        text_output = render_performance_text(adaptive_stats)

        # Both should show the same concurrency limit format
        assert "Max concurrency limit used: 5" in markdown_output
        assert "Max concurrency limit used: 5" in text_output
