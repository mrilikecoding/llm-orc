"""Tests for JSON-first rendering architecture."""

from llm_orc.cli_modules.utils.json_renderer import (
    render_json_as_markdown,
    render_json_as_text,
    transform_to_execution_json,
)


class TestExecutionResultsTransformation:
    """Test transformation of raw execution data to structured JSON."""

    def test_transform_complete_execution_data(self) -> None:
        """Test transformation of complete execution result with all data types."""
        raw_results = {
            "agent1": {"content": "Result 1", "status": "success"},
            "agent2": {"content": "Result 2", "status": "success"},
        }

        raw_usage = {
            "agents": {
                "agent1": {"total_tokens": 150, "total_cost_usd": 0.001},
                "agent2": {"total_tokens": 200, "total_cost_usd": 0.002},
            },
            "totals": {"total_tokens": 350, "total_cost_usd": 0.003, "agents_count": 2},
        }

        raw_metadata = {
            "adaptive_resource_management": {
                "management_type": "user_configured",
                "concurrency_decisions": [{"configured_limit": 5, "agent_count": 2}],
                "execution_metrics": {
                    "peak_cpu": 45.2,
                    "avg_cpu": 32.1,
                    "peak_memory": 78.5,
                    "avg_memory": 65.3,
                    "sample_count": 12,
                },
                "phase_metrics": [
                    {
                        "phase_index": 0,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                        "peak_cpu": 40.0,
                        "avg_cpu": 30.0,
                    },
                    {
                        "phase_index": 1,
                        "agent_names": ["agent2"],
                        "duration_seconds": 0.8,
                        "final_cpu_percent": 25.0,
                    },
                ],
            }
        }

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)

        # Test top-level structure
        assert "execution_summary" in json_result
        assert "resource_management" in json_result
        assert "agent_results" in json_result
        assert "usage_summary" in json_result

        # Test execution summary
        summary = json_result["execution_summary"]
        assert summary["total_agents"] == 2
        assert summary["successful_agents"] == 2
        assert summary["failed_agents"] == 0

        # Test resource management with 1-based phase numbering
        rm = json_result["resource_management"]
        assert rm["type"] == "user_configured"
        assert rm["concurrency_limit"] == 5
        assert rm["execution_metrics"]["peak_cpu"] == 45.2
        assert len(rm["phases"]) == 2
        assert rm["phases"][0]["phase_number"] == 1  # 1-based for users
        assert rm["phases"][1]["phase_number"] == 2

        # Test agent results
        agents = json_result["agent_results"]
        assert len(agents) == 2
        assert agents[0]["name"] == "agent1"
        assert agents[0]["status"] == "success"
        assert agents[0]["content"] == "Result 1"

        # Test usage summary
        usage = json_result["usage_summary"]
        assert usage["total_tokens"] == 350
        assert usage["total_cost_usd"] == 0.003
        assert len(usage["per_agent"]) == 2

    def test_transform_handles_missing_data(self) -> None:
        """Test transformation handles missing or incomplete data gracefully."""
        # Minimal data
        raw_results = {"agent1": {"content": "Result", "status": "success"}}
        raw_usage = {"totals": {"total_tokens": 100}}
        raw_metadata: dict[
            str, dict[str, str | list[dict[str, str | int | float]]]
        ] = {}

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)

        # Should still have required structure with defaults
        assert json_result["execution_summary"]["total_agents"] == 1
        assert json_result["resource_management"]["type"] == "unknown"
        assert json_result["resource_management"]["concurrency_limit"] == 1
        assert json_result["usage_summary"]["total_tokens"] == 100


class TestJSONToPresentation:
    """Test rendering of structured JSON to text and markdown formats."""

    def test_render_json_as_text_consistent_formatting(self) -> None:
        """Test that JSON-to-text rendering produces consistent format."""
        structured_json = {
            "execution_summary": {"total_agents": 2, "successful_agents": 2},
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 5,
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                    }
                ],
            },
            "usage_summary": {"total_tokens": 350, "total_cost_usd": 0.003},
        }

        text_output = render_json_as_text(structured_json)

        # Test key formatting requirements
        assert "Phase 1:" in text_output  # 1-based numbering
        assert "Max concurrency limit used: 5" in text_output
        assert "Total tokens: 350" in text_output

    def test_render_json_as_markdown_consistent_formatting(self) -> None:
        """Test that JSON-to-markdown rendering produces consistent format."""
        structured_json = {
            "execution_summary": {"total_agents": 2, "successful_agents": 2},
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 5,
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                    }
                ],
            },
            "usage_summary": {"total_tokens": 350, "total_cost_usd": 0.003},
        }

        markdown_output = render_json_as_markdown(structured_json)

        # Test key formatting requirements
        assert "Phase 1:" in markdown_output  # 1-based numbering
        # Account for markdown bold
        assert "concurrency limit used:** 5" in markdown_output
        assert "tokens:** 350" in markdown_output

    def test_text_and_markdown_consistency(self) -> None:
        """Test text and markdown renderers produce consistent data representation."""
        structured_json = {
            "execution_summary": {"total_agents": 1},
            "resource_management": {"concurrency_limit": 3, "phases": []},
            "usage_summary": {"total_tokens": 100},
        }

        text_output = render_json_as_text(structured_json)
        markdown_output = render_json_as_markdown(structured_json)

        # Both should show same concurrency limit value (accounting for formatting)
        assert "concurrency limit used: 3" in text_output.lower()
        assert "concurrency limit used:** 3" in markdown_output.lower()

        # Both should show same token count
        assert "100" in text_output
        assert "100" in markdown_output


class TestRichMarkdownRendering:
    """Test comprehensive rich markdown rendering with full feature parity."""

    def test_render_complete_performance_summary(self) -> None:
        """Test that markdown renderer includes comprehensive performance data."""
        execution_json = {
            "execution_summary": {
                "total_agents": 3,
                "successful_agents": 3,
                "failed_agents": 0,
            },
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 5,
                "execution_metrics": {
                    "peak_cpu": 45.2,
                    "avg_cpu": 32.1,
                    "peak_memory": 78.5,
                    "avg_memory": 65.3,
                    "sample_count": 12,
                },
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1", "agent2"],
                        "duration_seconds": 1.5,
                        "peak_cpu": 40.0,
                        "avg_cpu": 30.0,
                        "peak_memory": 70.0,
                        "avg_memory": 60.0,
                        "sample_count": 5,
                    },
                    {
                        "phase_number": 2,
                        "agent_names": ["agent3"],
                        "duration_seconds": 0.8,
                        "final_cpu_percent": 25.0,
                        "final_memory_percent": 45.0,
                    },
                ],
            },
            "usage_summary": {
                "total_tokens": 450,
                "total_cost_usd": 0.005,
                "per_agent": [
                    {"name": "agent1", "tokens": 150, "cost_usd": 0.001},
                    {"name": "agent2", "tokens": 200, "cost_usd": 0.002},
                    {"name": "agent3", "tokens": 100, "cost_usd": 0.002},
                ],
            },
        }

        from llm_orc.cli_modules.utils.json_renderer import (
            render_comprehensive_markdown,
        )

        markdown_output = render_comprehensive_markdown(execution_json)

        # Test resource management section
        assert "### Resource Management" in markdown_output
        # Account for markdown bold formatting
        assert "concurrency limit used:** 5" in markdown_output
        assert "Peak usage:** CPU 45.2%" in markdown_output
        assert "Memory 78.5%" in markdown_output

        # Test per-phase metrics section
        assert "#### Per-Phase Performance" in markdown_output
        assert "**Phase 1**" in markdown_output
        assert "**Phase 2**" in markdown_output
        assert "agent1, agent2" in markdown_output  # Phase 1 agents
        assert "agent3" in markdown_output  # Phase 2 agents
        assert "1.5 seconds" in markdown_output  # Phase 1 duration
        assert "0.8 seconds" in markdown_output  # Phase 2 duration

        # Test per-agent usage section
        assert "### Per-Agent Usage" in markdown_output
        assert "**agent1**" in markdown_output
        assert "150 tokens" in markdown_output
        assert "$0.001" in markdown_output

    def test_render_handles_missing_phase_data_gracefully(self) -> None:
        """Test rendering handles missing or incomplete phase data."""
        execution_json = {
            "execution_summary": {"total_agents": 1},
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 3,
                "phases": [],  # No phase data
            },
            "usage_summary": {"total_tokens": 100, "per_agent": []},
        }

        from llm_orc.cli_modules.utils.json_renderer import (
            render_comprehensive_markdown,
        )

        markdown_output = render_comprehensive_markdown(execution_json)

        # Should still have main sections but gracefully handle missing data
        assert "### Resource Management" in markdown_output
        # Account for markdown bold formatting
        assert "concurrency limit used:** 3" in markdown_output
        # Should not crash or show empty phase sections
        assert "Phase 1" not in markdown_output
