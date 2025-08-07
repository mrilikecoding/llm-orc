"""Backward compatibility tests for CLI visualization utilities."""

from typing import Any

from llm_orc.cli_modules.utils.visualization import (
    _calculate_agent_level,
    # Performance metrics functions
    _format_adaptive_resource_metrics,
    _format_execution_metrics,
    _format_execution_summary,
    _group_agents_by_dependency_level,
    _has_code_content,
    _update_agent_progress_status,
    _update_agent_status_by_names,
    # Dependency functions
    create_dependency_graph,
    create_dependency_tree,
    display_results,
    find_final_agent,
    run_streaming_execution,
)


class TestBackwardCompatibility:
    """Test backward compatibility of visualization module imports."""

    def test_all_functions_importable(self) -> None:
        """Test that all functions are importable from the main visualization module."""
        # This test passes if all imports above succeed
        assert callable(create_dependency_graph)
        assert callable(display_results)
        assert callable(run_streaming_execution)
        assert callable(_format_adaptive_resource_metrics)

    def test_basic_functionality_works(self) -> None:
        """Test that basic functionality still works after refactoring."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # Test dependency graph creation
        graph = create_dependency_graph(agents)
        assert isinstance(graph, str)
        assert "agent_a" in graph or "agent_b" in graph

        # Test dependency tree creation
        tree = create_dependency_tree(agents)
        assert tree is not None

        # Test final agent finding
        results = {
            "agent_a": {"status": "success"},
            "coordinator": {"status": "success"},
        }
        final = find_final_agent(results)
        assert final == "coordinator"

    def test_helper_functions_work(self) -> None:
        """Test that helper functions work correctly."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]

        # Test grouping by dependency level
        grouped = _group_agents_by_dependency_level(agents)
        assert 0 in grouped
        assert 1 in grouped

        # Test calculating agent level
        level = _calculate_agent_level(agents[0], agents)
        assert level == 0

        level = _calculate_agent_level(agents[1], agents)
        assert level == 1

    def test_code_content_detection(self) -> None:
        """Test code content detection function."""
        assert _has_code_content("def hello(): pass")
        assert _has_code_content("class MyClass: pass")
        assert _has_code_content("```python\nprint('hi')\n```")
        assert not _has_code_content("Just regular text")
        assert not _has_code_content("")

    def test_execution_metrics_formatting(self) -> None:
        """Test execution metrics formatting functions."""
        metrics = {
            "peak_cpu": 80.5,
            "avg_cpu": 65.2,
            "peak_memory": 75.0,
            "avg_memory": 60.0,
            "sample_count": 10
        }

        formatted = _format_execution_metrics(metrics)
        assert isinstance(formatted, list)
        assert len(formatted) > 0

        summary = _format_execution_summary(metrics)
        assert isinstance(summary, list)

    def test_adaptive_resource_metrics_formatting(self) -> None:
        """Test adaptive resource metrics formatting."""
        stats = {
            "management_type": "adaptive",
            "adaptive_used": True,
            "execution_metrics": {
                "peak_cpu": 90.0,
                "avg_cpu": 75.0
            }
        }

        formatted = _format_adaptive_resource_metrics(stats)
        assert isinstance(formatted, list)

    def test_agent_status_functions(self) -> None:
        """Test agent status update functions."""
        agent_progress: dict[str, dict[str, Any]] = {}

        # Test single agent status update
        _update_agent_progress_status("agent_a", "✅ Completed", agent_progress)
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "✅ Completed"

        # Test multiple agent status update
        _update_agent_status_by_names(
            ["agent_b", "agent_c"], "🔄 Running", agent_progress
        )
        assert "agent_b" in agent_progress
        assert "agent_c" in agent_progress
        assert agent_progress["agent_b"]["status"] == "🔄 Running"
        assert agent_progress["agent_c"]["status"] == "🔄 Running"
