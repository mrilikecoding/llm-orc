"""Backward compatibility tests for CLI visualization utilities."""

import pytest

from llm_orc.cli_modules.utils.visualization import (
    # Dependency functions
    create_dependency_graph,
    create_dependency_tree,
    create_dependency_graph_with_status,
    find_final_agent,
    _group_agents_by_dependency_level,
    _calculate_agent_level,
    _create_plain_text_dependency_graph,
    _create_structured_dependency_info,
    _create_agent_statuses,
    _build_dependency_levels,
    
    # Results display functions
    display_results,
    display_plain_text_results,
    display_simplified_results,
    _process_agent_results, 
    _format_performance_metrics,
    _display_detailed_plain_text,
    _display_simplified_plain_text,
    _display_plain_text_dependency_graph,
    _has_code_content,
    
    # Performance metrics functions
    _format_adaptive_resource_metrics,
    _format_per_phase_metrics,
    _format_adaptive_with_decisions,
    _format_adaptive_decision_details,
    _format_static_with_decisions,
    _format_adaptive_no_decisions,
    _format_execution_metrics,
    _format_execution_summary,
    _format_static_no_decisions,
    _display_adaptive_resource_metrics_text,
    _display_phase_statistics,
    _display_phase_resource_usage, 
    _display_phase_timing,
    _display_performance_guidance,
    _display_execution_metrics,
    _display_simplified_metrics,
    _display_raw_samples,
    
    # Streaming functions
    run_streaming_execution,
    run_standard_execution, 
    _run_text_json_execution,
    _handle_streaming_event,
    _process_execution_completed_event,
    _update_agent_progress_status,
    _update_agent_status_by_names,
    _handle_fallback_started_event,
    _handle_fallback_completed_event,
    _handle_fallback_failed_event,
    _handle_text_fallback_started,
    _handle_text_fallback_completed,
    _handle_text_fallback_failed,
)


class TestBackwardCompatibility:
    """Test backward compatibility of visualization module imports."""

    def test_all_functions_importable(self):
        """Test that all functions are importable from the main visualization module."""
        # This test passes if all imports above succeed
        assert callable(create_dependency_graph)
        assert callable(display_results)
        assert callable(run_streaming_execution)
        assert callable(_format_adaptive_resource_metrics)

    def test_basic_functionality_works(self):
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

    def test_helper_functions_work(self):
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

    def test_code_content_detection(self):
        """Test code content detection function."""
        assert _has_code_content("def hello(): pass") == True
        assert _has_code_content("class MyClass: pass") == True
        assert _has_code_content("```python\nprint('hi')\n```") == True
        assert _has_code_content("Just regular text") == False
        assert _has_code_content("") == False

    def test_execution_metrics_formatting(self):
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

    def test_adaptive_resource_metrics_formatting(self):
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

    def test_agent_status_functions(self):
        """Test agent status update functions."""
        agent_progress = {}
        
        # Test single agent status update
        _update_agent_progress_status("agent_a", "✅ Completed", agent_progress)
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "✅ Completed"
        
        # Test multiple agent status update
        _update_agent_status_by_names(["agent_b", "agent_c"], "🔄 Running", agent_progress)
        assert "agent_b" in agent_progress
        assert "agent_c" in agent_progress
        assert agent_progress["agent_b"]["status"] == "🔄 Running"
        assert agent_progress["agent_c"]["status"] == "🔄 Running"