"""CLI visualization utilities for dependency graphs and execution display.

This module provides visualization utilities for the CLI, organized into logical submodules:
- dependency: Dependency graph and tree visualization
- results_display: Results display and formatting 
- performance_metrics: Performance and resource metrics display
- streaming: Streaming execution and event handling
"""

# Import all public functions to maintain backward compatibility
from .dependency import (
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
)

from .results_display import (
    display_results,
    display_plain_text_results,
    display_simplified_results,
    _process_agent_results,
    _format_performance_metrics,
    _display_detailed_plain_text,
    _display_simplified_plain_text,
    _display_plain_text_dependency_graph,
    _has_code_content,
)

from .performance_metrics import (
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
)

from .streaming import (
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

# Re-export everything for backward compatibility
__all__ = [
    # Dependency visualization
    "create_dependency_graph",
    "create_dependency_tree", 
    "create_dependency_graph_with_status",
    "find_final_agent",
    "_group_agents_by_dependency_level",
    "_calculate_agent_level",
    "_create_plain_text_dependency_graph",
    "_create_structured_dependency_info",
    "_create_agent_statuses",
    "_build_dependency_levels",
    
    # Results display
    "display_results",
    "display_plain_text_results",
    "display_simplified_results",
    "_process_agent_results", 
    "_format_performance_metrics",
    "_display_detailed_plain_text",
    "_display_simplified_plain_text",
    "_display_plain_text_dependency_graph",
    "_has_code_content",
    
    # Performance metrics
    "_format_adaptive_resource_metrics",
    "_format_per_phase_metrics",
    "_format_adaptive_with_decisions",
    "_format_adaptive_decision_details",
    "_format_static_with_decisions",
    "_format_adaptive_no_decisions",
    "_format_execution_metrics",
    "_format_execution_summary",
    "_format_static_no_decisions",
    "_display_adaptive_resource_metrics_text",
    "_display_phase_statistics",
    "_display_phase_resource_usage", 
    "_display_phase_timing",
    "_display_performance_guidance",
    "_display_execution_metrics",
    "_display_simplified_metrics",
    "_display_raw_samples",
    
    # Streaming execution
    "run_streaming_execution",
    "run_standard_execution", 
    "_run_text_json_execution",
    "_handle_streaming_event",
    "_process_execution_completed_event",
    "_update_agent_progress_status",
    "_update_agent_status_by_names",
    "_handle_fallback_started_event",
    "_handle_fallback_completed_event",
    "_handle_fallback_failed_event",
    "_handle_text_fallback_started",
    "_handle_text_fallback_completed",
    "_handle_text_fallback_failed",
]