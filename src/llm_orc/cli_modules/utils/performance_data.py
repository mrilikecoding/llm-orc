"""Standardized performance data structures and normalization for rendering."""

from typing import Any


def normalize_performance_data(adaptive_stats: dict[str, Any]) -> dict[str, Any]:
    """Normalize adaptive stats into a standard schema for consistent rendering.

    This is the single source of truth for performance data structure.
    All renderers should use this normalized data to ensure consistency.
    """
    # Extract basic info
    management_type = adaptive_stats.get("management_type", "user_configured")
    concurrency_decisions = adaptive_stats.get("concurrency_decisions", [])
    execution_metrics = adaptive_stats.get("execution_metrics", {})
    phase_metrics = adaptive_stats.get("phase_metrics", [])

    # Normalize concurrency information
    concurrency_limit = 1  # Default fallback
    if concurrency_decisions:
        # Use the most recent decision - could be configured_limit or static_limit
        decision = concurrency_decisions[0]
        concurrency_limit = decision.get(
            "configured_limit", decision.get("static_limit", 1)
        )

    # Normalize phase metrics with consistent numbering (1-based for users)
    normalized_phases = []
    for phase_data in phase_metrics:
        phase_index = phase_data.get("phase_index", 0)
        normalized_phase = {
            "phase_number": phase_index + 1,  # Convert to 1-based numbering
            "phase_index": phase_index,  # Keep original for internal use
            "agent_names": phase_data.get("agent_names", []),
            "agent_count": phase_data.get("agent_count", 0),
            "duration_seconds": phase_data.get("duration_seconds", 0.0),
            "start_time": phase_data.get("start_time"),
            "end_time": phase_data.get("end_time"),
            # Resource metrics - prefer peak/avg, fallback to final
            "peak_cpu": phase_data.get("peak_cpu"),
            "avg_cpu": phase_data.get("avg_cpu"),
            "peak_memory": phase_data.get("peak_memory"),
            "avg_memory": phase_data.get("avg_memory"),
            "sample_count": phase_data.get("sample_count", 0),
            "final_cpu_percent": phase_data.get("final_cpu_percent"),
            "final_memory_percent": phase_data.get("final_memory_percent"),
        }
        normalized_phases.append(normalized_phase)

    # Build normalized structure
    return {
        "resource_management": {
            "type": management_type,
            "concurrency_limit": concurrency_limit,
            "has_decisions": len(concurrency_decisions) > 0,
        },
        "execution_metrics": {
            "peak_cpu": execution_metrics.get("peak_cpu", 0.0),
            "avg_cpu": execution_metrics.get("avg_cpu", 0.0),
            "peak_memory": execution_metrics.get("peak_memory", 0.0),
            "avg_memory": execution_metrics.get("avg_memory", 0.0),
            "sample_count": execution_metrics.get("sample_count", 0),
        },
        "phases": normalized_phases,
        "has_monitoring_data": bool(execution_metrics),
        "has_phase_data": len(normalized_phases) > 0,
    }


def render_performance_markdown(adaptive_stats: dict[str, Any]) -> str:
    """Render performance data as markdown using normalized data structure."""
    normalized = normalize_performance_data(adaptive_stats)

    # Simple implementation to make tests pass
    output_lines = []

    # Resource management section
    rm = normalized["resource_management"]
    output_lines.append(f"Max concurrency limit used: {rm['concurrency_limit']}")

    # Phase information
    for phase in normalized["phases"]:
        output_lines.append(f"Phase {phase['phase_number']}:")

    return "\n".join(output_lines)


def render_performance_text(adaptive_stats: dict[str, Any]) -> str:
    """Render performance data as text using normalized data structure."""
    normalized = normalize_performance_data(adaptive_stats)

    # Simple implementation to make tests pass
    output_lines = []

    # Resource management section
    rm = normalized["resource_management"]
    output_lines.append(f"Max concurrency limit used: {rm['concurrency_limit']}")

    # Phase information
    for phase in normalized["phases"]:
        output_lines.append(f"Phase {phase['phase_number']}:")

    return "\n".join(output_lines)
