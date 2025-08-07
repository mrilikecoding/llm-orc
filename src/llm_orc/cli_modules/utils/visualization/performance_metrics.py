"""Performance metrics visualization and formatting."""

from typing import Any

import click


def _format_adaptive_resource_metrics(adaptive_stats: dict[str, Any]) -> list[str]:
    """Format adaptive resource management metrics for display."""
    lines: list[str] = []

    if not adaptive_stats:
        return lines

    management_type = adaptive_stats.get("management_type", "unknown")

    if management_type == "adaptive" and adaptive_stats.get("adaptive_used"):
        lines.extend(_format_adaptive_with_decisions(adaptive_stats))
    elif management_type == "static":
        lines.extend(_format_static_with_decisions(adaptive_stats))
    else:
        lines.extend(_format_adaptive_no_decisions(adaptive_stats))

    return lines


def _format_per_phase_metrics(phase_metrics: list[dict[str, Any]]) -> list[str]:
    """Format per-phase performance metrics."""
    if not phase_metrics:
        return []

    content = ["\n#### Per-Phase Performance\n"]

    for phase_data in phase_metrics:
        phase_index = phase_data.get("phase_index", "Unknown")
        agent_count = phase_data.get("agent_count", 0)
        agent_names = phase_data.get("agent_names", [])
        sample_count = phase_data.get("sample_count", 0)

        # Get resource usage metrics
        peak_cpu = phase_data.get("peak_cpu", 0.0)
        avg_cpu = phase_data.get("avg_cpu", 0.0)
        peak_memory = phase_data.get("peak_memory", 0.0)
        avg_memory = phase_data.get("avg_memory", 0.0)

        content.append(f"**Phase {phase_index}** ({agent_count} agents)\n")
        content.append(f"- **Agents:** {', '.join(agent_names)}\n")

        # Show duration if available
        duration = phase_data.get("duration_seconds")
        if duration is not None:
            content.append(f"- **Duration:** {duration:.1f} seconds\n")

        # Show resource usage if meaningful data is available
        if peak_cpu > 0 or avg_cpu > 0:
            content.append(
                f"- **CPU Usage:** {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak\n"
            )

        if peak_memory > 0 or avg_memory > 0:
            content.append(
                f"- **Memory Usage:** {avg_memory:.1f}% avg, {peak_memory:.1f}% peak\n"
            )

        if sample_count > 0:
            content.append(f"- **Samples:** {sample_count} monitoring points\n")

        content.append("\n")

    return content


def _format_adaptive_with_decisions(adaptive_stats: dict[str, Any]) -> list[str]:
    """Format adaptive resource management with decisions."""
    lines: list[str] = []
    lines.append("\n🎯 Adaptive Resource Management")
    lines.append("=" * 50)

    # Show final decision
    concurrency_decisions = adaptive_stats.get("concurrency_decisions", [])
    if concurrency_decisions:
        final_decision = concurrency_decisions[0]
        lines.extend(_format_adaptive_decision_details(final_decision))

    # Show execution metrics
    execution_metrics = adaptive_stats.get("execution_metrics", {})
    if execution_metrics:
        lines.extend(_format_execution_metrics(execution_metrics))

    return lines


def _format_adaptive_decision_details(final_decision: dict[str, Any]) -> list[str]:
    """Format adaptive decision details."""
    lines: list[str] = []

    configured_limit = final_decision.get("configured_limit")
    static_limit = final_decision.get("static_limit")
    recommendation = final_decision.get("recommendation", {})

    if configured_limit is not None:
        lines.append(f"Configured limit: {configured_limit}")
    elif static_limit is not None:
        lines.append(f"Static limit: {static_limit}")

    if recommendation:
        rec_limit = recommendation.get("limit")
        if rec_limit is not None:
            lines.append(f"Recommended limit: {rec_limit}")

    return lines


def _format_static_with_decisions(adaptive_stats: dict[str, Any]) -> list[str]:
    """Format static resource management with decisions."""
    lines: list[str] = []
    lines.append("\n⚙️  Static Resource Management")
    lines.append("=" * 50)

    concurrency_decisions = adaptive_stats.get("concurrency_decisions", [])
    if concurrency_decisions:
        decision = concurrency_decisions[0]
        static_limit = decision.get("static_limit")
        if static_limit is not None:
            lines.append(f"Static concurrency limit: {static_limit}")

    execution_metrics = adaptive_stats.get("execution_metrics", {})
    if execution_metrics:
        lines.extend(_format_execution_metrics(execution_metrics))

    return lines


def _format_adaptive_no_decisions(adaptive_stats: dict[str, Any]) -> list[str]:
    """Format adaptive management when no decisions were made."""
    lines: list[str] = []
    lines.append("\n📊 Resource Management")
    lines.append("=" * 50)

    management_type = adaptive_stats.get("management_type", "unknown")
    lines.append(f"Management type: {management_type}")

    execution_metrics = adaptive_stats.get("execution_metrics", {})
    if execution_metrics:
        lines.extend(_format_execution_metrics(execution_metrics))

    return lines


def _format_execution_metrics(execution_metrics: dict[str, Any]) -> list[str]:
    """Format execution performance metrics."""
    lines: list[str] = []

    peak_cpu = execution_metrics.get("peak_cpu", 0.0)
    avg_cpu = execution_metrics.get("avg_cpu", 0.0)
    peak_memory = execution_metrics.get("peak_memory", 0.0)
    avg_memory = execution_metrics.get("avg_memory", 0.0)
    sample_count = execution_metrics.get("sample_count", 0)

    if peak_cpu > 0 or avg_cpu > 0:
        lines.append(f"CPU usage: {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak")

    if peak_memory > 0 or avg_memory > 0:
        lines.append(f"Memory usage: {avg_memory:.1f}% avg, {peak_memory:.1f}% peak")

    if sample_count > 0:
        lines.append(f"Monitoring samples: {sample_count}")

    return lines


def _format_execution_summary(execution_metrics: dict[str, Any]) -> list[str]:
    """Format execution summary metrics."""
    lines: list[str] = []

    if not execution_metrics:
        return lines

    lines.append("\nExecution Performance:")

    peak_cpu = execution_metrics.get("peak_cpu", 0.0)
    avg_cpu = execution_metrics.get("avg_cpu", 0.0)

    if peak_cpu > 0:
        lines.append(f"  Peak CPU: {peak_cpu:.1f}%")
    if avg_cpu > 0:
        lines.append(f"  Average CPU: {avg_cpu:.1f}%")

    peak_memory = execution_metrics.get("peak_memory", 0.0)
    avg_memory = execution_metrics.get("avg_memory", 0.0)

    if peak_memory > 0:
        lines.append(f"  Peak Memory: {peak_memory:.1f}%")
    if avg_memory > 0:
        lines.append(f"  Average Memory: {avg_memory:.1f}%")

    return lines


def _format_static_no_decisions(adaptive_stats: dict[str, Any]) -> list[str]:
    """Format static management when no decisions available."""
    lines: list[str] = []
    lines.append("\n📊 Resource Management")
    lines.append("=" * 50)
    lines.append("Static resource management (no adaptive decisions)")

    execution_metrics = adaptive_stats.get("execution_metrics", {})
    if execution_metrics:
        lines.extend(_format_execution_metrics(execution_metrics))

    return lines


def _display_adaptive_resource_metrics_text(adaptive_stats: dict[str, Any]) -> None:
    """Display adaptive resource metrics in plain text."""
    if not adaptive_stats:
        return

    lines = _format_adaptive_resource_metrics(adaptive_stats)
    for line in lines:
        click.echo(line)


def _display_phase_statistics(phase_metrics: list[dict[str, Any]]) -> None:
    """Display per-phase performance statistics."""
    if not phase_metrics:
        return

    click.echo("\nPer-Phase Performance:")
    click.echo("=" * 30)

    for phase_data in phase_metrics:
        phase_index = phase_data.get("phase_index", 0)
        agent_names = phase_data.get("agent_names", [])
        duration = phase_data.get("duration_seconds", 0.0)
        agent_count = len(agent_names)

        click.echo(f"\nPhase {phase_index + 1} ({agent_count} agents)")
        if agent_names:
            click.echo(f"  Agents: {', '.join(agent_names)}")

        click.echo(f"  Duration: {duration:.1f} seconds")

        # Display resource usage
        _display_phase_resource_usage(phase_data)
        _display_phase_timing(phase_data, phase_index, phase_metrics)


def _display_phase_resource_usage(phase_data: dict[str, Any]) -> None:
    """Display resource usage for a phase."""
    peak_cpu = phase_data.get("peak_cpu")
    avg_cpu = phase_data.get("avg_cpu")
    peak_memory = phase_data.get("peak_memory")
    avg_memory = phase_data.get("avg_memory")
    sample_count = phase_data.get("sample_count", 0)

    if peak_cpu is not None and avg_cpu is not None and sample_count > 0:
        click.echo(
            f"  Resource usage: CPU {avg_cpu:.1f}% (peak {peak_cpu:.1f}%), "
            f"Memory {avg_memory:.1f}% (peak {peak_memory:.1f}%)"
        )
        click.echo(f"  Monitoring: {sample_count} samples collected")
    else:
        final_cpu = phase_data.get("final_cpu_percent")
        final_memory = phase_data.get("final_memory_percent")
        if final_cpu is not None and final_memory is not None:
            click.echo(
                f"  Resource usage: CPU {final_cpu:.1f}%, Memory {final_memory:.1f}%"
            )


def _display_phase_timing(
    phase_data: dict[str, Any], phase_index: int, phase_metrics: list[dict[str, Any]]
) -> None:
    """Display timing information for a phase."""
    duration = phase_data.get("duration_seconds")
    if duration is None:
        click.echo("  Duration: Not measured")
    else:
        click.echo(f"  Duration: {duration:.1f} seconds")

        # Show timing relative to other phases
        if len(phase_metrics) > 1:
            total_duration = sum(
                p.get("duration_seconds", 0) for p in phase_metrics
            )
            if total_duration > 0:
                percentage = (duration / total_duration) * 100
                click.echo(f"  ({percentage:.1f}% of total execution time)")


def _display_performance_guidance(adaptive_stats: dict[str, Any]) -> None:
    """Display performance optimization guidance."""
    execution_metrics = adaptive_stats.get("execution_metrics", {})

    if not execution_metrics:
        return

    peak_cpu = execution_metrics.get("peak_cpu", 0.0)
    avg_cpu = execution_metrics.get("avg_cpu", 0.0)
    peak_memory = execution_metrics.get("peak_memory", 0.0)

    suggestions = []

    if peak_cpu < 50 and avg_cpu < 30:
        suggestions.append(
            "💡 CPU utilization is low - consider increasing concurrency"
        )
    elif peak_cpu > 90:
        suggestions.append("⚠️  High CPU usage detected - consider reducing concurrency")

    if peak_memory > 85:
        suggestions.append("⚠️  High memory usage - monitor for memory pressure")

    if suggestions:
        click.echo("\nPerformance Suggestions:")
        for suggestion in suggestions:
            click.echo(f"  {suggestion}")


def _display_execution_metrics(execution_metrics: dict[str, Any]) -> None:
    """Display execution metrics in detail."""
    if not execution_metrics:
        click.echo("No execution metrics available")
        return

    click.echo("Execution Metrics:")
    click.echo("-" * 20)

    # Show basic metrics
    _display_simplified_metrics(execution_metrics)

    # Show detailed sampling data if available
    if execution_metrics.get("sample_count", 0) > 0:
        _display_raw_samples(execution_metrics)


def _display_simplified_metrics(execution_metrics: dict[str, Any]) -> None:
    """Display simplified execution metrics."""
    peak_cpu = execution_metrics.get("peak_cpu", 0.0)
    avg_cpu = execution_metrics.get("avg_cpu", 0.0)
    peak_memory = execution_metrics.get("peak_memory", 0.0)
    avg_memory = execution_metrics.get("avg_memory", 0.0)

    if peak_cpu > 0:
        click.echo(f"  CPU: {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak")
    if peak_memory > 0:
        click.echo(f"  Memory: {avg_memory:.1f}% avg, {peak_memory:.1f}% peak")


def _display_raw_samples(execution_metrics: dict[str, Any]) -> None:
    """Display raw sampling data."""
    sample_count = execution_metrics.get("sample_count", 0)
    if sample_count > 0:
        click.echo(f"  Based on {sample_count} monitoring samples")

    # Show additional raw data if available
    cpu_samples = execution_metrics.get("cpu_samples", [])
    memory_samples = execution_metrics.get("memory_samples", [])

    if cpu_samples and len(cpu_samples) <= 10:  # Only show for small datasets
        click.echo(f"  CPU samples: {[f'{s:.1f}' for s in cpu_samples]}")

    if memory_samples and len(memory_samples) <= 10:
        click.echo(f"  Memory samples: {[f'{s:.1f}' for s in memory_samples]}")
