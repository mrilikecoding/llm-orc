"""JSON-first rendering architecture for consistent output across all formats."""

from typing import Any


def transform_to_execution_json(
    results: dict[str, Any], usage: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Transform raw execution data into structured JSON format.

    This is the canonical data transformation that serves as single source of truth
    for all presentation formats.
    """
    # Extract adaptive resource management data
    adaptive_stats = metadata.get("adaptive_resource_management", {})

    # Build execution summary
    successful_agents = sum(
        1 for result in results.values() if result.get("status") == "success"
    )
    total_agents = len(results)

    execution_summary = {
        "total_agents": total_agents,
        "successful_agents": successful_agents,
        "failed_agents": total_agents - successful_agents,
    }

    # Build resource management with normalized data
    concurrency_decisions = adaptive_stats.get("concurrency_decisions", [])
    concurrency_limit = 1
    if concurrency_decisions:
        decision = concurrency_decisions[0]
        concurrency_limit = decision.get(
            "configured_limit", decision.get("static_limit", 1)
        )

    execution_metrics = adaptive_stats.get("execution_metrics", {})
    phase_metrics = adaptive_stats.get("phase_metrics", [])

    # Normalize phases with 1-based numbering
    normalized_phases = []
    for phase_data in phase_metrics:
        phase_index = phase_data.get("phase_index", 0)
        normalized_phase = {
            "phase_number": phase_index + 1,  # 1-based for users
            "agent_names": phase_data.get("agent_names", []),
            "duration_seconds": phase_data.get("duration_seconds", 0.0),
            "peak_cpu": phase_data.get("peak_cpu"),
            "avg_cpu": phase_data.get("avg_cpu"),
            "peak_memory": phase_data.get("peak_memory"),
            "avg_memory": phase_data.get("avg_memory"),
            "sample_count": phase_data.get("sample_count", 0),
            "final_cpu_percent": phase_data.get("final_cpu_percent"),
            "final_memory_percent": phase_data.get("final_memory_percent"),
        }
        normalized_phases.append(normalized_phase)

    # Compute overall execution metrics from phase data if available
    # This provides more accurate and representative overall statistics
    computed_execution_metrics = {
        "peak_cpu": 0.0,
        "avg_cpu": 0.0,
        "peak_memory": 0.0,
        "avg_memory": 0.0,
        "has_phase_data": False,
    }

    if normalized_phases:
        # Take maximum peak values across all phases
        peak_cpus: list[float] = []
        peak_memories: list[float] = []

        for p in normalized_phases:
            peak_cpu = p.get("peak_cpu")
            if peak_cpu is not None:
                peak_cpus.append(float(peak_cpu))
            peak_memory = p.get("peak_memory")
            if peak_memory is not None:
                peak_memories.append(float(peak_memory))

        if peak_cpus:
            computed_execution_metrics["peak_cpu"] = max(peak_cpus)
        if peak_memories:
            computed_execution_metrics["peak_memory"] = max(peak_memories)

        # Compute weighted average of averages (weighted by duration)
        total_duration = sum(p.get("duration_seconds", 0.0) for p in normalized_phases)
        if total_duration > 0:
            weighted_cpu = sum(
                p.get("avg_cpu", 0.0) * p.get("duration_seconds", 0.0)
                for p in normalized_phases
                if p.get("avg_cpu") is not None
            )
            weighted_memory = sum(
                p.get("avg_memory", 0.0) * p.get("duration_seconds", 0.0)
                for p in normalized_phases
                if p.get("avg_memory") is not None
            )
            computed_execution_metrics["avg_cpu"] = weighted_cpu / total_duration
            computed_execution_metrics["avg_memory"] = weighted_memory / total_duration
            computed_execution_metrics["has_phase_data"] = True

    # Use computed metrics if we have phase data, otherwise fall back to raw
    final_execution_metrics = (
        computed_execution_metrics
        if computed_execution_metrics["has_phase_data"]
        else {
            "peak_cpu": execution_metrics.get("peak_cpu", 0.0),
            "avg_cpu": execution_metrics.get("avg_cpu", 0.0),
            "peak_memory": execution_metrics.get("peak_memory", 0.0),
            "avg_memory": execution_metrics.get("avg_memory", 0.0),
            "sample_count": execution_metrics.get("sample_count", 0),
            "has_phase_data": False,
        }
    )

    resource_management = {
        "type": adaptive_stats.get("management_type", "unknown"),
        "concurrency_limit": concurrency_limit,
        "execution_metrics": final_execution_metrics,
        "phases": normalized_phases,
    }

    # Build agent results with model information from usage data
    agents_usage = usage.get("agents", {})
    agent_results = []
    for agent_name, result in results.items():
        # Get model information for this agent
        agent_usage = agents_usage.get(agent_name, {})
        model = agent_usage.get("model", "")
        model_profile = agent_usage.get("model_profile", "")
        duration_ms = agent_usage.get("duration_ms", 0)

        # Create model display string
        model_display = ""
        if model_profile and model:
            model_display = f" ({model_profile} → {model})"
        elif model:
            model_display = f" ({model})"

        agent_results.append(
            {
                "name": agent_name,
                "status": result.get("status", "unknown"),
                "content": result.get("response", result.get("content", "")),
                "model": model,
                "model_profile": model_profile,
                "model_display": model_display,
                "duration_ms": duration_ms,
                "error": (
                    result.get("error", "")
                    if result.get("status") != "success"
                    else ""
                ),
            }
        )

    # Build usage summary
    totals = usage.get("totals", {})
    agents_usage = usage.get("agents", {})

    per_agent_usage = []
    for agent_name, agent_usage in agents_usage.items():
        model = agent_usage.get("model", "unknown")
        model_profile = agent_usage.get("model_profile", "unknown")

        # Show both model profile and actual model to detect fallbacks
        model_display = (
            f"{model_profile} → {model}" if model_profile != "unknown" else model
        )

        per_agent_usage.append(
            {
                "name": agent_name,
                "tokens": agent_usage.get("total_tokens", 0),
                "cost_usd": agent_usage.get("total_cost_usd", 0.0),
                "duration_ms": agent_usage.get("duration_ms", 0),
                "model": model,
                "model_profile": model_profile,
                "model_display": model_display,
                "peak_cpu": agent_usage.get("peak_cpu"),
                "avg_cpu": agent_usage.get("avg_cpu"),
                "peak_memory": agent_usage.get("peak_memory"),
                "avg_memory": agent_usage.get("avg_memory"),
            }
        )

    usage_summary = {
        "total_tokens": totals.get("total_tokens", 0),
        "total_cost_usd": totals.get("total_cost_usd", 0.0),
        "per_agent": per_agent_usage,
    }

    return {
        "execution_summary": execution_summary,
        "resource_management": resource_management,
        "agent_results": agent_results,
        "usage_summary": usage_summary,
    }


def render_json_as_text(structured_json: dict[str, Any]) -> str:
    """Render structured JSON as text format."""
    lines = []

    # Execution summary
    summary = structured_json.get("execution_summary", {})
    lines.append(f"Total agents: {summary.get('total_agents', 0)}")

    # Resource management
    rm = structured_json.get("resource_management", {})
    lines.append(f"Max concurrency limit used: {rm.get('concurrency_limit', 1)}")

    # Phases
    for phase in rm.get("phases", []):
        phase_num = phase.get("phase_number", 1)
        lines.append(f"Phase {phase_num}:")

    # Usage
    usage = structured_json.get("usage_summary", {})
    lines.append(f"Total tokens: {usage.get('total_tokens', 0)}")

    return "\n".join(lines)


def render_json_as_markdown(structured_json: dict[str, Any]) -> str:
    """Render structured JSON as markdown format."""
    lines = []

    # Execution summary
    summary = structured_json.get("execution_summary", {})
    lines.append(f"**Total agents:** {summary.get('total_agents', 0)}")

    # Resource management
    rm = structured_json.get("resource_management", {})
    lines.append(f"**Max concurrency limit used:** {rm.get('concurrency_limit', 1)}")

    # Phases
    for phase in rm.get("phases", []):
        phase_num = phase.get("phase_number", 1)
        lines.append(f"**Phase {phase_num}:**")

    # Usage
    usage = structured_json.get("usage_summary", {})
    lines.append(f"**Total tokens:** {usage.get('total_tokens', 0)}")

    return "\n".join(lines)


def render_comprehensive_text(structured_json: dict[str, Any]) -> str:
    """Render structured JSON as comprehensive plain text with full feature parity."""
    lines = []

    # Agent Results Section - show ALL agents with model info
    agent_results = structured_json.get("agent_results", [])
    if agent_results:
        for agent in agent_results:
            agent_name = agent.get("name", "unknown")
            status = agent.get("status", "unknown")
            content = agent.get("content", "")
            model_display = agent.get("model_display", "")
            error = agent.get("error", "")

            if status == "success":
                lines.append(f"{agent_name}{model_display}:")
                lines.append("-" * (len(agent_name) + len(model_display) + 1) + ":")
                if content:
                    lines.append(content)
                else:
                    lines.append("*No content provided*")
                lines.append("")
            else:
                lines.append(f"❌ {agent_name}{model_display}:")
                lines.append("-" * (len(agent_name) + len(model_display) + 3) + ":")
                if error:
                    lines.append(f"Error: {error}")
                else:
                    lines.append("Error: Unknown error")
                lines.append("")

    # Resource Management Section
    rm = structured_json.get("resource_management", {})
    if rm:
        lines.append("Resource Management")
        lines.append("==================")
        lines.append(f"Type: {rm.get('type', 'unknown')} (fixed concurrency limits)")
        lines.append(f"Max concurrency limit used: {rm.get('concurrency_limit', 1)}")

        # Execution metrics - show if we have meaningful data
        exec_metrics = rm.get("execution_metrics", {})
        has_phase_data = exec_metrics.get("has_phase_data", False)
        peak_cpu = exec_metrics.get("peak_cpu", 0.0)
        avg_cpu = exec_metrics.get("avg_cpu", 0.0)
        peak_memory = exec_metrics.get("peak_memory", 0.0)
        avg_memory = exec_metrics.get("avg_memory", 0.0)

        # Show metrics if we have any meaningful data
        if peak_cpu > 0 or peak_memory > 0 or avg_cpu > 0 or avg_memory > 0:
            lines.append(f"Peak usage: CPU {peak_cpu:.1f}%, Memory {peak_memory:.1f}%")
            lines.append(f"Average usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%")

            # Only show sample count if using raw execution metrics
            if not has_phase_data:
                sample_count = exec_metrics.get("sample_count", 0)
                if sample_count > 0:
                    lines.append(f"Monitoring: {sample_count} samples collected")
            else:
                lines.append("Computed from per-phase performance data")

    # Per-Phase Performance Section
    phases = rm.get("phases", [])
    if phases:
        lines.append("")
        lines.append("Per-Phase Performance")
        lines.append("====================")

        for phase in phases:
            phase_num = phase.get("phase_number", 1)
            agent_names = phase.get("agent_names", [])
            duration = phase.get("duration_seconds", 0.0)
            agent_count = len(agent_names)

            lines.append(f"Phase {phase_num} ({agent_count} agents)")
            if agent_names:
                agent_list = ", ".join(agent_names)
                lines.append(f"  Agents: {agent_list}")

            lines.append(f"  Duration: {duration:.1f} seconds")

            # Resource usage - prefer peak/avg, fallback to final
            peak_cpu = phase.get("peak_cpu")
            avg_cpu = phase.get("avg_cpu")
            peak_memory = phase.get("peak_memory")
            avg_memory = phase.get("avg_memory")
            sample_count = phase.get("sample_count", 0)

            if peak_cpu is not None and avg_cpu is not None and sample_count > 0:
                lines.append(
                    f"  Resource usage: CPU {avg_cpu:.1f}% (peak {peak_cpu:.1f}%), "
                    f"Memory {avg_memory:.1f}% (peak {peak_memory:.1f}%)"
                )
                lines.append(f"  Monitoring: {sample_count} samples collected")
            else:
                final_cpu = phase.get("final_cpu_percent")
                final_memory = phase.get("final_memory_percent")
                if final_cpu is not None and final_memory is not None:
                    lines.append(
                        f"  Resource usage: CPU {final_cpu:.1f}%, "
                        f"Memory {final_memory:.1f}%"
                    )

            lines.append("")

    # Per-Agent Usage Section
    usage = structured_json.get("usage_summary", {})
    per_agent = usage.get("per_agent", [])
    if per_agent:
        lines.append("Per-Agent Usage")
        lines.append("===============")

        for agent in per_agent:
            name = agent.get("name", "unknown")
            tokens = agent.get("tokens", 0)
            cost = agent.get("cost_usd", 0.0)
            duration = agent.get("duration_ms", 0)
            model_display = agent.get("model_display", "unknown")
            peak_cpu = agent.get("peak_cpu")
            avg_cpu = agent.get("avg_cpu")
            peak_memory = agent.get("peak_memory")
            avg_memory = agent.get("avg_memory")

            # Base usage line
            usage_line = (
                f"{name} ({model_display}): {tokens:,} tokens, "
                f"${cost:.4f}, {duration}ms"
            )

            # Add resource metrics if available
            if peak_cpu is not None and avg_cpu is not None:
                usage_line += (
                    f", CPU {avg_cpu:.1f}% (peak {peak_cpu:.1f}%), "
                    f"Memory {avg_memory:.1f}% (peak {peak_memory:.1f}%)"
                )

            lines.append(usage_line)

    return "\n".join(lines)


def render_comprehensive_markdown(structured_json: dict[str, Any]) -> str:
    """Render structured JSON as comprehensive markdown with full feature parity."""
    sections = []

    # Agent Results Section - show ALL agents with model info
    agent_results = structured_json.get("agent_results", [])
    if agent_results:
        for agent in agent_results:
            agent_name = agent.get("name", "unknown")
            status = agent.get("status", "unknown")
            content = agent.get("content", "")
            model_display = agent.get("model_display", "")
            error = agent.get("error", "")

            if status == "success":
                sections.append(f"## {agent_name}{model_display}\n\n")
                if content:
                    # Format as code block if it looks like code
                    # otherwise as regular text
                    code_keywords = ["def ", "class ", "```", "import ", "function"]
                    if any(keyword in content.lower() for keyword in code_keywords):
                        sections.append(f"```\n{content}\n```\n\n")
                    else:
                        sections.append(f"{content}\n\n")
                else:
                    sections.append("*No content provided*\n\n")
            else:
                sections.append(f"## \u274c {agent_name}{model_display}\n\n")
                if error:
                    sections.append(f"**Error:** {error}\n\n")
                else:
                    sections.append("**Error:** Unknown error\n\n")

    # Resource Management Section
    rm = structured_json.get("resource_management", {})
    if rm:
        sections.append("### Resource Management\n")
        sections.append(
            f"- **Type:** {rm.get('type', 'unknown')} (fixed concurrency limits)\n"
        )
        sections.append(
            f"- **Max concurrency limit used:** {rm.get('concurrency_limit', 1)}\n"
        )

        # Execution metrics - show if we have meaningful data
        exec_metrics = rm.get("execution_metrics", {})
        has_phase_data = exec_metrics.get("has_phase_data", False)
        peak_cpu = exec_metrics.get("peak_cpu", 0.0)
        avg_cpu = exec_metrics.get("avg_cpu", 0.0)
        peak_memory = exec_metrics.get("peak_memory", 0.0)
        avg_memory = exec_metrics.get("avg_memory", 0.0)

        # Show metrics if we have any meaningful data
        if peak_cpu > 0 or peak_memory > 0 or avg_cpu > 0 or avg_memory > 0:
            sections.append(
                f"- **Peak usage:** CPU {peak_cpu:.1f}%, Memory {peak_memory:.1f}%\n"
            )
            sections.append(
                f"- **Average usage:** CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%\n"
            )

            # Only show sample count if using raw execution metrics
            if not has_phase_data:
                sample_count = exec_metrics.get("sample_count", 0)
                if sample_count > 0:
                    sections.append(
                        f"- **Monitoring:** {sample_count} samples collected\n"
                    )
            else:
                sections.append("- **Computed from per-phase performance data**\n")

    # Per-Phase Performance Section
    phases = rm.get("phases", [])
    if phases:
        sections.append("\n#### Per-Phase Performance\n")

        for phase in phases:
            phase_num = phase.get("phase_number", 1)
            agent_names = phase.get("agent_names", [])
            duration = phase.get("duration_seconds", 0.0)
            agent_count = len(agent_names)

            sections.append(f"**Phase {phase_num}** ({agent_count} agents)\n")
            if agent_names:
                agent_list = ", ".join(agent_names)
                sections.append(f"- **Agents:** {agent_list}\n")

            sections.append(f"- **Duration:** {duration:.1f} seconds\n")

            # Resource usage - prefer peak/avg, fallback to final
            peak_cpu = phase.get("peak_cpu")
            avg_cpu = phase.get("avg_cpu")
            peak_memory = phase.get("peak_memory")
            avg_memory = phase.get("avg_memory")
            sample_count = phase.get("sample_count", 0)

            if peak_cpu is not None and avg_cpu is not None and sample_count > 0:
                sections.append(
                    f"- **Resource usage:** CPU {avg_cpu:.1f}% (peak {peak_cpu:.1f}%), "
                    f"Memory {avg_memory:.1f}% (peak {peak_memory:.1f}%)\n"
                )
                sections.append(f"- **Monitoring:** {sample_count} samples collected\n")
            else:
                final_cpu = phase.get("final_cpu_percent")
                final_memory = phase.get("final_memory_percent")
                if final_cpu is not None and final_memory is not None:
                    sections.append(
                        f"- **Resource usage:** CPU {final_cpu:.1f}%, "
                        f"Memory {final_memory:.1f}%\n"
                    )

            sections.append("\n")

    # Per-Agent Usage Section
    usage = structured_json.get("usage_summary", {})
    per_agent = usage.get("per_agent", [])
    if per_agent:
        sections.append("### Per-Agent Usage\n")

        for agent in per_agent:
            name = agent.get("name", "unknown")
            tokens = agent.get("tokens", 0)
            cost = agent.get("cost_usd", 0.0)
            duration = agent.get("duration_ms", 0)
            model_display = agent.get("model_display", "unknown")
            peak_cpu = agent.get("peak_cpu")
            avg_cpu = agent.get("avg_cpu")
            peak_memory = agent.get("peak_memory")
            avg_memory = agent.get("avg_memory")

            # Base usage line
            usage_line = (
                f"- **{name}** ({model_display}): {tokens:,} tokens, "
                f"${cost:.4f}, {duration}ms"
            )

            # Add resource metrics if available
            if peak_cpu is not None and avg_cpu is not None:
                usage_line += (
                    f", CPU {avg_cpu:.1f}% (peak {peak_cpu:.1f}%), "
                    f"Memory {avg_memory:.1f}% (peak {peak_memory:.1f}%)"
                )

            sections.append(usage_line + "\n")

    return "".join(sections)
