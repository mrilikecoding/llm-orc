"""Results display utilities for CLI visualization."""

from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown

from .dependency import (
    _create_plain_text_dependency_graph, 
    find_final_agent
)


def display_results(
    results: dict[str, Any],
    metadata: dict[str, Any],
    agents: list[dict[str, Any]],
    detailed: bool = False,
) -> None:
    """Display execution results with Rich formatting."""
    # Enhanced console config to prevent any truncation
    console = Console(
        soft_wrap=True, 
        width=None,  # Use full terminal width
        force_terminal=True,
        no_color=False,
        legacy_windows=False,
        markup=True,
        highlight=False,
        _environ={},
        stderr=False,
        file=None
    )

    if detailed:
        # Add Results header
        console.print("\n[bold blue]📋 Results[/bold blue]")
        console.print("=" * 50)
        
        # Process and display agent results
        processed_results = _process_agent_results(results)
        for agent_name, result in processed_results.items():
            _display_agent_result(console, agent_name, result, agents, metadata)

        # Display performance metrics
        performance_lines = _format_performance_metrics(metadata)
        if performance_lines:
            console.print("\n" + "\n".join(performance_lines))
    else:
        # Simplified display - show final agent result
        final_agent = find_final_agent(results)
        if final_agent and results[final_agent].get("response"):
            # Get model info for the final agent
            model_info = _get_agent_model_info(final_agent, agents, results[final_agent], metadata)
            model_display = f" ({model_info})" if model_info else ""
            
            console.print(f"\n[bold blue]📋 Final Result[/bold blue]")
            console.print("=" * 30)
            console.print(f"\n[bold green]✓ {final_agent}[/bold green][dim]{model_display}[/dim]")
            
            response = results[final_agent]["response"]
            # Use direct stdout to prevent truncation
            import sys
            sys.stdout.write(response)
            sys.stdout.write('\n\n')
            sys.stdout.flush()
        else:
            console.print("No results to display")


def display_plain_text_results(
    results: dict[str, Any],
    metadata: dict[str, Any],
    detailed: bool = False,
    agents: list[dict[str, Any]] | None = None,
) -> None:
    """Display results in plain text format."""
    if detailed:
        _display_detailed_plain_text(results, metadata, agents or [])
    else:
        _display_simplified_plain_text(results, metadata)


def display_simplified_results(
    results: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Display simplified results showing only the final output using markdown."""
    console = Console(soft_wrap=True, width=None, force_terminal=True)

    # Find the final agent result to display
    successful_agents = [
        name for name, result in results.items() 
        if result.get("status") == "success"
    ]

    if successful_agents:
        last_agent = successful_agents[-1]
        response = results[last_agent]["response"]
        click.echo(f"Result from {last_agent}:")
        click.echo(response)
    else:
        click.echo("❌ No successful results found")

    # Show minimal performance summary
    if "usage" in metadata:
        totals = metadata["usage"].get("totals", {})
        agents_count = totals.get("agents_count", 0)
        duration = metadata.get("duration", "unknown")
        click.echo()
        click.echo(f"⚡ {agents_count} agents completed in {duration}")


def _process_agent_results(results: dict[str, Any]) -> dict[str, Any]:
    """Process agent results for display."""
    processed = {}
    
    for agent_name, result in results.items():
        status = result.get("status", "unknown")
        response = result.get("response", result.get("content", ""))
        error = result.get("error", "")
        
        processed[agent_name] = {
            "status": status,
            "response": response,
            "error": error,
            "has_code": _has_code_content(response)
        }
    
    return processed


def _display_agent_result(console: Console, agent_name: str, result: dict[str, Any], agents: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
    """Display a single agent result."""
    status = result["status"]
    response = result["response"]
    error = result["error"]
    
    # Find model information from agent config, result, and metadata
    model_info = _get_agent_model_info(agent_name, agents, result, metadata)
    model_display = f" ({model_info})" if model_info else ""
    
    # Extract individual agent performance info from result
    timing_info = _extract_agent_timing_info(result)
    timing_display = f" [dim]• {timing_info}[/dim]" if timing_info else ""
    
    if status == "success":
        console.print(f"\n[bold green]✓ {agent_name}[/bold green][dim]{model_display}{timing_display}[/dim]")
        if response:
            # Use direct stdout approach to bypass all Rich limitations
            import sys
            sys.stdout.write(response)
            sys.stdout.write('\n\n')
            sys.stdout.flush()
        else:
            console.print("[dim]No response provided[/dim]")
    else:
        console.print(f"\n[bold red]✗ {agent_name}[/bold red][dim]{model_display}{timing_display}[/dim]")
        if error:
            console.print(f"[red]Error: {error}[/red]", soft_wrap=True)
        else:
            console.print("[red]Unknown error occurred[/red]")


def _display_detailed_plain_text(
    results: dict[str, Any],
    metadata: dict[str, Any],
    agents: list[dict[str, Any]],
) -> None:
    """Display detailed plain text results."""
    # Display dependency graph
    _display_plain_text_dependency_graph(agents)
    
    # Add Results header
    click.echo("\n📋 Results")
    click.echo("=" * 50)
    
    # Display each agent result
    for agent_name, result in results.items():
        status = result.get("status", "unknown")
        response = result.get("response", "")
        error = result.get("error", "")
        
        # Get model information
        model_info = _get_agent_model_info(agent_name, agents, result)
        model_display = f" ({model_info})" if model_info else ""
        
        click.echo(f"\n{agent_name} ({status}){model_display}:")
        click.echo("-" * (len(agent_name) + len(status) + len(model_display) + 3))
        
        if status == "success" and response:
            click.echo(response)
        elif error:
            click.echo(f"Error: {error}")
        else:
            click.echo("No output")

    # Display performance summary
    performance_lines = _format_performance_metrics(metadata)
    if performance_lines:
        click.echo("\n" + "\n".join(performance_lines))


def _display_simplified_plain_text(
    results: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Display simplified plain text results."""
    successful_agents = [
        name for name, result in results.items() 
        if result.get("status") == "success"
    ]
    
    if successful_agents:
        last_agent = successful_agents[-1]
        response = results[last_agent]["response"]
        click.echo(f"Result from {last_agent}:")
        click.echo(response)
    else:
        click.echo("❌ No successful results found")


def _display_plain_text_dependency_graph(agents: list[dict[str, Any]]) -> None:
    """Display dependency graph in plain text."""
    if not agents:
        return
        
    click.echo("Agent Dependencies:")
    graph_lines = _create_plain_text_dependency_graph(agents)
    for line in graph_lines:
        click.echo(f"  {line}")
    click.echo()


def _format_performance_metrics(metadata: dict[str, Any]) -> list[str]:
    """Format comprehensive performance metrics for display."""
    lines = []
    
    # Usage summary
    if "usage" in metadata:
        usage = metadata["usage"]
        totals = usage.get("totals", {})
        
        lines.append("\n📊 Performance Summary")
        lines.append("=" * 60)
        lines.append(f"Total agents: {totals.get('agents_count', len(usage.get('agents', {})))}")
        lines.append(f"Total tokens: {totals.get('total_tokens', 0):,}")
        lines.append(f"Total cost: ${totals.get('total_cost_usd', 0.0):.4f}")
        
        if "duration" in metadata:
            lines.append(f"Duration: {metadata['duration']}")
        
        # Per-agent usage details
        agents_usage = usage.get("agents", {})
        if agents_usage:
            lines.append(f"\n👥 Per-Agent Performance")
            lines.append("-" * 50)
            
            for agent_name, agent_usage in agents_usage.items():
                tokens = agent_usage.get("total_tokens", 0)
                cost = agent_usage.get("total_cost_usd", 0.0)
                input_tokens = agent_usage.get("input_tokens", 0)
                output_tokens = agent_usage.get("output_tokens", 0)
                duration = agent_usage.get("duration_seconds")
                peak_cpu = agent_usage.get("peak_cpu")
                avg_cpu = agent_usage.get("avg_cpu")
                peak_memory = agent_usage.get("peak_memory")
                avg_memory = agent_usage.get("avg_memory")
                
                lines.append(f"  {agent_name}:")
                if input_tokens or output_tokens:
                    lines.append(f"    Tokens: {tokens:,} (input: {input_tokens:,}, output: {output_tokens:,})")
                else:
                    lines.append(f"    Tokens: {tokens:,}")
                lines.append(f"    Cost: ${cost:.4f}")
                
                if duration is not None:
                    lines.append(f"    Duration: {duration:.2f}s")
                    
                if peak_cpu is not None:
                    lines.append(f"    Peak CPU: {peak_cpu:.1f}%")
                if avg_cpu is not None:
                    lines.append(f"    Avg CPU: {avg_cpu:.1f}%")
                if peak_memory is not None:
                    lines.append(f"    Peak memory: {peak_memory:.1f} MB")
                if avg_memory is not None:
                    lines.append(f"    Avg memory: {avg_memory:.1f} MB")
    
    # Adaptive resource management metrics
    if "adaptive_resource_management" in metadata:
        arm = metadata["adaptive_resource_management"]
        
        # Concurrency information
        if "concurrency_decisions" in arm:
            decisions = arm["concurrency_decisions"]
            if decisions:
                configured_limit = decisions[0].get("configured_limit")
                if configured_limit:
                    lines.append(f"Max concurrency limit used: {configured_limit}")
        
        # Execution metrics (CPU usage, etc.)
        if "execution_metrics" in arm:
            exec_metrics = arm["execution_metrics"]
            lines.append("\n🔧 System Resource Usage")
            lines.append("-" * 40)
            
            peak_cpu = exec_metrics.get("peak_cpu")
            avg_cpu = exec_metrics.get("avg_cpu")
            sample_count = exec_metrics.get("sample_count")
            peak_memory = exec_metrics.get("peak_memory_mb")
            avg_memory = exec_metrics.get("avg_memory_mb")
            
            if peak_cpu is not None:
                lines.append(f"Peak CPU usage: {peak_cpu:.1f}%")
            if avg_cpu is not None:
                lines.append(f"Average CPU usage: {avg_cpu:.1f}%")
            if peak_memory is not None:
                lines.append(f"Peak memory: {peak_memory:.1f} MB")
            if avg_memory is not None:
                lines.append(f"Average memory: {avg_memory:.1f} MB")
            if sample_count is not None:
                lines.append(f"Sample count: {sample_count}")
        
        # Phase metrics (execution phases)
        if "phase_metrics" in arm:
            phase_metrics = arm["phase_metrics"]
            if phase_metrics:
                lines.append("\n⚡ Execution Phases")
                lines.append("-" * 35)
                
                for i, phase in enumerate(phase_metrics):
                    phase_num = i + 1
                    duration = phase.get("duration_seconds")
                    agent_names = phase.get("agent_names", [])
                    peak_cpu = phase.get("peak_cpu")
                    avg_cpu = phase.get("avg_cpu")
                    peak_memory = phase.get("peak_memory")
                    avg_memory = phase.get("avg_memory")
                    
                    # Phase header with duration
                    if duration is not None:
                        lines.append(f"Phase {phase_num}: {duration:.2f}s")
                    else:
                        lines.append(f"Phase {phase_num}:")
                    
                    # Agent list (with better formatting for long lists)
                    if agent_names:
                        if len(agent_names) > 4:
                            agent_display = ", ".join(agent_names[:4]) + f" (+{len(agent_names)-4} more)"
                        else:
                            agent_display = ", ".join(agent_names)
                        lines.append(f"  Agents: {agent_display}")
                    
                    # Resource usage metrics
                    resource_lines = []
                    if peak_cpu is not None:
                        resource_lines.append(f"Peak CPU: {peak_cpu:.1f}%")
                    if avg_cpu is not None:
                        resource_lines.append(f"Avg CPU: {avg_cpu:.1f}%")
                    if peak_memory is not None:
                        resource_lines.append(f"Peak memory: {peak_memory:.1f} MB")
                    if avg_memory is not None:
                        resource_lines.append(f"Avg memory: {avg_memory:.1f} MB")
                    
                    # Display resource metrics (combine if multiple exist)
                    if resource_lines:
                        if len(resource_lines) <= 2:
                            lines.append(f"  {' • '.join(resource_lines)}")
                        else:
                            for resource_line in resource_lines:
                                lines.append(f"  {resource_line}")
                    
                    # Add spacing between phases for readability
                    if i < len(phase_metrics) - 1:
                        lines.append("")
    
    return lines


def _extract_agent_timing_info(result: dict[str, Any]) -> str:
    """Extract timing and performance info from agent result."""
    info_parts = []
    
    # Check for duration/timing information
    duration = result.get("duration")
    if duration:
        if isinstance(duration, (int, float)):
            info_parts.append(f"{duration:.2f}s")
        else:
            info_parts.append(str(duration))
    
    # Check for token usage
    tokens = result.get("tokens") or result.get("usage", {}).get("total_tokens")
    if tokens:
        info_parts.append(f"{tokens:,} tokens")
    
    # Check for cost information
    cost = result.get("cost") or result.get("usage", {}).get("total_cost_usd")
    if cost:
        info_parts.append(f"${cost:.4f}")
    
    # Check for model used (actual model from execution, not config)
    actual_model = result.get("model_used") or result.get("model")
    if actual_model:
        info_parts.append(f"via {actual_model}")
    
    return " • ".join(info_parts)


def _get_agent_model_info(agent_name: str, agents: list[dict[str, Any]], result: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> str:
    """Get comprehensive model information for an agent."""
    for agent in agents:
        if agent.get("name") == agent_name:
            # Get profile from agent config
            model_profile = agent.get("model_profile")
            
            # Get actual model from usage metadata (JSON single source of truth)
            model = None
            if metadata and "usage" in metadata:
                usage = metadata["usage"]
                agents_usage = usage.get("agents", {})
                if agent_name in agents_usage:
                    agent_usage = agents_usage[agent_name]
                    model = agent_usage.get("model")
            
            # Fallback to agent config if not in metadata
            if not model:
                model = agent.get("model")
            
            # Build model display
            if model_profile and model:
                return f"{model_profile} -> {model}"
            elif model_profile:
                return model_profile
            elif model:
                return model
            elif agent.get("provider"):
                return agent["provider"]
    
    return ""


def _has_code_content(text: str) -> bool:
    """Check if text contains code-like content."""
    if not text:
        return False
    
    code_indicators = ["def ", "class ", "```", "import ", "function", "{", "}", ";"]
    return any(indicator in text.lower() for indicator in code_indicators)