"""CLI visualization utilities for dependency graphs and execution display."""

from typing import Any

import click
from rich.console import Console

from llm_orc.ensemble_config import EnsembleConfig


def create_dependency_graph(agents: list[dict[str, Any]]) -> str:
    """Create horizontal dependency graph: A,B,C → D → E,F → G"""
    return create_dependency_graph_with_status(agents, {})


def create_dependency_graph_with_status(
    agents: list[dict[str, Any]], agent_statuses: dict[str, str]
) -> str:
    """Create horizontal dependency graph with status indicators."""
    # Group agents by dependency level
    agents_by_level: dict[int, list[dict[str, Any]]] = {}

    for agent in agents:
        dependencies = agent.get("depends_on", [])
        level = _calculate_agent_level(agent["name"], dependencies, agents)

        if level not in agents_by_level:
            agents_by_level[level] = []
        agents_by_level[level].append(agent)

    # Build horizontal graph: A,B,C → D → E,F → G
    graph_parts = []
    max_level = max(agents_by_level.keys()) if agents_by_level else 0

    for level in range(max_level + 1):
        if level not in agents_by_level:
            continue

        level_agents = agents_by_level[level]
        agent_displays = []

        for agent in level_agents:
            name = agent["name"]
            status = agent_statuses.get(name, "pending")

            # Status indicators without dots
            if status == "running":
                agent_displays.append(f"[yellow]{name}[/yellow]")
            elif status == "completed":
                agent_displays.append(f"[green]{name}[/green]")
            elif status == "failed":
                agent_displays.append(f"[red]{name}[/red]")
            else:
                agent_displays.append(f"[dim]{name}[/dim]")

        # Join agents at same level with commas
        level_text = ", ".join(agent_displays)
        graph_parts.append(level_text)

    # Join levels with arrows
    return " → ".join(graph_parts)


def _calculate_agent_level(
    agent_name: str, dependencies: list[str], all_agents: list[dict[str, Any]]
) -> int:
    """Calculate the dependency level of an agent (0 = no dependencies)."""
    if not dependencies:
        return 0

    # Find the maximum level of all dependencies
    max_dep_level = 0
    for dep_name in dependencies:
        # Find the dependency agent
        dep_agent = next((a for a in all_agents if a["name"] == dep_name), None)
        if dep_agent:
            dep_dependencies = dep_agent.get("depends_on", [])
            dep_level = _calculate_agent_level(dep_name, dep_dependencies, all_agents)
            max_dep_level = max(max_dep_level, dep_level)

    return max_dep_level + 1


def display_results(
    results: dict[str, Any], metadata: dict[str, Any], detailed: bool = False
) -> None:
    """Display results in a formatted way."""
    if detailed:
        # Show detailed results with all agents
        click.echo("\n📋 Results:")
        click.echo("=" * 50)

        for agent_name, result in results.items():
            if result.get("status") == "success":
                click.echo(f"\n✅ {agent_name}:")
                click.echo(f"   {result['response']}")
            else:
                click.echo(f"\n❌ {agent_name}:")
                click.echo(f"   Error: {result.get('error', 'Unknown error')}")

        # Show performance metrics
        if "usage" in metadata:
            usage = metadata["usage"]
            totals = usage.get("totals", {})
            click.echo("\n📊 Performance Metrics:")
            click.echo(f"   Duration: {metadata['duration']}")
            click.echo(f"   Total tokens: {totals.get('total_tokens', 0):,}")
            click.echo(f"   Total cost: ${totals.get('total_cost_usd', 0.0):.4f}")
            click.echo(f"   Agents: {totals.get('agents_count', 0)}")

            # Show per-agent usage
            agents_usage = usage.get("agents", {})
            if agents_usage:
                click.echo("\n   Per-Agent Usage:")
                for agent_name, agent_usage in agents_usage.items():
                    tokens = agent_usage.get("total_tokens", 0)
                    cost = agent_usage.get("cost_usd", 0.0)
                    duration = agent_usage.get("duration_ms", 0)
                    model = agent_usage.get("model", "unknown")
                    click.echo(
                        f"     {agent_name} ({model}): {tokens:,} tokens, "
                        f"${cost:.4f}, {duration}ms"
                    )
    else:
        # Simplified output: just show final synthesis/result
        display_simplified_results(results, metadata)


def display_simplified_results(
    results: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Display simplified results showing only the final output."""
    # Find the final agent (the one with no dependents)
    final_agent = find_final_agent(results)

    if final_agent and results[final_agent].get("status") == "success":
        click.echo(f"\n{results[final_agent]['response']}")
    else:
        # Fallback: show last successful agent
        successful_agents = [
            name
            for name, result in results.items()
            if result.get("status") == "success"
        ]
        if successful_agents:
            last_agent = successful_agents[-1]
            click.echo(f"\n✅ Result from {last_agent}:")
            click.echo(f"{results[last_agent]['response']}")
        else:
            click.echo("\n❌ No successful results found")

    # Show minimal performance summary
    if "usage" in metadata:
        totals = metadata["usage"].get("totals", {})
        agents_count = totals.get("agents_count", 0)
        duration = metadata.get("duration", "unknown")
        click.echo(f"\n⚡ {agents_count} agents completed in {duration}")
        click.echo("   Use --detailed flag for full results and metrics")


def find_final_agent(results: dict[str, Any]) -> str | None:
    """Find the final agent in the dependency chain (the one with no dependents)."""
    # For now, use a simple heuristic: the agent with the highest token count
    # is likely the final agent (since it got input from all previous agents)
    final_agent = None

    for agent_name in results.keys():
        # This is a simple heuristic - in practice we'd want to track dependencies
        # But for now, we can assume the last successful agent is often the final one
        if results[agent_name].get("status") == "success":
            final_agent = agent_name

    return final_agent


async def run_streaming_execution(
    executor: Any,
    ensemble_config: EnsembleConfig,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> None:
    """Run streaming execution with Rich status display."""
    console = Console()
    agent_statuses: dict[str, str] = {}

    # Initialize with Rich status
    with console.status("Starting execution...", spinner="dots") as status:
        async for event in executor.execute_streaming(ensemble_config, input_data):
            if output_format == "json":
                import json
                click.echo(json.dumps(event, indent=2))
            else:
                event_type = event["type"]
                if event_type == "agent_progress":
                    # Extract agent status from progress data
                    completed_agents = event["data"].get("completed_agents", 0)
                    total_agents = event["data"].get(
                        "total_agents", len(ensemble_config.agents)
                    )

                    # Mark first N agents as completed, rest as pending
                    for i, agent in enumerate(ensemble_config.agents):
                        if i < completed_agents:
                            agent_statuses[agent["name"]] = "completed"
                        elif i == completed_agents and completed_agents < total_agents:
                            agent_statuses[agent["name"]] = "running"
                        else:
                            agent_statuses[agent["name"]] = "pending"

                    # Update status display with current dependency graph
                    current_graph = create_dependency_graph_with_status(
                        ensemble_config.agents, agent_statuses
                    )
                    status.update(current_graph)

                elif event_type == "execution_completed":
                    # Final update with all completed
                    final_statuses = {
                        agent["name"]: "completed"
                        for agent in ensemble_config.agents
                    }
                    final_graph = create_dependency_graph_with_status(
                        ensemble_config.agents, final_statuses
                    )
                    console.print(f"Final: {final_graph}")
                    console.print(f"Completed in {event['data']['duration']:.2f}s")

                    if output_format == "text":
                        display_results(
                            event["data"]["results"],
                            event["data"]["metadata"],
                            detailed,
                        )


async def run_standard_execution(
    executor: Any,
    ensemble_config: EnsembleConfig,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> None:
    """Run standard execution without streaming."""
    result = await executor.execute(ensemble_config, input_data)
    if output_format == "json":
        import json
        click.echo(json.dumps(result, indent=2))
    else:
        display_results(result["results"], result["metadata"], detailed)
