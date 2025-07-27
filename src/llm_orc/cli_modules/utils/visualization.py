"""CLI visualization utilities for dependency graphs and execution display."""

from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.tree import Tree

from llm_orc.core.config.ensemble_config import EnsembleConfig


def create_dependency_graph(agents: list[dict[str, Any]]) -> str:
    """Create horizontal dependency graph: A,B,C → D → E,F → G"""
    return create_dependency_graph_with_status(agents, {})


def create_dependency_tree(
    agents: list[dict[str, Any]], agent_statuses: dict[str, str] | None = None
) -> Tree:
    """Create a tree visualization of agent dependencies by execution levels."""
    if agent_statuses is None:
        agent_statuses = {}

    # Group agents by dependency level
    agents_by_level = _group_agents_by_dependency_level(agents)
    tree = Tree("[bold blue]Orchestrating Agent Responses[/bold blue]")

    max_level = max(agents_by_level.keys()) if agents_by_level else 0

    # Create each level as a single line with agents grouped together
    for level in range(max_level + 1):
        if level not in agents_by_level:
            continue

        level_agents = agents_by_level[level]

        # Create agent status strings for this level
        agent_labels = []
        for agent in level_agents:
            agent_name = agent["name"]
            status = agent_statuses.get(agent_name, "pending")

            if status == "running":
                symbol = "[yellow]◐[/yellow]"
                style = "yellow"
            elif status == "completed":
                symbol = "[green]✓[/green]"
                style = "green"
            elif status == "failed":
                symbol = "[red]✗[/red]"
                style = "red"
            else:
                symbol = "[dim]○[/dim]"
                style = "dim"

            agent_label = f"{symbol} [{style}]{agent_name}[/{style}]"
            agent_labels.append(agent_label)

        # Join all agents at this level into a single line
        level_line = "  ".join(agent_labels)
        tree.add(level_line)

    return tree


def _group_agents_by_dependency_level(
    agents: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group agents by their dependency level (0 = no dependencies)."""
    agents_by_level: dict[int, list[dict[str, Any]]] = {}

    for agent in agents:
        dependencies = agent.get("depends_on", [])
        level = _calculate_agent_level(agent["name"], dependencies, agents)

        if level not in agents_by_level:
            agents_by_level[level] = []
        agents_by_level[level].append(agent)

    return agents_by_level


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

            # Status indicators with symbols
            if status == "running":
                agent_displays.append(f"[yellow]◐[/yellow] [yellow]{name}[/yellow]")
            elif status == "completed":
                agent_displays.append(f"[green]✓[/green] [green]{name}[/green]")
            elif status == "failed":
                agent_displays.append(f"[red]✗[/red] [red]{name}[/red]")
            else:
                agent_displays.append(f"[dim]○[/dim] [dim]{name}[/dim]")

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


def _process_agent_results(
    results: dict[str, Any], metadata: dict[str, Any]
) -> list[str]:
    """Process agent results and generate markdown content.

    Args:
        results: Dictionary of agent results
        metadata: Metadata dictionary containing usage information

    Returns:
        List of markdown content strings for agent results
    """
    markdown_content = []

    # Get agent usage data for model information
    agents_usage = metadata.get("usage", {}).get("agents", {})

    for agent_name, result in results.items():
        # Get model information for this agent
        agent_usage = agents_usage.get(agent_name, {})
        model = agent_usage.get("model", "")
        model_profile = agent_usage.get("model_profile", "")

        # Create model display string
        model_display = ""
        if model_profile and model:
            model_display = f" ({model_profile} → {model})"
        elif model:
            model_display = f" ({model})"

        if result.get("status") == "success":
            markdown_content.append(f"## {agent_name}{model_display}\n")
            # Format the response as a code block if it looks like code,
            # otherwise as regular text (let Rich handle wrapping)
            response = result["response"]
            code_keywords = ["def ", "class ", "```", "import ", "function"]
            if any(keyword in response.lower() for keyword in code_keywords):
                markdown_content.append(f"```\n{response}\n```\n")
            else:
                markdown_content.append(f"{response}\n")
        else:
            markdown_content.append(f"## ❌ {agent_name}{model_display}\n")
            error_msg = result.get("error", "Unknown error")
            markdown_content.append(f"**Error:** {error_msg}\n")

    return markdown_content


def _format_performance_metrics(metadata: dict[str, Any]) -> list[str]:
    """Format performance metrics into markdown content.

    Args:
        metadata: Metadata dictionary containing usage information

    Returns:
        List of markdown content strings for performance metrics
    """
    markdown_content: list[str] = []

    if "usage" not in metadata:
        return markdown_content

    usage = metadata["usage"]
    totals = usage.get("totals", {})

    markdown_content.append("## Performance Metrics\n")
    markdown_content.append(f"- **Duration:** {metadata['duration']}\n")

    total_tokens = totals.get("total_tokens", 0)
    total_cost = totals.get("total_cost_usd", 0.0)
    markdown_content.append(f"- **Total tokens:** {total_tokens:,}\n")
    markdown_content.append(f"- **Total cost:** ${total_cost:.4f}\n")
    markdown_content.append(f"- **Agents:** {totals.get('agents_count', 0)}\n")

    # Show per-agent usage
    agents_usage = usage.get("agents", {})
    if agents_usage:
        markdown_content.append("\n### Per-Agent Usage\n")
        for agent_name, agent_usage in agents_usage.items():
            tokens = agent_usage.get("total_tokens", 0)
            cost = agent_usage.get("cost_usd", 0.0)
            duration = agent_usage.get("duration_ms", 0)
            model = agent_usage.get("model", "unknown")
            model_profile = agent_usage.get("model_profile", "unknown")

            # Show both model profile and actual model to detect fallbacks
            model_display = (
                f"{model_profile} → {model}" if model_profile != "unknown" else model
            )

            markdown_content.append(
                f"- **{agent_name}** ({model_display}): {tokens:,} tokens, "
                f"${cost:.4f}, {duration}ms\n"
            )

    return markdown_content


def display_results(
    results: dict[str, Any], metadata: dict[str, Any], detailed: bool = False
) -> None:
    """Display results in a formatted way using Rich markdown rendering."""
    console = Console(soft_wrap=True, width=None, force_terminal=True)

    if detailed:
        # Build markdown content for detailed results using helper methods
        markdown_content = ["# Results\n"]

        # Process agent results using helper method
        markdown_content.extend(_process_agent_results(results, metadata))

        # Format performance metrics using helper method
        markdown_content.extend(_format_performance_metrics(metadata))

        # Render the markdown - Rich will handle soft wrapping
        markdown_text = "".join(markdown_content)
        markdown_obj = Markdown(markdown_text)
        console.print(markdown_obj, overflow="ellipsis", crop=True, no_wrap=False)
    else:
        # Simplified output: just show final synthesis/result
        display_simplified_results(results, metadata)


def display_simplified_results(
    results: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Display simplified results showing only the final output using markdown."""
    console = Console(soft_wrap=True, width=None, force_terminal=True)

    # Find the final agent (the one with no dependents)
    final_agent = find_final_agent(results)

    markdown_content = []

    if final_agent and results[final_agent].get("status") == "success":
        response = results[final_agent]["response"]
        # Format as code block if it looks like code, otherwise as regular text
        code_keywords = ["def ", "class ", "```", "import ", "function"]
        if any(keyword in response.lower() for keyword in code_keywords):
            markdown_content.append(f"```\n{response}\n```\n")
        else:
            markdown_content.append(f"{response}\n")
    else:
        # Fallback: show last successful agent
        successful_agents = [
            name
            for name, result in results.items()
            if result.get("status") == "success"
        ]
        if successful_agents:
            last_agent = successful_agents[-1]
            response = results[last_agent]["response"]
            markdown_content.append(f"## Result from {last_agent}\n")
            code_keywords = ["def ", "class ", "```", "import ", "function"]
            if any(keyword in response.lower() for keyword in code_keywords):
                markdown_content.append(f"```\n{response}\n```\n")
            else:
                markdown_content.append(f"{response}\n")
        else:
            markdown_content.append("**❌ No successful results found**\n")

    # Show minimal performance summary
    if "usage" in metadata:
        totals = metadata["usage"].get("totals", {})
        agents_count = totals.get("agents_count", 0)
        duration = metadata.get("duration", "unknown")
        summary = f"\n⚡ **{agents_count} agents completed in {duration}**\n"
        markdown_content.append(summary)
        markdown_content.append("*Use --detailed flag for full results and metrics*\n")

    # Render the markdown
    if markdown_content:
        markdown_text = "".join(markdown_content)
        console.print(
            Markdown(markdown_text), overflow="ellipsis", crop=True, no_wrap=False
        )


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


def _update_agent_progress_status(
    agents: list[dict[str, Any]],
    completed_agents: int,
    total_agents: int,
    agent_statuses: dict[str, str],
) -> None:
    """Update agent progress statuses based on completion count.

    Args:
        agents: List of agent configurations
        completed_agents: Number of completed agents
        total_agents: Total number of agents
        agent_statuses: Dictionary to update with agent statuses
    """
    # Mark first N agents as completed, rest as pending
    for i, agent in enumerate(agents):
        if i < completed_agents:
            agent_statuses[agent["name"]] = "completed"
        elif i == completed_agents and completed_agents < total_agents:
            agent_statuses[agent["name"]] = "running"
        else:
            agent_statuses[agent["name"]] = "pending"


def _update_agent_status_by_names(
    agents: list[dict[str, Any]],
    started_agent_names: list[str],
    completed_agent_names: list[str],
    agent_statuses: dict[str, str],
) -> None:
    """Update agent statuses based on specific agent names that have started/completed.

    Args:
        agents: List of agent configurations
        started_agent_names: Names of agents that have started
        completed_agent_names: Names of agents that have completed
        agent_statuses: Dictionary to update with agent statuses
    """
    # Convert to sets for faster lookup
    started_set = set(started_agent_names)
    completed_set = set(completed_agent_names)

    for agent in agents:
        agent_name = agent["name"]
        if agent_name in completed_set:
            agent_statuses[agent_name] = "completed"
        elif agent_name in started_set:
            agent_statuses[agent_name] = "running"
        else:
            agent_statuses[agent_name] = "pending"


def _process_execution_completed_event(
    console: Console,
    status: Any,
    agents: list[dict[str, Any]],
    event_data: dict[str, Any],
    output_format: str,
    detailed: bool,
) -> bool:
    """Process execution completed event and display final results.

    Args:
        console: Rich console instance
        status: Rich status object
        agents: List of agent configurations
        event_data: Event data containing results and metadata
        output_format: Output format (text/json)
        detailed: Whether to show detailed output

    Returns:
        bool: True to indicate loop should break
    """
    # Stop the status spinner and show final results
    status.stop()

    # Final update with all completed
    final_statuses = {agent["name"]: "completed" for agent in agents}
    final_tree = create_dependency_tree(agents, final_statuses)
    console.print(final_tree)
    console.print(f"Completed in {event_data['duration']:.2f}s")

    if output_format == "text":
        display_results(event_data["results"], event_data["metadata"], detailed)

    return True


def _handle_fallback_started_event(
    console: Console, event_data: dict[str, Any]
) -> None:
    """Handle agent_fallback_started event display."""
    agent_name = event_data["agent_name"]
    failure_type = event_data.get("failure_type", "unknown")
    error_msg = event_data["original_error"]
    original_profile = event_data.get("original_model_profile", "unknown")
    fallback_model = event_data.get("fallback_model_name", "unknown")

    # Enhanced display with failure type
    if failure_type == "oauth_error":
        failure_emoji = "🔐"
    elif failure_type == "authentication_error":
        failure_emoji = "🔑"
    else:
        failure_emoji = "⚠️"

    console.print(
        f"\n{failure_emoji} Model profile '{original_profile}' "
        f"failed for agent '{agent_name}' ({failure_type}): {error_msg}"
    )
    console.print(
        f"🔄 Using fallback model '{fallback_model}' for "
        f"agent '{agent_name}'..."
    )


def _handle_fallback_completed_event(
    console: Console, event_data: dict[str, Any]
) -> None:
    """Handle agent_fallback_completed event display."""
    agent_name = event_data["agent_name"]
    fallback_model = event_data["fallback_model_name"]
    response_preview = event_data.get("response_preview", "")

    console.print(
        f"✅ Fallback successful for agent '{agent_name}' using {fallback_model}"
    )
    if response_preview:
        console.print(f"   Preview: {response_preview}")


def _handle_fallback_failed_event(
    console: Console, event_data: dict[str, Any]
) -> None:
    """Handle agent_fallback_failed event display."""
    agent_name = event_data["agent_name"]
    failure_type = event_data.get("failure_type", "unknown")
    fallback_error = event_data["fallback_error"]
    fallback_model = event_data.get("fallback_model_name", "unknown")

    console.print(
        f"❌ Fallback model '{fallback_model}' also failed for "
        f"agent '{agent_name}' ({failure_type}): {fallback_error}"
    )


async def run_streaming_execution(
    executor: Any,
    ensemble_config: EnsembleConfig,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> None:
    """Run streaming execution with Rich status display."""
    console = Console(soft_wrap=True, width=None, force_terminal=True)
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
                    # Extract detailed agent status from progress data
                    started_agent_names = event["data"].get("started_agent_names", [])
                    completed_agent_names = event["data"].get(
                        "completed_agent_names", []
                    )

                    # Update agent statuses based on actual agent states
                    _update_agent_status_by_names(
                        ensemble_config.agents,
                        started_agent_names,
                        completed_agent_names,
                        agent_statuses,
                    )

                    # Update status display with current dependency tree
                    current_tree = create_dependency_tree(
                        ensemble_config.agents, agent_statuses
                    )
                    status.update(current_tree)

                elif event_type == "agent_fallback_started":
                    _handle_fallback_started_event(console, event["data"])

                elif event_type == "agent_fallback_completed":
                    _handle_fallback_completed_event(console, event["data"])

                elif event_type == "agent_fallback_failed":
                    _handle_fallback_failed_event(console, event["data"])

                elif event_type == "execution_completed":
                    # Process execution completed event using helper method
                    if _process_execution_completed_event(
                        console,
                        status,
                        ensemble_config.agents,
                        event["data"],
                        output_format,
                        detailed,
                    ):
                        break


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
