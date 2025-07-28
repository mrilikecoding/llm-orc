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


def _create_plain_text_dependency_graph(
    agents: list[dict[str, Any]], agent_statuses: dict[str, str]
) -> str:
    """Create plain text dependency graph without Rich formatting."""
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

            # Status indicators with plain text symbols
            if status == "running":
                agent_displays.append(f"◐ {name}")
            elif status == "completed":
                agent_displays.append(f"✓ {name}")
            elif status == "failed":
                agent_displays.append(f"✗ {name}")
            else:
                agent_displays.append(f"○ {name}")

        # Join agents at same level with commas
        level_text = ", ".join(agent_displays)
        graph_parts.append(level_text)

    # Join levels with arrows
    return " → ".join(graph_parts)


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


def display_plain_text_results(
    results: dict[str, Any], 
    metadata: dict[str, Any], 
    detailed: bool = False,
    agents: list[dict[str, Any]] | None = None,
) -> None:
    """Display results in plain text format without Rich formatting."""
    # Show dependency graph if agents provided
    if agents:
        # Create agent status based on results (completed/failed)
        agent_statuses = {}
        for agent in agents:
            agent_name = agent["name"]
            if agent_name in results:
                status = results[agent_name].get("status", "pending")
                agent_statuses[agent_name] = "completed" if status == "success" else "failed"
            else:
                agent_statuses[agent_name] = "pending"
        
        # Show dependency graph (plain text version without Rich formatting)
        dependency_graph = _create_plain_text_dependency_graph(agents, agent_statuses)
        click.echo("Dependency Graph:")
        click.echo(dependency_graph)
        click.echo()
    
    if detailed:
        # Detailed plain text output
        click.echo("Results")
        click.echo("=======")
        click.echo()

        # Show agent results
        for agent_name, result in results.items():
            if result.get("status") == "success":
                click.echo(f"{agent_name}:")
                click.echo("-" * len(agent_name) + ":")
                click.echo(result["response"])
                click.echo()
            else:
                click.echo(f"❌ {agent_name}:")
                click.echo("-" * (len(agent_name) + 3) + ":")
                error_msg = result.get("error", "Unknown error")
                click.echo(f"Error: {error_msg}")
                click.echo()

        # Show performance metrics
        if "usage" in metadata:
            usage = metadata["usage"]
            totals = usage.get("totals", {})
            
            click.echo("Performance Metrics")
            click.echo("==================")
            click.echo(f"Duration: {metadata['duration']}")
            click.echo(f"Total tokens: {totals.get('total_tokens', 0):,}")
            click.echo(f"Total cost: ${totals.get('total_cost_usd', 0.0):.4f}")
            click.echo(f"Agents: {totals.get('agents_count', 0)}")
    else:
        # Simplified plain text output - just show final result
        final_agent = find_final_agent(results)
        
        if final_agent and results[final_agent].get("status") == "success":
            response = results[final_agent]["response"]
            click.echo(response)
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
            click.echo("Use --detailed flag for full results and metrics")


def display_simplified_results(
    results: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Display simplified results showing only the final output using markdown."""
    console = Console(soft_wrap=True, width=None, force_terminal=True)

    # Find the final agent (the one with no dependents)
    final_agent = find_final_agent(results)
    
    # Debug: print to see what's happening
    # print(f"DEBUG: final_agent={final_agent}, results keys={list(results.keys())}")  # Uncomment for debugging

    markdown_content = []

    if final_agent and results[final_agent].get("status") == "success":
        response = results[final_agent]["response"]
        # Add a clear header to indicate this is the final result
        markdown_content.append(f"## Result\n\n")
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


def _create_structured_dependency_info(
    agents: list[dict[str, Any]], results: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create structured dependency information for JSON output."""
    # Group agents by dependency level
    agents_by_level = _group_agents_by_dependency_level(agents)
    
    # Create agent status if results provided
    agent_statuses = {}
    if results:
        for agent in agents:
            agent_name = agent["name"]
            if agent_name in results:
                status = results[agent_name].get("status", "pending")
                agent_statuses[agent_name] = "completed" if status == "success" else "failed"
            else:
                agent_statuses[agent_name] = "pending"
    
    # Build structured dependency info
    dependency_levels = []
    for level in sorted(agents_by_level.keys()):
        level_agents = []
        for agent in agents_by_level[level]:
            agent_info = {
                "name": agent["name"],
                "dependencies": agent.get("depends_on", [])
            }
            if results and agent["name"] in agent_statuses:
                agent_info["status"] = agent_statuses[agent["name"]]
            level_agents.append(agent_info)
        
        dependency_levels.append({
            "level": level,
            "agents": level_agents
        })
    
    return {
        "dependency_levels": dependency_levels,
        "dependency_graph": _create_plain_text_dependency_graph(agents, agent_statuses) if results else create_dependency_graph(agents)
    }


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

    # Always display results for Rich interface (output_format is None or "rich")
    # The text/JSON outputs are handled separately in _run_text_json_execution
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
        f"{failure_emoji} Model profile '{original_profile}' "
        f"failed for agent '{agent_name}' ({failure_type}): {error_msg}"
    )
    console.print(
        f"🔄 Using fallback model '{fallback_model}' for agent '{agent_name}'..."
    )
    console.print("─" * 50)


def _handle_fallback_completed_event(
    console: Console, event_data: dict[str, Any]
) -> None:
    """Handle agent_fallback_completed event display."""
    agent_name = event_data["agent_name"]
    fallback_model = event_data["fallback_model_name"]
    response_preview = event_data.get("response_preview", "")

    # Use print() to bypass Rich buffering and show immediately
    print(
        f"✅ SUCCESS: Fallback model '{fallback_model}' succeeded for "
        f"agent '{agent_name}'"
    )


def _handle_fallback_failed_event(console: Console, event_data: dict[str, Any]) -> None:
    """Handle agent_fallback_failed event display."""
    agent_name = event_data["agent_name"]
    failure_type = event_data.get("failure_type", "unknown")
    fallback_error = event_data["fallback_error"]
    fallback_model = event_data.get("fallback_model_name", "unknown")

    # Use print() to bypass Rich buffering and show immediately
    print(
        f"❌ FAILED: Fallback model '{fallback_model}' also failed for "
        f"agent '{agent_name}' ({failure_type}): {fallback_error}"
    )


def _handle_text_fallback_started(event_data: dict[str, Any]) -> None:
    """Handle agent_fallback_started event for text output."""
    agent_name = event_data["agent_name"]
    failure_type = event_data.get("failure_type", "unknown")
    error_msg = event_data["original_error"]
    original_profile = event_data.get("original_model_profile", "unknown")
    fallback_model = event_data.get("fallback_model_name", "unknown")

    click.echo(
        f"WARNING: Model profile '{original_profile}' failed for "
        f"agent '{agent_name}' ({failure_type}): {error_msg}"
    )
    click.echo(f"Using fallback model '{fallback_model}' for agent '{agent_name}'...")
    click.echo("─" * 50)


def _handle_text_fallback_completed(event_data: dict[str, Any]) -> None:
    """Handle agent_fallback_completed event for text output."""
    agent_name = event_data["agent_name"]
    fallback_model = event_data["fallback_model_name"]
    response_preview = event_data.get("response_preview", "")

    click.echo(
        f"SUCCESS: Fallback model '{fallback_model}' succeeded for agent '{agent_name}'"
    )


def _handle_streaming_event(
    event_type: str,
    event: dict[str, Any],
    agent_statuses: dict[str, str],
    ensemble_config: EnsembleConfig,
    status: Any,
    console: Console,
    output_format: str = "rich",
    detailed: bool = False,
) -> bool:
    """Handle a single streaming event and update status display.

    Returns True if execution should continue, False if it should break.
    """
    if event_type == "agent_progress":
        # Extract detailed agent status from progress data
        started_agent_names = event["data"].get("started_agent_names", [])
        completed_agent_names = event["data"].get("completed_agent_names", [])

        # Update agent statuses based on actual agent states
        _update_agent_status_by_names(
            ensemble_config.agents,
            started_agent_names,
            completed_agent_names,
            agent_statuses,
        )

        # Update status display with current dependency tree
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

    elif event_type == "execution_started":
        # Show initial dependency tree when execution starts
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

    elif event_type == "agent_started":
        # Agent has started execution
        event_data = event["data"]
        agent_name = event_data["agent_name"]
        agent_statuses[agent_name] = "running"
        
        # Update status display with current dependency tree
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)
        
    elif event_type == "agent_completed":
        # Agent has completed execution
        event_data = event["data"]
        agent_name = event_data["agent_name"]
        agent_statuses[agent_name] = "completed"
        
        # Update status display with current dependency tree
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

    elif event_type == "execution_progress":
        # Just update the tree without elapsed time overlay to avoid flickering
        # The elapsed time will be shown in the final completion message
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

    elif event_type == "agent_fallback_started":
        # Show fallback message without disrupting the tree display
        event_data = event["data"]
        agent_name = event_data["agent_name"]

        # Update agent status to running (attempting fallback)
        agent_statuses[agent_name] = "running"

        # Update status display with current dependency tree
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

        _handle_fallback_started_event(console, event_data)

    elif event_type == "agent_fallback_completed":
        # Update agent status to completed
        event_data = event["data"]
        agent_name = event_data["agent_name"]
        agent_statuses[agent_name] = "completed"

        # Update status display with current dependency tree
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

    elif event_type == "agent_fallback_failed":
        # Update agent status to failed
        event_data = event["data"]
        agent_name = event_data["agent_name"]
        agent_statuses[agent_name] = "failed"

        # Update status with current tree
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

        _handle_fallback_failed_event(console, event_data)

    elif event_type == "execution_completed":
        # Process execution completed event using helper method
        return _process_execution_completed_event(
            console,
            status,
            ensemble_config.agents,
            event["data"],
            output_format,
            detailed,
        )

    return True


def _handle_text_fallback_failed(event_data: dict[str, Any]) -> None:
    """Handle agent_fallback_failed event for text output."""
    agent_name = event_data["agent_name"]
    failure_type = event_data.get("failure_type", "unknown")
    fallback_error = event_data["fallback_error"]
    fallback_model = event_data.get("fallback_model_name", "unknown")

    click.echo(
        f"ERROR: Fallback model '{fallback_model}' also failed for "
        f"agent '{agent_name}' ({failure_type}): {fallback_error}"
    )


async def _run_text_json_execution(
    executor: Any,
    ensemble_config: EnsembleConfig,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> None:
    """Run execution with text or JSON output (no Rich interface)."""
    if output_format == "json":
        import json
        
        # For JSON output, collect all meaningful events and output as single JSON
        collected_events: list[dict[str, Any]] = []
        meaningful_events = {
            "execution_started",
            "execution_completed",
            "agent_started",
            "agent_completed", 
            "agent_fallback_started", 
            "agent_fallback_completed",
            "agent_fallback_failed",
            "phase_started",
            "phase_completed"
        }
        
        final_result = None
        async for event in executor.execute_streaming(ensemble_config, input_data):
            event_type = event["type"]
            
            if event_type in meaningful_events:
                collected_events.append(event)
            
            # Store final result for consolidated output
            if event_type == "execution_completed":
                final_result = event["data"]
        
        # Add structured dependency information
        dependency_info = _create_structured_dependency_info(
            ensemble_config.agents, 
            final_result["results"] if final_result else None
        )
        
        # Output consolidated JSON structure
        output_data = {
            "events": collected_events,
            "result": final_result,
            "dependency_info": dependency_info
        }
        click.echo(json.dumps(output_data, indent=2))
        
    else:
        # Handle text output - stream events as they come
        async for event in executor.execute_streaming(ensemble_config, input_data):
            event_type = event["type"]
            if event_type == "agent_fallback_started":
                _handle_text_fallback_started(event["data"])
            elif event_type == "agent_fallback_completed":
                _handle_text_fallback_completed(event["data"])
            elif event_type == "agent_fallback_failed":
                _handle_text_fallback_failed(event["data"])
            elif event_type == "execution_completed":
                # Show final results for text output using plain text formatting
                display_plain_text_results(
                    event["data"]["results"], 
                    event["data"]["metadata"], 
                    detailed,
                    ensemble_config.agents
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

    # Use Rich interface by default (None), but simple processing for explicit text/json
    if output_format in ["json", "text"]:
        # Direct processing without Rich status for JSON/text output
        await _run_text_json_execution(
            executor, ensemble_config, input_data, output_format, detailed
        )
    else:
        # Rich interface for default output (None = streaming)
        # Initialize with dependency tree in status display
        initial_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        
        with console.status(initial_tree, spinner="dots") as status:
            async for event in executor.execute_streaming(ensemble_config, input_data):
                event_type = event["type"]

                # Handle the event and check if execution should continue
                should_continue = _handle_streaming_event(
                    event_type,
                    event,
                    agent_statuses,
                    ensemble_config,
                    status,
                    console,
                    output_format,
                    detailed,
                )

                # If event handler returns False, break the loop
                if not should_continue:
                    break


async def run_standard_execution(
    executor: Any,
    ensemble_config: EnsembleConfig,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> None:
    """Run standard execution without streaming but with fallback event monitoring."""
    # Set up fallback event monitoring for text/JSON output
    fallback_events: list[dict[str, Any]] = []
    
    def capture_fallback_event(event_type: str, data: dict[str, Any]) -> None:
        """Capture fallback events during standard execution."""
        if event_type in ["agent_fallback_started", "agent_fallback_completed", "agent_fallback_failed"]:
            fallback_events.append({"type": event_type, "data": data})
    
    # Temporarily replace the event emission to capture fallback events
    original_emit = executor._emit_performance_event
    executor._emit_performance_event = capture_fallback_event
    
    try:
        result = await executor.execute(ensemble_config, input_data)
        
        # Process any captured fallback events
        if fallback_events and output_format == "text":
            for event in fallback_events:
                event_type = event["type"]
                event_data = event["data"]
                if event_type == "agent_fallback_started":
                    _handle_text_fallback_started(event_data)
                elif event_type == "agent_fallback_failed":
                    _handle_text_fallback_failed(event_data)
        
        # Display results
        if output_format == "json":
            import json
            
            # Add structured dependency information  
            dependency_info = _create_structured_dependency_info(
                ensemble_config.agents, 
                result["results"]
            )
            
            # Include fallback events and dependency info in JSON output
            output_data = result.copy()
            output_data["dependency_info"] = dependency_info
            if fallback_events:
                output_data["fallback_events"] = fallback_events
            click.echo(json.dumps(output_data, indent=2))
        elif output_format == "text":
            # Use plain text output for clean piping
            display_plain_text_results(
                result["results"], 
                result["metadata"], 
                detailed, 
                ensemble_config.agents
            )
        else:
            # Use Rich formatting for default output
            display_results(result["results"], result["metadata"], detailed)
            
    finally:
        # Restore original event emission
        executor._emit_performance_event = original_emit
