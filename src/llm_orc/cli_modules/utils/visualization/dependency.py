"""Dependency graph and tree visualization utilities."""

from typing import Any

from rich.tree import Tree


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

            agent_labels.append(f"{symbol} [{style}]{agent_name}[/{style}]")

        # Create level label and add all agents
        level_label = f"Phase {level + 1}"
        level_node = tree.add(f"[bold]{level_label}[/bold]")

        # Add all agents on the same line, grouped
        agents_text = " | ".join(agent_labels)
        level_node.add(agents_text)

    return tree


def create_dependency_graph_with_status(
    agents: list[dict[str, Any]], agent_statuses: dict[str, str]
) -> str:
    """Create dependency graph with status indicators."""
    if not agents:
        return "No agents to display"

    # Group agents by level
    agents_by_level = _group_agents_by_dependency_level(agents)
    if not agents_by_level:
        return "No dependency levels found"

    max_level = max(agents_by_level.keys())

    for level in range(max_level + 1):
        if level not in agents_by_level:
            continue

        level_agents = agents_by_level[level]
        agent_displays = []

        for agent in level_agents:
            name = agent["name"]
            status = agent_statuses.get(name, "pending")

            if status == "running":
                symbol = "◐"
            elif status == "completed":
                symbol = "✓"
            elif status == "failed":
                symbol = "✗"
            else:
                symbol = "○"

            agent_displays.append(f"{symbol} {name}")

    # Return a simple representation for text mode
    return " → ".join(
        [
            ", ".join([a["name"] for a in level_agents])
            for level_agents in agents_by_level.values()
        ]
    )


def find_final_agent(results: dict[str, Any]) -> str | None:
    """Find the final agent that should be displayed."""
    # Priority order: coordinator > synthesizer > last successful agent
    successful_agents = [
        name for name, result in results.items() if result.get("status") == "success"
    ]

    if not successful_agents:
        return None

    # Check for special agent names first
    if "coordinator" in successful_agents:
        return "coordinator"
    if "synthesizer" in successful_agents:
        return "synthesizer"

    # Return the last successful agent
    return successful_agents[-1]


def _group_agents_by_dependency_level(
    agents: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group agents by their dependency level."""
    agents_by_level: dict[int, list[dict[str, Any]]] = {}

    for agent in agents:
        level = _calculate_agent_level(agent, agents)
        if level not in agents_by_level:
            agents_by_level[level] = []
        agents_by_level[level].append(agent)

    return agents_by_level


def _calculate_agent_level(
    agent: dict[str, Any], all_agents: list[dict[str, Any]]
) -> int:
    """Calculate the dependency level of an agent."""
    dependencies = agent.get("depends_on", [])
    if not dependencies:
        return 0

    # Find the maximum level of dependencies
    max_dep_level = 0
    for dep_name in dependencies:
        for dep_agent in all_agents:
            if dep_agent["name"] == dep_name:
                dep_level = _calculate_agent_level(dep_agent, all_agents)
                max_dep_level = max(max_dep_level, dep_level)

    return max_dep_level + 1


def _create_plain_text_dependency_graph(agents: list[dict[str, Any]]) -> list[str]:
    """Create a plain text dependency graph."""
    lines = []
    agents_by_level = _group_agents_by_dependency_level(agents)

    if not agents_by_level:
        return ["No agents found"]

    max_level = max(agents_by_level.keys())

    # Build the graph level by level
    for level in range(max_level + 1):
        if level not in agents_by_level:
            continue

        level_agents = agents_by_level[level]
        agent_names = [agent["name"] for agent in level_agents]

        if level == 0:
            # First level
            lines.append(" | ".join(agent_names))
        else:
            # Subsequent levels with arrow
            lines.append(" ↓ ")
            lines.append(" | ".join(agent_names))

    return lines


def _create_structured_dependency_info(
    agents: list[dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, str]]:
    """Create structured dependency information for display."""
    agents_by_level = _group_agents_by_dependency_level(agents)
    agent_statuses = _create_agent_statuses(agents)
    return agents_by_level, agent_statuses


def _create_agent_statuses(agents: list[dict[str, Any]]) -> dict[str, str]:
    """Create initial agent status mapping."""
    return {agent["name"]: "pending" for agent in agents}


def _build_dependency_levels(
    agents: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Build dependency levels for agents."""
    return _group_agents_by_dependency_level(agents)
