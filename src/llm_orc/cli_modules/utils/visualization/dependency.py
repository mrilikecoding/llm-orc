"""Dependency graph and tree visualization utilities."""

from typing import Any

from rich.tree import Tree

from llm_orc.core.execution.utils import dep_name
from llm_orc.schemas.agent_config import AgentConfig


def create_dependency_graph(agents: list[AgentConfig]) -> str:
    """Create horizontal dependency graph: A,B,C → D → E,F → G"""
    return create_dependency_graph_with_status(agents, {})


def create_dependency_tree(
    agents: list[AgentConfig], agent_statuses: dict[str, str] | None = None
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
            agent_name = agent.name
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
    agents: list[AgentConfig], agent_statuses: dict[str, str]
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
            name = agent.name
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
            ", ".join([a.name for a in level_agents])
            for level_agents in agents_by_level.values()
        ]
    )


def find_final_agent(
    results: dict[str, Any],
    agents: list[AgentConfig] | None = None,
) -> str | None:
    """Find the final agent that should be displayed.

    When ``agents`` config is provided, picks the successful agent at the
    highest dependency level.  Falls back to name-based heuristics when
    no config is available.
    """
    successful_agents = [
        name for name, result in results.items() if result.get("status") == "success"
    ]

    if not successful_agents:
        return None

    # Check for well-known coordinator/synthesizer names first
    if "coordinator" in successful_agents:
        return "coordinator"
    if "synthesizer" in successful_agents:
        return "synthesizer"

    # Use dependency graph when agents config is available
    if agents:
        found = _find_highest_level_successful_agent(agents, successful_agents)
        if found:
            return found

    # Fallback: last successful agent by dict order
    return successful_agents[-1]


def _find_highest_level_successful_agent(
    agents: list[AgentConfig],
    successful_agents: list[str],
) -> str | None:
    """Return the successful agent at the highest dependency level."""
    agents_by_level = _group_agents_by_dependency_level(agents)
    if not agents_by_level:
        return None

    max_level = max(agents_by_level.keys())
    for level in range(max_level, -1, -1):
        for agent in agents_by_level.get(level, []):
            if agent.name in successful_agents:
                return str(agent.name)
    return None


def _group_agents_by_dependency_level(
    agents: list[AgentConfig],
) -> dict[int, list[AgentConfig]]:
    """Group agents by their dependency level."""
    agents_by_level: dict[int, list[AgentConfig]] = {}
    cache: dict[str, int] = {}

    for agent in agents:
        level = _calculate_agent_level(agent, agents, cache)
        if level not in agents_by_level:
            agents_by_level[level] = []
        agents_by_level[level].append(agent)

    return agents_by_level


def _calculate_agent_level(
    agent: AgentConfig,
    all_agents: list[AgentConfig],
    _cache: dict[str, int] | None = None,
) -> int:
    """Calculate the dependency level of an agent."""
    if _cache is None:
        _cache = {}

    name = agent.name
    if name in _cache:
        return _cache[name]

    dependencies = agent.depends_on
    if not dependencies:
        _cache[name] = 0
        return 0

    # Find the maximum level of dependencies
    max_dep_level = 0
    for dep in dependencies:
        dep_agent_name = dep_name(dep)
        for dep_agent in all_agents:
            if dep_agent.name == dep_agent_name:
                dep_level = _calculate_agent_level(dep_agent, all_agents, _cache)
                max_dep_level = max(max_dep_level, dep_level)

    level = max_dep_level + 1
    _cache[name] = level
    return level


def _create_plain_text_dependency_graph(agents: list[AgentConfig]) -> list[str]:
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
        agent_names = [agent.name for agent in level_agents]

        if level == 0:
            # First level
            lines.append(" | ".join(agent_names))
        else:
            # Subsequent levels with arrow
            lines.append(" ↓ ")
            lines.append(" | ".join(agent_names))

    return lines


def _create_structured_dependency_info(
    agents: list[AgentConfig],
) -> tuple[dict[int, list[AgentConfig]], dict[str, str]]:
    """Create structured dependency information for display."""
    agents_by_level = _group_agents_by_dependency_level(agents)
    agent_statuses = _create_agent_statuses(agents)
    return agents_by_level, agent_statuses


def _create_agent_statuses(agents: list[AgentConfig]) -> dict[str, str]:
    """Create initial agent status mapping."""
    return {agent.name: "pending" for agent in agents}
