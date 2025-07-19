"""Simple horizontal dependency graph visualization."""

from typing import Any

from rich.console import Console
from rich.live import Live

from .config import SimpleVisualizationConfig
from .events import ExecutionEvent, ExecutionEventType
from .stream import EventStream


class SimpleVisualizer:
    """Simple horizontal dependency graph visualization."""

    def __init__(self, config: SimpleVisualizationConfig | None = None):
        self.config = config or SimpleVisualizationConfig()
        self.console = Console()
        self.live_display: Live | None = None

        # State tracking
        self.execution_state: dict[str, Any] = {
            "ensemble_name": "",
            "execution_id": "",
            "status": "starting",
            "start_time": None,
            "total_agents": 0,
            "completed_agents": 0,
            "failed_agents": 0,
            "agents": {},  # agent_name -> agent_info
        }

    async def visualize_execution(self, stream: EventStream) -> None:
        """Visualize ensemble execution with simple dependency graph."""
        try:
            dependency_tree_shown = False

            # Subscribe to all events
            async for event in stream.subscribe():
                await self._process_event(event)

                # Show complete dependency tree once we have agent info
                if not dependency_tree_shown and self.execution_state["agents"]:
                    # Show the complete dependency tree before execution starts
                    full_tree = self._create_complete_dependency_tree()
                    self.console.print(f"🔗 Dependency flow: {full_tree}")
                    dependency_tree_shown = True

                # Check if execution is complete
                if self._is_execution_complete(event):
                    break

        except Exception as e:
            self.console.print(f"❌ Visualization error: {e}")

    async def _process_event(self, event: ExecutionEvent) -> None:
        """Process a single execution event."""
        if event.event_type == ExecutionEventType.ENSEMBLE_STARTED:
            await self._handle_ensemble_started(event)
        elif event.event_type == ExecutionEventType.AGENT_STARTED:
            await self._handle_agent_started(event)
        elif event.event_type == ExecutionEventType.AGENT_COMPLETED:
            await self._handle_agent_completed(event)
        elif event.event_type == ExecutionEventType.AGENT_FAILED:
            await self._handle_agent_failed(event)
        elif event.event_type == ExecutionEventType.ENSEMBLE_COMPLETED:
            await self._handle_ensemble_completed(event)

    async def _handle_ensemble_started(self, event: ExecutionEvent) -> None:
        """Handle ensemble started event."""
        self.execution_state.update(
            {
                "ensemble_name": event.ensemble_name or "Unknown",
                "execution_id": event.execution_id or "Unknown",
                "status": "running",
                "start_time": event.timestamp,
                "total_agents": event.data.get("total_agents", 0),
            }
        )

        # Populate all agents with pending status
        agents_config = event.data.get("agents_config", [])
        for agent_config in agents_config:
            agent_name = agent_config.get("name", "unknown")
            self.execution_state["agents"][agent_name] = {
                "name": agent_name,
                "status": "pending",  # Not yet started
                "dependencies": agent_config.get("depends_on", []),
                "model_profile": agent_config.get("model_profile", "default"),
            }

    async def _handle_agent_started(self, event: ExecutionEvent) -> None:
        """Handle agent started event."""
        agent_name = event.agent_name
        if not agent_name or agent_name not in self.execution_state["agents"]:
            return

        # Update agent state to running
        self.execution_state["agents"][agent_name].update(
            {
                "status": "running",
                "model": event.data.get("model", "Unknown"),
                "start_time": event.timestamp,
            }
        )

    async def _handle_agent_completed(self, event: ExecutionEvent) -> None:
        """Handle agent completed event."""
        agent_name = event.agent_name
        if not agent_name or agent_name not in self.execution_state["agents"]:
            return

        # Update agent state
        self.execution_state["agents"][agent_name]["status"] = "completed"
        self.execution_state["completed_agents"] += 1

    async def _handle_agent_failed(self, event: ExecutionEvent) -> None:
        """Handle agent failed event."""
        agent_name = event.agent_name
        if not agent_name or agent_name not in self.execution_state["agents"]:
            return

        # Update agent state
        self.execution_state["agents"][agent_name]["status"] = "failed"
        self.execution_state["failed_agents"] += 1

    async def _handle_ensemble_completed(self, event: ExecutionEvent) -> None:
        """Handle ensemble completed event."""
        self.execution_state["status"] = "completed"

    def _create_dependency_graph(self) -> str:
        """Create the horizontal dependency graph: A,B,C → D → E,F → G"""
        if not self.execution_state["agents"]:
            return "Agents loading..."

        # Group agents by dependency level
        agents_by_level = self._group_agents_by_level()

        if not agents_by_level:
            return "No agents to display"

        # Build horizontal graph: A,B,C → D → E,F → G
        graph_parts = []
        max_level = max(agents_by_level.keys())

        for level in range(max_level + 1):
            if level not in agents_by_level:
                continue

            level_agents = agents_by_level[level]
            agent_displays = []

            for agent_info in level_agents:
                status = agent_info["status"]
                name = agent_info["name"]

                # Status indicators as requested
                if status == "running":
                    # Use animated spinner while processing
                    agent_displays.append(f"[yellow]⠋[/yellow] {name}")
                elif status == "completed":
                    # Use checkmark when complete
                    agent_displays.append(f"[green]✓[/green] {name}")
                elif status == "failed":
                    # Use X for failed
                    agent_displays.append(f"[red]✗[/red] {name}")
                else:
                    # Use static dot for pending/not yet started
                    agent_displays.append(f"[dim]⦁[/dim] {name}")

            # Join agents at same level with commas
            level_text = ", ".join(agent_displays)
            graph_parts.append(level_text)

        # Join levels with arrows
        return " → ".join(graph_parts)

    def _create_complete_dependency_tree(self) -> str:
        """Create complete dependency tree showing ALL agents with static symbols."""
        if not self.execution_state["agents"]:
            return "No agents configured"

        # Group agents by dependency level
        agents_by_level = self._group_agents_by_level()

        if not agents_by_level:
            return "No agents to display"

        # Build horizontal graph with all agents using static symbols
        graph_parts = []
        max_level = max(agents_by_level.keys())

        for level in range(max_level + 1):
            if level not in agents_by_level:
                continue

            level_agents = agents_by_level[level]
            agent_displays = []

            for agent_info in level_agents:
                name = agent_info["name"]
                # Show all agents with static dot initially
                agent_displays.append(f"[dim]⦁[/dim] {name}")

            # Join agents at same level with commas
            level_text = ", ".join(agent_displays)
            graph_parts.append(level_text)

        # Join levels with arrows
        return " → ".join(graph_parts)

    def _group_agents_by_level(self) -> dict[int, list[dict[str, Any]]]:
        """Group agents by their dependency level."""
        agents_by_level: dict[int, list[dict[str, Any]]] = {}

        for agent_name, agent_info in self.execution_state["agents"].items():
            dependencies = agent_info.get("dependencies", [])
            level = self._calculate_agent_level(agent_name, dependencies)

            if level not in agents_by_level:
                agents_by_level[level] = []
            agents_by_level[level].append(agent_info)

        return agents_by_level

    def _calculate_agent_level(self, agent_name: str, dependencies: list[str]) -> int:
        """Calculate the dependency level of an agent (0 = no dependencies)."""
        if not dependencies:
            return 0

        # Find the maximum level of all dependencies
        max_dep_level = 0
        for dep_name in dependencies:
            if dep_name in self.execution_state["agents"]:
                dep_info = self.execution_state["agents"][dep_name]
                dep_dependencies = dep_info.get("dependencies", [])
                dep_level = self._calculate_agent_level(dep_name, dep_dependencies)
                max_dep_level = max(max_dep_level, dep_level)

        return max_dep_level + 1

    def _is_execution_complete(self, event: ExecutionEvent) -> bool:
        """Check if execution is complete."""
        return (
            event.event_type == ExecutionEventType.ENSEMBLE_COMPLETED
            or event.event_type == ExecutionEventType.ENSEMBLE_FAILED
        )

    def print_summary(self) -> None:
        """Print execution summary."""
        if not self.execution_state["ensemble_name"]:
            return

        # Final dependency graph showing completed status
        final_graph = self._create_dependency_graph()
        self.console.print(f"✅ Final: {final_graph}")
