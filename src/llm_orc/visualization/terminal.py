"""Terminal-based visualization for ensemble execution."""

from datetime import datetime
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .config import VisualizationConfig
from .events import ExecutionEvent, ExecutionEventType
from .stream import EventStream


class TerminalVisualizer:
    """Terminal-based visualization using Rich library."""

    def __init__(self, config: VisualizationConfig | None = None):
        self.config = config or VisualizationConfig()
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
            "performance": {
                "total_cost": 0.0,
                "total_duration": 0,
                "memory_usage": 0,
            },
            "live_results": [],
        }

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.overall_task_id: TaskID | None = None
        self.agent_task_ids: dict[str, TaskID] = {}

    async def visualize_execution(self, stream: EventStream) -> None:
        """Visualize ensemble execution from event stream."""
        try:
            # Start live display
            with Live(
                self.create_layout(),
                console=self.console,
                auto_refresh=True,
                refresh_per_second=10,
            ) as live:
                self.live_display = live

                # Subscribe to all events
                async for event in stream.subscribe():
                    await self._process_event(event)

                    # Update display
                    layout = self.create_layout()
                    live.update(layout)

                    # Check if execution is complete
                    if self._is_execution_complete(event):
                        break

        except Exception as e:
            self.console.print(f"❌ Visualization error: {e}")
        finally:
            self.live_display = None

    async def _process_event(self, event: ExecutionEvent) -> None:
        """Process a single execution event."""
        if event.event_type == ExecutionEventType.ENSEMBLE_STARTED:
            await self._handle_ensemble_started(event)
        elif event.event_type == ExecutionEventType.AGENT_STARTED:
            await self._handle_agent_started(event)
        elif event.event_type == ExecutionEventType.AGENT_PROGRESS:
            await self._handle_agent_progress(event)
        elif event.event_type == ExecutionEventType.AGENT_COMPLETED:
            await self._handle_agent_completed(event)
        elif event.event_type == ExecutionEventType.AGENT_FAILED:
            await self._handle_agent_failed(event)
        elif event.event_type == ExecutionEventType.ENSEMBLE_COMPLETED:
            await self._handle_ensemble_completed(event)
        elif event.event_type == ExecutionEventType.PERFORMANCE_METRIC:
            await self._handle_performance_metric(event)

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

        # Create overall progress task
        self.overall_task_id = self.progress.add_task(
            f"🎭 {self.execution_state['ensemble_name']} Ensemble",
            total=self.execution_state["total_agents"],
        )

    async def _handle_agent_started(self, event: ExecutionEvent) -> None:
        """Handle agent started event."""
        agent_name = event.agent_name
        if not agent_name:
            return

        # Update agent state
        self.execution_state["agents"][agent_name] = {
            "name": agent_name,
            "status": "running",
            "model": event.data.get("model", "Unknown"),
            "dependencies": event.data.get("depends_on", []),
            "start_time": event.timestamp,
            "progress": 0.0,
            "result": None,
            "error": None,
            "duration": 0,
            "cost": 0.0,
        }

        # Create progress task for agent
        if self.config.terminal.show_agent_progress:
            task_id = self.progress.add_task(
                f"🤖 {agent_name}",
                total=100,
            )
            self.agent_task_ids[agent_name] = task_id

    async def _handle_agent_progress(self, event: ExecutionEvent) -> None:
        """Handle agent progress event."""
        agent_name = event.agent_name
        if not agent_name or agent_name not in self.execution_state["agents"]:
            return

        progress = event.data.get("progress_percentage", 0.0)
        self.execution_state["agents"][agent_name]["progress"] = progress

        # Update progress bar
        if agent_name in self.agent_task_ids:
            self.progress.update(self.agent_task_ids[agent_name], completed=progress)

        # Add intermediate result if available
        intermediate_result = event.data.get("intermediate_result")
        if intermediate_result and self.config.terminal.show_live_results:
            self.execution_state["live_results"].append(
                {
                    "agent": agent_name,
                    "result": intermediate_result,
                    "timestamp": event.timestamp,
                }
            )

    async def _handle_agent_completed(self, event: ExecutionEvent) -> None:
        """Handle agent completed event."""
        agent_name = event.agent_name
        if not agent_name or agent_name not in self.execution_state["agents"]:
            return

        # Update agent state
        agent_info = self.execution_state["agents"][agent_name]
        cost_usd = event.data.get("cost_usd")
        agent_info.update(
            {
                "status": "completed",
                "result": event.data.get("result", ""),
                "duration": event.data.get("duration_ms", 0),
                "cost": cost_usd if cost_usd is not None else 0.0,
                "progress": 100.0,
            }
        )

        # Update progress bars
        if agent_name in self.agent_task_ids:
            self.progress.update(self.agent_task_ids[agent_name], completed=100)

        # Update overall progress
        self.execution_state["completed_agents"] += 1
        if self.overall_task_id:
            self.progress.update(
                self.overall_task_id, completed=self.execution_state["completed_agents"]
            )

        # Update performance metrics
        self.execution_state["performance"]["total_cost"] += agent_info["cost"]
        self.execution_state["performance"]["total_duration"] += agent_info["duration"]

        # Add result to live results
        if self.config.terminal.show_live_results:
            self.execution_state["live_results"].append(
                {
                    "agent": agent_name,
                    "result": (
                        f"✅ {agent_name}: Completed in {agent_info['duration']}ms"
                    ),
                    "timestamp": event.timestamp,
                }
            )

    async def _handle_agent_failed(self, event: ExecutionEvent) -> None:
        """Handle agent failed event."""
        agent_name = event.agent_name
        if not agent_name or agent_name not in self.execution_state["agents"]:
            return

        # Update agent state
        agent_info = self.execution_state["agents"][agent_name]
        agent_info.update(
            {
                "status": "failed",
                "error": event.data.get("error", "Unknown error"),
                "duration": event.data.get("duration_ms", 0),
                "progress": 0.0,
            }
        )

        # Update progress bars
        if agent_name in self.agent_task_ids:
            self.progress.update(self.agent_task_ids[agent_name], completed=0)

        # Update counters
        self.execution_state["failed_agents"] += 1
        if self.overall_task_id:
            self.progress.update(
                self.overall_task_id, completed=self.execution_state["completed_agents"]
            )

        # Add error to live results
        self.execution_state["live_results"].append(
            {
                "agent": agent_name,
                "result": f"❌ {agent_name}: {agent_info['error']}",
                "timestamp": event.timestamp,
            }
        )

    async def _handle_ensemble_completed(self, event: ExecutionEvent) -> None:
        """Handle ensemble completed event."""
        self.execution_state["status"] = "completed"

        # Complete overall progress
        if self.overall_task_id:
            self.progress.update(
                self.overall_task_id, completed=self.execution_state["total_agents"]
            )

    async def _handle_performance_metric(self, event: ExecutionEvent) -> None:
        """Handle performance metric event."""
        metric_name = event.data.get("metric_name")
        metric_value = event.data.get("metric_value")

        if metric_name == "memory_usage":
            self.execution_state["performance"]["memory_usage"] = metric_value

    def create_layout(self) -> Layout:
        """Create the terminal layout."""
        layout = Layout()

        # Split into sections
        if self.config.terminal.compact_mode:
            layout.split_column(
                Layout(self.create_header(), name="header", size=3),
                Layout(self.create_dependency_graph(), name="graph", size=3),
                Layout(self.create_metrics_section(), name="metrics", size=3),
            )
        else:
            layout.split_column(
                Layout(self.create_header(), name="header", size=3),
                Layout(self.create_dependency_graph(), name="graph", size=3),
                Layout(self.create_progress_section(), name="progress"),
                Layout(self.create_agents_section(), name="agents"),
                Layout(self.create_results_section(), name="results"),
                Layout(self.create_metrics_section(), name="metrics", size=3),
            )

        return layout

    def create_header(self) -> Panel:
        """Create the header section."""
        ensemble_name = self.execution_state["ensemble_name"]
        status = self.execution_state["status"]

        # Status emoji
        status_emoji = {
            "starting": "🚀",
            "running": "⚡",
            "completed": "✅",
            "failed": "❌",
        }.get(status, "🔄")

        # Calculate duration
        duration_text = ""
        if self.execution_state["start_time"]:
            duration = (
                datetime.now() - self.execution_state["start_time"]
            ).total_seconds()
            duration_text = f"({duration:.1f}s)"

        header_text = (
            f"{status_emoji} {ensemble_name} - {status.title()} {duration_text}"
        )

        return Panel(
            Align.center(Text(header_text, style="bold")),
            title="🎭 Ensemble Execution",
            border_style="blue",
        )

    def create_dependency_graph(self) -> Panel:
        """Create a simple horizontal dependency graph visualization."""
        if not self.execution_state["agents"]:
            return Panel(
                "Dependency graph will appear here...", 
                title="🔗 Dependency Flow",
                border_style="yellow"
            )
        
        # Group agents by dependency level
        agents_by_level = self._group_agents_by_level()
        
        if not agents_by_level:
            return Panel(
                "No agents to display", 
                title="🔗 Dependency Flow",
                border_style="yellow"
            )
        
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
                
                # Status indicators
                if status == "running":
                    indicator = "🔄"  # Spinner for active
                elif status == "completed":
                    indicator = "✅"  # Checkmark for done
                elif status == "failed":
                    indicator = "❌"  # X for failed
                else:
                    indicator = "⏳"  # Hourglass for waiting
                
                agent_displays.append(f"{indicator} {name}")
            
            # Join agents at same level with commas
            level_text = ", ".join(agent_displays)
            graph_parts.append(level_text)
        
        # Join levels with arrows
        graph_text = " → ".join(graph_parts)
        
        return Panel(
            Text(graph_text, style="bold"),
            title="🔗 Dependency Flow",
            border_style="yellow"
        )
    
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
    
    def _calculate_agent_level(
        self, 
        agent_name: str, 
        dependencies: list[str]
    ) -> int:
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

    def create_progress_section(self) -> Panel:
        """Create the progress section."""
        if not self.config.terminal.show_progress_bars:
            return Panel("Progress tracking disabled", title="📊 Progress")

        # Overall progress
        completed = self.execution_state["completed_agents"]
        total = self.execution_state["total_agents"]
        failed = self.execution_state["failed_agents"]

        progress_text = f"Overall: {completed}/{total} completed"
        if failed > 0:
            progress_text += f", {failed} failed"

        return Panel(
            self.progress,
            title=f"📊 {progress_text}",
            border_style="green",
        )

    def create_agents_section(self) -> Panel:
        """Create the agents status section."""
        if not self.config.terminal.show_agent_status:
            return Panel("Agent status tracking disabled", title="🤖 Agents")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan", width=20)
        table.add_column("Status", width=12)
        table.add_column("Model", style="yellow", width=15)
        table.add_column("Progress", width=10)
        table.add_column("Duration", width=10)
        table.add_column("Cost", width=8)

        for agent_name, agent_info in self.execution_state["agents"].items():
            # Status with emoji
            status = agent_info["status"]
            status_emoji = {
                "waiting": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(status, "❓")

            # Progress bar
            progress = agent_info["progress"]
            progress_bar = "█" * int(progress / 10) + "░" * (10 - int(progress / 10))
            progress_text = f"{progress_bar} {progress:.0f}%"

            # Duration
            duration = agent_info["duration"]
            duration_text = f"{duration}ms" if duration > 0 else "-"

            # Cost
            cost = agent_info["cost"]
            cost_text = f"${cost:.4f}" if cost > 0 else "-"

            table.add_row(
                agent_name,
                f"{status_emoji} {status}",
                agent_info["model"],
                progress_text,
                duration_text,
                cost_text,
            )

        return Panel(table, title="🤖 Agent Status", border_style="cyan")

    def create_results_section(self) -> Panel:
        """Create the live results section."""
        if not self.config.terminal.show_live_results:
            return Panel("Live results disabled", title="📋 Live Results")

        # Show last 5 results
        recent_results = self.execution_state["live_results"][-5:]

        if not recent_results:
            return Panel("No results yet...", title="📋 Live Results")

        results_text = "\n".join([f"• {result['result']}" for result in recent_results])

        return Panel(results_text, title="📋 Live Results", border_style="yellow")

    def create_metrics_section(self) -> Panel:
        """Create the performance metrics section."""
        if not self.config.terminal.show_performance_metrics:
            return Panel("Performance metrics disabled", title="📈 Performance")

        perf = self.execution_state["performance"]

        # Format metrics
        cost_text = f"${perf['total_cost']:.4f}"
        duration_text = f"{perf['total_duration']:,}ms"

        # Calculate agents per second
        if self.execution_state["start_time"]:
            elapsed = (
                datetime.now() - self.execution_state["start_time"]
            ).total_seconds()
            agents_per_sec = (
                self.execution_state["completed_agents"] / elapsed if elapsed > 0 else 0
            )
        else:
            agents_per_sec = 0

        metrics_text = (
            f"Duration: {duration_text} | "
            f"Cost: {cost_text} | "
            f"Rate: {agents_per_sec:.1f} agents/sec"
        )

        return Panel(
            Align.center(Text(metrics_text)),
            title="📈 Performance Metrics",
            border_style="green",
        )

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

        # Create summary table
        table = Table(title="🎭 Ensemble Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        # Add summary rows
        table.add_row("Ensemble", self.execution_state["ensemble_name"])
        table.add_row("Status", self.execution_state["status"].title())
        table.add_row("Total Agents", str(self.execution_state["total_agents"]))
        table.add_row("Completed", str(self.execution_state["completed_agents"]))
        table.add_row("Failed", str(self.execution_state["failed_agents"]))

        # Performance metrics
        perf = self.execution_state["performance"]
        table.add_row("Total Cost", f"${perf['total_cost']:.4f}")
        table.add_row("Total Duration", f"{perf['total_duration']:,}ms")

        # Duration
        if self.execution_state["start_time"]:
            duration = (
                datetime.now() - self.execution_state["start_time"]
            ).total_seconds()
            table.add_row("Execution Time", f"{duration:.1f}s")

        self.console.print(table)

    def print_simple_progress(self, event: ExecutionEvent) -> None:
        """Print simple progress without Rich layout (fallback)."""
        if event.event_type == ExecutionEventType.ENSEMBLE_STARTED:
            self.console.print(f"🎭 Starting {event.ensemble_name} ensemble...")

        elif event.event_type == ExecutionEventType.AGENT_STARTED:
            self.console.print(f"🤖 Starting {event.agent_name}...")

        elif event.event_type == ExecutionEventType.AGENT_COMPLETED:
            duration = event.data.get("duration_ms", 0)
            self.console.print(f"✅ {event.agent_name} completed in {duration}ms")

        elif event.event_type == ExecutionEventType.AGENT_FAILED:
            error = event.data.get("error", "Unknown error")
            self.console.print(f"❌ {event.agent_name} failed: {error}")

        elif event.event_type == ExecutionEventType.ENSEMBLE_COMPLETED:
            self.console.print(f"✅ {event.ensemble_name} ensemble completed!")
