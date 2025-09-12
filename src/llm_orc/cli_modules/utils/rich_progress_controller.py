"""Rich console progress controller for synchronous user input handling."""

from typing import Any

from rich.console import Console

from llm_orc.cli_modules.utils.visualization import create_dependency_tree
from llm_orc.core.execution.progress_controller import ProgressController


class RichProgressController(ProgressController):
    """Progress controller that directly controls Rich console status display."""

    def __init__(
        self,
        console: Console,
        status: Any,
        agent_statuses: dict[str, str],
        ensemble_agents: list[dict[str, Any]],
    ):
        """Initialize with Rich console components.

        Args:
            console: Rich console instance
            status: Rich status display object
            agent_statuses: Dictionary tracking agent statuses
            ensemble_agents: List of ensemble agent configurations
        """
        self._console = console
        self._status = status
        self._agent_statuses = agent_statuses
        self._ensemble_agents = ensemble_agents

    def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
        """Pause the progress display to allow clean user input."""

        # Update agent status to waiting_input
        self._agent_statuses[agent_name] = "waiting_input"

        # Stop the spinner
        self._status.stop()

        # Clear the terminal for clean user input
        self._console.clear()

        # Show a clean status tree indicating the agent is waiting for input
        current_tree = create_dependency_tree(
            self._ensemble_agents, self._agent_statuses
        )
        self._console.print(current_tree)

        # Display the prompt if provided, otherwise show generic message
        if prompt:
            self._console.print(f"[blue]⏸  {agent_name} is waiting for input...[/blue]")
            self._console.print(f"[yellow]{prompt}[/yellow]", end=" ")
        else:
            self._console.print(
                f"[blue]⏸  {agent_name} is waiting for user input...[/blue]"
            )

    def resume_from_user_input(self, agent_name: str) -> None:
        """Resume the progress display after user input is complete."""
        # Update agent status back to running
        self._agent_statuses[agent_name] = "running"

        # Update the tree display and restart the status
        current_tree = create_dependency_tree(
            self._ensemble_agents, self._agent_statuses
        )
        self._status.update(current_tree)
        self._status.start()
