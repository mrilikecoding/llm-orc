"""Progress display controller interface for synchronous user input handling."""

from abc import ABC, abstractmethod


class ProgressController(ABC):
    """Abstract interface for controlling progress display during user input."""

    @abstractmethod
    def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
        """Pause the progress display to allow clean user input.

        Args:
            agent_name: Name of the agent requesting user input
            prompt: The prompt text to display to the user
        """
        pass

    @abstractmethod
    def resume_from_user_input(self, agent_name: str) -> None:
        """Resume the progress display after user input is complete.

        Args:
            agent_name: Name of the agent that completed user input
        """
        pass

    @abstractmethod
    async def start_ensemble(self, ensemble_name: str) -> None:
        """Start tracking ensemble execution progress.

        Args:
            ensemble_name: Name of the ensemble being executed
        """
        pass

    @abstractmethod
    async def update_agent_progress(self, agent_name: str, status: str) -> None:
        """Update progress for a specific agent.

        Args:
            agent_name: Name of the agent
            status: Current status of the agent execution
        """
        pass

    @abstractmethod
    async def complete_ensemble(self) -> None:
        """Complete ensemble execution tracking."""
        pass


class NoOpProgressController(ProgressController):
    """No-op implementation for when no progress display is available."""

    def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
        """No-op pause."""
        pass

    def resume_from_user_input(self, agent_name: str) -> None:
        """No-op resume."""
        pass

    async def start_ensemble(self, ensemble_name: str) -> None:
        """No-op ensemble start."""
        pass

    async def update_agent_progress(self, agent_name: str, status: str) -> None:
        """No-op agent progress update."""
        pass

    async def complete_ensemble(self) -> None:
        """No-op ensemble completion."""
        pass
