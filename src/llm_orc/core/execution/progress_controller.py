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


class NoOpProgressController(ProgressController):
    """No-op implementation for when no progress display is available."""

    def pause_for_user_input(self, agent_name: str, prompt: str = "") -> None:
        """No-op pause."""
        pass

    def resume_from_user_input(self, agent_name: str) -> None:
        """No-op resume."""
        pass
