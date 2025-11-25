"""Script interaction events with Pydantic validation."""

from typing import Any

from llm_orc.core.events.base import Event


class UserInputRequiredEvent(Event):
    """Event emitted when a script agent requires user input."""

    prompt: str
    script_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = super().to_dict()
        result.update(
            {
                "prompt": self.prompt,
                "script_path": self.script_path,
            }
        )
        return result
