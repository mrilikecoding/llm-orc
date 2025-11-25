"""Base Event classes with Pydantic validation."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base class for all events in llm-orc system."""

    event_type: str
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
        }
