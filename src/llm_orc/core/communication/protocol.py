"""Agent communication protocol and message handling."""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Message:
    """Represents a message between agents."""

    id: str
    sender: str
    recipient: str
    content: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class ConversationManager:
    """Manages conversation state between agents."""

    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.active_conversations: list[str] = []

    def start_conversation(self, participants: list[str], topic: str) -> str:
        """Start a new conversation between agents."""
        conversation_id = str(uuid.uuid4())

        self.conversations[conversation_id] = {
            "participants": participants,
            "topic": topic,
            "messages": [],
            "created_at": datetime.now(),
        }

        self.active_conversations.append(conversation_id)
        return conversation_id

    def add_message(self, conversation_id: str, message: Message) -> None:
        """Add message to existing conversation."""
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation '{conversation_id}' not found")

        self.conversations[conversation_id]["messages"].append(message)

    def get_conversation_history(self, conversation_id: str) -> list[Message]:
        """Retrieve conversation history."""
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation '{conversation_id}' not found")

        messages = self.conversations[conversation_id]["messages"]
        return [msg for msg in messages if isinstance(msg, Message)]


class MessageProtocol:
    """Handles message protocol and turn-taking coordination."""

    def __init__(self, conversation_manager: ConversationManager, timeout: float = 5.0):
        self.conversation_manager = conversation_manager
        self.pending_messages: list[Message] = []
        self.timeout = timeout
        self.current_speakers: dict[str, str] = {}  # conversation_id -> current_speaker

    async def send_message(
        self, sender: str, recipient: str, content: str, conversation_id: str
    ) -> Message:
        """Send message and handle turn-taking."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            content=content,
            timestamp=datetime.now(),
        )

        # Add to conversation
        self.conversation_manager.add_message(conversation_id, message)

        # Update turn-taking
        self.current_speakers[conversation_id] = recipient

        # Deliver message (async) with timeout
        try:
            await asyncio.wait_for(self.deliver_message(message), timeout=self.timeout)
        except TimeoutError:
            raise TimeoutError(
                f"Message delivery timed out after {self.timeout} seconds"
            ) from None

        return message

    async def deliver_message(self, message: Message) -> None:
        """Deliver message to recipient (placeholder for actual delivery)."""
        # In a real implementation, this would send to the appropriate agent
        # For now, we'll simulate delivery
        await asyncio.sleep(0.001)  # Simulate network delay

    def get_current_speaker(self, conversation_id: str) -> str | None:
        """Get the agent expected to speak next in the conversation."""
        return self.current_speakers.get(conversation_id)
