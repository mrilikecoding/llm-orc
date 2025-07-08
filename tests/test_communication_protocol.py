"""Test suite for agent communication protocol."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from llm_orc.communication import ConversationManager, Message, MessageProtocol


class TestMessage:
    """Test the Message data structure."""

    def test_message_creation(self) -> None:
        """Should create a message with required fields."""
        message = Message(
            id="msg-123",
            sender="agent-1",
            recipient="agent-2",
            content="Hello there!",
            timestamp=datetime.now(),
        )

        assert message.id == "msg-123"
        assert message.sender == "agent-1"
        assert message.recipient == "agent-2"
        assert message.content == "Hello there!"
        assert isinstance(message.timestamp, datetime)

    def test_message_to_dict(self) -> None:
        """Should convert message to dictionary."""
        timestamp = datetime.now()
        message = Message(
            id="msg-123",
            sender="agent-1",
            recipient="agent-2",
            content="Hello there!",
            timestamp=timestamp,
        )

        result = message.to_dict()

        assert result["id"] == "msg-123"
        assert result["sender"] == "agent-1"
        assert result["recipient"] == "agent-2"
        assert result["content"] == "Hello there!"
        assert result["timestamp"] == timestamp.isoformat()

    def test_message_from_dict(self) -> None:
        """Should create message from dictionary."""
        timestamp = datetime.now()
        data = {
            "id": "msg-123",
            "sender": "agent-1",
            "recipient": "agent-2",
            "content": "Hello there!",
            "timestamp": timestamp.isoformat(),
        }

        message = Message.from_dict(data)

        assert message.id == "msg-123"
        assert message.sender == "agent-1"
        assert message.recipient == "agent-2"
        assert message.content == "Hello there!"
        assert message.timestamp == timestamp


class TestConversationManager:
    """Test conversation state management."""

    def test_conversation_manager_initialization(self) -> None:
        """Should initialize with empty conversation state."""
        manager = ConversationManager()

        assert len(manager.conversations) == 0
        assert len(manager.active_conversations) == 0

    def test_start_conversation(self) -> None:
        """Should start a new conversation between agents."""
        manager = ConversationManager()

        conversation_id = manager.start_conversation(
            participants=["agent-1", "agent-2"], topic="Test discussion"
        )

        assert conversation_id in manager.conversations
        assert conversation_id in manager.active_conversations
        conversation = manager.conversations[conversation_id]
        assert conversation["participants"] == ["agent-1", "agent-2"]
        assert conversation["topic"] == "Test discussion"
        assert len(conversation["messages"]) == 0

    def test_add_message_to_conversation(self) -> None:
        """Should add message to existing conversation."""
        manager = ConversationManager()
        conversation_id = manager.start_conversation(
            participants=["agent-1", "agent-2"], topic="Test discussion"
        )

        message = Message(
            id="msg-123",
            sender="agent-1",
            recipient="agent-2",
            content="Hello there!",
            timestamp=datetime.now(),
        )

        manager.add_message(conversation_id, message)

        conversation = manager.conversations[conversation_id]
        assert len(conversation["messages"]) == 1
        assert conversation["messages"][0] == message

    def test_get_conversation_history(self) -> None:
        """Should retrieve conversation history."""
        manager = ConversationManager()
        conversation_id = manager.start_conversation(
            participants=["agent-1", "agent-2"], topic="Test discussion"
        )

        message1 = Message(
            id="msg-1",
            sender="agent-1",
            recipient="agent-2",
            content="Hello!",
            timestamp=datetime.now(),
        )
        message2 = Message(
            id="msg-2",
            sender="agent-2",
            recipient="agent-1",
            content="Hi there!",
            timestamp=datetime.now(),
        )

        manager.add_message(conversation_id, message1)
        manager.add_message(conversation_id, message2)

        history = manager.get_conversation_history(conversation_id)

        assert len(history) == 2
        assert history[0] == message1
        assert history[1] == message2


class TestMessageProtocol:
    """Test message protocol and turn-taking coordination."""

    def test_message_protocol_initialization(self) -> None:
        """Should initialize with conversation manager."""
        manager = ConversationManager()
        protocol = MessageProtocol(manager)

        assert protocol.conversation_manager == manager
        assert len(protocol.pending_messages) == 0

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        """Should send message and handle turn-taking."""
        manager = ConversationManager()
        protocol = MessageProtocol(manager)

        # Create conversation first
        conversation_id = manager.start_conversation(
            participants=["agent-1", "agent-2"], topic="Test discussion"
        )

        # Mock the message delivery
        from unittest.mock import patch

        with patch.object(
            protocol, "deliver_message", new_callable=AsyncMock
        ) as mock_deliver:
            result = await protocol.send_message(
                sender="agent-1",
                recipient="agent-2",
                content="Hello there!",
                conversation_id=conversation_id,
            )

            assert result is not None
            assert result.sender == "agent-1"
            assert result.recipient == "agent-2"
            assert result.content == "Hello there!"
            mock_deliver.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_turn_taking(self) -> None:
        """Should coordinate turn-taking between agents."""
        manager = ConversationManager()
        protocol = MessageProtocol(manager)

        # Start a conversation
        conversation_id = manager.start_conversation(
            participants=["agent-1", "agent-2"], topic="Test discussion"
        )

        # Agent 1 sends first message
        await protocol.send_message(
            sender="agent-1",
            recipient="agent-2",
            content="Hello!",
            conversation_id=conversation_id,
        )

        # Check that agent-2 is now expected to respond
        assert protocol.get_current_speaker(conversation_id) == "agent-2"

        # Agent 2 responds
        await protocol.send_message(
            sender="agent-2",
            recipient="agent-1",
            content="Hi there!",
            conversation_id=conversation_id,
        )

        # Check that turn has switched back to agent-1
        assert protocol.get_current_speaker(conversation_id) == "agent-1"

    @pytest.mark.asyncio
    async def test_message_timeout_handling(self) -> None:
        """Should handle message timeouts gracefully."""
        manager = ConversationManager()
        protocol = MessageProtocol(manager, timeout=0.1)  # 100ms timeout

        # Create conversation first
        conversation_id = manager.start_conversation(
            participants=["agent-1", "agent-2"], topic="Test discussion"
        )

        # Mock a slow message delivery that exceeds timeout
        from typing import Any

        async def slow_delivery(msg: Any) -> None:
            await asyncio.sleep(0.2)  # Longer than 0.1s timeout

        from unittest.mock import patch

        with patch.object(
            protocol,
            "deliver_message",
            new_callable=AsyncMock,
            side_effect=slow_delivery,
        ):
            with pytest.raises(TimeoutError):
                await protocol.send_message(
                    sender="agent-1",
                    recipient="agent-2",
                    content="Hello there!",
                    conversation_id=conversation_id,
                )
