"""Tests for base Event Pydantic models using TDD methodology."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from llm_orc.core.events.base import Event


class TestEvent:
    """Test suite for base Event model."""

    def test_event_creation_with_required_fields(self) -> None:
        """Test that Event can be created with all required fields."""
        # Given
        event_type = "test_event"
        agent_name = "test_agent"

        # When
        event = Event(event_type=event_type, agent_name=agent_name)

        # Then
        assert event.event_type == event_type
        assert event.agent_name == agent_name
        assert isinstance(event.timestamp, datetime)

    def test_event_timestamp_defaults_to_now(self) -> None:
        """Test that Event timestamp defaults to current time."""
        # Given
        before = datetime.now()

        # When
        event = Event(event_type="test", agent_name="agent")

        # Then
        after = datetime.now()
        assert before <= event.timestamp <= after

    def test_event_to_dict_serialization(self) -> None:
        """Test that Event can be serialized to dict."""
        # Given
        event = Event(event_type="test_event", agent_name="test_agent")

        # When
        result = event.to_dict()

        # Then
        assert isinstance(result, dict)
        assert result["event_type"] == "test_event"
        assert result["agent_name"] == "test_agent"
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)  # Should be ISO string

    def test_event_validation_requires_event_type(self) -> None:
        """Test that Event validation fails without event_type."""
        # When/Then
        with pytest.raises(ValidationError):
            Event(agent_name="test_agent")  # type: ignore[call-arg]

    def test_event_validation_requires_agent_name(self) -> None:
        """Test that Event validation fails without agent_name."""
        # When/Then
        with pytest.raises(ValidationError):
            Event(event_type="test_event")  # type: ignore[call-arg]
