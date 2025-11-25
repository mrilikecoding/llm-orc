"""Unit tests for base event classes."""

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from llm_orc.core.events.base import Event


class TestEvent:
    """Test suite for Event base class."""

    def test_event_creation_with_required_fields(self) -> None:
        """Test creating an event with required fields."""
        event = Event(event_type="test_event", agent_name="test_agent")

        assert event.event_type == "test_event"
        assert event.agent_name == "test_agent"
        assert isinstance(event.timestamp, datetime)

    def test_event_creation_with_custom_timestamp(self) -> None:
        """Test creating an event with custom timestamp."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        event = Event(
            event_type="custom_event", agent_name="custom_agent", timestamp=custom_time
        )

        assert event.event_type == "custom_event"
        assert event.agent_name == "custom_agent"
        assert event.timestamp == custom_time

    def test_event_timestamp_auto_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before_creation = datetime.now()
        event = Event(event_type="test", agent_name="agent")
        after_creation = datetime.now()

        assert before_creation <= event.timestamp <= after_creation

    def test_event_validation_missing_event_type(self) -> None:
        """Test validation fails when event_type is missing."""
        with pytest.raises(ValidationError) as exc_info:
            Event(agent_name="test_agent")  # type: ignore

        assert "event_type" in str(exc_info.value)

    def test_event_validation_missing_agent_name(self) -> None:
        """Test validation fails when agent_name is missing."""
        with pytest.raises(ValidationError) as exc_info:
            Event(event_type="test_event")  # type: ignore

        assert "agent_name" in str(exc_info.value)

    def test_event_validation_invalid_types(self) -> None:
        """Test validation fails with invalid field types."""
        # Invalid event_type
        with pytest.raises(ValidationError):
            Event(event_type=123, agent_name="agent")  # type: ignore

        # Invalid agent_name
        with pytest.raises(ValidationError):
            Event(event_type="event", agent_name=456)  # type: ignore

        # Invalid timestamp
        with pytest.raises(ValidationError):
            Event(
                event_type="event",
                agent_name="agent",
                timestamp="invalid",  # type: ignore
            )

    def test_event_to_dict(self) -> None:
        """Test converting event to dictionary."""
        timestamp = datetime(2023, 6, 15, 14, 30, 45)
        event = Event(
            event_type="conversion_test",
            agent_name="converter_agent",
            timestamp=timestamp,
        )

        result = event.to_dict()

        expected = {
            "event_type": "conversion_test",
            "agent_name": "converter_agent",
            "timestamp": "2023-06-15T14:30:45",
        }

        assert result == expected

    def test_event_to_dict_with_auto_timestamp(self) -> None:
        """Test converting event with auto-generated timestamp to dictionary."""
        event = Event(event_type="auto_time", agent_name="time_agent")

        result = event.to_dict()

        assert result["event_type"] == "auto_time"
        assert result["agent_name"] == "time_agent"
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)
        # Should be valid ISO format
        datetime.fromisoformat(result["timestamp"])

    def test_event_serialization_consistency(self) -> None:
        """Test that serialization is consistent."""
        event = Event(event_type="serial_test", agent_name="serial_agent")

        dict1 = event.to_dict()
        dict2 = event.to_dict()

        assert dict1 == dict2

    def test_event_with_empty_strings(self) -> None:
        """Test event with empty string values."""
        event = Event(event_type="", agent_name="")

        assert event.event_type == ""
        assert event.agent_name == ""

        result = event.to_dict()
        assert result["event_type"] == ""
        assert result["agent_name"] == ""

    def test_event_with_special_characters(self) -> None:
        """Test event with special characters in fields."""
        event = Event(
            event_type="test/event:123",
            agent_name="agent@domain.com",
        )

        assert event.event_type == "test/event:123"
        assert event.agent_name == "agent@domain.com"

        result = event.to_dict()
        assert result["event_type"] == "test/event:123"
        assert result["agent_name"] == "agent@domain.com"

    def test_event_pydantic_features(self) -> None:
        """Test that Event works with Pydantic features."""
        # Test model validation
        data = {
            "event_type": "pydantic_test",
            "agent_name": "pydantic_agent",
            "timestamp": "2023-07-20T10:30:00",
        }

        event = Event.model_validate(data)
        assert event.event_type == "pydantic_test"
        assert event.agent_name == "pydantic_agent"
        assert event.timestamp == datetime(2023, 7, 20, 10, 30, 0)

    def test_event_model_dump(self) -> None:
        """Test Pydantic model_dump functionality."""
        timestamp = datetime(2023, 8, 10, 16, 45, 30)
        event = Event(
            event_type="dump_test",
            agent_name="dump_agent",
            timestamp=timestamp,
        )

        # Test default dump
        dumped = event.model_dump()
        assert dumped["event_type"] == "dump_test"
        assert dumped["agent_name"] == "dump_agent"
        assert dumped["timestamp"] == timestamp

    def test_event_json_serialization(self) -> None:
        """Test JSON serialization capabilities."""
        timestamp = datetime(2023, 9, 5, 9, 15, 45)
        event = Event(
            event_type="json_test",
            agent_name="json_agent",
            timestamp=timestamp,
        )

        # Test model_dump_json
        json_str = event.model_dump_json()
        assert '"event_type":"json_test"' in json_str
        assert '"agent_name":"json_agent"' in json_str

        # Test round-trip
        reconstructed = Event.model_validate_json(json_str)
        assert reconstructed.event_type == event.event_type
        assert reconstructed.agent_name == event.agent_name
        assert reconstructed.timestamp == event.timestamp

    def test_event_equality(self) -> None:
        """Test event equality comparison."""
        timestamp = datetime(2023, 10, 12, 13, 20, 15)

        event1 = Event(
            event_type="equality_test",
            agent_name="equal_agent",
            timestamp=timestamp,
        )

        event2 = Event(
            event_type="equality_test",
            agent_name="equal_agent",
            timestamp=timestamp,
        )

        event3 = Event(
            event_type="different_test",
            agent_name="equal_agent",
            timestamp=timestamp,
        )

        assert event1 == event2
        assert event1 != event3

    def test_event_inheritance_compatibility(self) -> None:
        """Test that Event can be inherited from."""

        class CustomEvent(Event):
            custom_field: str

            def to_dict(self) -> dict[str, Any]:
                base_dict = super().to_dict()
                base_dict["custom_field"] = self.custom_field
                return base_dict

        custom_event = CustomEvent(
            event_type="custom",
            agent_name="custom_agent",
            custom_field="custom_value",
        )

        assert custom_event.event_type == "custom"
        assert custom_event.agent_name == "custom_agent"
        assert custom_event.custom_field == "custom_value"

        result = custom_event.to_dict()
        assert result["custom_field"] == "custom_value"

    def test_event_field_access(self) -> None:
        """Test accessing event fields."""
        event = Event(event_type="access_test", agent_name="access_agent")

        # Test field access
        assert hasattr(event, "event_type")
        assert hasattr(event, "agent_name")
        assert hasattr(event, "timestamp")

        # Test field modification
        event.event_type = "modified_event"
        assert event.event_type == "modified_event"

    def test_event_timestamp_precision(self) -> None:
        """Test timestamp precision in serialization."""
        # Create event with microseconds
        timestamp = datetime(2023, 11, 8, 14, 25, 33, 123456)
        event = Event(
            event_type="precision_test",
            agent_name="precision_agent",
            timestamp=timestamp,
        )

        result = event.to_dict()
        # ISO format should include microseconds
        assert result["timestamp"] == "2023-11-08T14:25:33.123456"
