"""Tests for script interaction events using TDD methodology."""

import pytest
from pydantic import ValidationError

from llm_orc.core.events.base import Event
from llm_orc.core.events.script_interaction import UserInputRequiredEvent


class TestUserInputRequiredEvent:
    """Test suite for UserInputRequiredEvent model."""

    def test_user_input_required_event_creation_with_required_fields(self) -> None:
        """Test that UserInputRequiredEvent can be created with all required fields."""
        # Given
        event_type = "user_input_required"
        agent_name = "script_agent"
        prompt = "Enter your name:"
        script_path = "/path/to/script.py"

        # When
        event = UserInputRequiredEvent(
            event_type=event_type,
            agent_name=agent_name,
            prompt=prompt,
            script_path=script_path,
        )

        # Then
        assert event.event_type == event_type
        assert event.agent_name == agent_name
        assert event.prompt == prompt
        assert event.script_path == script_path

    def test_user_input_required_event_inherits_from_base_event(self) -> None:
        """Test that UserInputRequiredEvent inherits from base Event."""
        # When/Then
        assert issubclass(UserInputRequiredEvent, Event)

    def test_user_input_required_event_to_dict_includes_prompt_and_script_path(
        self,
    ) -> None:
        """Test that UserInputRequiredEvent to_dict includes prompt and script_path."""
        # Given
        event = UserInputRequiredEvent(
            event_type="user_input_required",
            agent_name="script_agent",
            prompt="Enter value:",
            script_path="/test/script.py",
        )

        # When
        result = event.to_dict()

        # Then
        assert result["prompt"] == "Enter value:"
        assert result["script_path"] == "/test/script.py"

    def test_user_input_required_event_validation_requires_prompt(self) -> None:
        """Test that UserInputRequiredEvent validation fails without prompt."""
        # When/Then
        with pytest.raises(ValidationError):
            UserInputRequiredEvent(  # type: ignore[call-arg]
                event_type="user_input_required",
                agent_name="script_agent",
                script_path="/test/script.py",
            )

    def test_user_input_required_event_validation_requires_script_path(self) -> None:
        """Test that UserInputRequiredEvent validation fails without script_path."""
        # When/Then
        with pytest.raises(ValidationError):
            UserInputRequiredEvent(  # type: ignore[call-arg]
                event_type="user_input_required",
                agent_name="script_agent",
                prompt="Enter value:",
            )
