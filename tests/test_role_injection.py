"""Tests for automatic role injection in OAuth models following TDD approach."""

from unittest.mock import Mock

import pytest

from llm_orc.models import OAuthClaudeModel


class TestRoleInjection:
    """Test automatic role injection for OAuth models."""

    @pytest.fixture
    def mock_oauth_client(self) -> Mock:
        """Create a mock OAuth client."""
        mock_client = Mock()
        mock_client.create_message.return_value = {
            "content": [{"text": "Mock response"}],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }
        return mock_client

    @pytest.fixture
    def oauth_model(self, mock_oauth_client: Mock) -> OAuthClaudeModel:
        """Create OAuth model with mocked client."""
        model = OAuthClaudeModel(
            access_token="test_token", refresh_token="test_refresh"
        )
        model.client = mock_oauth_client
        return model

    @pytest.mark.asyncio
    async def test_role_injection_on_first_call(
        self, oauth_model: OAuthClaudeModel, mock_oauth_client: Mock
    ) -> None:
        """Test that role is automatically injected on first call with a role."""
        # Given
        role_prompt = "You are a financial analyst specializing in startup valuations."
        user_message = "Analyze this startup."

        # When
        await oauth_model.generate_response(user_message, role_prompt)

        # Then
        mock_oauth_client.create_message.assert_called_once()
        call_args = mock_oauth_client.create_message.call_args

        # Check system prompt is unchanged (OAuth requirement)
        assert (
            call_args.kwargs["system"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )

        # Check messages include role establishment
        messages = call_args.kwargs["messages"]
        assert len(messages) >= 3  # Role setup + user message

        # First message should establish role
        assert "act as" in messages[0]["content"].lower()
        assert role_prompt in messages[0]["content"]
        assert messages[0]["role"] == "user"

        # Second message should be assistant acknowledgment
        assert messages[1]["role"] == "assistant"

        # Third message should be the actual user message
        assert messages[2]["content"] == user_message
        assert messages[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_role_not_reinjected_on_subsequent_calls(
        self, oauth_model: OAuthClaudeModel, mock_oauth_client: Mock
    ) -> None:
        """Test that role is not reinjected if already established."""
        # Given
        role_prompt = "You are a financial analyst."

        # When - make two calls with same role
        await oauth_model.generate_response("First message", role_prompt)
        mock_oauth_client.create_message.reset_mock()

        await oauth_model.generate_response("Second message", role_prompt)

        # Then - second call should not include role establishment
        call_args = mock_oauth_client.create_message.call_args
        messages = call_args.kwargs["messages"]

        # Should have: role setup (2 msgs) + first message + response + second message
        assert len(messages) == 5

        # Last message should be the second user message
        assert messages[-1]["content"] == "Second message"
        assert messages[-1]["role"] == "user"

        # No new role establishment in this call
        recent_messages = [msg["content"] for msg in messages[-2:]]
        assert not any("act as" in msg.lower() for msg in recent_messages)

    @pytest.mark.asyncio
    async def test_role_reinjected_when_role_changes(
        self, oauth_model: OAuthClaudeModel, mock_oauth_client: Mock
    ) -> None:
        """Test that role is reinjected when a different role is provided."""
        # Given
        first_role = "You are a financial analyst."
        second_role = "You are a marketing strategist."

        # When - make calls with different roles
        await oauth_model.generate_response("Financial question", first_role)
        await oauth_model.generate_response("Marketing question", second_role)

        # Then
        call_args = mock_oauth_client.create_message.call_args
        messages = call_args.kwargs["messages"]

        # Should contain two role establishments
        role_messages = [
            msg
            for msg in messages
            if msg["role"] == "user" and "act as" in msg["content"].lower()
        ]
        assert len(role_messages) == 2

        # First role establishment
        assert first_role in role_messages[0]["content"]

        # Second role establishment
        assert second_role in role_messages[1]["content"]

    @pytest.mark.asyncio
    async def test_no_role_injection_for_oauth_system_prompt(
        self, oauth_model: OAuthClaudeModel, mock_oauth_client: Mock
    ) -> None:
        """Test no role injection when role_prompt is OAuth system prompt."""
        # Given
        oauth_system_prompt = (
            "You are Claude Code, Anthropic's official CLI for Claude."
        )
        user_message = "Hello"

        # When
        await oauth_model.generate_response(user_message, oauth_system_prompt)

        # Then
        call_args = mock_oauth_client.create_message.call_args
        messages = call_args.kwargs["messages"]

        # Should only have the user message, no role establishment
        assert len(messages) == 1
        assert messages[0]["content"] == user_message
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_role_injection_format(
        self, oauth_model: OAuthClaudeModel, mock_oauth_client: Mock
    ) -> None:
        """Test the specific format of role injection message."""
        # Given
        role_prompt = "You are a data scientist specializing in machine learning."

        # When
        await oauth_model.generate_response("Test message", role_prompt)

        # Then
        call_args = mock_oauth_client.create_message.call_args
        messages = call_args.kwargs["messages"]

        role_message = messages[0]["content"]

        # Check format: "For this conversation, please act as: [role]"
        assert role_message.startswith("For this conversation, please act as:")
        assert role_prompt in role_message

        # Check assistant acknowledgment is reasonable
        assistant_response = messages[1]["content"]
        assert len(assistant_response) > 0  # Should have some acknowledgment

    @pytest.mark.asyncio
    async def test_conversation_history_includes_role_establishment(
        self, oauth_model: OAuthClaudeModel
    ) -> None:
        """Test that conversation history includes role establishment messages."""
        # Given
        role_prompt = "You are a legal advisor."

        # When
        await oauth_model.generate_response("Legal question", role_prompt)

        # Then
        history = oauth_model.get_conversation_history()

        # Should include role establishment and user message
        assert len(history) >= 3

        # First message should be role establishment
        assert "act as" in history[0]["content"].lower()
        assert role_prompt in history[0]["content"]

        # Check roles are correct
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
