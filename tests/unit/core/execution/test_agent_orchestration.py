"""Test suite for agent orchestration."""

from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.config.roles import RoleDefinition
from llm_orc.models.base import ModelInterface


class TestAgent:
    """Test the Agent class."""

    def test_agent_creation(self) -> None:
        """Should create an agent with name, role, and model."""
        # Arrange
        role = RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare, the renowned playwright.",
        )
        model = Mock(spec=ModelInterface)
        model.name = "test-model"

        # Act
        from llm_orc.core.execution.orchestration import Agent

        agent = Agent(name="shakespeare", role=role, model=model)

        # Assert
        assert agent.name == "shakespeare"
        assert agent.role == role
        assert agent.model == model
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_agent_respond_to_message(self) -> None:
        """Should generate response using role and model."""
        # Arrange
        role = RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare, the renowned playwright.",
        )
        model = AsyncMock(spec=ModelInterface)
        model.generate_response.return_value = (
            "Hark! What light through yonder window breaks?"
        )

        from llm_orc.core.execution.orchestration import Agent

        agent = Agent(name="shakespeare", role=role, model=model)

        # Act - This will fail because respond_to_message doesn't exist yet
        response = await agent.respond_to_message("Tell me about beauty.")

        # Assert
        assert response == "Hark! What light through yonder window breaks?"
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["message"] == "Tell me about beauty."
        assert agent.conversation_history[0]["response"] == response

        # Verify model was called with correct parameters
        model.generate_response.assert_called_once_with(
            message="Tell me about beauty.", role_prompt=role.prompt
        )
