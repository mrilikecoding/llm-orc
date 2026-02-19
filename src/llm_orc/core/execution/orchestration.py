"""Agent orchestration for multi-agent conversations."""

from datetime import datetime
from typing import Any

from llm_orc.core.config.roles import RoleDefinition
from llm_orc.models.base import ModelInterface


class Agent:
    """Agent that combines role, model, and conversation capabilities."""

    def __init__(self, name: str, role: RoleDefinition, model: ModelInterface):
        self.name = name
        self.role = role
        self.model = model
        self.conversation_history: list[dict[str, Any]] = []

    async def respond_to_message(self, message: str) -> str:
        """Generate response to a message using the agent's role and model."""
        # Generate response using model
        response = await self.model.generate_response(
            message=message, role_prompt=self.role.prompt
        )

        # Store in conversation history
        self.conversation_history.append(
            {"message": message, "response": response, "timestamp": datetime.now()}
        )

        return response
