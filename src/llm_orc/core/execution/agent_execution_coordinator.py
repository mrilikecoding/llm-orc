"""Agent execution coordination with timeout management."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import AgentConfig


class AgentExecutionCoordinator:
    """Coordinates agent execution with timeout control."""

    def __init__(
        self,
        performance_config: dict[str, Any],
        agent_executor: Callable[
            [AgentConfig, str], Awaitable[tuple[str, ModelInterface | None]]
        ],
    ) -> None:
        """Initialize coordinator with performance config and agent executor."""
        self._performance_config = performance_config
        self._execute_agent = agent_executor

    async def execute_agent_with_timeout(
        self,
        agent_config: AgentConfig,
        input_data: str,
        timeout_seconds: int | None,
    ) -> tuple[str, ModelInterface | None]:
        """Execute an agent with optional timeout."""
        if timeout_seconds is None:
            return await self._execute_agent(agent_config, input_data)

        try:
            return await asyncio.wait_for(
                self._execute_agent(agent_config, input_data),
                timeout=timeout_seconds,
            )
        except TimeoutError as e:
            raise Exception(
                f"Agent execution timed out after {timeout_seconds} seconds"
            ) from e
