"""Agent execution coordination with timeout and concurrency management."""

import asyncio
from collections.abc import Callable
from typing import Any

from llm_orc.models.base import ModelInterface


class AgentExecutionCoordinator:
    """Coordinates agent execution with timeout and concurrency control."""

    def __init__(
        self,
        performance_config: dict[str, Any],
        agent_executor: Callable[
            [dict[str, Any], str], tuple[str, ModelInterface | None]
        ],
    ) -> None:
        """Initialize coordinator with performance config and agent executor."""
        self._performance_config = performance_config
        self._execute_agent = agent_executor

    async def execute_agent_with_timeout(
        self,
        agent_config: dict[str, Any],
        input_data: str,
        timeout_seconds: int | None,
    ) -> tuple[str, ModelInterface | None]:
        """Execute an agent with optional timeout."""
        if timeout_seconds is None:
            # No timeout specified, execute normally
            return await self._execute_agent(agent_config, input_data)

        try:
            return await asyncio.wait_for(
                self._execute_agent(agent_config, input_data), timeout=timeout_seconds
            )
        except TimeoutError as e:
            raise Exception(
                f"Agent execution timed out after {timeout_seconds} seconds"
            ) from e

    def get_effective_concurrency_limit(self, agent_count: int) -> int:
        """Get effective concurrency limit based on configuration and agent count."""
        # Check performance configuration first
        configured_limit = self._performance_config.get("concurrency", {}).get(
            "max_concurrent_agents", 0
        )

        # If explicitly configured and > 0, use it
        if isinstance(configured_limit, int) and configured_limit > 0:
            return configured_limit

        # Otherwise use smart defaults based on agent count and system resources
        if agent_count <= 3:
            return agent_count  # Small ensembles: run all in parallel
        elif agent_count <= 10:
            return 5  # Medium ensembles: limit to 5 concurrent
        elif agent_count <= 20:
            return 8  # Large ensembles: limit to 8 concurrent
        else:
            return 10  # Very large ensembles: cap at 10 concurrent

    def get_agent_timeout(
        self, agent_config: dict[str, Any], enhanced_config: dict[str, Any]
    ) -> int:
        """Get timeout for an agent based on config hierarchy."""
        # Priority: enhanced_config -> performance_config default
        timeout = enhanced_config.get("timeout_seconds")
        if timeout is not None:
            return timeout

        return self._performance_config.get("execution", {}).get("default_timeout", 60)

    def should_use_concurrency_limit(
        self, agent_count: int, max_concurrent: int
    ) -> bool:
        """Determine if concurrency limiting should be applied."""
        return agent_count > max_concurrent

    async def execute_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        agent_config: dict[str, Any],
        input_data: str,
        timeout_seconds: int | None,
    ) -> tuple[str, ModelInterface | None]:
        """Execute agent with semaphore-based concurrency control."""
        async with semaphore:
            return await self.execute_agent_with_timeout(
                agent_config, input_data, timeout_seconds
            )

    def create_semaphore(self, max_concurrent: int) -> asyncio.Semaphore:
        """Create a semaphore for concurrency control."""
        return asyncio.Semaphore(max_concurrent)

    def get_timeout_strategy(self, agent_config: dict[str, Any]) -> dict[str, Any]:
        """Get timeout strategy information for an agent."""
        timeout_info = {
            "has_custom_timeout": "timeout_seconds" in agent_config,
            "timeout_source": (
                "custom" if "timeout_seconds" in agent_config else "default"
            ),
            "default_timeout": self._performance_config.get("execution", {}).get(
                "default_timeout", 60
            ),
        }

        if timeout_info["has_custom_timeout"]:
            timeout_info["custom_timeout"] = agent_config["timeout_seconds"]

        return timeout_info

    def validate_timeout_config(self, timeout_seconds: int | None) -> bool:
        """Validate timeout configuration."""
        if timeout_seconds is None:
            return True
        return isinstance(timeout_seconds, int) and timeout_seconds > 0

    def get_concurrency_strategy(self, agent_count: int) -> dict[str, Any]:
        """Get concurrency strategy information for the given agent count."""
        effective_limit = self.get_effective_concurrency_limit(agent_count)

        strategy = {
            "agent_count": agent_count,
            "effective_concurrency_limit": effective_limit,
            "uses_semaphore": self.should_use_concurrency_limit(
                agent_count, effective_limit
            ),
            "concurrency_source": "configured"
            if self._performance_config.get("concurrency", {}).get(
                "max_concurrent_agents", 0
            ) > 0
            else "calculated",
        }

        if strategy["concurrency_source"] == "configured":
            strategy["configured_limit"] = self._performance_config["concurrency"][
                "max_concurrent_agents"
            ]

        return strategy

    async def execute_multiple_with_coordination(
        self,
        agents: list[dict[str, Any]],
        input_data_func: Callable[[dict[str, Any]], str],
        timeout_func: Callable[[dict[str, Any]], int],
    ) -> list[tuple[str, Any]]:
        """Execute multiple agents with coordinated timeout and concurrency."""
        if not agents:
            return []

        # Determine concurrency strategy
        max_concurrent = self.get_effective_concurrency_limit(len(agents))
        use_semaphore = self.should_use_concurrency_limit(len(agents), max_concurrent)

        if use_semaphore:
            return await self._execute_with_semaphore_coordination(
                agents, input_data_func, timeout_func, max_concurrent
            )
        else:
            return await self._execute_unlimited_coordination(
                agents, input_data_func, timeout_func
            )

    async def _execute_with_semaphore_coordination(
        self,
        agents: list[dict[str, Any]],
        input_data_func: Callable[[dict[str, Any]], str],
        timeout_func: Callable[[dict[str, Any]], int],
        max_concurrent: int,
    ) -> list[tuple[str, Any]]:
        """Execute agents with semaphore coordination."""
        semaphore = self.create_semaphore(max_concurrent)

        async def execute_with_coordination(
            agent_config: dict[str, Any]
        ) -> tuple[str, Any]:
            input_data = input_data_func(agent_config)
            timeout = timeout_func(agent_config)
            return await self.execute_with_semaphore(
                semaphore, agent_config, input_data, timeout
            )

        tasks = [
            asyncio.create_task(execute_with_coordination(agent))
            for agent in agents
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_unlimited_coordination(
        self,
        agents: list[dict[str, Any]],
        input_data_func: Callable[[dict[str, Any]], str],
        timeout_func: Callable[[dict[str, Any]], int],
    ) -> list[tuple[str, Any]]:
        """Execute agents without concurrency limits."""

        async def execute_with_coordination(
            agent_config: dict[str, Any]
        ) -> tuple[str, Any]:
            input_data = input_data_func(agent_config)
            timeout = timeout_func(agent_config)
            return await self.execute_agent_with_timeout(
                agent_config, input_data, timeout
            )

        tasks = [
            asyncio.create_task(execute_with_coordination(agent))
            for agent in agents
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_execution_plan(
        self, agents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Get execution plan with timeout and concurrency details."""
        concurrency_info = self.get_concurrency_strategy(len(agents))

        agent_timeouts = []
        for agent in agents:
            timeout_info = self.get_timeout_strategy(agent)
            agent_timeouts.append({
                "agent_name": agent["name"],
                "timeout_info": timeout_info,
            })

        return {
            "total_agents": len(agents),
            "concurrency": concurrency_info,
            "agent_timeouts": agent_timeouts,
            "execution_mode": (
                "semaphore" if concurrency_info["uses_semaphore"] else "unlimited"
            ),
        }
