"""Conversational dependency resolver with condition support (ADR-005)."""

from collections.abc import Callable

from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationalDependency,
    ConversationState,
)


class ConversationalDependencyResolver(DependencyResolver):
    """Extends DependencyResolver with conversational condition support."""

    def __init__(self, role_resolver: Callable[[str], str | None]) -> None:
        """Initialize resolver with role description function."""
        super().__init__(role_resolver)

    def get_ready_agents_for_conversation(
        self,
        agents: list[ConversationalAgent],
        conversation_state: ConversationState,
    ) -> list[ConversationalAgent]:
        """Get agents ready to execute based on dependency conditions and state.

        Args:
            agents: List of conversational agents
            conversation_state: Current conversation state

        Returns:
            List of agents ready to execute
        """
        ready_agents = []

        for agent in agents:
            # Check if agent can execute more times
            if not self._can_agent_execute_more_times(agent, conversation_state):
                continue

            # Check if all dependencies are satisfied
            if self._has_dependencies_satisfied_for_agent(agent, conversation_state):
                ready_agents.append(agent)

        return ready_agents

    def _has_dependencies_satisfied_for_agent(
        self,
        agent: ConversationalAgent,
        conversation_state: ConversationState,
    ) -> bool:
        """Check if all dependencies for an agent are satisfied.

        Args:
            agent: The agent to check
            conversation_state: Current conversation state

        Returns:
            True if all dependencies are satisfied
        """
        if not agent.dependencies:
            return True

        satisfied_count = 0
        total_dependencies = len(agent.dependencies)

        for dependency in agent.dependencies:
            if self._is_dependency_satisfied(dependency, conversation_state):
                satisfied_count += 1

        # Check requires_all logic - use the first dependency's setting
        # In practice, all dependencies should have the same requires_all value
        requires_all = agent.dependencies[0].requires_all

        if requires_all:
            # AND logic: all dependencies must be satisfied
            return satisfied_count == total_dependencies
        else:
            # OR logic: at least one dependency must be satisfied
            return satisfied_count > 0

    def _is_dependency_satisfied(
        self,
        dependency: ConversationalDependency,
        conversation_state: ConversationState,
    ) -> bool:
        """Check if a single dependency is satisfied.

        Args:
            dependency: The dependency to check
            conversation_state: Current conversation state

        Returns:
            True if dependency is satisfied
        """
        # Check if required agent has executed at least once
        agent_executed = (
            conversation_state.agent_execution_count.get(dependency.agent_name, 0) > 0
        )

        if not agent_executed:
            return False

        # Evaluate condition if present (agent must have executed AND condition met)
        if dependency.condition:
            return self._evaluate_dependency_condition(
                dependency.condition, conversation_state
            )

        # If no condition, dependency is satisfied once agent has executed
        return True

    def _evaluate_dependency_condition(
        self,
        condition: str,
        conversation_state: ConversationState,
    ) -> bool:
        """Safely evaluate a dependency condition against conversation state.

        Args:
            condition: Python expression to evaluate
            conversation_state: Current conversation state

        Returns:
            True if condition evaluates to True, False otherwise or on error
        """
        return conversation_state.evaluate_condition(condition)

    def _can_agent_execute_more_times(
        self,
        agent: ConversationalAgent,
        conversation_state: ConversationState,
    ) -> bool:
        """Check if agent can execute more times based on its execution limits.

        Args:
            agent: The agent to check
            conversation_state: Current conversation state

        Returns:
            True if agent can execute more times
        """
        current_executions = conversation_state.agent_execution_count.get(agent.name, 0)

        # Calculate max executions from dependencies
        max_executions = 1  # Default
        if agent.dependencies:
            max_executions = max(dep.max_executions for dep in agent.dependencies)

        return current_executions < max_executions
