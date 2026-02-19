"""Conversational ensemble executor extending existing EnsembleExecutor."""

import time
from datetime import datetime
from typing import Any

from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationalEnsemble,
    ConversationResult,
    ConversationState,
    ConversationTurn,
)


class ConversationalEnsembleExecutor(EnsembleExecutor):
    """Executor supporting multi-turn agent conversations."""

    async def execute_conversation(
        self,
        ensemble: ConversationalEnsemble,
        initial_context: dict[str, Any] | None = None,
    ) -> ConversationResult:
        """Execute ensemble as a multi-turn conversation."""
        conversation_state = ConversationState(
            accumulated_context=initial_context or {}
        )

        # Main conversation loop - execute turns until completion
        while self._should_continue_conversation(conversation_state, ensemble):
            # Get agents ready for this turn
            ready_agents = self._get_ready_agents_for_conversation(
                ensemble.agents, conversation_state
            )

            if not ready_agents:
                break  # Conversation complete - no more agents can execute

            # Execute ready agents in this turn
            await self._execute_conversation_turn(ready_agents, conversation_state)

            # Check if we should continue the conversation
            if not conversation_state.can_continue_conversation(
                ensemble.conversation_limits
            ):
                break

        # Determine completion reason
        completion_reason = self._determine_completion_reason(
            conversation_state, ensemble
        )

        return ConversationResult(
            final_state=conversation_state.accumulated_context,
            conversation_history=conversation_state.conversation_history,
            turn_count=conversation_state.turn_count,
            completion_reason=completion_reason,
        )

    def _should_continue_conversation(
        self, state: ConversationState, ensemble: ConversationalEnsemble
    ) -> bool:
        """Check if conversation should continue based on turn limits.

        Args:
            state: Current conversation state
            ensemble: Ensemble configuration

        Returns:
            True if conversation should continue
        """
        return state.turn_count < ensemble.conversation_limits.max_total_turns

    async def _execute_conversation_turn(
        self,
        ready_agents: list[ConversationalAgent],
        conversation_state: ConversationState,
    ) -> None:
        """Execute one turn of conversation with ready agents.

        Args:
            ready_agents: Agents ready to execute in this turn
            conversation_state: Current conversation state
        """
        for agent in ready_agents:
            try:
                await self._execute_agent_turn(agent, conversation_state)
            except Exception as e:
                self._record_agent_error(agent, conversation_state, e)

    async def _execute_agent_turn(
        self, agent: ConversationalAgent, conversation_state: ConversationState
    ) -> None:
        """Execute a single agent turn.

        Args:
            agent: Agent to execute
            conversation_state: Current conversation state
        """
        turn_start_time = time.time()

        # Convert ConversationalAgent to dict for EnsembleExecutor
        agent_config = {
            "name": agent.name,
            "script": agent.script,
            "model_profile": agent.model_profile,
            "prompt": agent.prompt,
            "config": agent.config,
        }

        # Prepare input data from accumulated context
        input_data = self._prepare_agent_input(agent, conversation_state)

        # Execute agent using existing infrastructure
        result = await self._execute_single_agent(agent_config, input_data)

        # Record turn in conversation state using existing method
        execution_time = time.time() - turn_start_time
        conversation_state.record_agent_turn(
            agent,
            {"context": input_data},
            {"result": result[0] if result else ""},
            execution_time,
        )

        # Update accumulated context
        self._update_conversation_context(agent, result, conversation_state)

    def _update_conversation_context(
        self,
        agent: ConversationalAgent,
        result: tuple[str, Any],
        conversation_state: ConversationState,
    ) -> None:
        """Update accumulated context with agent result.

        Args:
            agent: Agent that was executed
            result: Agent execution result
            conversation_state: Current conversation state
        """
        result_text = result[0] if result else ""
        if agent.conversation and agent.conversation.state_key:
            conversation_state.accumulated_context[agent.conversation.state_key] = (
                result_text
            )
        else:
            conversation_state.accumulated_context[agent.name] = result_text

    def _record_agent_error(
        self,
        agent: ConversationalAgent,
        conversation_state: ConversationState,
        error: Exception,
    ) -> None:
        """Record an agent execution error in conversation history.

        Args:
            agent: Agent that failed
            conversation_state: Current conversation state
            error: Exception that occurred
        """
        error_turn = ConversationTurn(
            turn_number=conversation_state.turn_count + 1,
            agent_name=agent.name,
            input_data={},
            output_data={"error": str(error)},
            execution_time=0.0,
            timestamp=datetime.now(),
        )
        conversation_state.conversation_history.append(error_turn)
        conversation_state.turn_count += 1

    def _prepare_agent_input(
        self, agent: "ConversationalAgent", state: ConversationState
    ) -> str:
        """Prepare input data for agent based on conversation state."""
        # For minimal implementation, return JSON representation of context
        import json

        context_data = {
            "turn_count": state.turn_count,
            "accumulated_context": state.accumulated_context,
            "agent_execution_count": state.agent_execution_count,
        }

        try:
            return json.dumps(context_data, indent=2)
        except Exception:
            return str(context_data)

    def _get_ready_agents_for_conversation(
        self,
        agents: list[ConversationalAgent],
        conversation_state: ConversationState,
    ) -> list[ConversationalAgent]:
        """Get agents ready to execute considering conversation state."""
        ready_agents = []

        for agent in agents:
            # Check if agent should execute based on conversation limits
            if not conversation_state.should_execute_agent(agent):
                continue

            # For now, simple dependency resolution - can be extended later
            # All agents are ready if they haven't exceeded their limits
            ready_agents.append(agent)

        return ready_agents

    def _determine_completion_reason(
        self, conversation_state: ConversationState, ensemble: ConversationalEnsemble
    ) -> str:
        """Determine why the conversation completed."""
        if (
            conversation_state.turn_count
            >= ensemble.conversation_limits.max_total_turns
        ):
            return "max_turns_reached"
        elif not ensemble.agents:
            return "no_agents"
        else:
            return "completed"

    async def _execute_single_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, Any]:
        """Execute single agent using existing infrastructure."""
        # Use existing agent execution logic from parent class
        try:
            # execute_agent_with_timeout requires timeout_seconds parameter
            timeout_seconds = agent_config.get("timeout_seconds", 30)
            return await self._execution_coordinator.execute_agent_with_timeout(
                agent_config, input_data, timeout_seconds
            )
        except Exception:
            # Return empty result for minimal implementation
            return ("", None)
