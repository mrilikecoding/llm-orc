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

        # Execute all agents in the ensemble
        for agent in ensemble.agents:
            # Check if agent should execute based on conversation limits
            if not conversation_state.should_execute_agent(agent):
                continue

            # Check global turn limit
            if (
                conversation_state.turn_count
                >= ensemble.conversation_limits.max_total_turns
            ):
                break

            # Execute agent
            try:
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

                # Record turn in conversation state
                execution_time = time.time() - turn_start_time
                turn = ConversationTurn(
                    turn_number=conversation_state.turn_count + 1,
                    agent_name=agent.name,
                    input_data={"context": input_data},
                    output_data={"result": result[0] if result else ""},
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                )

                conversation_state.conversation_history.append(turn)
                conversation_state.turn_count += 1

                # Update agent execution count
                current_count = conversation_state.agent_execution_count.get(
                    agent.name, 0
                )
                conversation_state.agent_execution_count[agent.name] = current_count + 1

                # Update accumulated context
                if agent.conversation and agent.conversation.state_key:
                    conversation_state.accumulated_context[
                        agent.conversation.state_key
                    ] = result[0] if result else ""
                else:
                    conversation_state.accumulated_context[agent.name] = (
                        result[0] if result else ""
                    )

            except Exception as e:
                # Error handling with exception chaining (ADR-003)
                error_turn = ConversationTurn(
                    turn_number=conversation_state.turn_count + 1,
                    agent_name=agent.name,
                    input_data={},
                    output_data={"error": str(e)},
                    execution_time=0.0,
                    timestamp=datetime.now(),
                )
                conversation_state.conversation_history.append(error_turn)
                conversation_state.turn_count += 1

        # Determine completion reason
        completion_reason = "completed"
        if (
            conversation_state.turn_count
            >= ensemble.conversation_limits.max_total_turns
        ):
            completion_reason = "max_turns_reached"
        elif not ensemble.agents:
            completion_reason = "no_agents"

        return ConversationResult(
            final_state=conversation_state.accumulated_context,
            conversation_history=conversation_state.conversation_history,
            turn_count=conversation_state.turn_count,
            completion_reason=completion_reason,
        )

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

    async def _execute_single_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, Any]:
        """Execute single agent using existing infrastructure."""
        # Use existing agent execution logic from parent class
        try:
            # _execute_agent_with_timeout requires timeout_seconds parameter
            timeout_seconds = agent_config.get("timeout_seconds", 30)
            return await self._execute_agent_with_timeout(
                agent_config, input_data, timeout_seconds
            )
        except Exception:
            # Return empty result for minimal implementation
            return ("", None)
