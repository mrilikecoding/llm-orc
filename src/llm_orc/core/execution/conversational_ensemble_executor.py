"""Conversational ensemble executor extending existing EnsembleExecutor."""

import time
from datetime import datetime
from typing import Any

from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.schemas.conversational_agent import (
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

        # Start with a simple single-turn execution for minimal implementation
        start_time = time.time()

        try:
            # Execute first agent only for minimal implementation
            if ensemble.agents:
                first_agent = ensemble.agents[0]

                # Convert ConversationalAgent to dict for EnsembleExecutor
                agent_config = {
                    "name": first_agent.name,
                    "script": first_agent.script,
                    "model_profile": first_agent.model_profile,
                    "prompt": first_agent.prompt,
                    "config": first_agent.config,
                }

                # Use existing ensemble execution for single agent
                result = await self._execute_single_agent(agent_config, "")

                # Record turn in conversation state
                turn = ConversationTurn(
                    turn_number=1,
                    agent_name=first_agent.name,
                    input_data={},
                    output_data={"result": result[0] if result else ""},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                )

                conversation_state.conversation_history.append(turn)
                conversation_state.turn_count = 1
                conversation_state.agent_execution_count[first_agent.name] = 1
                conversation_state.accumulated_context[first_agent.name] = (
                    result[0] if result else ""
                )

        except Exception:
            # Basic error handling
            pass

        return ConversationResult(
            final_state=conversation_state.accumulated_context,
            conversation_history=conversation_state.conversation_history,
            turn_count=conversation_state.turn_count,
            completion_reason="minimal_implementation",
        )

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
