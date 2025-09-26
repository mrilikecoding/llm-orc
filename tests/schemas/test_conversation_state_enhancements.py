"""Additional enhancement tests for ConversationState (ADR-005).

These tests validate additional methods that enhance ConversationState
functionality for better conversation flow control and debugging.
"""

from datetime import datetime

from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationConfig,
    ConversationLimits,
    ConversationState,
    ConversationTurn,
)


class TestConversationStateEnhancements:
    """Test enhanced ConversationState methods."""

    def test_record_agent_turn_updates_state_correctly(self) -> None:
        """record_agent_turn should update all state components atomically."""
        state = ConversationState()

        # Create test agent
        agent = ConversationalAgent(
            name="test_agent",
            script="test.py",
            conversation=ConversationConfig(max_turns=3),
        )

        input_data = {"context": "test input"}
        output_data = {"result": "test output"}
        execution_time = 1.5

        # This method should be added to ConversationState
        state.record_agent_turn(
            agent=agent,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
        )

        # Verify all state updates
        assert state.turn_count == 1
        assert state.agent_execution_count["test_agent"] == 1
        assert len(state.conversation_history) == 1

        turn = state.conversation_history[0]
        assert turn.turn_number == 1
        assert turn.agent_name == "test_agent"
        assert turn.input_data == input_data
        assert turn.output_data == output_data
        assert turn.execution_time == execution_time

    def test_record_agent_turn_with_state_key_updates_context(self) -> None:
        """record_agent_turn should use state_key when provided."""
        state = ConversationState()

        agent = ConversationalAgent(
            name="analyzer",
            script="analyze.py",
            conversation=ConversationConfig(max_turns=2, state_key="analysis_result"),
        )

        output_data = {"analysis": "completed", "confidence": 0.95}

        state.record_agent_turn(
            agent=agent, input_data={}, output_data=output_data, execution_time=0.8
        )

        # Should use state_key for context storage
        assert state.accumulated_context["analysis_result"] == output_data

    def test_record_agent_turn_without_state_key_uses_agent_name(self) -> None:
        """record_agent_turn should use agent name when no state_key provided."""
        state = ConversationState()

        agent = ConversationalAgent(name="processor", model_profile="test-model")

        output_data = {"processed": True}

        state.record_agent_turn(
            agent=agent, input_data={}, output_data=output_data, execution_time=0.3
        )

        # Should use agent name for context storage
        assert state.accumulated_context["processor"] == output_data

    def test_can_continue_conversation_checks_global_limits(self) -> None:
        """can_continue_conversation should check against global limits."""
        limits = ConversationLimits(
            max_total_turns=5, max_agent_executions={"agent1": 2, "agent2": 1}
        )

        state = ConversationState()

        # Should continue when under limits
        assert state.can_continue_conversation(limits) is True

        # Should not continue when at turn limit
        state.turn_count = 5
        assert state.can_continue_conversation(limits) is False

        # Reset turn count, test agent execution limits
        state.turn_count = 2
        state.agent_execution_count = {"agent1": 2, "agent2": 0}

        # Should not continue when any agent hits its limit
        assert state.can_continue_conversation(limits) is False

    def test_get_recent_turns_returns_last_n_turns(self) -> None:
        """get_recent_turns should return the most recent N turns."""
        state = ConversationState()

        # Add several turns
        for i in range(5):
            turn = ConversationTurn(
                turn_number=i + 1,
                agent_name=f"agent{i}",
                input_data={"turn": i},
                output_data={"result": f"output{i}"},
                execution_time=0.1,
                timestamp=datetime.now(),
            )
            state.conversation_history.append(turn)

        # Get last 3 turns
        recent_turns = state.get_recent_turns(3)

        assert len(recent_turns) == 3
        assert recent_turns[0].turn_number == 3  # Third turn
        assert recent_turns[1].turn_number == 4  # Fourth turn
        assert recent_turns[2].turn_number == 5  # Fifth turn

    def test_get_recent_turns_handles_fewer_turns_available(self) -> None:
        """get_recent_turns should handle when fewer turns exist than requested."""
        state = ConversationState()

        # Add only 2 turns
        for i in range(2):
            turn = ConversationTurn(
                turn_number=i + 1,
                agent_name=f"agent{i}",
                input_data={},
                output_data={},
                execution_time=0.1,
                timestamp=datetime.now(),
            )
            state.conversation_history.append(turn)

        # Request 5 turns, should return all 2
        recent_turns = state.get_recent_turns(5)
        assert len(recent_turns) == 2

    def test_get_agent_last_output_returns_most_recent_result(self) -> None:
        """get_agent_last_output should return the most recent output from agent."""
        state = ConversationState()

        # Add turns for multiple agents
        agents = ["agent1", "agent2", "agent1", "agent3", "agent1"]
        for i, agent_name in enumerate(agents):
            turn = ConversationTurn(
                turn_number=i + 1,
                agent_name=agent_name,
                input_data={},
                output_data={"result": f"{agent_name}_output_{i}"},
                execution_time=0.1,
                timestamp=datetime.now(),
            )
            state.conversation_history.append(turn)

        # Should return most recent output from agent1 (turn 5)
        last_output = state.get_agent_last_output("agent1")
        assert last_output == {"result": "agent1_output_4"}

        # Should return output from agent2 (turn 2)
        last_output = state.get_agent_last_output("agent2")
        assert last_output == {"result": "agent2_output_1"}

    def test_get_agent_last_output_returns_none_for_nonexistent_agent(self) -> None:
        """get_agent_last_output should return None for agents that haven't executed."""
        state = ConversationState()

        # Add turn for one agent
        turn = ConversationTurn(
            turn_number=1,
            agent_name="agent1",
            input_data={},
            output_data={"result": "test"},
            execution_time=0.1,
            timestamp=datetime.now(),
        )
        state.conversation_history.append(turn)

        # Should return None for non-existent agent
        assert state.get_agent_last_output("nonexistent") is None

    def test_has_agent_executed_checks_execution_history(self) -> None:
        """has_agent_executed should check if agent has executed at least once."""
        state = ConversationState()

        # Initially no agents executed
        assert state.has_agent_executed("agent1") is False

        # Add execution count
        state.agent_execution_count["agent1"] = 2
        state.agent_execution_count["agent2"] = 0  # Explicit zero

        assert state.has_agent_executed("agent1") is True
        assert state.has_agent_executed("agent2") is False
        assert state.has_agent_executed("agent3") is False  # Not in dict

    def test_reset_conversation_clears_all_state(self) -> None:
        """reset_conversation should clear all conversation state."""
        state = ConversationState()

        # Set up some state
        state.turn_count = 5
        state.agent_execution_count = {"agent1": 3, "agent2": 2}
        state.accumulated_context = {"key": "value", "analysis": {"complete": True}}

        turn = ConversationTurn(
            turn_number=1,
            agent_name="agent1",
            input_data={},
            output_data={},
            execution_time=0.1,
            timestamp=datetime.now(),
        )
        state.conversation_history.append(turn)

        # Reset conversation
        state.reset_conversation()

        # All state should be cleared
        assert state.turn_count == 0
        assert state.agent_execution_count == {}
        assert state.accumulated_context == {}
        assert state.conversation_history == []
