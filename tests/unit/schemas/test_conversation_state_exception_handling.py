"""Exception handling tests for ConversationState (ADR-003 compliance).

These tests validate that ConversationState properly implements exception
chaining as required by ADR-003 for testable contracts.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationConfig,
    ConversationState,
    ConversationTurn,
)


class TestConversationStateExceptionChaining:
    """Test ADR-003 exception chaining compliance."""

    def test_record_agent_turn_chains_datetime_error(self) -> None:
        """record_agent_turn should chain datetime errors per ADR-003."""
        state = ConversationState()

        agent = ConversationalAgent(
            name="test_agent",
            script="test.py",
        )

        # Mock datetime.now() to raise an exception
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.side_effect = RuntimeError("DateTime system error")

            # Should chain the original exception
            with pytest.raises(RuntimeError, match="DateTime system error"):
                state.record_agent_turn(
                    agent=agent,
                    input_data={},
                    output_data={},
                    execution_time=0.1,
                )

    def test_record_agent_turn_handles_conversation_turn_creation_error(self) -> None:
        """record_agent_turn should handle ConversationTurn creation errors properly."""
        state = ConversationState()

        agent = ConversationalAgent(
            name="test_agent",
            script="test.py",
        )

        # Mock ConversationTurn to raise validation error
        with patch(
            "llm_orc.schemas.conversational_agent.ConversationTurn"
        ) as mock_turn:
            mock_turn.side_effect = ValueError("Invalid turn data")

            # Should chain the original exception
            with pytest.raises(ValueError, match="Invalid turn data"):
                state.record_agent_turn(
                    agent=agent,
                    input_data={},
                    output_data={},
                    execution_time=0.1,
                )

    def test_evaluate_condition_graceful_error_handling(self) -> None:
        """evaluate_condition should handle errors gracefully without chaining."""
        state = ConversationState()

        # These should all return False gracefully (not raise exceptions)
        error_conditions = [
            "undefined_function()",
            "context['nonexistent_key'].method()",
            "1 / 0",  # Division by zero
            "raise ValueError('test error')",
        ]

        for condition in error_conditions:
            # Should not raise exception, should return False
            result = state.evaluate_condition(condition)
            assert result is False, (
                f"Condition '{condition}' should return False on error"
            )

    def test_evaluate_condition_prevents_exception_leakage(self) -> None:
        """evaluate_condition should prevent internal exceptions from leaking."""
        state = ConversationState()

        # Even malicious conditions should be safely handled
        malicious_conditions = [
            "__import__('sys').exit()",
            "exec('import os; os.system(\"echo test\")')",
            "eval('__import__(\"sys\").exit()')",
        ]

        for condition in malicious_conditions:
            try:
                result = state.evaluate_condition(condition)
                # Should return False without raising
                assert result is False
            except SystemExit:
                pytest.fail(
                    f"Malicious condition '{condition}' should not cause system exit"
                )
            except Exception as e:
                pytest.fail(f"Condition '{condition}' should not raise: {e}")


class TestConversationStateErrorResilience:
    """Test error resilience in ConversationState methods."""

    def test_get_agent_last_output_handles_corrupted_history(self) -> None:
        """get_agent_last_output should handle corrupted conversation history."""
        state = ConversationState()

        # Add a turn with missing output_data
        corrupted_turn = ConversationTurn(
            turn_number=1,
            agent_name="test_agent",
            input_data={},
            output_data={},  # Empty output data
            execution_time=0.1,
            timestamp=datetime.now(),
        )
        state.conversation_history.append(corrupted_turn)

        # Should handle gracefully
        result = state.get_agent_last_output("test_agent")
        assert result == {}  # Should return the empty dict

    def test_get_recent_turns_handles_negative_count(self) -> None:
        """get_recent_turns should handle negative count gracefully."""
        state = ConversationState()

        # Add some turns
        for i in range(3):
            turn = ConversationTurn(
                turn_number=i + 1,
                agent_name=f"agent{i}",
                input_data={},
                output_data={},
                execution_time=0.1,
                timestamp=datetime.now(),
            )
            state.conversation_history.append(turn)

        # Should handle negative count gracefully
        result = state.get_recent_turns(-5)
        assert result == []

        # Should handle zero count gracefully
        result = state.get_recent_turns(0)
        assert result == []

    def test_should_execute_agent_handles_none_conversation_config(self) -> None:
        """should_execute_agent should handle None conversation config gracefully."""
        state = ConversationState()

        # Agent with None conversation config
        agent = ConversationalAgent(
            name="simple_agent",
            script="simple.py",
            conversation=None,
        )

        # Should default to 1 execution and not crash
        assert state.should_execute_agent(agent) is True

        # After one execution, should not execute again
        state.agent_execution_count["simple_agent"] = 1
        assert state.should_execute_agent(agent) is False

    def test_record_agent_turn_handles_none_conversation_config(self) -> None:
        """record_agent_turn should handle agent with None conversation config."""
        state = ConversationState()

        agent = ConversationalAgent(
            name="simple_agent",
            script="simple.py",
            conversation=None,
        )

        # Should use agent name as context key when no conversation config
        state.record_agent_turn(
            agent=agent,
            input_data={"test": "input"},
            output_data={"test": "output"},
            execution_time=0.5,
        )

        # Should update state correctly
        assert state.turn_count == 1
        assert state.agent_execution_count["simple_agent"] == 1
        assert state.accumulated_context["simple_agent"] == {"test": "output"}
        assert len(state.conversation_history) == 1

    def test_record_agent_turn_handles_none_state_key(self) -> None:
        """record_agent_turn should handle conversation config with None state_key."""
        state = ConversationState()

        agent = ConversationalAgent(
            name="test_agent",
            script="test.py",
            conversation=ConversationConfig(max_turns=2, state_key=None),
        )

        output_data = {"result": "test output"}

        state.record_agent_turn(
            agent=agent,
            input_data={},
            output_data=output_data,
            execution_time=0.3,
        )

        # Should use agent name when state_key is None
        assert state.accumulated_context["test_agent"] == output_data
