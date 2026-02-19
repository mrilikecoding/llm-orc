"""Unit tests for ConversationState management system (ADR-005).

This test module follows strict TDD methodology to validate ConversationState
functionality as specified in ADR-005. Tests ensure:

1. Turn Counting: Track conversation turns and agent executions
2. Context Accumulation: Persistent state across turns
3. Condition Evaluation: Safe evaluation of conditional dependencies
4. Execution Limits: Check if agents should execute based on turn limits
5. History Tracking: Record of conversation turns with metadata

All tests follow ADR-001 Pydantic compliance and ADR-003 exception chaining.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationConfig,
    ConversationLimits,
    ConversationState,
    ConversationTurn,
)


class TestConversationStateTurnCounting:
    """Test turn counting functionality."""

    def test_initial_state_has_zero_turns(self) -> None:
        """ConversationState should initialize with zero turns."""
        state = ConversationState()

        assert state.turn_count == 0
        assert state.agent_execution_count == {}
        assert state.accumulated_context == {}
        assert state.conversation_history == []

    def test_turn_count_tracks_conversation_progress(self) -> None:
        """ConversationState should track turn progression correctly."""
        state = ConversationState()

        # Simulate turn progression
        state.turn_count = 3
        state.agent_execution_count = {"agent1": 2, "agent2": 1}

        assert state.turn_count == 3
        assert state.agent_execution_count["agent1"] == 2
        assert state.agent_execution_count["agent2"] == 1

    def test_agent_execution_count_initializes_correctly(self) -> None:
        """Agent execution counts should initialize as empty dict."""
        state = ConversationState()

        # Should handle getting count for non-existent agent
        count = state.agent_execution_count.get("non_existent", 0)
        assert count == 0


class TestConversationStateContextAccumulation:
    """Test context accumulation across turns."""

    def test_accumulated_context_persists_data(self) -> None:
        """Accumulated context should persist data across turns."""
        state = ConversationState()

        # Add context data
        state.accumulated_context["agent1_output"] = "test data"
        state.accumulated_context["analysis_result"] = {
            "score": 0.85,
            "confidence": "high",
        }

        assert state.accumulated_context["agent1_output"] == "test data"
        assert state.accumulated_context["analysis_result"]["score"] == 0.85
        assert state.accumulated_context["analysis_result"]["confidence"] == "high"

    def test_context_supports_complex_data_structures(self) -> None:
        """Context should support nested dicts, lists, and mixed types."""
        state = ConversationState()

        complex_data = {
            "nested_dict": {"key1": "value1", "key2": {"deep": "value"}},
            "list_data": [1, 2, {"item": "value"}],
            "mixed_types": {"str": "text", "int": 42, "bool": True, "none": None},
        }

        state.accumulated_context["complex"] = complex_data

        # Verify data integrity
        assert (
            state.accumulated_context["complex"]["nested_dict"]["key2"]["deep"]
            == "value"
        )
        assert state.accumulated_context["complex"]["list_data"][2]["item"] == "value"
        assert state.accumulated_context["complex"]["mixed_types"]["int"] == 42


class TestConversationStateExecutionLimits:
    """Test execution limit enforcement."""

    def test_should_execute_agent_respects_max_turns(self) -> None:
        """should_execute_agent should respect agent max_turns configuration."""
        state = ConversationState()

        # Create agent with 2 max turns
        agent = ConversationalAgent(
            name="test_agent",
            script="test.py",
            conversation=ConversationConfig(max_turns=2),
        )

        # Should execute when under limit
        state.agent_execution_count["test_agent"] = 0
        assert state.should_execute_agent(agent) is True

        state.agent_execution_count["test_agent"] = 1
        assert state.should_execute_agent(agent) is True

        # Should not execute when at limit
        state.agent_execution_count["test_agent"] = 2
        assert state.should_execute_agent(agent) is False

    def test_should_execute_agent_defaults_to_one_turn(self) -> None:
        """Agents without conversation config should default to 1 execution."""
        state = ConversationState()

        # Agent without conversation config
        agent = ConversationalAgent(name="simple_agent", script="simple.py")

        # Should execute once
        assert state.should_execute_agent(agent) is True

        # Should not execute after first execution
        state.agent_execution_count["simple_agent"] = 1
        assert state.should_execute_agent(agent) is False

    def test_should_execute_agent_handles_missing_execution_count(self) -> None:
        """should_execute_agent should handle agents not yet in execution count."""
        state = ConversationState()

        agent = ConversationalAgent(
            name="new_agent",
            script="new.py",
            conversation=ConversationConfig(max_turns=3),
        )

        # Should execute when agent not in execution count (implies 0 executions)
        assert state.should_execute_agent(agent) is True


class TestConversationStateConditionEvaluation:
    """Test safe condition evaluation."""

    def test_evaluate_condition_returns_true_for_empty_condition(self) -> None:
        """Empty or None conditions should evaluate to True."""
        state = ConversationState()

        assert state.evaluate_condition("") is True
        assert state.evaluate_condition(None) is True  # type: ignore

    def test_evaluate_condition_accesses_turn_count(self) -> None:
        """Conditions should safely access turn_count."""
        state = ConversationState()
        state.turn_count = 5

        assert state.evaluate_condition("turn_count > 3") is True
        assert state.evaluate_condition("turn_count < 3") is False
        assert state.evaluate_condition("turn_count == 5") is True

    def test_evaluate_condition_accesses_accumulated_context(self) -> None:
        """Conditions should safely access accumulated context."""
        state = ConversationState()
        state.accumulated_context = {
            "needs_clarification": True,
            "analysis_complete": False,
            "score": 0.85,
        }

        assert (
            state.evaluate_condition("context.get('needs_clarification', False)")
            is True
        )
        assert (
            state.evaluate_condition("context.get('analysis_complete', True)") is False
        )
        assert state.evaluate_condition("context.get('score', 0) > 0.8") is True

    def test_evaluate_condition_accesses_conversation_history(self) -> None:
        """Conditions should safely access conversation history."""
        state = ConversationState()

        # Add some history
        turn1 = ConversationTurn(
            turn_number=1,
            agent_name="agent1",
            input_data={"input": "test"},
            output_data={"output": "result"},
            execution_time=0.5,
            timestamp=datetime.now(),
        )
        state.conversation_history.append(turn1)

        assert state.evaluate_condition("len(history) > 0") is True
        assert state.evaluate_condition("len(history) == 1") is True
        assert state.evaluate_condition("len(history) > 5") is False

    def test_evaluate_condition_prevents_dangerous_operations(self) -> None:
        """Condition evaluation should prevent dangerous operations."""
        state = ConversationState()

        # These should all return False due to restricted context
        dangerous_conditions = [
            "__import__('os')",
            "open('/etc/passwd')",
            "exec('print(\"evil\")')",
            "eval('1+1')",  # eval inside eval should be blocked
            "__builtins__['print']('test')",
        ]

        for dangerous_condition in dangerous_conditions:
            result = state.evaluate_condition(dangerous_condition)
            assert result is False, (
                f"Dangerous condition '{dangerous_condition}' should return False"
            )

    def test_evaluate_condition_allows_safe_functions(self) -> None:
        """Condition evaluation should allow whitelisted safe functions."""
        state = ConversationState()
        state.accumulated_context["items"] = [1, 2, 3, 4, 5]

        # len() should be available
        assert state.evaluate_condition("len(context['items']) == 5") is True
        assert state.evaluate_condition("len(context['items']) > 3") is True

    def test_evaluate_condition_handles_syntax_errors_gracefully(self) -> None:
        """Malformed conditions should fail gracefully and return False."""
        state = ConversationState()

        malformed_conditions = [
            "invalid syntax here (",
            "context['key'",  # Missing closing bracket
            "turn_count >",  # Incomplete expression
            "1 + * 1",  # Invalid operator sequence
            "nonexistent_variable > 0",
        ]

        for condition in malformed_conditions:
            result = state.evaluate_condition(condition)
            assert result is False, (
                f"Malformed condition '{condition}' should return False"
            )


class TestConversationStateHistoryTracking:
    """Test conversation history management."""

    def test_conversation_history_maintains_turn_order(self) -> None:
        """Conversation history should maintain chronological turn order."""
        state = ConversationState()

        # Add turns in order
        for i in range(1, 4):
            turn = ConversationTurn(
                turn_number=i,
                agent_name=f"agent{i}",
                input_data={"input": f"data{i}"},
                output_data={"output": f"result{i}"},
                execution_time=0.1 * i,
                timestamp=datetime.now(),
            )
            state.conversation_history.append(turn)

        # Verify order maintained
        assert len(state.conversation_history) == 3
        for i, turn in enumerate(state.conversation_history):
            assert turn.turn_number == i + 1
            assert turn.agent_name == f"agent{i + 1}"

    def test_conversation_turn_stores_complete_metadata(self) -> None:
        """ConversationTurn should store all required metadata."""
        turn = ConversationTurn(
            turn_number=1,
            agent_name="test_agent",
            input_data={"context": "input data"},
            output_data={"result": "output data"},
            execution_time=1.25,
            timestamp=datetime.now(),
        )

        assert turn.turn_number == 1
        assert turn.agent_name == "test_agent"
        assert turn.input_data["context"] == "input data"
        assert turn.output_data["result"] == "output data"
        assert turn.execution_time == 1.25
        assert isinstance(turn.timestamp, datetime)


class TestConversationStatePydanticCompliance:
    """Test ADR-001 Pydantic compliance."""

    def test_conversation_state_validates_field_types(self) -> None:
        """ConversationState should validate field types per ADR-001."""
        # Valid initialization
        state = ConversationState(
            turn_count=5,
            agent_execution_count={"agent1": 2},
            accumulated_context={"key": "value"},
            conversation_history=[],
        )

        assert state.turn_count == 5
        assert state.agent_execution_count == {"agent1": 2}
        assert state.accumulated_context == {"key": "value"}

    def test_conversation_state_provides_field_defaults(self) -> None:
        """ConversationState should provide proper field defaults."""
        state = ConversationState()

        # All fields should have proper defaults
        assert isinstance(state.turn_count, int)
        assert isinstance(state.agent_execution_count, dict)
        assert isinstance(state.accumulated_context, dict)
        assert isinstance(state.conversation_history, list)

    def test_conversation_state_serializes_correctly(self) -> None:
        """ConversationState should serialize to/from JSON per ADR-001."""
        original_state = ConversationState(
            turn_count=3,
            agent_execution_count={"agent1": 2, "agent2": 1},
            accumulated_context={"result": "test", "score": 0.85},
            conversation_history=[],
        )

        # Should serialize to dict
        state_dict = original_state.model_dump()
        assert state_dict["turn_count"] == 3
        assert state_dict["agent_execution_count"]["agent1"] == 2
        assert state_dict["accumulated_context"]["score"] == 0.85

        # Should deserialize from dict
        restored_state = ConversationState.model_validate(state_dict)
        assert restored_state.turn_count == original_state.turn_count
        assert (
            restored_state.agent_execution_count == original_state.agent_execution_count
        )
        assert restored_state.accumulated_context == original_state.accumulated_context


class TestConversationStateIntegration:
    """Integration tests for ConversationState functionality."""

    def test_conversation_state_supports_full_workflow(self) -> None:
        """ConversationState should support complete conversation workflows."""
        state = ConversationState()

        # Create test agents
        analyzer = ConversationalAgent(
            name="analyzer",
            script="analyze.py",
            conversation=ConversationConfig(max_turns=2),
        )

        reviewer = ConversationalAgent(
            name="reviewer",
            model_profile="test-model",
            conversation=ConversationConfig(max_turns=1),
        )

        # Test first turn - analyzer should execute
        assert state.should_execute_agent(analyzer) is True
        assert state.should_execute_agent(reviewer) is True

        # Simulate first turn execution
        state.turn_count = 1
        state.agent_execution_count["analyzer"] = 1
        state.accumulated_context["analysis"] = {"status": "needs_review"}

        # Add turn to history
        turn1 = ConversationTurn(
            turn_number=1,
            agent_name="analyzer",
            input_data={},
            output_data={"analysis": "initial results"},
            execution_time=0.5,
            timestamp=datetime.now(),
        )
        state.conversation_history.append(turn1)

        # Test condition evaluation for next turn
        assert (
            state.evaluate_condition(
                "context.get('analysis', {}).get('status') == 'needs_review'"
            )
            is True
        )

        # Analyzer can still execute (max_turns=2), reviewer can execute once
        assert state.should_execute_agent(analyzer) is True
        assert state.should_execute_agent(reviewer) is True

        # Simulate second turn
        state.turn_count = 2
        state.agent_execution_count["analyzer"] = 2
        state.agent_execution_count["reviewer"] = 1

        # Now analyzer is at limit, reviewer is at limit
        assert state.should_execute_agent(analyzer) is False
        assert state.should_execute_agent(reviewer) is False

        # Verify conversation state integrity
        assert state.turn_count == 2
        assert len(state.conversation_history) == 1
        assert state.accumulated_context["analysis"]["status"] == "needs_review"


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
        """record_agent_turn should handle ConversationTurn creation errors."""
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
