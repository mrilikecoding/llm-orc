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

from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationConfig,
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
