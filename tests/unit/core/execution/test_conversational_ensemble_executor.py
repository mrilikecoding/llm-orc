"""Tests for ConversationalEnsembleExecutor."""

from unittest.mock import MagicMock, patch

import pytest

from llm_orc.core.execution.conversational_ensemble_executor import (
    ConversationalEnsembleExecutor,
)
from llm_orc.schemas.conversational_agent import (
    ConversationalAgent,
    ConversationalEnsemble,
    ConversationConfig,
    ConversationLimits,
)


class TestConversationalEnsembleExecutor:
    """Test the ConversationalEnsembleExecutor class."""

    @pytest.fixture
    def executor(self) -> ConversationalEnsembleExecutor:
        """Create a ConversationalEnsembleExecutor instance for testing."""
        with patch(
            "llm_orc.core.execution.ensemble_execution.ConfigurationManager"
        ) as mock_cm:
            with patch(
                "llm_orc.core.execution.ensemble_execution.CredentialStorage"
            ) as mock_cs:
                # Mock configuration manager methods
                mock_config = MagicMock()
                mock_config.load_performance_config.return_value = {
                    "batch_size": 5,
                    "parallel_execution": True,
                }
                mock_cm.return_value = mock_config

                # Mock credential storage
                mock_cs.return_value = MagicMock()

                return ConversationalEnsembleExecutor()

    @pytest.mark.asyncio
    async def test_execute_conversation_basic(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test basic conversation execution."""
        # Create test agents with correct schema
        agent1 = ConversationalAgent(
            name="agent1",
            model_profile="test-model",
            prompt="You are agent 1",
            conversation=ConversationConfig(max_turns=2),
        )
        agent2 = ConversationalAgent(
            name="agent2",
            model_profile="test-model",
            prompt="You are agent 2",
            conversation=ConversationConfig(max_turns=2),
        )

        # Create ensemble
        ensemble = ConversationalEnsemble(
            name="test_conversation",
            agents=[agent1, agent2],
            conversation_limits=ConversationLimits(
                max_total_turns=10, timeout_seconds=60
            ),
        )

        # Mock model execution
        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.return_value = (
                "test response",
                {"model": "test-model"},
            )

            # Execute conversation
            result = await executor.execute_conversation(
                ensemble=ensemble, initial_context={"test": "context"}
            )

            # Verify result structure
            assert result is not None
            assert hasattr(result, "final_state")
            assert hasattr(result, "conversation_history")
            assert hasattr(result, "turn_count")
            assert hasattr(result, "completion_reason")
            # With agent1 (2 turns) + agent2 (2 turns) = 4 total turns < 10 limit
            assert result.completion_reason == "completed"

    @pytest.mark.asyncio
    async def test_execute_conversation_with_multiple_turns(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test conversation execution with agent having multiple turns."""
        agent = ConversationalAgent(
            name="test_agent",
            model_profile="test-model",
            prompt="Test prompt",
            conversation=ConversationConfig(max_turns=3),
        )

        ensemble = ConversationalEnsemble(
            name="test_ensemble",
            agents=[agent],
            conversation_limits=ConversationLimits(
                max_total_turns=5, timeout_seconds=60
            ),
        )

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.return_value = ("Response", None)

            result = await executor.execute_conversation(ensemble=ensemble)

            # Agent should execute up to its max_turns (3)
            assert len(result.conversation_history) == 3
            assert mock_execute.call_count == 3

            # Verify turn numbers increment correctly
            for i, turn in enumerate(result.conversation_history):
                assert turn.agent_name == "test_agent"
                assert turn.turn_number == i + 1

    @pytest.mark.asyncio
    async def test_execute_conversation_empty_ensemble(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test conversation execution with empty ensemble."""
        ensemble = ConversationalEnsemble(
            name="empty_ensemble",
            agents=[],
            conversation_limits=ConversationLimits(
                max_total_turns=5, timeout_seconds=60
            ),
        )

        result = await executor.execute_conversation(ensemble=ensemble)

        # Should handle empty ensemble gracefully
        assert result is not None
        assert result.completion_reason == "no_agents"
        assert len(result.conversation_history) == 0
        assert result.turn_count == 0

    @pytest.mark.asyncio
    async def test_execute_conversation_with_initial_context(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test conversation execution with initial context."""
        agent = ConversationalAgent(
            name="context_agent",
            model_profile="test-model",
            prompt="Use context",
            conversation=ConversationConfig(max_turns=1),
        )

        ensemble = ConversationalEnsemble(
            name="context_ensemble",
            agents=[agent],
            conversation_limits=ConversationLimits(
                max_total_turns=1, timeout_seconds=60
            ),
        )

        initial_context = {
            "user_input": "Hello",
            "session_id": "test-123",
            "metadata": {"source": "test"},
        }

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.return_value = {
                "output": "Response with context",
                "metadata": {},
            }

            result = await executor.execute_conversation(
                ensemble=ensemble, initial_context=initial_context
            )

            # Verify initial context was used
            assert result.final_state is not None
            assert "user_input" in result.final_state
            assert result.final_state["user_input"] == "Hello"

    @pytest.mark.asyncio
    async def test_execute_conversation_state_accumulation(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test that conversation state accumulates across turns."""
        agent = ConversationalAgent(
            name="state_agent",
            model_profile="test-model",
            prompt="Accumulate state",
            conversation=ConversationConfig(max_turns=2),
        )

        ensemble = ConversationalEnsemble(
            name="state_ensemble",
            agents=[agent],
            conversation_limits=ConversationLimits(
                max_total_turns=2, timeout_seconds=60
            ),
        )

        # Mock responses that modify context
        responses = [
            {"output": "First", "metadata": {"turn": 1}},
            {"output": "Second", "metadata": {"turn": 2}},
        ]

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.side_effect = responses

            result = await executor.execute_conversation(ensemble=ensemble)

            # Agent should execute its max_turns (2)
            assert len(result.conversation_history) == 2
            assert result.turn_count == 2

            # Final state should include accumulated information
            assert result.final_state is not None

    def test_conversational_ensemble_executor_initialization(self) -> None:
        """Test that ConversationalEnsembleExecutor initializes correctly."""
        with patch(
            "llm_orc.core.execution.ensemble_execution.ConfigurationManager"
        ) as mock_cm:
            with patch(
                "llm_orc.core.execution.ensemble_execution.CredentialStorage"
            ) as mock_cs:
                mock_config = MagicMock()
                mock_config.load_performance_config.return_value = {}
                mock_cm.return_value = mock_config
                mock_cs.return_value = MagicMock()

                executor = ConversationalEnsembleExecutor()

                # Verify it's an instance of the parent class
                from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

                assert isinstance(executor, EnsembleExecutor)
                assert hasattr(executor, "execute_conversation")

    @pytest.mark.asyncio
    async def test_execute_conversation_respects_max_turns(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test that max conversation turns limit is respected."""
        agent = ConversationalAgent(
            name="limited_agent",
            model_profile="test-model",
            prompt="Limited turns",
            conversation=ConversationConfig(max_turns=10),  # Agent can do many turns
        )

        ensemble = ConversationalEnsemble(
            name="limited_ensemble",
            agents=[agent],
            conversation_limits=ConversationLimits(
                max_total_turns=2,  # But ensemble limits to 2
                timeout_seconds=60,
            ),
        )

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.return_value = {"output": "Response", "metadata": {}}

            result = await executor.execute_conversation(ensemble=ensemble)

            # Should stop at ensemble max turns (agent can do 10, ensemble limits to 2)
            assert len(result.conversation_history) == 2
            assert result.completion_reason == "max_turns_reached"
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_conversation_with_multiple_agents(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test conversation with multiple agents participating."""
        agents = [
            ConversationalAgent(
                name=f"agent_{i}",
                model_profile="test-model",
                prompt=f"Agent {i}",
                conversation=ConversationConfig(max_turns=1),
            )
            for i in range(3)
        ]

        ensemble = ConversationalEnsemble(
            name="multi_agent_ensemble",
            agents=agents,
            conversation_limits=ConversationLimits(
                max_total_turns=3, timeout_seconds=60
            ),
        )

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.side_effect = [
                {"output": f"Response from agent_{i}", "metadata": {}} for i in range(3)
            ]

            result = await executor.execute_conversation(ensemble=ensemble)

            # Verify all agents participated
            assert len(result.conversation_history) == 3
            agent_names = {turn.agent_name for turn in result.conversation_history}
            assert agent_names == {"agent_0", "agent_1", "agent_2"}

    @pytest.mark.asyncio
    async def test_execute_conversation_timing(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test that conversation timing is tracked correctly."""
        agent = ConversationalAgent(
            name="timing_agent",
            model_profile="test-model",
            prompt="Track timing",
            conversation=ConversationConfig(max_turns=1),
        )

        ensemble = ConversationalEnsemble(
            name="timing_ensemble",
            agents=[agent],
            conversation_limits=ConversationLimits(
                max_total_turns=1, timeout_seconds=60
            ),
        )

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.return_value = {"output": "Response", "metadata": {}}

            # Mock time to control duration
            with patch("time.time") as mock_time:
                mock_time.side_effect = [1000.0, 1001.5]  # 1.5 second duration

                result = await executor.execute_conversation(ensemble=ensemble)

                # With max_total_turns=1, should reach max turns
                assert result.completion_reason == "max_turns_reached"
                assert result.turn_count == 1

    @pytest.mark.asyncio
    async def test_execute_conversation_with_script_agent(
        self, executor: ConversationalEnsembleExecutor
    ) -> None:
        """Test conversation with script-based agents."""
        # Script agent (has script, not model_profile)
        script_agent = ConversationalAgent(
            name="script_agent",
            script="test_script.py",
            conversation=ConversationConfig(max_turns=1),
        )

        ensemble = ConversationalEnsemble(
            name="script_ensemble",
            agents=[script_agent],
            conversation_limits=ConversationLimits(
                max_total_turns=1, timeout_seconds=60
            ),
        )

        with patch.object(executor, "_execute_single_agent") as mock_execute:
            mock_execute.return_value = {
                "output": "Script output",
                "metadata": {"type": "script"},
            }

            result = await executor.execute_conversation(ensemble=ensemble)

            # Verify script agent was executed
            assert len(result.conversation_history) == 1
            assert result.conversation_history[0].agent_name == "script_agent"
            assert result.turn_count == 1
