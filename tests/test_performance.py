"""Performance tests for agent orchestration."""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.models import ModelInterface
from llm_orc.orchestration import Agent, ConversationOrchestrator
from llm_orc.roles import RoleDefinition


class TestMessageRoutingPerformance:
    """Test message routing performance requirements."""

    @pytest.mark.asyncio
    async def test_message_routing_latency_under_50ms(self) -> None:
        """Should route messages between agents in under 50ms."""
        # Arrange - Create fast mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Fast response"

        role = RoleDefinition(
            name="test_agent", prompt="You are a test agent that responds quickly."
        )

        agent1 = Agent("agent1", role, mock_model)
        agent2 = Agent("agent2", role, mock_model)

        orchestrator = ConversationOrchestrator()
        # Mock message delivery to avoid async timeout issues
        orchestrator.message_protocol.deliver_message = AsyncMock()  # type: ignore[method-assign]

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Start conversation
        conversation_id = await orchestrator.start_conversation(
            participants=["agent1", "agent2"], topic="Performance Test"
        )

        # Act - Measure message routing time
        start_time = time.perf_counter()

        response = await orchestrator.send_agent_message(
            sender="agent1",
            recipient="agent2",
            content="Hello, how are you?",
            conversation_id=conversation_id,
        )

        end_time = time.perf_counter()
        routing_time_ms = (end_time - start_time) * 1000

        # Assert - Should be under 50ms
        assert response == "Fast response"
        assert routing_time_ms < 50.0, (
            f"Message routing took {routing_time_ms:.2f}ms, should be under 50ms"
        )

        # Verify agent responded
        mock_model.generate_response.assert_called_once()
        assert len(agent2.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_multi_agent_conversation_performance(self) -> None:
        """Should handle multi-agent conversations efficiently."""
        # Arrange - Create 3 agents with fast mock models
        agents = []
        for i in range(3):
            mock_model = AsyncMock(spec=ModelInterface)
            mock_model.generate_response.return_value = f"Response from agent {i + 1}"

            role = RoleDefinition(
                name=f"agent_{i + 1}", prompt=f"You are agent {i + 1}."
            )

            agent = Agent(f"agent_{i + 1}", role, mock_model)
            agents.append(agent)

        orchestrator = ConversationOrchestrator()
        orchestrator.message_protocol.deliver_message = AsyncMock()  # type: ignore[method-assign]

        for agent in agents:
            orchestrator.register_agent(agent)

        conversation_id = await orchestrator.start_conversation(
            participants=["agent_1", "agent_2", "agent_3"],
            topic="Multi-Agent Performance Test",
        )

        # Act - Measure conversation with multiple message exchanges
        start_time = time.perf_counter()

        # Round 1: agent_1 -> agent_2
        await orchestrator.send_agent_message(
            sender="agent_1",
            recipient="agent_2",
            content="Hello agent 2",
            conversation_id=conversation_id,
        )

        # Round 2: agent_2 -> agent_3
        await orchestrator.send_agent_message(
            sender="agent_2",
            recipient="agent_3",
            content="Hello agent 3",
            conversation_id=conversation_id,
        )

        # Round 3: agent_3 -> agent_1
        await orchestrator.send_agent_message(
            sender="agent_3",
            recipient="agent_1",
            content="Hello agent 1",
            conversation_id=conversation_id,
        )

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Assert - 3 message exchanges should complete quickly
        assert total_time_ms < 150.0, (
            f"Multi-agent conversation took {total_time_ms:.2f}ms, "
            f"should be under 150ms"
        )

        # Verify all agents participated
        for agent in agents:
            assert len(agent.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_agent_response_generation_performance(self) -> None:
        """Should generate agent responses efficiently."""
        # Arrange
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = "Quick response"

        role = RoleDefinition(
            name="performance_agent", prompt="You are a performance test agent."
        )

        agent = Agent("performance_agent", role, mock_model)

        # Act - Measure response generation time
        start_time = time.perf_counter()

        response = await agent.respond_to_message("Test message")

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Assert - Should be very fast with mock model
        assert response == "Quick response"
        assert response_time_ms < 10.0, (
            f"Response generation took {response_time_ms:.2f}ms, should be under 10ms"
        )
        assert len(agent.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_agent_registration_performance(self) -> None:
        """Should register agents efficiently."""
        # Arrange
        orchestrator = ConversationOrchestrator()
        agents = []

        # Create 10 agents
        for i in range(10):
            mock_model = Mock(spec=ModelInterface)
            mock_model.name = f"test-model-{i}"

            role = RoleDefinition(name=f"agent_{i}", prompt=f"You are agent {i}.")

            agent = Agent(f"agent_{i}", role, mock_model)
            agents.append(agent)

        # Act - Measure registration time
        start_time = time.perf_counter()

        for agent in agents:
            orchestrator.register_agent(agent)

        end_time = time.perf_counter()
        registration_time_ms = (end_time - start_time) * 1000

        # Assert - Should register quickly
        assert registration_time_ms < 10.0, (
            f"Registering 10 agents took {registration_time_ms:.2f}ms, "
            "should be under 10ms"
        )
        assert len(orchestrator.agents) == 10

        # Verify all agents are registered
        for i in range(10):
            assert f"agent_{i}" in orchestrator.agents


class TestPRReviewPerformance:
    """Test PR review orchestration performance."""

    @pytest.mark.asyncio
    async def test_pr_review_orchestration_performance(self) -> None:
        """Should orchestrate PR reviews efficiently."""
        # Arrange - Create PR review orchestrator with fast mock agents
        from llm_orc.orchestration import PRReviewOrchestrator

        pr_orchestrator = PRReviewOrchestrator()

        # Create 3 fast reviewer agents
        reviewers = []
        for _, specialty in enumerate(["senior_dev", "security_expert", "ux_reviewer"]):
            mock_model = AsyncMock(spec=ModelInterface)
            mock_model.generate_response.return_value = f"Fast review from {specialty}"

            role = RoleDefinition(
                name=specialty,
                prompt=f"You are a {specialty} reviewer.",
                context={"specialties": [specialty]},
            )

            agent = Agent(specialty, role, mock_model)
            reviewers.append(agent)
            pr_orchestrator.register_reviewer(agent)

        # Mock PR data
        pr_data = {
            "title": "Performance test PR",
            "description": "Testing PR review performance",
            "diff": "Simple diff content",
            "files_changed": ["test.py"],
            "additions": 10,
            "deletions": 5,
        }

        # Act - Measure PR review time
        start_time = time.perf_counter()

        review_results = await pr_orchestrator.review_pr(pr_data)

        end_time = time.perf_counter()
        review_time_ms = (end_time - start_time) * 1000

        # Assert - Should complete review quickly with mock models
        assert review_time_ms < 100.0, (
            f"PR review took {review_time_ms:.2f}ms, should be under 100ms"
        )
        assert len(review_results["reviews"]) == 3
        assert review_results["total_reviewers"] == 3

        # Verify all reviewers were called
        for reviewer in reviewers:
            assert len(reviewer.conversation_history) == 1
