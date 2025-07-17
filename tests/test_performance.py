"""Performance tests for agent orchestration."""

import time
from typing import Any
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


class TestEnsembleExecutionPerformance:
    """Test ensemble execution performance requirements."""

    @pytest.mark.asyncio
    async def test_parallel_execution_performance_improvement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should execute independent agents in parallel for significant speedup."""
        from llm_orc.ensemble_config import EnsembleConfig
        from llm_orc.ensemble_execution import EnsembleExecutor

        # Arrange - Create ensemble with 3 independent agents (no dependencies)
        agent_configs: list[dict[str, str]] = []

        for i in range(3):
            agent_config = {
                "name": f"agent_{i}",
                "model": f"mock-model-{i}",
                "provider": "mock",
            }
            agent_configs.append(agent_config)

        config = EnsembleConfig(
            name="parallel-test-ensemble",
            description="Test parallel execution performance",
            agents=agent_configs,
            coordinator={"synthesis_prompt": "Combine all results"},
        )

        executor = EnsembleExecutor()

        # Mock the model loading to use fast mock models
        fast_mock_model = AsyncMock(spec=ModelInterface)
        fast_mock_model.generate_response.return_value = "Fast mock response"
        fast_mock_model.get_last_usage.return_value = {
            "total_tokens": 10,
            "input_tokens": 5,
            "output_tokens": 5,
            "cost_usd": 0.001,
            "duration_ms": 1,
        }

        mock_load_model = AsyncMock(return_value=fast_mock_model)
        monkeypatch.setattr(executor, "_load_model", mock_load_model)

        # Act - Measure execution time for parallel-capable ensemble
        start_time = time.perf_counter()

        result = await executor.execute(config, "Test input for parallel execution")

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Assert - Should complete in reasonable time with mock models
        # This test will initially fail because current implementation is sequential
        assert result["status"] in ["completed", "completed_with_errors"]
        assert execution_time_ms < 100.0, (
            f"Parallel ensemble execution took {execution_time_ms:.2f}ms, "
            f"should be under 100ms with mock models"
        )

        # Verify all agents executed
        assert len(result["results"]) == 3
        for i in range(3):
            assert f"agent_{i}" in result["results"]

    @pytest.mark.asyncio
    async def test_dependency_aware_execution_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should execute agents in correct order when dependencies exist."""
        from llm_orc.ensemble_config import EnsembleConfig
        from llm_orc.ensemble_execution import EnsembleExecutor

        # Arrange - Create ensemble where synthesizer depends on 3 reviewers
        agent_configs: list[dict[str, Any]] = [
            # Independent agents (should run in parallel)
            {"name": "security_reviewer", "model": "mock-security", "provider": "mock"},
            {"name": "performance_reviewer", "model": "mock-perf", "provider": "mock"},
            {"name": "style_reviewer", "model": "mock-style", "provider": "mock"},
            # Dependent agent (should run after the above 3)
            {
                "name": "synthesizer",
                "model": "mock-synthesizer",
                "provider": "mock",
                "dependencies": [
                    "security_reviewer",
                    "performance_reviewer", 
                    "style_reviewer"
                ],
            },
        ]

        config = EnsembleConfig(
            name="dependency-test-ensemble",
            description="Test dependency-aware execution",
            agents=agent_configs,
            coordinator={"synthesis_prompt": "Final coordination"},
        )

        executor = EnsembleExecutor()

        # Track execution order with mock that records call times
        execution_times: dict[str, float] = {}
        call_count = 0

        def track_execution_time(model_name: str) -> str:
            nonlocal call_count
            call_count += 1
            execution_times[model_name] = time.perf_counter()
            return f"Response from {model_name}"

        # Mock the model loading to track execution order
        fast_mock_model = AsyncMock(spec=ModelInterface)
        fast_mock_model.generate_response.side_effect = lambda message, role_prompt=None: track_execution_time(
            fast_mock_model._model_name
        )
        fast_mock_model.get_last_usage.return_value = {
            "total_tokens": 10,
            "input_tokens": 5,
            "output_tokens": 5,
            "cost_usd": 0.001,
            "duration_ms": 1,
        }

        def create_tracked_model(model_name: str) -> AsyncMock:
            mock = AsyncMock(spec=ModelInterface)
            mock._model_name = model_name  # Store model name for tracking
            mock.generate_response.side_effect = lambda message, role_prompt=None: track_execution_time(model_name)
            mock.get_last_usage.return_value = fast_mock_model.get_last_usage.return_value
            return mock

        # Mock model loading to return tracked models
        async def mock_load_model(model_name: str, provider: str | None = None) -> AsyncMock:
            return create_tracked_model(model_name)

        monkeypatch.setattr(executor, "_load_model", mock_load_model)

        # Act - Execute ensemble with dependencies
        time.perf_counter()
        result = await executor.execute(config, "Test dependency execution")
        time.perf_counter()

        # Assert - Verify execution order and timing
        assert result["status"] in ["completed", "completed_with_errors"]

        # All agents should have executed
        assert len(result["results"]) == 4
        expected_agents = ["security_reviewer", "performance_reviewer", "style_reviewer", "synthesizer"]
        for agent_name in expected_agents:
            assert agent_name in result["results"]

        # Critical test: synthesizer should execute AFTER all its dependencies
        synthesizer_time = execution_times.get("mock-synthesizer")
        security_time = execution_times.get("mock-security")
        perf_time = execution_times.get("mock-perf")
        style_time = execution_times.get("mock-style")

        assert synthesizer_time is not None, "Synthesizer should have executed"
        assert security_time is not None, "Security reviewer should have executed"
        assert perf_time is not None, "Performance reviewer should have executed"
        assert style_time is not None, "Style reviewer should have executed"

        # Synthesizer must execute after ALL dependencies complete
        assert synthesizer_time > security_time, "Synthesizer must execute after security reviewer"
        assert synthesizer_time > perf_time, "Synthesizer must execute after performance reviewer"
        assert synthesizer_time > style_time, "Synthesizer must execute after style reviewer"

        # Debug: Print execution times
        print("\nExecution times:")
        print(f"  Security: {security_time:.6f}")
        print(f"  Performance: {perf_time:.6f}")
        print(f"  Style: {style_time:.6f}")
        print(f"  Synthesizer: {synthesizer_time:.6f}")

        # Independent agents should execute in parallel (similar start times)
        independent_times = [security_time, perf_time, style_time]
        time_spread = max(independent_times) - min(independent_times)
        print(f"  Time spread between independent agents: {time_spread:.6f}s")

        # Critical test: Independent agents should run nearly simultaneously (parallel)
        # But current implementation runs them sequentially - this should FAIL
        assert time_spread < 0.001, (
            f"Independent agents should run in parallel (time_spread < 0.001s), "
            f"but got {time_spread:.6f}s. Current implementation is fully sequential!"
        )

        # Synthesizer should start after dependencies, but only slightly after latest independent
        latest_independent_time = max(independent_times)
        time_gap = synthesizer_time - latest_independent_time
        print(f"  Time gap between latest independent and synthesizer: {time_gap:.6f}s")

        # This should pass once we implement proper dependency analysis
        assert time_gap > 0.001, (
            f"Synthesizer should start after dependencies complete. "
            f"Gap: {time_gap:.6f}s (should be > 0.001s)"
        )
