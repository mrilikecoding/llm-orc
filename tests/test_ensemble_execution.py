"""Tests for ensemble execution."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.models import ModelInterface
from llm_orc.roles import RoleDefinition


class TestEnsembleExecutor:
    """Test ensemble execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_ensemble(self) -> None:
        """Test executing a simple ensemble with mock agents."""
        # Create ensemble config
        config = EnsembleConfig(
            name="test_ensemble",
            description="Test ensemble",
            agents=[
                {"name": "agent1", "role": "tester", "model": "mock-model"},
                {"name": "agent2", "role": "reviewer", "model": "mock-model"},
            ],
            coordinator={
                "synthesis_prompt": "Combine the results from both agents",
                "output_format": "json",
            },
        )

        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.side_effect = [
            "Agent 1 response: This looks good",
            "Agent 2 response: I found some issues",
        ]
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        # Create role definitions
        role1 = RoleDefinition(name="tester", prompt="You are a tester")
        role2 = RoleDefinition(name="reviewer", prompt="You are a reviewer")

        # Create executor with mock dependencies
        executor = EnsembleExecutor()

        # Create mock synthesis model
        mock_synthesis_model = AsyncMock(spec=ModelInterface)
        mock_synthesis_model.generate_response.return_value = (
            "Synthesis: Combined results from both agents"
        )
        mock_synthesis_model.get_last_usage.return_value = {
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "cost_usd": 0.01,
            "duration_ms": 100,
        }

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
            patch.object(
                executor, "_get_synthesis_model", new_callable=AsyncMock
            ) as mock_get_synthesis_model,
        ):
            mock_load_role.side_effect = [role1, role2]
            mock_load_model.return_value = mock_model
            mock_get_synthesis_model.return_value = mock_synthesis_model

            # Execute ensemble
            result = await executor.execute(config, input_data="Test this code")

        # Verify result structure
        assert result["ensemble"] == "test_ensemble"
        assert result["status"] == "completed"
        assert "input" in result
        assert "results" in result
        assert "synthesis" in result
        assert "metadata" in result

        # Verify agent results
        agent_results = result["results"]
        assert len(agent_results) == 2
        assert "agent1" in agent_results
        assert "agent2" in agent_results

        # Verify metadata
        metadata = result["metadata"]
        assert "duration" in metadata
        assert metadata["agents_used"] == 2

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_different_models(self) -> None:
        """Test executing ensemble with different models per agent."""
        config = EnsembleConfig(
            name="multi_model_ensemble",
            description="Ensemble with different models",
            agents=[
                {"name": "claude_agent", "role": "analyst", "model": "claude-3-sonnet"},
                {"name": "local_agent", "role": "checker", "model": "llama3"},
            ],
            coordinator={
                "synthesis_prompt": "Compare the analysis and check",
                "output_format": "structured",
            },
        )

        # Mock different models
        claude_model = AsyncMock(spec=ModelInterface)
        claude_model.generate_response.return_value = "Claude analysis result"
        claude_model.get_last_usage.return_value = {
            "total_tokens": 40,
            "input_tokens": 25,
            "output_tokens": 15,
            "cost_usd": 0.008,
            "duration_ms": 80,
        }

        llama_model = AsyncMock(spec=ModelInterface)
        llama_model.generate_response.return_value = "Llama check result"
        llama_model.get_last_usage.return_value = {
            "total_tokens": 25,
            "input_tokens": 15,
            "output_tokens": 10,
            "cost_usd": 0.003,
            "duration_ms": 60,
        }

        # Mock role
        analyst_role = RoleDefinition(name="analyst", prompt="Analyze this")
        checker_role = RoleDefinition(name="checker", prompt="Check this")

        executor = EnsembleExecutor()

        # Create mock synthesis model
        mock_synthesis_model = AsyncMock(spec=ModelInterface)
        mock_synthesis_model.generate_response.return_value = (
            "Synthesis: The analysis and check both confirm the feature is solid"
        )
        mock_synthesis_model.get_last_usage.return_value = {
            "total_tokens": 35,
            "input_tokens": 20,
            "output_tokens": 15,
            "cost_usd": 0.006,
            "duration_ms": 70,
        }

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
            patch.object(
                executor, "_get_synthesis_model", new_callable=AsyncMock
            ) as mock_get_synthesis_model,
        ):
            mock_load_role.side_effect = [analyst_role, checker_role]
            mock_load_model.side_effect = [claude_model, llama_model]
            mock_get_synthesis_model.return_value = mock_synthesis_model

            result = await executor.execute(config, input_data="Analyze this feature")

        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert "claude_agent" in result["results"]
        assert "local_agent" in result["results"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_handles_agent_failure(self) -> None:
        """Test that ensemble execution handles individual agent failures."""
        config = EnsembleConfig(
            name="test_ensemble_with_failure",
            description="Test ensemble with one failing agent",
            agents=[
                {"name": "working_agent", "role": "tester", "model": "mock-model"},
                {"name": "failing_agent", "role": "reviewer", "model": "mock-model"},
            ],
            coordinator={
                "synthesis_prompt": "Combine available results",
                "output_format": "json",
            },
        )

        # Create mock models - one works, one fails
        working_model = AsyncMock(spec=ModelInterface)
        working_model.generate_response.return_value = "Working agent response"

        failing_model = AsyncMock(spec=ModelInterface)
        failing_model.generate_response.side_effect = Exception("Model failed")

        # Mock roles
        role = RoleDefinition(name="tester", prompt="You are a tester")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.side_effect = [working_model, failing_model]

            result = await executor.execute(config, input_data="Test input")

        # Should still complete but mark failures
        assert result["status"] == "completed_with_errors"
        assert len(result["results"]) == 2
        assert "working_agent" in result["results"]
        assert "failing_agent" in result["results"]

        # Working agent should have response
        assert "response" in result["results"]["working_agent"]

        # Failing agent should have error
        assert "error" in result["results"]["failing_agent"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_synthesis(self) -> None:
        """Test that ensemble execution includes synthesis of results."""
        config = EnsembleConfig(
            name="synthesis_test",
            description="Test synthesis functionality",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "mock-model"},
            ],
            coordinator={
                "synthesis_prompt": "Summarize the analysis results",
                "output_format": "json",
            },
        )

        # Mock agent response
        agent_model = AsyncMock(spec=ModelInterface)
        agent_model.generate_response.return_value = "Detailed analysis result"

        # Mock synthesis model
        synthesis_model = AsyncMock(spec=ModelInterface)
        synthesis_model.generate_response.return_value = "Synthesized summary"

        role = RoleDefinition(name="analyst", prompt="Analyze")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
            patch.object(
                executor, "_get_synthesis_model", new_callable=AsyncMock
            ) as mock_get_synthesis_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = agent_model
            mock_get_synthesis_model.return_value = synthesis_model

            result = await executor.execute(config, input_data="Test analysis")

        assert result["synthesis"] == "Synthesized summary"

        # Verify synthesis model was called with agent results
        synthesis_model.generate_response.assert_called_once()
        synthesis_call_args = synthesis_model.generate_response.call_args[1]
        assert "Summarize the analysis results" in synthesis_call_args["role_prompt"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_tracks_usage_metrics(self) -> None:
        """Test that ensemble execution tracks LLM usage metrics."""
        config = EnsembleConfig(
            name="usage_tracking_test",
            description="Test usage tracking",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "claude-3-sonnet"},
                {"name": "agent2", "role": "reviewer", "model": "llama3"},
            ],
            coordinator={
                "synthesis_prompt": "Combine results",
                "output_format": "json",
            },
        )

        # Mock models with usage tracking
        claude_model = AsyncMock(spec=ModelInterface)
        claude_model.generate_response.return_value = "Claude response"
        claude_model.get_last_usage.return_value = {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
            "cost_usd": 0.0045,
            "duration_ms": 1200,
            "model": "claude-3-sonnet",
        }

        llama_model = AsyncMock(spec=ModelInterface)
        llama_model.generate_response.return_value = "Llama response"
        llama_model.get_last_usage.return_value = {
            "input_tokens": 45,
            "output_tokens": 80,
            "total_tokens": 125,
            "cost_usd": 0.0,  # Local model, no cost
            "duration_ms": 800,
            "model": "llama3",
        }

        synthesis_model = AsyncMock(spec=ModelInterface)
        synthesis_model.generate_response.return_value = "Synthesized result"
        synthesis_model.get_last_usage.return_value = {
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250,
            "cost_usd": 0.0075,
            "duration_ms": 1500,
            "model": "llama3",
        }

        role = RoleDefinition(name="test", prompt="Test role")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
            patch.object(
                executor, "_get_synthesis_model", new_callable=AsyncMock
            ) as mock_get_synthesis_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.side_effect = [claude_model, llama_model]
            mock_get_synthesis_model.return_value = synthesis_model

            result = await executor.execute(config, input_data="Test usage tracking")

        # Verify usage tracking is included in results
        assert "usage" in result["metadata"]
        usage = result["metadata"]["usage"]

        # Check individual agent usage
        assert "agents" in usage
        agent_usage = usage["agents"]

        assert "agent1" in agent_usage
        assert agent_usage["agent1"]["model"] == "claude-3-sonnet"
        assert agent_usage["agent1"]["input_tokens"] == 50
        assert agent_usage["agent1"]["output_tokens"] == 100
        assert agent_usage["agent1"]["total_tokens"] == 150
        assert agent_usage["agent1"]["cost_usd"] == 0.0045
        assert agent_usage["agent1"]["duration_ms"] == 1200

        assert "agent2" in agent_usage
        assert agent_usage["agent2"]["model"] == "llama3"
        assert agent_usage["agent2"]["total_tokens"] == 125
        assert agent_usage["agent2"]["cost_usd"] == 0.0

        # Check synthesis usage
        assert "synthesis" in usage
        synthesis_usage = usage["synthesis"]
        assert synthesis_usage["total_tokens"] == 250
        assert synthesis_usage["cost_usd"] == 0.0075

        # Check totals
        assert "totals" in usage
        totals = usage["totals"]
        assert totals["total_tokens"] == 525  # 150 + 125 + 250
        assert totals["total_cost_usd"] == 0.012  # 0.0045 + 0.0 + 0.0075
        assert totals["total_duration_ms"] == 3500  # 1200 + 800 + 1500
        assert totals["agents_count"] == 2

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_timeout(self) -> None:
        """Test that ensemble execution respects timeout settings."""
        config = EnsembleConfig(
            name="timeout_test",
            description="Test timeout functionality",
            agents=[
                {"name": "slow_agent", "role": "analyst", "model": "slow-model"},
            ],
            coordinator={
                "synthesis_prompt": "Summarize results",
                "output_format": "json",
                "timeout_seconds": 5,  # 5 second timeout
            },
        )

        # Mock model that takes too long
        slow_model = AsyncMock(spec=ModelInterface)

        async def slow_response(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(10)  # Takes 10 seconds, longer than 5 second timeout
            return "This should timeout"

        slow_model.generate_response = slow_response
        slow_model.get_last_usage.return_value = {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
            "cost_usd": 0.001,
            "duration_ms": 10000,
            "model": "slow-model",
        }

        role = RoleDefinition(name="analyst", prompt="Analyze")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = slow_model

            result = await executor.execute(config, input_data="Test timeout")

        # Should complete with errors due to timeout
        assert result["status"] == "completed_with_errors"
        assert "slow_agent" in result["results"]
        agent_result = result["results"]["slow_agent"]
        assert agent_result["status"] == "failed"
        assert "timed out" in agent_result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_per_agent_timeout(self) -> None:
        """Test that individual agents can have their own timeout settings."""
        config = EnsembleConfig(
            name="per_agent_timeout_test",
            description="Test per-agent timeout functionality",
            agents=[
                {
                    "name": "fast_agent",
                    "role": "analyst",
                    "model": "fast-model",
                    "timeout_seconds": 10,
                },
                {
                    "name": "slow_agent",
                    "role": "reviewer",
                    "model": "slow-model",
                    "timeout_seconds": 2,
                },
            ],
            coordinator={
                "synthesis_prompt": "Combine results",
                "output_format": "json",
            },
        )

        # Fast model
        fast_model = AsyncMock(spec=ModelInterface)
        fast_model.generate_response.return_value = "Fast response"
        fast_model.get_last_usage.return_value = {
            "input_tokens": 20,
            "output_tokens": 30,
            "total_tokens": 50,
            "cost_usd": 0.001,
            "duration_ms": 500,
            "model": "fast-model",
        }

        # Slow model that exceeds its timeout
        slow_model = AsyncMock(spec=ModelInterface)

        async def slow_response(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(5)  # Takes 5 seconds, longer than 2 second timeout
            return "This should timeout"

        slow_model.generate_response = slow_response
        slow_model.get_last_usage.return_value = {
            "input_tokens": 30,
            "output_tokens": 40,
            "total_tokens": 70,
            "cost_usd": 0.002,
            "duration_ms": 5000,
            "model": "slow-model",
        }

        role = RoleDefinition(name="test", prompt="Test")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.side_effect = [fast_model, slow_model]

            result = await executor.execute(config, input_data="Test per-agent timeout")

        # Should complete with errors due to one agent timing out
        assert result["status"] == "completed_with_errors"

        # Fast agent should succeed
        assert result["results"]["fast_agent"]["status"] == "success"
        assert result["results"]["fast_agent"]["response"] == "Fast response"

        # Slow agent should fail with timeout
        assert result["results"]["slow_agent"]["status"] == "failed"
        assert "timed out" in result["results"]["slow_agent"]["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_ensemble_with_synthesis_timeout(self) -> None:
        """Test that synthesis step respects timeout settings."""
        config = EnsembleConfig(
            name="synthesis_timeout_test",
            description="Test synthesis timeout",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "fast-model"},
            ],
            coordinator={
                "synthesis_prompt": "Synthesize results",
                "output_format": "json",
                "synthesis_timeout_seconds": 3,
            },
        )

        # Fast agent
        fast_model = AsyncMock(spec=ModelInterface)
        fast_model.generate_response.return_value = "Agent response"
        fast_model.get_last_usage.return_value = {
            "input_tokens": 20,
            "output_tokens": 30,
            "total_tokens": 50,
            "cost_usd": 0.001,
            "duration_ms": 500,
            "model": "fast-model",
        }

        # Slow synthesis model
        slow_synthesis_model = AsyncMock(spec=ModelInterface)

        async def slow_synthesis(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(5)  # Takes 5 seconds, longer than 3 second timeout
            return "This should timeout"

        slow_synthesis_model.generate_response = slow_synthesis
        slow_synthesis_model.get_last_usage.return_value = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.003,
            "duration_ms": 5000,
            "model": "synthesis-model",
        }

        role = RoleDefinition(name="analyst", prompt="Analyze")

        executor = EnsembleExecutor()

        # Mock the role and model loading methods
        with (
            patch.object(
                executor, "_load_role", new_callable=AsyncMock
            ) as mock_load_role,
            patch.object(
                executor, "_load_model", new_callable=AsyncMock
            ) as mock_load_model,
            patch.object(
                executor, "_get_synthesis_model", new_callable=AsyncMock
            ) as mock_get_synthesis_model,
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = fast_model
            mock_get_synthesis_model.return_value = slow_synthesis_model

            result = await executor.execute(config, input_data="Test synthesis timeout")

        # Should complete with errors due to synthesis timeout
        assert result["status"] == "completed_with_errors"
        assert result["results"]["agent1"]["status"] == "success"
        assert (
            "synthesis failed" in result["synthesis"].lower()
            or "timeout" in result["synthesis"].lower()
        )
