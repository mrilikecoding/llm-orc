"""Tests for ensemble execution."""

import json
from unittest.mock import AsyncMock

import pytest

from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.models import ModelInterface
from llm_orc.roles import RoleDefinition


class TestEnsembleExecutor:
    """Test ensemble execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_ensemble(self):
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
                "output_format": "json"
            }
        )
        
        # Create mock model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.side_effect = [
            "Agent 1 response: This looks good",
            "Agent 2 response: I found some issues"
        ]
        
        # Create role definitions
        role1 = RoleDefinition(name="tester", prompt="You are a tester")
        role2 = RoleDefinition(name="reviewer", prompt="You are a reviewer")
        
        # Create executor with mock dependencies
        executor = EnsembleExecutor()
        
        # Mock the role and model loading
        executor._load_role = AsyncMock(side_effect=[role1, role2])
        executor._load_model = AsyncMock(return_value=mock_model)
        
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
    async def test_execute_ensemble_with_different_models(self):
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
                "output_format": "structured"
            }
        )
        
        # Mock different models
        claude_model = AsyncMock(spec=ModelInterface)
        claude_model.generate_response.return_value = "Claude analysis result"
        
        llama_model = AsyncMock(spec=ModelInterface)
        llama_model.generate_response.return_value = "Llama check result"
        
        # Mock role
        analyst_role = RoleDefinition(name="analyst", prompt="Analyze this")
        checker_role = RoleDefinition(name="checker", prompt="Check this")
        
        executor = EnsembleExecutor()
        executor._load_role = AsyncMock(side_effect=[analyst_role, checker_role])
        executor._load_model = AsyncMock(side_effect=[claude_model, llama_model])
        
        result = await executor.execute(config, input_data="Analyze this feature")
        
        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert "claude_agent" in result["results"]
        assert "local_agent" in result["results"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_handles_agent_failure(self):
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
                "output_format": "json"
            }
        )
        
        # Create mock models - one works, one fails
        working_model = AsyncMock(spec=ModelInterface)
        working_model.generate_response.return_value = "Working agent response"
        
        failing_model = AsyncMock(spec=ModelInterface)
        failing_model.generate_response.side_effect = Exception("Model failed")
        
        # Mock roles
        role = RoleDefinition(name="tester", prompt="You are a tester")
        
        executor = EnsembleExecutor()
        executor._load_role = AsyncMock(return_value=role)
        executor._load_model = AsyncMock(side_effect=[working_model, failing_model])
        
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
    async def test_execute_ensemble_synthesis(self):
        """Test that ensemble execution includes synthesis of results."""
        config = EnsembleConfig(
            name="synthesis_test",
            description="Test synthesis functionality",
            agents=[
                {"name": "agent1", "role": "analyst", "model": "mock-model"},
            ],
            coordinator={
                "synthesis_prompt": "Summarize the analysis results",
                "output_format": "json"
            }
        )
        
        # Mock agent response
        agent_model = AsyncMock(spec=ModelInterface)
        agent_model.generate_response.return_value = "Detailed analysis result"
        
        # Mock synthesis model
        synthesis_model = AsyncMock(spec=ModelInterface)
        synthesis_model.generate_response.return_value = "Synthesized summary"
        
        role = RoleDefinition(name="analyst", prompt="Analyze")
        
        executor = EnsembleExecutor()
        executor._load_role = AsyncMock(return_value=role)
        executor._load_model = AsyncMock(return_value=agent_model)
        executor._get_synthesis_model = AsyncMock(return_value=synthesis_model)
        
        result = await executor.execute(config, input_data="Test analysis")
        
        assert result["synthesis"] == "Synthesized summary"
        
        # Verify synthesis model was called with agent results
        synthesis_model.generate_response.assert_called_once()
        synthesis_call_args = synthesis_model.generate_response.call_args[1]
        assert "Summarize the analysis results" in synthesis_call_args["role_prompt"]

    @pytest.mark.asyncio
    async def test_execute_ensemble_tracks_usage_metrics(self):
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
                "output_format": "json"
            }
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
            "model": "claude-3-sonnet"
        }
        
        llama_model = AsyncMock(spec=ModelInterface)
        llama_model.generate_response.return_value = "Llama response"
        llama_model.get_last_usage.return_value = {
            "input_tokens": 45,
            "output_tokens": 80,
            "total_tokens": 125,
            "cost_usd": 0.0,  # Local model, no cost
            "duration_ms": 800,
            "model": "llama3"
        }
        
        synthesis_model = AsyncMock(spec=ModelInterface)
        synthesis_model.generate_response.return_value = "Synthesized result"
        synthesis_model.get_last_usage.return_value = {
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250,
            "cost_usd": 0.0075,
            "duration_ms": 1500,
            "model": "llama3"
        }
        
        role = RoleDefinition(name="test", prompt="Test role")
        
        executor = EnsembleExecutor()
        executor._load_role = AsyncMock(return_value=role)
        executor._load_model = AsyncMock(side_effect=[claude_model, llama_model])
        executor._get_synthesis_model = AsyncMock(return_value=synthesis_model)
        
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