"""Tests for agent orchestrator."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.agent_orchestrator import AgentOrchestrator
from llm_orc.core.execution.input_enhancer import InputEnhancer


class TestAgentOrchestrator:
    """Test agent orchestration functionality."""

    def setup_orchestrator(self) -> tuple[AgentOrchestrator, dict[str, Mock | AsyncMock]]:
        """Set up orchestrator with mocked dependencies."""
        mocks = {
            "performance_config": {"execution": {"default_timeout": 60}},
            "event_emitter": Mock(),
            "config_resolver": AsyncMock(),
            "agent_executor": AsyncMock(),
            "input_enhancer": Mock(spec=InputEnhancer),
            "results_processor": Mock(),
        }
        
        orchestrator = AgentOrchestrator(
            performance_config=mocks["performance_config"],
            event_emitter=mocks["event_emitter"],
            config_resolver=mocks["config_resolver"],
            agent_executor=mocks["agent_executor"],
            input_enhancer=mocks["input_enhancer"],
            results_processor=mocks["results_processor"],
        )
        
        return orchestrator, mocks

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_empty_list(self) -> None:
        """Test parallel execution with empty agent list."""
        orchestrator, mocks = self.setup_orchestrator()
        
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        success = await orchestrator.execute_agents_parallel(
            [], "test input", results_dict, agent_usage
        )
        
        assert success is True
        assert results_dict == {}
        assert agent_usage == {}
        mocks["results_processor"].assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_single_success(self) -> None:
        """Test parallel execution with single successful agent."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].return_value = {
            "name": "agent1", 
            "timeout_seconds": 30
        }
        mocks["agent_executor"].return_value = ("Agent response", None)
        mocks["input_enhancer"].get_agent_input.return_value = "Enhanced input"
        
        agents = [{"name": "agent1", "role": "test", "model": "mock"}]
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        success = await orchestrator.execute_agents_parallel(
            agents, "test input", results_dict, agent_usage
        )
        
        # Verify calls
        mocks["config_resolver"].assert_called_once()
        mocks["agent_executor"].assert_called_once_with(
            {"name": "agent1", "role": "test", "model": "mock"},
            "Enhanced input",
            30
        )
        mocks["input_enhancer"].get_agent_input.assert_called_once_with(
            "test input", "agent1"
        )
        
        # Verify events emitted
        assert mocks["event_emitter"].call_count >= 2  # started + completed
        event_calls = mocks["event_emitter"].call_args_list
        assert event_calls[0][0][0] == "agent_started"
        assert event_calls[1][0][0] == "agent_completed"
        
        # Verify results processing
        mocks["results_processor"].assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_multiple_agents(self) -> None:
        """Test parallel execution with multiple agents."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].side_effect = [
            {"name": "agent1", "timeout_seconds": 30},
            {"name": "agent2", "timeout_seconds": 45},
        ]
        mocks["agent_executor"].side_effect = [
            ("Response 1", None),
            ("Response 2", None),
        ]
        mocks["input_enhancer"].get_agent_input.side_effect = [
            "Enhanced input 1",
            "Enhanced input 2",
        ]
        
        agents = [
            {"name": "agent1", "role": "test", "model": "mock"},
            {"name": "agent2", "role": "test", "model": "mock"},
        ]
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        success = await orchestrator.execute_agents_parallel(
            agents, "test input", results_dict, agent_usage
        )
        
        # Verify both agents were processed
        assert mocks["config_resolver"].call_count == 2
        assert mocks["agent_executor"].call_count == 2
        
        # Verify events for both agents
        assert mocks["event_emitter"].call_count >= 4  # 2 started + 2 completed
        
        mocks["results_processor"].assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_with_agent_failure(self) -> None:
        """Test parallel execution when one agent fails."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks - first succeeds, second fails
        mocks["config_resolver"].side_effect = [
            {"name": "agent1", "timeout_seconds": 30},
            {"name": "agent2", "timeout_seconds": 30},
        ]
        mocks["agent_executor"].side_effect = [
            ("Response 1", None),
            RuntimeError("Agent 2 failed"),
        ]
        mocks["input_enhancer"].get_agent_input.side_effect = [
            "Enhanced input 1",
            "Enhanced input 2",
        ]
        
        agents = [
            {"name": "agent1", "role": "test", "model": "mock"},
            {"name": "agent2", "role": "test", "model": "mock"},
        ]
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        success = await orchestrator.execute_agents_parallel(
            agents, "test input", results_dict, agent_usage
        )
        
        # Verify both agents were attempted
        assert mocks["config_resolver"].call_count == 2
        assert mocks["agent_executor"].call_count == 2
        
        # Verify error events emitted
        event_calls = mocks["event_emitter"].call_args_list
        error_events = [call for call in event_calls if "error" in str(call)]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_with_dict_input(self) -> None:
        """Test parallel execution with dictionary input (dependency-based)."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].return_value = {
            "name": "agent1", 
            "timeout_seconds": 30
        }
        mocks["agent_executor"].return_value = ("Agent response", None)
        
        agents = [{"name": "agent1", "role": "test", "model": "mock"}]
        input_data = {"agent1": "Specific input for agent1"}
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        success = await orchestrator.execute_agents_parallel(
            agents, input_data, results_dict, agent_usage
        )
        
        # Should use dict input directly, not call input_enhancer
        mocks["input_enhancer"].get_agent_input.assert_not_called()
        
        # Verify agent executor was called with dict input
        mocks["agent_executor"].assert_called_once_with(
            {"name": "agent1", "role": "test", "model": "mock"},
            "Specific input for agent1",
            30
        )

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_with_default_timeout(self) -> None:
        """Test parallel execution uses default timeout when not specified."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks - no timeout in config
        mocks["config_resolver"].return_value = {"name": "agent1"}
        mocks["agent_executor"].return_value = ("Agent response", None)
        mocks["input_enhancer"].get_agent_input.return_value = "Enhanced input"
        
        agents = [{"name": "agent1", "role": "test", "model": "mock"}]
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        success = await orchestrator.execute_agents_parallel(
            agents, "test input", results_dict, agent_usage
        )
        
        # Should use default timeout of 60
        mocks["agent_executor"].assert_called_once_with(
            {"name": "agent1", "role": "test", "model": "mock"},
            "Enhanced input",
            60  # Default timeout
        )

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_gather_exception(self) -> None:
        """Test parallel execution handles gather exceptions."""
        orchestrator, mocks = self.setup_orchestrator()
        
        agents = [{"name": "agent1", "role": "test", "model": "mock"}]
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        
        # Mock asyncio.gather to raise an exception
        with pytest.MonkeyPatch.context() as mp:
            async def failing_gather(*tasks, return_exceptions=True):
                raise RuntimeError("Gather operation failed")
            
            mp.setattr("asyncio.gather", failing_gather)
            
            success = await orchestrator.execute_agents_parallel(
                agents, "test input", results_dict, agent_usage
            )
        
        # Should return False and mark agent as failed
        assert success is False
        assert "agent1" in results_dict
        assert results_dict["agent1"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_agent_task_success(self) -> None:
        """Test individual agent task execution success."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].return_value = {
            "name": "agent1", 
            "timeout_seconds": 30
        }
        mocks["agent_executor"].return_value = ("Success response", None)
        mocks["input_enhancer"].get_agent_input.return_value = "Enhanced input"
        
        agent_config = {"name": "agent1", "role": "test", "model": "mock"}
        
        agent_name, result = await orchestrator._execute_agent_task(
            agent_config, "test input"
        )
        
        assert agent_name == "agent1"
        assert result == ("Success response", None)
        
        # Verify started and completed events
        assert mocks["event_emitter"].call_count == 2
        event_calls = mocks["event_emitter"].call_args_list
        assert event_calls[0][0][0] == "agent_started"
        assert event_calls[1][0][0] == "agent_completed"
        assert "duration_ms" in event_calls[1][0][1]

    @pytest.mark.asyncio
    async def test_execute_agent_task_with_exception(self) -> None:
        """Test individual agent task execution with exception."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].return_value = {
            "name": "agent1", 
            "timeout_seconds": 30
        }
        mocks["agent_executor"].side_effect = RuntimeError("Agent execution failed")
        mocks["input_enhancer"].get_agent_input.return_value = "Enhanced input"
        
        agent_config = {"name": "agent1", "role": "test", "model": "mock"}
        
        agent_name, result = await orchestrator._execute_agent_task(
            agent_config, "test input"
        )
        
        assert agent_name == "agent1"
        assert isinstance(result, RuntimeError)
        assert str(result) == "Agent execution failed"
        
        # Verify started and error completion events
        assert mocks["event_emitter"].call_count == 2
        event_calls = mocks["event_emitter"].call_args_list
        assert event_calls[0][0][0] == "agent_started"
        assert event_calls[1][0][0] == "agent_completed"
        assert "error" in event_calls[1][0][1]

    @pytest.mark.asyncio
    async def test_execute_agent_task_with_dict_input(self) -> None:
        """Test individual agent task with dictionary input."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].return_value = {
            "name": "agent1", 
            "timeout_seconds": 30
        }
        mocks["agent_executor"].return_value = ("Success response", None)
        
        agent_config = {"name": "agent1", "role": "test", "model": "mock"}
        input_data = {"agent1": "Specific input", "agent2": "Other input"}
        
        agent_name, result = await orchestrator._execute_agent_task(
            agent_config, input_data
        )
        
        # Should not call input enhancer for dict input
        mocks["input_enhancer"].get_agent_input.assert_not_called()
        
        # Should use specific input for agent1
        mocks["agent_executor"].assert_called_once_with(
            agent_config, "Specific input", 30
        )

    @pytest.mark.asyncio
    async def test_execute_agent_task_missing_dict_key(self) -> None:
        """Test individual agent task with dict input missing agent key."""
        orchestrator, mocks = self.setup_orchestrator()
        
        # Setup mocks
        mocks["config_resolver"].return_value = {
            "name": "agent1", 
            "timeout_seconds": 30
        }
        mocks["agent_executor"].return_value = ("Success response", None)
        
        agent_config = {"name": "agent1", "role": "test", "model": "mock"}
        input_data = {"agent2": "Other input"}  # Missing agent1 key
        
        agent_name, result = await orchestrator._execute_agent_task(
            agent_config, input_data
        )
        
        # Should use empty string for missing key
        mocks["agent_executor"].assert_called_once_with(
            agent_config, "", 30
        )