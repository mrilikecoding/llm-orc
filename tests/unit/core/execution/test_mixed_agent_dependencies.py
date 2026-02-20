"""Tests for mixed script-LLM agent dependencies.

Tests the core issue where script→LLM→script workflows fail
due to type-based phased execution instead of dependency-based execution.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


@pytest.fixture(autouse=True)
def mock_expensive_dependencies() -> Generator[None, None, None]:
    """Mock expensive dependencies for mixed agent dependency tests."""
    with patch(
        "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_config"
    ):
        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_ensembles"
        ):
            with patch(
                "llm_orc.core.config.config_manager.ConfigurationManager._copy_profile_templates"
            ):
                yield


class TestMixedAgentDependencies:
    """Test mixed script-LLM agent dependency chains."""

    @pytest.mark.asyncio
    async def test_script_to_llm_to_script_dependency_chain(self) -> None:
        """Test script→LLM→script workflow respects dependency order.

        RED: This test should fail until we implement dependency-based execution.
        The current phased execution runs all scripts first, then all LLMs,
        which breaks the dependency chain.
        """
        config = EnsembleConfig(
            name="script_llm_script_chain",
            description="Test script→LLM→script dependency chain",
            agents=[
                ScriptAgentConfig(
                    name="data_collector",
                    script='echo "Initial data: user_count=100"',
                ),
                LlmAgentConfig(
                    name="data_analyzer",
                    model_profile="test-analyzer",
                    depends_on=["data_collector"],
                ),
                ScriptAgentConfig(
                    name="report_generator",
                    # Script receives JSON with dependencies dict via stdin
                    # Uses Python to parse and extract dependency data
                    script=(
                        'python3 -c "'
                        "import sys,json; "
                        "d=json.load(sys.stdin); "
                        "deps=d.get('dependencies',{}); "
                        "r=deps.get('data_analyzer',{}).get('response','EMPTY'); "
                        "print(json.dumps({'success':True,'data':'Report: '+r}))"
                        '"'
                    ),
                    depends_on=["data_analyzer"],
                ),
            ],
        )

        # Mock the LLM model response
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = (
            "Analysis: 100 users is above average"
        )
        mock_model.get_last_usage.return_value = {
            "total_tokens": 30,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.005,
            "duration_ms": 50,
        }

        role = RoleDefinition(name="analyzer", prompt="Analyze the data")
        executor = EnsembleExecutor()

        # Mock dependencies
        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch.object(executor, "_artifact_manager"),
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            result = await executor.execute(config, "Test dependency chain")

        # Verify execution order was respected
        assert result["status"] == "completed"
        assert len(result["results"]) == 3

        # Verify all agents executed successfully
        assert result["results"]["data_collector"]["status"] == "success"
        assert result["results"]["data_analyzer"]["status"] == "success"
        assert result["results"]["report_generator"]["status"] == "success"

        # The key test: verify the dependency chain was followed
        # The LLM analyzer should have received the script collector's output
        # This will fail with current phased execution because scripts run first
        analyzer_response = result["results"]["data_analyzer"]["response"]
        assert "Analysis: 100 users is above average" == analyzer_response

        # The report generator should have received the analyzer's output
        # Script agents now receive JSON with dependencies dict
        report_response = result["results"]["report_generator"]["response"]
        assert "Report:" in report_response
        assert "100 users" in report_response

    @pytest.mark.asyncio
    async def test_parallel_script_agents_with_llm_dependency(self) -> None:
        """Test parallel script agents that both feed into an LLM.

        RED: This test should fail due to false circular dependency detection.
        """
        config = EnsembleConfig(
            name="parallel_script_to_llm",
            description="Parallel scripts feeding LLM",
            agents=[
                ScriptAgentConfig(
                    name="metrics_collector",
                    script='echo "metrics: cpu=50%"',
                ),
                ScriptAgentConfig(
                    name="logs_collector",
                    script='echo "logs: 10 errors found"',
                ),
                LlmAgentConfig(
                    name="system_analyzer",
                    model_profile="test-analyzer",
                    depends_on=["metrics_collector", "logs_collector"],
                ),
                ScriptAgentConfig(
                    name="alert_generator",
                    script='echo "Alert: $1"',
                    depends_on=["system_analyzer"],
                ),
            ],
        )

        # Mock the LLM model
        mock_model = AsyncMock(spec=ModelInterface)
        mock_model.generate_response.return_value = (
            "CRITICAL: High CPU with errors detected"
        )
        mock_model.get_last_usage.return_value = {
            "total_tokens": 40,
            "input_tokens": 25,
            "output_tokens": 15,
            "cost_usd": 0.008,
            "duration_ms": 80,
        }

        role = RoleDefinition(name="analyzer", prompt="Analyze system status")
        executor = EnsembleExecutor()

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ) as mock_load_role,
            patch.object(
                executor._model_factory,
                "load_model_from_agent_config",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch.object(executor, "_artifact_manager"),
        ):
            mock_load_role.return_value = role
            mock_load_model.return_value = mock_model

            result = await executor.execute(config, "Monitor system")

        # Should complete successfully
        assert result["status"] == "completed"
        assert len(result["results"]) == 4

        # Verify execution phases respected dependencies
        # Phase 1: Both script collectors (parallel)
        # Phase 2: LLM analyzer (depends on both collectors)
        # Phase 3: Alert generator script (depends on analyzer)

        for agent_name in [
            "metrics_collector",
            "logs_collector",
            "system_analyzer",
            "alert_generator",
        ]:
            assert result["results"][agent_name]["status"] == "success"

    @pytest.mark.asyncio
    async def test_circular_dependency_detection_mixed_agents(self) -> None:
        """Test proper circular dependency detection with mixed agents.

        This should legitimately fail due to real circular dependency.
        """
        config = EnsembleConfig(
            name="circular_dependency_test",
            description="Test circular dependency detection",
            agents=[
                ScriptAgentConfig(
                    name="script_a",
                    script='echo "Script A"',
                    depends_on=["llm_b"],  # Creates circle
                ),
                LlmAgentConfig(
                    name="llm_b",
                    model_profile="test-analyzer",
                    depends_on=["script_a"],  # Creates circle
                ),
            ],
        )

        executor = EnsembleExecutor()

        # This should raise a circular dependency error
        with (
            patch.object(executor, "_artifact_manager"),
            pytest.raises(ValueError, match="Circular dependency detected"),
        ):
            await executor.execute(config, "Test circular detection")
