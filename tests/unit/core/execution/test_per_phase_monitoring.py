"""Tests verifying per-phase monitoring is not implemented in simplified
architecture."""

from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.adaptive_resource_manager import (
    SystemResourceMonitor,
)
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


class TestPerPhaseMonitoring:
    """Test that per-phase monitoring is not needed in simplified architecture."""

    @pytest.fixture
    def mock_monitor(self) -> Mock:
        """Create a mock system resource monitor."""
        monitor = Mock(spec=SystemResourceMonitor)
        monitor.start_execution_monitoring = AsyncMock()
        monitor.stop_execution_monitoring = AsyncMock(
            return_value={
                "peak_cpu": 15.5,
                "avg_cpu": 12.3,
                "peak_memory": 78.2,
                "avg_memory": 75.1,
                "sample_count": 25,
                "raw_cpu_samples": [10.1, 12.5, 15.5, 14.2, 11.8],
                "raw_memory_samples": [75.0, 76.1, 78.2, 77.5, 74.9],
            }
        )
        monitor.get_current_metrics = AsyncMock(
            return_value={
                "cpu_percent": 8.5,
                "memory_percent": 72.3,
            }
        )
        monitor.collect_phase_metrics = AsyncMock(
            return_value={
                "phase_specific_data": "test_data",
                "collection_timestamp": 1234567890,
            }
        )
        return monitor

    @pytest.fixture
    def mock_ensemble_execution(self) -> Mock:
        """Create a mock ensemble execution."""
        execution = Mock(spec=EnsembleExecutor)
        execution._execute_agents_in_phase_parallel = AsyncMock(
            return_value={
                "agent1": {"response": "test response 1", "status": "success"},
                "agent2": {"response": "test response 2", "status": "success"},
            }
        )
        # Mock the new method with expected return value
        execution.execute_dependency_phases_with_monitoring = AsyncMock(
            return_value={
                "phase_metrics": [
                    {"phase_index": 0, "adaptive_limit": 3, "sample_count": 15},
                    {"phase_index": 1, "adaptive_limit": 3, "sample_count": 10},
                ],
                "results": {
                    "agent1": {"response": "test response 1", "status": "success"},
                    "agent2": {"response": "test response 2", "status": "success"},
                    "agent3": {"response": "test response 3", "status": "success"},
                },
            }
        )
        return execution

    def test_per_phase_monitoring_not_implemented_by_design(self) -> None:
        """Test that per-phase monitoring is intentionally not implemented.

        In the simplified architecture, we use user-configured limits
        and don't need complex per-phase adaptive calculations.
        """
        from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

        # Verify that complex per-phase monitoring methods don't exist
        assert not hasattr(
            EnsembleExecutor, "execute_dependency_phases_with_monitoring"
        )
        assert not hasattr(
            EnsembleExecutor, "_execute_dependency_phases_with_monitoring"
        )

    def test_adaptive_resource_manager_removed_by_design(self) -> None:
        """Test that AdaptiveResourceManager class has been removed.

        The simplified architecture uses SystemResourceMonitor directly.
        """
        import llm_orc.core.execution.adaptive_resource_manager as arm_module

        assert not hasattr(arm_module, "AdaptiveResourceManager")

    @pytest.mark.asyncio
    async def test_basic_metrics_collection_for_guidance(self) -> None:
        """Test that basic metrics collection still works for user guidance."""
        from llm_orc.core.execution.adaptive_resource_manager import (
            SystemResourceMonitor,
        )

        monitor = SystemResourceMonitor(polling_interval=0.1)

        # Test basic metrics collection (used for performance feedback)
        current_metrics = await monitor.get_current_metrics()

        assert isinstance(current_metrics, dict)
        assert "cpu_percent" in current_metrics
        assert "memory_percent" in current_metrics

        # Test phase metrics collection still works but is simpler
        phase_metrics = await monitor.collect_phase_metrics(
            phase_index=0,
            phase_name="test_phase",
            agent_count=3,
        )
        assert phase_metrics["phase_index"] == 0
        assert phase_metrics["phase_name"] == "test_phase"

    @pytest.mark.asyncio
    async def test_simplified_execution_workflow_works(self) -> None:
        """Test that the simplified execution workflow provides adequate monitoring.

        Instead of complex per-phase adaptive calculations, we provide:
        - User-controlled concurrency limits
        - Performance feedback and guidance
        - Simple resource monitoring for optimization hints
        """
        from llm_orc.core.execution.adaptive_resource_manager import (
            SystemResourceMonitor,
        )

        monitor = SystemResourceMonitor(polling_interval=0.1)

        # Test that monitoring still works for performance feedback
        await monitor.start_execution_monitoring()
        metrics = await monitor.stop_execution_monitoring()

        # Should provide basic performance data for user guidance
        assert "peak_cpu" in metrics
        assert "avg_cpu" in metrics
        assert "sample_count" in metrics
