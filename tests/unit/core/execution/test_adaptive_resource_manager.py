"""Tests for simplified resource management (Issue #55)."""

import pytest

from llm_orc.core.execution.adaptive_resource_manager import (
    SystemResourceMonitor,
)


class TestSystemResourceMonitor:
    """Test the system resource monitoring component."""

    @pytest.fixture
    def monitor(self) -> SystemResourceMonitor:
        """Create a SystemResourceMonitor instance."""
        return SystemResourceMonitor(polling_interval=0.1)

    def test_monitor_initialization(self, monitor: SystemResourceMonitor) -> None:
        """Test that monitor initializes correctly."""
        assert monitor.polling_interval == 0.1
        assert not monitor.is_monitoring

    @pytest.mark.asyncio
    async def test_get_current_metrics(self, monitor: SystemResourceMonitor) -> None:
        """Test that current metrics can be retrieved."""
        metrics = await monitor.get_current_metrics()

        # Should include CPU and memory usage
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert 0 <= metrics["cpu_percent"] <= 100
        assert 0 <= metrics["memory_percent"] <= 100

    @pytest.mark.asyncio
    async def test_execution_monitoring_lifecycle(
        self, monitor: SystemResourceMonitor
    ) -> None:
        """Test that execution monitoring can be started and stopped."""
        # Start monitoring
        await monitor.start_execution_monitoring()
        assert monitor.is_monitoring

        # Stop monitoring and get metrics
        metrics = await monitor.stop_execution_monitoring()
        assert not monitor.is_monitoring

        # Should have collected metrics
        assert "peak_cpu" in metrics
        assert "avg_cpu" in metrics
        assert "peak_memory" in metrics
        assert "avg_memory" in metrics
        assert "sample_count" in metrics
        assert metrics["sample_count"] >= 0

    @pytest.mark.asyncio
    async def test_collect_phase_metrics(self, monitor: SystemResourceMonitor) -> None:
        """Test phase metrics collection."""
        metrics = await monitor.collect_phase_metrics(0, "test_phase", 3)

        assert metrics["phase_index"] == 0
        assert metrics["phase_name"] == "test_phase"
        assert metrics["agent_count"] == 3
        assert "phase_start_time" in metrics


class TestSimplifiedArchitecture:
    """Test the overall simplified architecture approach."""

    def test_monitoring_focused_approach(self) -> None:
        """Test that the approach is focused on monitoring, not adaptive decisions."""
        monitor = SystemResourceMonitor()

        # Should have monitoring capabilities
        assert hasattr(monitor, "start_execution_monitoring")
        assert hasattr(monitor, "stop_execution_monitoring")
        assert hasattr(monitor, "get_current_metrics")
