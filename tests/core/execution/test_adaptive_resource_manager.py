"""Tests for simplified resource management (Issue #55)."""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.adaptive_resource_manager import (
    AdaptiveResourceManager,
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


class TestAdaptiveResourceManager:
    """Test the simplified adaptive resource manager."""

    @pytest.fixture
    def mock_monitor(self) -> Mock:
        """Create a mock system resource monitor."""
        monitor = Mock()
        monitor.get_current_metrics = AsyncMock(return_value={
            "cpu_percent": 25.0,
            "memory_percent": 60.0
        })
        monitor.start_execution_monitoring = AsyncMock()
        monitor.stop_execution_monitoring = AsyncMock(return_value={
            "peak_cpu": 30.0,
            "avg_cpu": 25.0,
            "peak_memory": 65.0,
            "avg_memory": 60.0,
            "sample_count": 10
        })
        monitor.collect_phase_metrics = AsyncMock(return_value={
            "phase_index": 0,
            "phase_name": "test_phase",
            "agent_count": 3,
            "phase_start_time": time.time()
        })
        return monitor

    @pytest.fixture
    def resource_manager(self, mock_monitor: Mock) -> AdaptiveResourceManager:
        """Create an adaptive resource manager instance."""
        return AdaptiveResourceManager(
            base_limit=5,
            monitor=mock_monitor,
            min_limit=1,
            max_limit=10
        )

    def test_initialization(
        self,
        resource_manager: AdaptiveResourceManager,
        mock_monitor: Mock
    ) -> None:
        """Test that resource manager initializes correctly."""
        assert resource_manager.monitor == mock_monitor
        assert resource_manager.base_limit == 5
        assert resource_manager.min_limit == 1
        assert resource_manager.max_limit == 10

    def test_backward_compatibility_attributes(
        self, resource_manager: AdaptiveResourceManager
    ) -> None:
        """Test that backward compatibility attributes are maintained."""
        # These are kept for backward compatibility but not used in simplified approach
        assert hasattr(resource_manager, 'base_limit')
        assert hasattr(resource_manager, 'min_limit')
        assert hasattr(resource_manager, 'max_limit')
        assert hasattr(resource_manager, 'monitor')

    @pytest.mark.asyncio
    async def test_monitor_delegation(
        self,
        resource_manager: AdaptiveResourceManager,
        mock_monitor: Mock
    ) -> None:
        """Test that the resource manager properly delegates to the monitor."""
        # Test that we can access monitor methods through the resource manager
        await resource_manager.monitor.start_execution_monitoring()
        mock_monitor.start_execution_monitoring.assert_called_once()

        metrics = await resource_manager.monitor.stop_execution_monitoring()
        mock_monitor.stop_execution_monitoring.assert_called_once()
        assert metrics["sample_count"] == 10

        phase_metrics = await resource_manager.monitor.collect_phase_metrics(
            0, "test", 3
        )
        mock_monitor.collect_phase_metrics.assert_called_once_with(0, "test", 3)
        assert phase_metrics["phase_name"] == "test_phase"  # Mock returns test_phase


class TestSimplifiedArchitecture:
    """Test the overall simplified architecture approach."""

    @pytest.mark.asyncio
    async def test_no_complex_adaptive_calculations(self) -> None:
        """Test that complex adaptive calculations have been removed."""
        monitor = SystemResourceMonitor()
        manager = AdaptiveResourceManager(base_limit=5, monitor=monitor)

        # Should not have complex methods like get_adaptive_limit
        assert not hasattr(manager, 'get_adaptive_limit')
        assert not hasattr(manager, 'circuit_breaker')
        assert not hasattr(manager, 'get_adaptive_limit_for_phase')

    def test_monitoring_focused_approach(self) -> None:
        """Test that the approach is focused on monitoring, not adaptive decisions."""
        monitor = SystemResourceMonitor()
        manager = AdaptiveResourceManager(base_limit=5, monitor=monitor)

        # Should have monitoring capabilities
        assert hasattr(manager, 'monitor')
        assert hasattr(manager.monitor, 'start_execution_monitoring')
        assert hasattr(manager.monitor, 'stop_execution_monitoring')
        assert hasattr(manager.monitor, 'get_current_metrics')
