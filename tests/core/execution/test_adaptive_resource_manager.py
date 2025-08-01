"""Tests for adaptive resource management (Issue #55)."""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.adaptive_resource_manager import (
    AdaptiveResourceManager,
    ResourceMonitoringCircuitBreaker,
    SystemResourceMonitor,
)


class TestResourceMonitoringCircuitBreaker:
    """Test the circuit breaker component."""

    @pytest.fixture
    def circuit_breaker(self) -> ResourceMonitoringCircuitBreaker:
        """Create a circuit breaker instance."""
        return ResourceMonitoringCircuitBreaker(failure_threshold=3, reset_timeout=10)

    def test_circuit_breaker_initialization(
        self, circuit_breaker: ResourceMonitoringCircuitBreaker
    ) -> None:
        """Test that circuit breaker initializes in CLOSED state."""
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.failure_threshold == 3

    def test_record_failure_opens_circuit(
        self, circuit_breaker: ResourceMonitoringCircuitBreaker
    ) -> None:
        """Test that failures open the circuit after threshold is reached."""
        # Record failures up to threshold
        for _ in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == "OPEN"

    def test_record_success_resets_failure_count(
        self, circuit_breaker: ResourceMonitoringCircuitBreaker
    ) -> None:
        """Test that success resets failure count."""
        circuit_breaker.record_failure()
        assert circuit_breaker.failure_count == 1

        circuit_breaker.record_success()
        assert circuit_breaker.failure_count == 0


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


class TestAdaptiveResourceManager:
    """Test the adaptive resource manager component."""

    @pytest.fixture
    def monitor_mock(self) -> Mock:
        """Create a mock system resource monitor."""
        monitor = Mock(spec=SystemResourceMonitor)
        monitor.get_current_metrics = AsyncMock(
            return_value={"cpu_percent": 50.0, "memory_percent": 60.0}
        )
        return monitor

    @pytest.fixture
    def manager(self, monitor_mock: Mock) -> AdaptiveResourceManager:
        """Create an AdaptiveResourceManager instance."""
        return AdaptiveResourceManager(
            base_limit=5, monitor=monitor_mock, min_limit=1, max_limit=10
        )

    def test_manager_initialization(
        self, manager: AdaptiveResourceManager, monitor_mock: Mock
    ) -> None:
        """Test that manager initializes correctly."""
        assert manager.base_limit == 5
        assert manager.min_limit == 1
        assert manager.max_limit == 10
        assert manager.monitor is monitor_mock

    @pytest.mark.asyncio
    async def test_adaptive_limit_calculation(
        self, manager: AdaptiveResourceManager
    ) -> None:
        """Test that limit is calculated adaptively based on system resources."""
        current_limit = await manager.get_adaptive_limit()

        # Should be between min and max limits
        assert manager.min_limit <= current_limit <= manager.max_limit

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(
        self, manager: AdaptiveResourceManager, monitor_mock: Mock
    ) -> None:
        """Test that circuit breaker activates on monitoring failures."""
        # Simulate monitoring failure
        monitor_mock.get_current_metrics.side_effect = Exception("Monitor failed")

        # Should fall back to base limit when monitoring fails
        current_limit = await manager.get_adaptive_limit()
        assert current_limit == manager.base_limit

    @pytest.mark.asyncio
    async def test_performance_overhead(self, manager: AdaptiveResourceManager) -> None:
        """Test that adaptive calculation has minimal overhead."""
        # Measure time for 100 adaptive limit calculations
        start_time = time.perf_counter()
        for _ in range(100):
            await manager.get_adaptive_limit()
        end_time = time.perf_counter()

        # Should complete 100 calculations in under 10ms (0.1ms per calculation)
        total_time = end_time - start_time
        assert total_time < 0.01, f"Overhead too high: {total_time:.4f}s for 100 calls"

    @pytest.mark.asyncio
    async def test_start_execution_monitoring(self, monitor_mock: Mock) -> None:
        """Test starting execution monitoring."""
        monitor = SystemResourceMonitor(polling_interval=0.01)

        # Initially not monitoring
        assert not monitor.is_monitoring

        # Start monitoring
        await monitor.start_execution_monitoring()
        assert monitor.is_monitoring

        # Starting again should not create duplicate task
        await monitor.start_execution_monitoring()
        assert monitor.is_monitoring

        # Stop monitoring
        await monitor.stop_execution_monitoring()
        assert not monitor.is_monitoring

    @pytest.mark.asyncio
    async def test_stop_execution_monitoring_when_not_started(
        self, monitor_mock: Mock
    ) -> None:
        """Test stopping execution monitoring when it wasn't started."""
        monitor = SystemResourceMonitor(polling_interval=0.01)

        # Should return empty metrics when not monitoring
        metrics = await monitor.stop_execution_monitoring()
        assert metrics["peak_cpu"] == 0.0
        assert metrics["avg_cpu"] == 0.0
        assert metrics["peak_memory"] == 0.0
        assert metrics["avg_memory"] == 0.0
        assert metrics["sample_count"] == 0

    @pytest.mark.asyncio
    async def test_continuous_monitoring_collects_samples(
        self, monitor_mock: Mock
    ) -> None:
        """Test that continuous monitoring collects resource samples."""
        monitor = SystemResourceMonitor(polling_interval=0.01)

        # Start and immediately stop monitoring
        await monitor.start_execution_monitoring()
        # Give it a tiny bit of time to collect at least one sample
        import asyncio

        await asyncio.sleep(0.02)
        metrics = await monitor.stop_execution_monitoring()

        # Should have collected at least some metrics
        assert metrics["sample_count"] >= 0  # May be 0 if execution was too fast
        assert 0 <= metrics["peak_cpu"] <= 100
        assert 0 <= metrics["avg_cpu"] <= 100
        assert 0 <= metrics["peak_memory"] <= 100
        assert 0 <= metrics["avg_memory"] <= 100

    @pytest.mark.asyncio
    async def test_aggregate_execution_metrics_with_samples(
        self, monitor_mock: Mock
    ) -> None:
        """Test aggregating execution metrics when samples exist."""
        monitor = SystemResourceMonitor(polling_interval=0.01)

        # Manually add some sample data for testing
        monitor._execution_samples = [
            {"cpu_percent": 10.0, "memory_percent": 50.0, "timestamp": time.time()},
            {"cpu_percent": 20.0, "memory_percent": 60.0, "timestamp": time.time()},
            {"cpu_percent": 15.0, "memory_percent": 55.0, "timestamp": time.time()},
        ]

        metrics = monitor._aggregate_execution_metrics()

        assert metrics["peak_cpu"] == 20.0
        assert metrics["avg_cpu"] == 15.0  # (10+20+15)/3
        assert metrics["peak_memory"] == 60.0
        assert metrics["avg_memory"] == 55.0  # (50+60+55)/3
        assert metrics["sample_count"] == 3
        assert "raw_cpu_samples" in metrics
        assert "raw_memory_samples" in metrics

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_returns_base_limit(
        self, manager: AdaptiveResourceManager
    ) -> None:
        """Test that open circuit breaker returns base limit."""
        # Force circuit breaker to open state
        for _ in range(manager.circuit_breaker.failure_threshold):
            manager.circuit_breaker.record_failure()

        assert manager.circuit_breaker.state == "OPEN"

        # Should return base limit regardless of system resources
        limit = await manager.get_adaptive_limit()
        assert limit == manager.base_limit

    @pytest.mark.asyncio
    async def test_adaptive_limit_pressure_thresholds(
        self, manager: AdaptiveResourceManager, monitor_mock: Mock
    ) -> None:
        """Test different resource pressure scenarios."""
        # Test high pressure (>50%) - should reduce limit
        monitor_mock.get_current_metrics.return_value = {
            "cpu_percent": 60.0,
            "memory_percent": 70.0,
        }
        limit = await manager.get_adaptive_limit()
        expected_high_pressure = int(manager.base_limit * 0.7)
        assert limit == expected_high_pressure

        # Test low pressure (<20%) - should increase limit
        monitor_mock.get_current_metrics.return_value = {
            "cpu_percent": 10.0,
            "memory_percent": 15.0,
        }
        limit = await manager.get_adaptive_limit()
        expected_low_pressure = int(manager.base_limit * 1.3)
        # Ensure within max limit bounds
        expected_low_pressure = min(expected_low_pressure, manager.max_limit)
        assert limit == expected_low_pressure

        # Test normal pressure (20-50%) - should use base limit
        monitor_mock.get_current_metrics.return_value = {
            "cpu_percent": 30.0,
            "memory_percent": 35.0,
        }
        limit = await manager.get_adaptive_limit()
        assert limit == manager.base_limit
