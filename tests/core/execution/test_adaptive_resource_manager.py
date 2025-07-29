"""Tests for adaptive resource management (Issue #55)."""

import asyncio
import time
from unittest.mock import Mock, AsyncMock

import pytest

from llm_orc.core.execution.adaptive_resource_manager import (
    ResourceMonitoringCircuitBreaker,
    SystemResourceMonitor,
    AdaptiveResourceManager,
)


class TestResourceMonitoringCircuitBreaker:
    """Test the circuit breaker component."""

    @pytest.fixture
    def circuit_breaker(self) -> ResourceMonitoringCircuitBreaker:
        """Create a circuit breaker instance."""
        return ResourceMonitoringCircuitBreaker(
            failure_threshold=3, reset_timeout=10
        )

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
    def monitor_mock(self) -> SystemResourceMonitor:
        """Create a mock system resource monitor."""
        monitor = Mock(spec=SystemResourceMonitor)
        monitor.get_current_metrics = AsyncMock(return_value={
            "cpu_percent": 50.0,
            "memory_percent": 60.0
        })
        return monitor

    @pytest.fixture
    def manager(self, monitor_mock: SystemResourceMonitor) -> AdaptiveResourceManager:
        """Create an AdaptiveResourceManager instance."""
        return AdaptiveResourceManager(
            base_limit=5,
            monitor=monitor_mock,
            min_limit=1,
            max_limit=10
        )

    def test_manager_initialization(
        self, 
        manager: AdaptiveResourceManager,
        monitor_mock: SystemResourceMonitor
    ) -> None:
        """Test that manager initializes correctly."""
        assert manager.base_limit == 5
        assert manager.min_limit == 1
        assert manager.max_limit == 10
        assert manager.monitor is monitor_mock

    @pytest.mark.asyncio
    async def test_adaptive_limit_calculation(
        self, 
        manager: AdaptiveResourceManager
    ) -> None:
        """Test that limit is calculated adaptively based on system resources."""
        current_limit = await manager.get_adaptive_limit()
        
        # Should be between min and max limits
        assert manager.min_limit <= current_limit <= manager.max_limit

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(
        self, 
        manager: AdaptiveResourceManager,
        monitor_mock: SystemResourceMonitor
    ) -> None:
        """Test that circuit breaker activates on monitoring failures."""
        # Simulate monitoring failure
        monitor_mock.get_current_metrics.side_effect = Exception("Monitor failed")
        
        # Should fall back to base limit when monitoring fails
        current_limit = await manager.get_adaptive_limit()
        assert current_limit == manager.base_limit

    @pytest.mark.asyncio 
    async def test_performance_overhead(
        self,
        manager: AdaptiveResourceManager
    ) -> None:
        """Test that adaptive calculation has minimal overhead."""
        # Measure time for 100 adaptive limit calculations
        start_time = time.perf_counter()
        for _ in range(100):
            await manager.get_adaptive_limit()
        end_time = time.perf_counter()
        
        # Should complete 100 calculations in under 10ms (0.1ms per calculation)
        total_time = end_time - start_time
        assert total_time < 0.01, f"Overhead too high: {total_time:.4f}s for 100 calls"