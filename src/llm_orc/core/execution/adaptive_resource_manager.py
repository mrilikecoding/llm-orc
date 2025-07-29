"""Adaptive resource management system with circuit breaker patterns."""

import time

import psutil


class ResourceMonitoringCircuitBreaker:
    """Circuit breaker for resource monitoring failures."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60) -> None:
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before attempting reset
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.last_failure_time: float | None = None
        self.reset_timeout = reset_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        self.state = "CLOSED"


class SystemResourceMonitor:
    """Monitors system resources with hybrid polling/event-driven approach."""

    def __init__(self, polling_interval: float = 0.1) -> None:
        """Initialize the resource monitor.

        Args:
            polling_interval: Seconds between polling attempts
        """
        self.polling_interval = polling_interval
        self.is_monitoring = False

    async def get_current_metrics(self) -> dict[str, float]:
        """Get current system resource metrics.

        Returns:
            Dictionary containing cpu_percent and memory_percent
        """
        # Using psutil for system metrics - minimal overhead
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
        memory_info = psutil.virtual_memory()

        return {"cpu_percent": cpu_percent, "memory_percent": memory_info.percent}


class AdaptiveResourceManager:
    """Manages adaptive resource allocation with circuit breaker protection."""

    def __init__(
        self,
        base_limit: int,
        monitor: SystemResourceMonitor,
        min_limit: int = 1,
        max_limit: int = 10,
    ) -> None:
        """Initialize the adaptive resource manager.

        Args:
            base_limit: Default limit when adaptive management fails
            monitor: System resource monitor instance
            min_limit: Minimum allowed resource limit
            max_limit: Maximum allowed resource limit
        """
        self.base_limit = base_limit
        self.monitor = monitor
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.circuit_breaker = ResourceMonitoringCircuitBreaker()

    async def get_adaptive_limit(self) -> int:
        """Calculate adaptive resource limit based on system conditions.

        Returns:
            Calculated resource limit within configured bounds
        """
        # If circuit breaker is open, fall back to base limit
        if self.circuit_breaker.state == "OPEN":
            return self.base_limit

        try:
            metrics = await self.monitor.get_current_metrics()
            self.circuit_breaker.record_success()

            # Simple adaptive algorithm: reduce limit when resources are high
            cpu_percent = metrics["cpu_percent"]
            memory_percent = metrics["memory_percent"]

            # Calculate adjustment factor based on resource usage
            resource_pressure = max(cpu_percent, memory_percent) / 100.0

            if resource_pressure > 0.8:
                # High pressure: reduce limit
                adjusted_limit = int(self.base_limit * 0.7)
            elif resource_pressure < 0.3:
                # Low pressure: increase limit
                adjusted_limit = int(self.base_limit * 1.3)
            else:
                # Normal pressure: use base limit
                adjusted_limit = self.base_limit

            # Ensure within bounds
            return max(self.min_limit, min(adjusted_limit, self.max_limit))

        except Exception:
            # Monitoring failed - record failure and fall back
            self.circuit_breaker.record_failure()
            return self.base_limit
