"""Adaptive resource management system with circuit breaker patterns."""

import asyncio
import time
from typing import Any

import psutil


# Removed ResourceMonitoringCircuitBreaker - no longer needed with simplified approach


class SystemResourceMonitor:
    """Monitors system resources with hybrid polling/event-driven approach."""

    def __init__(self, polling_interval: float = 0.1) -> None:
        """Initialize the resource monitor.

        Args:
            polling_interval: Seconds between polling attempts
        """
        self.polling_interval = polling_interval
        self.is_monitoring = False
        self._baseline_established = False
        self._monitoring_task: asyncio.Task[None] | None = None
        self._execution_samples: list[dict[str, float]] = []
        self._stop_monitoring = asyncio.Event()

    async def get_current_metrics(self) -> dict[str, float]:
        """Get current system resource metrics.

        Returns:
            Dictionary containing cpu_percent and memory_percent
        """
        # For more accurate CPU measurement, use a small blocking interval
        # This is more reliable than the baseline approach for short-lived processes
        cpu_percent = psutil.cpu_percent(interval=0.1)  # 100ms blocking measurement
        memory_info = psutil.virtual_memory()

        return {"cpu_percent": cpu_percent, "memory_percent": memory_info.percent}

    async def start_execution_monitoring(self) -> None:
        """Start continuous resource monitoring during agent execution."""
        if self.is_monitoring:
            return  # Already monitoring

        self.is_monitoring = True
        self._stop_monitoring.clear()
        self._execution_samples = []

        # Start background monitoring task
        self._monitoring_task = asyncio.create_task(self._continuous_monitoring_loop())

    async def stop_execution_monitoring(self) -> dict[str, float]:
        """Stop monitoring and return aggregated execution metrics."""
        if not self.is_monitoring:
            empty_metrics: dict[str, float] = {
                "peak_cpu": 0.0,
                "avg_cpu": 0.0,
                "peak_memory": 0.0,
                "avg_memory": 0.0,
                "sample_count": 0,
            }
            return empty_metrics

        # Signal monitoring to stop
        self._stop_monitoring.set()

        # Wait for monitoring task to finish
        if self._monitoring_task:
            await self._monitoring_task
            self._monitoring_task = None

        self.is_monitoring = False

        # Aggregate metrics from execution samples
        return self._aggregate_execution_metrics()

    async def _continuous_monitoring_loop(self) -> None:
        """Background loop that continuously samples resources during execution."""
        try:
            # Initialize baseline for continuous monitoring
            psutil.cpu_percent(interval=None)  # Establish baseline
            await asyncio.sleep(0.1)  # Let baseline settle

            while not self._stop_monitoring.is_set():
                # Sample current resources with blocking interval for measurement
                # Use a short blocking interval to get meaningful CPU readings
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()

                sample = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                self._execution_samples.append(sample)

                # Wait for next sampling interval - 100ms to match CPU sampling
                try:
                    await asyncio.wait_for(self._stop_monitoring.wait(), timeout=0.1)
                    break  # Stop signal received
                except TimeoutError:
                    continue  # Continue sampling
        except Exception:
            # Gracefully handle any monitoring errors
            pass

    def _aggregate_execution_metrics(self) -> dict[str, float]:
        """Aggregate execution samples into summary metrics."""
        if not self._execution_samples:
            empty_metrics: dict[str, float] = {
                "peak_cpu": 0.0,
                "avg_cpu": 0.0,
                "peak_memory": 0.0,
                "avg_memory": 0.0,
                "sample_count": 0,
            }
            return empty_metrics

        cpu_values = [sample["cpu_percent"] for sample in self._execution_samples]
        memory_values = [sample["memory_percent"] for sample in self._execution_samples]

        metrics: dict[str, float] = {
            "peak_cpu": max(cpu_values) if cpu_values else 0.0,
            "avg_cpu": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            "peak_memory": max(memory_values) if memory_values else 0.0,
            "avg_memory": (
                sum(memory_values) / len(memory_values) if memory_values else 0.0
            ),
            "sample_count": len(self._execution_samples),
        }

        # For research purposes, include raw samples for debugging
        if cpu_values:
            # Type ignore because we're adding list values to a float dict
            metrics["raw_cpu_samples"] = cpu_values[:10]  # type: ignore[assignment]
            metrics["raw_memory_samples"] = memory_values[:10]  # type: ignore[assignment]

        return metrics

    async def collect_phase_metrics(
        self,
        phase_index: int,
        phase_name: str,
        agent_count: int,
    ) -> dict[str, Any]:
        """Collect phase-specific metrics.

        Args:
            phase_index: Index of the phase
            phase_name: Name of the phase
            agent_count: Number of agents in the phase

        Returns:
            Phase-specific metrics dictionary
        """
        return {
            "phase_index": phase_index,
            "phase_name": phase_name,
            "agent_count": agent_count,
            "phase_start_time": time.time(),
        }


class AdaptiveResourceManager:
    """Simple monitoring collector for performance feedback (no longer adaptive)."""

    def __init__(
        self,
        base_limit: int,
        monitor: SystemResourceMonitor,
        min_limit: int = 1,
        max_limit: int = 10,
    ) -> None:
        """Initialize the monitoring collector.

        Args:
            base_limit: Kept for backward compatibility (unused)
            monitor: System resource monitor instance
            min_limit: Kept for backward compatibility (unused)
            max_limit: Kept for backward compatibility (unused)
        """
        self.monitor = monitor
        # Keep these for backward compatibility but don't use them
        self.base_limit = base_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
