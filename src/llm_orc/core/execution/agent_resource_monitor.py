"""Agent execution resource monitoring and metrics collection."""

from typing import Any

from llm_orc.core.execution.system_resource_monitor import (
    SystemResourceMonitor,
)


class AgentResourceMonitor:
    """Provides resource monitoring and execution metrics for ensembles."""

    def __init__(
        self,
        performance_config: dict[str, Any],
    ) -> None:
        """Initialize the agent executor.

        Args:
            performance_config: Performance configuration settings
        """
        self._performance_config = performance_config

        # Initialize resource monitoring for performance feedback
        self.monitor = SystemResourceMonitor(polling_interval=0.1)

        # Track resource management statistics for this execution
        self.adaptive_stats: dict[str, Any] = {
            "adaptive_used": False,
            "management_type": "user_configured",
            "concurrency_decisions": [],
            "resource_metrics": [],
        }

        # Track per-phase metrics for detailed performance analysis
        self._phase_metrics: list[dict[str, Any]] = []

    def get_adaptive_stats(self) -> dict[str, Any]:
        """Get adaptive resource management statistics for this execution."""
        if "execution_metrics" not in self.adaptive_stats:
            try:
                import psutil

                current_cpu = psutil.cpu_percent(interval=None)
                current_memory = psutil.virtual_memory().percent
                self.adaptive_stats["execution_metrics"] = {
                    "peak_cpu": current_cpu,
                    "avg_cpu": current_cpu,
                    "peak_memory": current_memory,
                    "avg_memory": current_memory,
                    "sample_count": 1,
                    "raw_cpu_samples": [current_cpu],
                    "raw_memory_samples": [current_memory],
                }
                if not self.adaptive_stats["resource_metrics"]:
                    self.adaptive_stats["resource_metrics"].append(
                        {
                            "cpu_percent": current_cpu,
                            "memory_percent": current_memory,
                            "circuit_breaker_state": "N/A",
                            "measurement_point": "final_fallback",
                        }
                    )
            except Exception:
                self.adaptive_stats["execution_metrics"] = {
                    "peak_cpu": 0.0,
                    "avg_cpu": 0.0,
                    "peak_memory": 0.0,
                    "avg_memory": 0.0,
                    "sample_count": 0,
                }

        stats_copy = self.adaptive_stats.copy()
        if hasattr(self, "_phase_metrics") and self._phase_metrics:
            stats_copy["phase_metrics"] = self._phase_metrics

        return stats_copy
