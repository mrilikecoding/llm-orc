"""Usage tracking and aggregation for agent execution."""

import copy
import time
from typing import Any

import psutil

from llm_orc.models.base import ModelInterface


class UsageCollector:
    """Collects and aggregates usage metrics from agent execution."""

    def __init__(self) -> None:
        """Initialize usage collector."""
        self._agent_usage: dict[str, Any] = {}
        self._agent_resource_metrics: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        """Reset collected usage data."""
        self._agent_usage.clear()
        self._agent_resource_metrics.clear()

    def collect_agent_usage(
        self,
        agent_name: str,
        model_instance: ModelInterface | None,
        model_profile: str | None = None,
    ) -> None:
        """Collect usage metrics from a model instance."""
        if model_instance is not None and hasattr(model_instance, "get_last_usage"):
            usage = model_instance.get_last_usage()
            if usage:
                # Add model profile information to usage data
                if model_profile is not None:
                    usage["model_profile"] = model_profile
                elif hasattr(model_instance, "get_model_profile"):
                    # Try to get model profile from model instance if not provided
                    profile = model_instance.get_model_profile()
                    if profile:
                        usage["model_profile"] = profile

                # Merge with any collected resource metrics for this agent
                if agent_name in self._agent_resource_metrics:
                    usage.update(self._agent_resource_metrics[agent_name])

                self._agent_usage[agent_name] = usage

    def start_agent_resource_monitoring(self, agent_name: str) -> dict[str, Any]:
        """Start monitoring resources for an agent and return baseline metrics.

        Returns:
            Dictionary with baseline CPU and memory percentages
        """
        try:
            baseline_metrics = {
                "start_time": time.time(),
                "baseline_cpu": psutil.cpu_percent(interval=0.1),
                "baseline_memory": psutil.virtual_memory().percent,
                "sample_count": 0,
                "peak_cpu": 0.0,
                "avg_cpu": 0.0,
                "peak_memory": 0.0,
                "avg_memory": 0.0,
                "cpu_samples": [],
                "memory_samples": [],
            }
            self._agent_resource_metrics[agent_name] = baseline_metrics
            return baseline_metrics
        except Exception:
            # If resource monitoring fails, return empty baseline
            return {}

    def sample_agent_resources(self, agent_name: str) -> None:
        """Take a resource usage sample for the specified agent."""
        if agent_name not in self._agent_resource_metrics:
            return

        try:
            metrics = self._agent_resource_metrics[agent_name]
            current_cpu = psutil.cpu_percent(interval=None)  # Don't block
            current_memory = psutil.virtual_memory().percent

            # Store samples for averaging
            metrics["cpu_samples"].append(current_cpu)
            metrics["memory_samples"].append(current_memory)
            metrics["sample_count"] += 1

            # Update peaks
            metrics["peak_cpu"] = max(metrics["peak_cpu"], current_cpu)
            metrics["peak_memory"] = max(metrics["peak_memory"], current_memory)

            # Update running averages
            cpu_samples = metrics["cpu_samples"]
            memory_samples = metrics["memory_samples"]
            metrics["avg_cpu"] = sum(cpu_samples) / len(cpu_samples)
            metrics["avg_memory"] = sum(memory_samples) / len(memory_samples)

        except Exception:
            # Silently handle monitoring errors
            pass

    def finalize_agent_resource_monitoring(self, agent_name: str) -> dict[str, Any]:
        """Finalize resource monitoring for an agent and return the metrics.

        Returns:
            Dictionary with final resource usage metrics
        """
        if agent_name not in self._agent_resource_metrics:
            return {}

        try:
            metrics = self._agent_resource_metrics[agent_name]

            # Take final sample
            self.sample_agent_resources(agent_name)

            # Calculate final duration
            end_time = time.time()
            duration_seconds = end_time - metrics.get("start_time", end_time)

            # Return cleaned up metrics (remove internal tracking data)
            final_metrics = {
                "peak_cpu": metrics["peak_cpu"],
                "avg_cpu": metrics["avg_cpu"],
                "peak_memory": metrics["peak_memory"],
                "avg_memory": metrics["avg_memory"],
                "resource_duration_seconds": duration_seconds,
                "resource_sample_count": metrics["sample_count"],
            }

            # Update stored metrics with final values
            self._agent_resource_metrics[agent_name].update(final_metrics)

            return final_metrics

        except Exception:
            return {}

    def get_agent_usage(self) -> dict[str, Any]:
        """Get collected agent usage data."""
        return copy.deepcopy(self._agent_usage)

    def calculate_usage_summary(
        self, synthesis_usage: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Calculate aggregated usage summary.

        Args:
            synthesis_usage: Optional synthesis usage to include in totals

        Returns:
            Dictionary containing agents usage, totals, and optional synthesis
        """
        summary = {
            "agents": copy.deepcopy(self._agent_usage),
            "totals": {
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "total_duration_ms": 0,
                "agents_count": len(self._agent_usage),
            },
        }

        # Aggregate agent usage
        for usage in self._agent_usage.values():
            summary["totals"]["total_tokens"] += usage.get("total_tokens", 0)
            summary["totals"]["total_input_tokens"] += usage.get("input_tokens", 0)
            summary["totals"]["total_output_tokens"] += usage.get("output_tokens", 0)
            summary["totals"]["total_cost_usd"] += usage.get("cost_usd", 0.0)
            summary["totals"]["total_duration_ms"] += usage.get("duration_ms", 0)

        # Add synthesis usage
        if synthesis_usage:
            summary["synthesis"] = synthesis_usage
            summary["totals"]["total_tokens"] += synthesis_usage.get("total_tokens", 0)
            summary["totals"]["total_input_tokens"] += synthesis_usage.get(
                "input_tokens", 0
            )
            summary["totals"]["total_output_tokens"] += synthesis_usage.get(
                "output_tokens", 0
            )
            summary["totals"]["total_cost_usd"] += synthesis_usage.get("cost_usd", 0.0)
            summary["totals"]["total_duration_ms"] += synthesis_usage.get(
                "duration_ms", 0
            )

        return summary

    def add_manual_usage(self, agent_name: str, usage: dict[str, Any]) -> None:
        """Manually add usage data for an agent.

        Useful for script agents or when bypassing model instances.
        """
        self._agent_usage[agent_name] = usage

    def merge_usage(self, other_usage: dict[str, Any]) -> None:
        """Merge usage data from another source."""
        self._agent_usage.update(other_usage)

    def get_total_tokens(self) -> int:
        """Get total tokens across all agents."""
        return sum(usage.get("total_tokens", 0) for usage in self._agent_usage.values())

    def get_total_cost(self) -> float:
        """Get total cost across all agents."""
        total: float = 0.0
        for usage in self._agent_usage.values():
            cost = usage.get("cost_usd", 0.0)
            if isinstance(cost, int | float):
                total += float(cost)
        return total

    def get_agent_count(self) -> int:
        """Get number of agents with usage data."""
        return len(self._agent_usage)

    def has_usage_for_agent(self, agent_name: str) -> bool:
        """Check if usage data exists for a specific agent."""
        return agent_name in self._agent_usage

    def get_agent_usage_data(self, agent_name: str) -> dict[str, Any] | None:
        """Get usage data for a specific agent."""
        return self._agent_usage.get(agent_name)

    def remove_agent_usage(self, agent_name: str) -> None:
        """Remove usage data for a specific agent."""
        self._agent_usage.pop(agent_name, None)

    def get_usage_breakdown_by_metric(self) -> dict[str, dict[str, Any]]:
        """Get usage breakdown organized by metric type."""
        breakdown: dict[str, dict[str, Any]] = {
            "tokens": {},
            "costs": {},
            "durations": {},
        }

        for agent_name, usage in self._agent_usage.items():
            # Token breakdown
            breakdown["tokens"][agent_name] = {
                "total_tokens": usage.get("total_tokens", 0),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }

            # Cost breakdown
            breakdown["costs"][agent_name] = {
                "cost_usd": usage.get("cost_usd", 0.0),
            }

            # Duration breakdown
            breakdown["durations"][agent_name] = {
                "duration_ms": usage.get("duration_ms", 0),
            }

        return breakdown
