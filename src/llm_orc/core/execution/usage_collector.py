"""Usage tracking and aggregation for agent execution."""

import copy
from typing import Any

from llm_orc.models.base import ModelInterface


class UsageCollector:
    """Collects and aggregates usage metrics from agent execution."""

    def __init__(self) -> None:
        """Initialize usage collector."""
        self._agent_usage: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset collected usage data."""
        self._agent_usage.clear()

    def collect_agent_usage(
        self, agent_name: str, model_instance: ModelInterface | None
    ) -> None:
        """Collect usage metrics from a model instance."""
        if model_instance is not None and hasattr(model_instance, "get_last_usage"):
            usage = model_instance.get_last_usage()
            if usage:
                self._agent_usage[agent_name] = usage

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
        return sum(usage.get("cost_usd", 0.0) for usage in self._agent_usage.values())

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
        breakdown = {
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

