"""Results processing and output formatting for ensemble execution."""

import time
from typing import Any

from llm_orc.core.execution.patterns import INSTANCE_PATTERN


def finalize_result(
    result: dict[str, Any],
    agent_usage: dict[str, Any],
    has_errors: bool,
    start_time: float,
    adaptive_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize execution result with metadata and usage summary."""
    # Calculate usage totals (no coordinator synthesis in dependency-based model)
    usage_summary = calculate_usage_summary(agent_usage, None)

    # Finalize result
    end_time = time.time()
    result["status"] = "completed_with_errors" if has_errors else "completed"
    metadata_dict: dict[str, Any] = result["metadata"]
    metadata_dict["duration"] = f"{(end_time - start_time):.2f}s"
    metadata_dict["completed_at"] = end_time
    metadata_dict["usage"] = usage_summary

    # Add adaptive resource management statistics if available
    if adaptive_stats:
        metadata_dict["adaptive_resource_management"] = adaptive_stats

    return result


def calculate_usage_summary(
    agent_usage: dict[str, Any], synthesis_usage: dict[str, Any] | None
) -> dict[str, Any]:
    """Calculate aggregated usage summary."""
    summary = {
        "agents": agent_usage,
        "totals": {
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "total_duration_ms": 0,
            "agents_count": len(agent_usage),
        },
    }

    # Aggregate agent usage
    for usage in agent_usage.values():
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
        summary["totals"]["total_duration_ms"] += synthesis_usage.get("duration_ms", 0)

    return summary


def process_agent_results(
    agent_results: list[Any],
    results_dict: dict[str, Any],
    agent_usage: dict[str, Any],
) -> None:
    """Process results from agent execution."""
    for execution_result in agent_results:
        if isinstance(execution_result, Exception):
            # If we can't determine agent name from exception, skip this result
            # The agent task should handle its own error recording
            continue
        else:
            # execution_result is tuple[str, Any]
            agent_name, result = execution_result
            if result is None:
                # Error was already recorded in execute_agent_task
                continue

            # result is tuple[str, ModelInterface | None]
            response, model_instance = result
            results_dict[agent_name] = {
                "response": response,
                "status": "success",
            }
            # Collect usage metrics (only for LLM agents)
            if model_instance is not None:
                usage = model_instance.get_last_usage()
                if usage:
                    agent_usage[agent_name] = usage


def create_initial_result(
    ensemble_name: str, input_data: str, agent_count: int
) -> dict[str, Any]:
    """Create initial result structure."""
    start_time = time.time()
    return {
        "ensemble": ensemble_name,
        "status": "running",
        "input": {"data": input_data},
        "results": {},
        "synthesis": None,
        "metadata": {"agents_used": agent_count, "started_at": start_time},
    }


def format_execution_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Format execution summary with key metrics."""
    metadata = result.get("metadata", {})
    usage = metadata.get("usage", {})
    totals = usage.get("totals", {})

    return {
        "ensemble_name": result.get("ensemble", "unknown"),
        "status": result.get("status", "unknown"),
        "agents_count": totals.get("agents_count", 0),
        "duration": metadata.get("duration", "0.00s"),
        "total_tokens": totals.get("total_tokens", 0),
        "total_cost_usd": totals.get("total_cost_usd", 0.0),
        "has_errors": result.get("status", "") == "completed_with_errors",
    }


def get_agent_statuses(results: dict[str, Any]) -> dict[str, str]:
    """Extract agent statuses from results."""
    agent_statuses = {}
    for agent_name, result in results.items():
        if isinstance(result, dict):
            agent_statuses[agent_name] = result.get("status", "unknown")
        else:
            agent_statuses[agent_name] = "unknown"
    return agent_statuses


def count_successful_agents(results: dict[str, Any]) -> int:
    """Count the number of successful agents."""
    return sum(
        1
        for result in results.values()
        if isinstance(result, dict) and result.get("status") == "success"
    )


def count_failed_agents(results: dict[str, Any]) -> int:
    """Count the number of failed agents."""
    return sum(
        1
        for result in results.values()
        if isinstance(result, dict) and result.get("status") == "failed"
    )


def extract_agent_responses(results: dict[str, Any]) -> dict[str, str]:
    """Extract just the response content from agent results."""
    responses = {}
    for agent_name, result in results.items():
        if isinstance(result, dict) and "response" in result:
            responses[agent_name] = result["response"]
    return responses


def extract_agent_errors(results: dict[str, Any]) -> dict[str, str]:
    """Extract error messages from failed agents."""
    errors = {}
    for agent_name, result in results.items():
        if (
            isinstance(result, dict)
            and result.get("status") == "failed"
            and "error" in result
        ):
            errors[agent_name] = result["error"]
    return errors


# ========== Fan-Out Support (Issue #73) ==========


def add_fan_out_metadata(
    result: dict[str, Any],
    fan_out_stats: dict[str, Any],
) -> None:
    """Add fan-out execution statistics to result metadata.

    Args:
        result: Result dict to modify
        fan_out_stats: Dict of agent_name -> instance stats
    """
    if fan_out_stats:
        result["metadata"]["fan_out"] = fan_out_stats


def count_fan_out_instances(
    results: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Count fan-out instances in results.

    Args:
        results: Dict of agent results keyed by agent name

    Returns:
        Dict mapping original agent names to instance counts
    """
    instance_counts: dict[str, dict[str, int]] = {}

    for agent_name, result in results.items():
        match = INSTANCE_PATTERN.match(agent_name)
        if not match:
            continue

        original_name = match.group(1)

        if original_name not in instance_counts:
            instance_counts[original_name] = {
                "total_instances": 0,
                "successful_instances": 0,
                "failed_instances": 0,
            }

        instance_counts[original_name]["total_instances"] += 1

        if isinstance(result, dict):
            if result.get("status") == "success":
                instance_counts[original_name]["successful_instances"] += 1
            elif result.get("status") == "failed":
                instance_counts[original_name]["failed_instances"] += 1

    return instance_counts
