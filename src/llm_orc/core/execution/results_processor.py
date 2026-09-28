"""Results processing and output formatting for ensemble execution."""

import time
from typing import Any

from llm_orc.core.execution.patterns import INSTANCE_PATTERN
from llm_orc.core.execution.result_types import (
    AgentResult,
    ExecutionMetadata,
    ExecutionResult,
)
from llm_orc.core.execution.utils import SUCCEEDED_STATUSES, terminal_agent_names
from llm_orc.schemas.agent_config import AgentConfig


def finalize_result(
    result: ExecutionResult,
    agent_usage: dict[str, Any],
    has_errors: bool,
    start_time: float,
    adaptive_stats: dict[str, Any] | None = None,
) -> ExecutionResult:
    """Finalize execution result with metadata and usage summary."""
    usage_summary = calculate_usage_summary(agent_usage)

    # Finalize result
    end_time = time.time()
    result.status = "completed_with_errors" if has_errors else "completed"
    result.has_errors = has_errors
    result.metadata.duration = f"{(end_time - start_time):.2f}s"
    result.metadata.completed_at = end_time
    result.metadata.usage = usage_summary

    # Add adaptive resource management statistics if available
    if adaptive_stats:
        result.metadata.adaptive_resource_management = adaptive_stats

    return result


def calculate_usage_summary(agent_usage: dict[str, Any]) -> dict[str, Any]:
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

    return summary


def resolve_deliverable(
    results: dict[str, Any],
    agents: list[AgentConfig],
) -> str | None:
    """Resolve the ensemble's single deliverable from its dependency DAG.

    The ensemble abstraction presents one output regardless of internal
    multi-agent structure (ADR-035 D1, FC-56): a terminal node's
    response — an agent no other agent depends on — when it succeeded
    with content. Multi-terminal DAGs take the last successful terminal
    by declaration order. Returns ``None`` when no terminal succeeded
    with content (the caller decides the degraded fallback) — an
    intermediate (non-terminal) agent's output never stands in for a
    missing deliverable (fail-closed-composition, caller contract: a
    skipped/failed terminal must not surface someone else's work, e.g.
    a decomposer's raw query list, as if it were the real result).

    Computed here, where ``depends_on`` is known, rather than
    reconstructed downstream from the results dict (the dispatch layer
    receives a projection that has already dropped the config).
    """
    terminals = terminal_agent_names(agents)

    for name in reversed(terminals):
        response = _successful_response(results.get(name))
        if response is not None:
            return response
    return None


def caller_status(raw_status: str | None) -> tuple[str, bool]:
    """The caller-facing ``(status, has_errors)`` pair every invocation
    surface reports (fail-closed-composition, caller contract): REST
    (``ExecutionHandler.invoke``), MCP ``invoke`` (streamed through
    ``execute_streaming``), and the CLI's ``--output-format json``.

    One vocabulary, ``"success"`` or ``"error"``, derived from the
    executor's own ``completed``/``completed_with_errors`` distinction
    (Invariant 13's ``has_errors``) rather than a second, independently
    maintained notion of failure: ``"completed"`` is the only value that
    reads as success. ``"completed_with_errors"`` (an agent failed, or a
    dependent was skipped because none of its dependencies succeeded —
    that skip is always downstream of an already-recorded failure, so
    it never adds new information here) and ``"failed"`` (the whole
    execution task raised, e.g. streaming's ``execution_failed`` event)
    both read as error, and so does a missing/unrecognized raw status —
    a caller never sees anything outside ``{"success", "error"}``.

    A plain ``when:``-false skip is not folded in here: it carries no
    ``reason`` and never sets ``has_errors`` upstream, so it is not an
    error by construction — nothing to special-case.
    """
    has_errors = raw_status != "completed"
    return ("error" if has_errors else "success", has_errors)


def _successful_response(agent_result: Any) -> str | None:
    """An agent's response when it succeeded (SF2: ``status`` in
    ``SUCCEEDED_STATUSES``) with non-empty text, else None.

    ``SUCCEEDED_STATUSES`` includes ``"partial"`` (addendum 2026-09-23
    NIT: the membership check alone is inert for it in practice — a
    gathered fan-out's ``response`` is a list, never a string, so
    ``_non_empty_text``'s ``isinstance(response, str)`` guard excludes
    it regardless of status. A partial fan-out terminal never becomes
    the deliverable through this path; only a plain (non-fan-out)
    ``succeeded`` terminal, or a fan-out where every instance succeeded
    (its own ``status`` reads ``"success"``, not ``"partial"``), does.

    Deliberately does NOT exclude ``handled_failure`` (X1, unlike
    ``result_succeeded``): a node that ran only via
    ``on_dependency_failure: run`` still produced the right thing to
    show the caller (the fail-closed-composition caller contract:
    "deliverable = handler output") even though it does not count as a
    succeeded terminal for parent-status purposes.

    Tolerates both ``AgentResult`` objects and their serialized dict
    form, matching ``ExecutionResult.to_dict``'s posture.
    """
    if isinstance(agent_result, AgentResult):
        return _non_empty_text(agent_result.response, agent_result.status)
    if isinstance(agent_result, dict):
        return _non_empty_text(agent_result.get("response"), agent_result.get("status"))
    return None


def _non_empty_text(response: Any, status: Any) -> str | None:
    """Narrow a succeeded agent's response to non-empty text — ``str``
    only, so a gathered fan-out's list ``response`` never qualifies even
    when its ``status`` is in ``SUCCEEDED_STATUSES`` (see
    ``_successful_response``)."""
    if status not in SUCCEEDED_STATUSES:
        return None
    if isinstance(response, str) and response:
        return response
    return None


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
) -> ExecutionResult:
    """Create initial result structure."""
    start_time = time.time()
    return ExecutionResult(
        ensemble=ensemble_name,
        status="running",
        input={"data": input_data},
        results={},
        metadata=ExecutionMetadata(
            agents_used=agent_count,
            started_at=start_time,
        ),
    )


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
        result: Result dict to modify (plain dict, used in tests)
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
