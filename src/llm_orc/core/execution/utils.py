"""Shared utility functions for execution components."""

from typing import Any

from llm_orc.schemas.agent_config import AgentConfig

# A gathered fan-out dependency where some instances succeeded and some
# failed (fan_out/gatherer.py's "partial") counts as a successful
# dependency: GuardEvaluator's cascade rule lets a consumer run on it, and
# DependencyResolver renders it through the same per-terminal path as a
# fully successful one. Only a fan-out where EVERY instance failed
# ("failed") does not (fail-closed-composition, partial fan-out decision).
SUCCEEDED_STATUSES = ("success", "partial")


def dep_name(dep: str | dict[str, Any]) -> str:
    """Extract the agent name from a dependency entry.

    Dependency entries are either a plain string or a dict with a single
    ``"agent_name"`` key, e.g. ``{"agent_name": "b"}`` for conditional deps.
    """
    if isinstance(dep, dict):
        return str(dep["agent_name"])
    return dep


def terminal_agent_names(agents: list[AgentConfig]) -> list[str]:
    """Names of agents no other agent in ``agents`` depends on, in
    declaration order — the DAG's terminal nodes.

    One definition, three readers: ``resolve_deliverable``'s single
    collapsed deliverable, ``EnsembleExecutor._terminal_agents_for_
    ensemble`` (every terminal, for DependencyResolver's per-terminal
    rendering), and the ``ensemble:`` agent's own success rule
    (fail-closed-composition B1) — FAILS when none of its child's
    terminals succeeded.
    """
    depended_on = {dep_name(dep) for agent in agents for dep in agent.depends_on}
    return [agent.name for agent in agents if agent.name not in depended_on]


def terminal_failure_summary(
    agents: list[AgentConfig], results: dict[str, Any]
) -> str | None:
    """None when at least one of ``agents``' terminal agents succeeded in
    ``results`` (fail-closed-composition B1); otherwise a summary of each
    terminal's status/error, naming why none did.

    Shared by ``EnsembleAgentRunner`` (a static ``ensemble:`` reference)
    and ``DynamicDispatchRunner`` (a runtime-resolved ``dispatch:``
    target) — both hand a child executor's full result upward as their
    own "success", so both need the same check: an intermediate agent
    failing does not fail the parent as long as a terminal still
    succeeded (Invariant 13 already lets a terminal run on its other
    successful dependencies) — only terminals are checked here. A child
    with no terminals at all (an empty ensemble) falls open: there is
    nothing to name a failure against.
    """
    terminals = terminal_agent_names(agents)
    if not terminals:
        return None
    if any(
        isinstance(results.get(name), dict) and results[name].get("status") == "success"
        for name in terminals
    ):
        return None
    return "; ".join(
        _terminal_status_text(name, results.get(name)) for name in terminals
    )


def _terminal_status_text(name: str, result: Any) -> str:
    """``name (status): error`` for a failed/skipped terminal, or
    ``name (missing)`` when the child never recorded it at all."""
    if not isinstance(result, dict):
        return f"{name} (missing)"
    status = result.get("status", "missing")
    error = result.get("error")
    return f"{name} ({status}): {error}" if error else f"{name} ({status})"


def resolve_agent_timeout(
    agent_config: dict[str, Any], performance_config: dict[str, Any]
) -> int:
    """Seconds an agent may run: its own ``timeout_seconds`` when set,
    else the operator's ``performance.execution.default_timeout``, else
    60.

    One rule, one home. The dispatcher applies it as an outer bound and
    the script-agent runner applies it as the subprocess bound; two
    answers to "how long may this agent take" is how a script agent came
    to have no bound at all (#157). ``None`` means unset and defers —
    never a value in its own right.
    """
    timeout = agent_config.get("timeout_seconds")
    if timeout is not None:
        return int(timeout)
    default = performance_config.get("execution", {}).get("default_timeout", 60)
    return int(default)
