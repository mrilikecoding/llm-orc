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


def result_succeeded(result: Any) -> bool:
    """One predicate for "this result counts as a succeeded terminal"
    (fail-closed-composition SF2/X1): ``status`` in ``SUCCEEDED_STATUSES``
    and not ``handled_failure``.

    ``handled_failure`` (X1) marks a node that executed ONLY because
    ``on_dependency_failure: run`` waived rule 1's cascade — none of its
    own dependencies succeeded. It still reports its OWN status honestly
    (usually ``success``: it ran and produced the refusal/deliverable),
    but that status must not read as forward progress to whatever
    consults it next — a sibling cascade check, or the parent
    ``ensemble:``/``dispatch:``/``loop:`` node's own success rule. Both
    read this same predicate: ``GuardEvaluator``'s dependency-succeeded
    check (so a consumer past the handler also sees the failure, not a
    false success) and ``terminal_failure_summary`` (so the WRAPPING
    node fails, naming the real upstream failure, instead of reporting
    success because its handler ran without incident).

    ``resolve_deliverable`` deliberately does NOT use this predicate: a
    handler's own output is still the right thing to show the caller
    (fail-closed-composition contract: "deliverable = handler output")
    even though it does not count as a succeeded terminal here.

    Tolerates both a plain dict result and an object exposing ``status``/
    ``handled_failure`` attributes (``GuardEvaluator`` sees both forms).
    """
    status = (
        result.get("status")
        if isinstance(result, dict)
        else (getattr(result, "status", None))
    )
    if status not in SUCCEEDED_STATUSES:
        return False
    handled_failure = (
        result.get("handled_failure")
        if isinstance(result, dict)
        else getattr(result, "handled_failure", False)
    )
    return not handled_failure


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
    ``results`` (fail-closed-composition B1, unified via
    ``result_succeeded``: partial counts, ``handled_failure`` does not —
    X1); otherwise a summary of each GENUINELY failed/cascade-skipped
    terminal's status/error, naming why none did.

    Shared by ``EnsembleAgentRunner`` (a static ``ensemble:`` reference),
    ``DynamicDispatchRunner`` (a runtime-resolved ``dispatch:`` target),
    and ``LoopAgentRunner`` (a ``loop:`` body's final iteration) — all
    three hand a child executor's full result upward as their own
    "success", so all three need the same check: an intermediate agent
    failing does not fail the parent as long as a terminal still
    succeeded (Invariant 13 already lets a terminal run on its other
    successful dependencies) — only terminals are checked here. A child
    with no terminals at all (an empty ensemble) falls open: there is
    nothing to name a failure against.

    A terminal skipped by a plain ``when:``-false guard (no ``reason`` —
    ``GuardEvaluator`` never gives a dependency-cascade skip a bare
    ``when:``) is neither success nor failure (SF2, the whenskip/
    whenparent decision): a child whose ONLY terminals are when-skipped
    reports no failure here either, matching how a top-level invocation
    of that same child already reads (``completed``, no deliverable) —
    nested and top-level agree instead of the nested wrapper alone
    treating an intentional skip as "produced no successful terminal
    agent". A mix — some terminals genuinely failed or were cascade-
    skipped, none succeeded — still fails, naming only the genuine
    failures.
    """
    terminals = terminal_agent_names(agents)
    if not terminals:
        return None
    entries = [(name, results.get(name)) for name in terminals]
    if any(result_succeeded(result) for _name, result in entries):
        return None
    failures = [(name, result) for name, result in entries if not _is_when_skip(result)]
    if not failures:
        return None
    return "; ".join(_terminal_status_text(name, result) for name, result in failures)


def _is_when_skip(result: Any) -> bool:
    """A plain ``when:``-false skip: ``status == "skipped"`` with no
    ``reason`` — ``GuardEvaluator`` only sets ``reason`` for a rule-1
    dependency-cascade skip, never for a ``when:`` guard (SF2)."""
    return (
        isinstance(result, dict)
        and result.get("status") == "skipped"
        and result.get("reason") is None
    )


def _terminal_status_text(name: str, result: Any) -> str:
    """``name (status): error`` for a failed/skipped terminal, or
    ``name (missing)`` when the child never recorded it at all.

    A ``handled_failure`` terminal (X1) names the real upstream failure
    it handled (``handled_failure_reason``) instead of its own
    unremarkable execution status — a parent failing over it should say
    what actually went wrong, not "worker (success)".
    """
    if not isinstance(result, dict):
        return f"{name} (missing)"
    if result.get("handled_failure"):
        reason = result.get("handled_failure_reason")
        detail = f": {reason}" if reason else ""
        return f"{name} (handled failure{detail})"
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
