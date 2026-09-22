"""Guard predicate evaluation for conditional node execution (control-flow).

The `when:` field on an agent is evaluated here against accumulated upstream
results to decide whether a node runs or is skipped at execution time. This is
the conditional-execution primitive: deterministic, no model involvement.
"""

from __future__ import annotations

from typing import Any

from llm_orc.core.execution.phases import predicate
from llm_orc.core.execution.phases.reference import resolve_reference
from llm_orc.core.execution.utils import SUCCEEDED_STATUSES, dep_name
from llm_orc.schemas.agent_config import AgentConfig


class GuardEvaluator:
    """Decides whether a node runs, given accumulated upstream results."""

    def should_run(
        self, agent_config: AgentConfig, results_dict: dict[str, Any]
    ) -> bool:
        if self._no_dependency_succeeded(agent_config, results_dict):
            return False
        when = agent_config.when
        if when is None:
            return True
        return predicate.evaluate(
            when, lambda token: resolve_reference(token, results_dict)
        )

    def dependency_skip_reason(
        self, agent_config: AgentConfig, results_dict: dict[str, Any]
    ) -> str | None:
        """Why the agent is skipped by dependency cascading (fail-closed-
        composition rule 1): every dependency is failed or skipped, in any
        mix — none succeeded. Names each upstream agent and its status,
        with error text for failed ones. None when the agent is not
        skipped by this rule — in particular, a `when:`-false skip has no
        reason here (rule 1 is evaluated before `when:`, so the two never
        both apply to the same skip).
        """
        if not self._no_dependency_succeeded(agent_config, results_dict):
            return None
        deps = [dep_name(d) for d in agent_config.depends_on]
        named = ", ".join(
            self._dep_status_text(dep, results_dict.get(dep)) for dep in deps
        )
        return f"no dependency succeeded: {named}"

    def _no_dependency_succeeded(
        self, agent_config: AgentConfig, results_dict: dict[str, Any]
    ) -> bool:
        deps = [dep_name(d) for d in agent_config.depends_on]
        return bool(deps) and not any(
            self._status_of(results_dict.get(d)) in SUCCEEDED_STATUSES for d in deps
        )

    @staticmethod
    def _status_of(result: Any) -> str | None:
        if isinstance(result, dict):
            status = result.get("status")
        else:
            status = getattr(result, "status", None)
        return status if isinstance(status, str) else None

    @staticmethod
    def _error_of(result: Any) -> str | None:
        if isinstance(result, dict):
            error = result.get("error")
        else:
            error = getattr(result, "error", None)
        return error if isinstance(error, str) else None

    @classmethod
    def _dep_status_text(cls, dep: str, result: Any) -> str:
        status = cls._status_of(result) or "missing"
        error = cls._error_of(result) if status == "failed" else None
        return f"{dep} ({status}: {error})" if error else f"{dep} ({status})"
