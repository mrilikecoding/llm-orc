"""Guard predicate evaluation for conditional node execution (control-flow).

The `when:` field on an agent is evaluated here against accumulated upstream
results to decide whether a node runs or is skipped at execution time. This is
the conditional-execution primitive: deterministic, no model involvement.
"""

from __future__ import annotations

from typing import Any

from llm_orc.core.execution.phases import predicate
from llm_orc.core.execution.phases.reference import resolve_reference
from llm_orc.core.execution.utils import dep_name
from llm_orc.schemas.agent_config import AgentConfig

SKIPPED = "skipped"


class GuardEvaluator:
    """Decides whether a node runs, given accumulated upstream results."""

    def should_run(
        self, agent_config: AgentConfig, results_dict: dict[str, Any]
    ) -> bool:
        deps = [dep_name(d) for d in agent_config.depends_on]
        if deps and all(self._is_skipped(results_dict.get(d)) for d in deps):
            return False
        when = agent_config.when
        if when is None:
            return True
        return predicate.evaluate(
            when, lambda token: resolve_reference(token, results_dict)
        )

    @staticmethod
    def _is_skipped(result: Any) -> bool:
        if isinstance(result, dict):
            return result.get("status") == SKIPPED
        return getattr(result, "status", None) == SKIPPED
