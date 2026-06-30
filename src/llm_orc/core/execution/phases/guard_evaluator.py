"""Guard predicate evaluation for conditional node execution (control-flow).

The `when:` field on an agent is evaluated here against accumulated upstream
results to decide whether a node runs or is skipped at execution time. This is
the conditional-execution primitive: deterministic, no model involvement.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_orc.core.execution.utils import dep_name
from llm_orc.schemas.agent_config import AgentConfig

_REF = re.compile(r"^\$\{([^.}]+)\.([^}]+)\}$")

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
        return self._evaluate(when.strip(), results_dict)

    @staticmethod
    def _is_skipped(result: Any) -> bool:
        if isinstance(result, dict):
            return result.get("status") == SKIPPED
        return getattr(result, "status", None) == SKIPPED

    def _evaluate(self, expr: str, results_dict: dict[str, Any]) -> bool:
        if "==" in expr:
            left, right = expr.split("==", 1)
            resolved = self._resolve(left.strip(), results_dict)
            return bool(resolved == self._literal(right.strip()))
        return bool(self._resolve(expr, results_dict))

    @staticmethod
    def _literal(token: str) -> Any:
        if token == "true":
            return True
        if token == "false":
            return False
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
            return token[1:-1]
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    def _resolve(self, token: str, results_dict: dict[str, Any]) -> Any:
        match = _REF.match(token)
        if not match:
            return token
        dep, field = match.group(1), match.group(2)
        result = results_dict.get(dep, {})
        parsed = json.loads(result.get("response", ""))
        value: Any = parsed
        for part in field.split("."):
            value = value[part]
        return value
