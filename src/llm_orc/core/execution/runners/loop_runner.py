"""Loop agent runner — re-runs a body ensemble under a bound (ADR control-flow).

Mirrors EnsembleAgentRunner: resolve the body ensemble, run it once per
iteration via a child executor, and drive LoopController until the `until`
predicate holds over the body's terminal output or the iteration bound trips.
`until`/`carry` are evaluated against the body output dict via the shared
predicate grammar.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.phases import predicate
from llm_orc.core.execution.phases.loop_controller import (
    Carry,
    LoopController,
    Predicate,
)
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import LoopAgentConfig

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

ResolveEnsembleFn = Callable[[str], EnsembleConfig]
_OUT_REF = re.compile(r"^\$\{([^}]+)\}$")


class LoopAgentRunner:
    """Runs loop agents by re-executing a body ensemble under a bound."""

    def __init__(
        self,
        ensemble_loader: ResolveEnsembleFn | None = None,
        parent_executor: EnsembleExecutor | None = None,
        controller: LoopController | None = None,
        current_depth: int = 0,
        depth_limit: int = 5,
    ) -> None:
        self._resolve = ensemble_loader
        self._parent = parent_executor
        self._controller = controller or LoopController()
        self._current_depth = current_depth
        self._depth_limit = depth_limit

    async def execute(
        self,
        agent_config: LoopAgentConfig,
        input_data: str,
    ) -> tuple[str, ModelInterface | None, bool]:
        """Run the loop and return its outcome as a JSON response."""
        child_depth = self._current_depth + 1
        if child_depth > self._depth_limit:
            msg = (
                f"Ensemble nesting depth limit exceeded: "
                f"depth {child_depth} > limit {self._depth_limit}"
            )
            raise RuntimeError(msg)
        if self._resolve is None or self._parent is None:
            msg = "LoopAgentRunner not configured"
            raise RuntimeError(msg)

        spec = agent_config.loop
        body_config = self._resolve(spec.body)
        parent = self._parent

        async def body_executor(inp: str) -> dict[str, Any]:
            child = parent.create_child_executor(depth=child_depth)
            child_result = await child.execute(body_config, inp)
            return self._terminal_output(child_result)

        outcome = await self._controller.run(
            body_executor,
            self._compile_until(spec.until),
            spec.max_iterations,
            self._compile_carry(spec.carry),
            input_data,
        )
        return (
            json.dumps(
                {
                    "output": outcome.output,
                    "iterations": outcome.iterations,
                    "terminated": outcome.terminated,
                }
            ),
            None,
            False,
        )

    @staticmethod
    def _terminal_output(child_result: dict[str, Any]) -> dict[str, Any]:
        deliverable = child_result.get("deliverable")
        if deliverable is None:
            return {}
        try:
            parsed = json.loads(deliverable)
        except (json.JSONDecodeError, TypeError):
            return {"value": deliverable}
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    def _compile_until(self, until: str) -> Predicate:
        return lambda output: predicate.evaluate(
            until, lambda token: self._resolve_output(token, output)
        )

    def _compile_carry(self, carry: str | None) -> Carry | None:
        if carry is None:
            return None
        return lambda output: self._stringify(self._resolve_output(carry, output))

    @staticmethod
    def _resolve_output(token: str, output: dict[str, Any]) -> Any:
        match = _OUT_REF.match(token)
        if not match:
            return token
        value: Any = output
        for part in match.group(1).split("."):
            value = value[part] if isinstance(value, dict) and part in value else None
        return value

    @staticmethod
    def _stringify(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value)
