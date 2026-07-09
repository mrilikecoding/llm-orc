"""Dynamic-dispatch runner — executes a runtime-resolved child ensemble.

The execution half of the dynamic-dispatch primitive. The target ensemble name
is resolved at the phase layer (guard-sibling ``${dep.field}`` resolution against
``results_dict``) and stashed on ``dispatch_resolved``; this runner reads it,
loads that ensemble, and executes it through a child executor. Mirrors
EnsembleAgentRunner, differing only in where the target name comes from
(runtime-resolved rather than the load-time ``ensemble`` reference).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import DynamicDispatchAgentConfig

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import EnsembleExecutor

# Callable that resolves an ensemble name to its config
ResolveEnsembleFn = Callable[[str], EnsembleConfig]


class DynamicDispatchRunner:
    """Runs dynamic-dispatch agents by executing a runtime-resolved ensemble.

    A thin adapter: read the phase-layer-resolved target name, resolve it to a
    config, create a child executor, execute, and return the result as JSON.
    """

    def __init__(
        self,
        ensemble_loader: ResolveEnsembleFn | None = None,
        parent_executor: EnsembleExecutor | None = None,
        current_depth: int = 0,
        depth_limit: int = 5,
    ) -> None:
        self._resolve = ensemble_loader
        self._parent = parent_executor
        self._current_depth = current_depth
        self._depth_limit = depth_limit

    async def execute(
        self,
        agent_config: DynamicDispatchAgentConfig,
        input_data: str,
    ) -> tuple[str, ModelInterface | None, bool]:
        """Execute the runtime-resolved ensemble and return its result as JSON.

        Returns:
            Tuple of (JSON-serialized result dict, None, False). Model instance
            and model_substituted are always None/False for dispatch agents.
        """
        child_depth = self._current_depth + 1
        if child_depth > self._depth_limit:
            msg = (
                f"Ensemble nesting depth limit exceeded: "
                f"depth {child_depth} > limit {self._depth_limit}"
            )
            raise RuntimeError(msg)

        if self._resolve is None or self._parent is None:
            msg = "DynamicDispatchRunner not configured"
            raise RuntimeError(msg)

        target = agent_config.dispatch_resolved
        if target is None:
            msg = (
                f"Dispatch target for '{agent_config.name}' was not resolved "
                f"(dispatch: {agent_config.dispatch!r})"
            )
            raise RuntimeError(msg)

        child_config = self._resolve(target)
        child_executor = self._parent.create_child_executor(depth=child_depth)
        child_result = await child_executor.execute(child_config, input_data)

        return json.dumps(child_result), None, False
