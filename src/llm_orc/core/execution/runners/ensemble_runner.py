"""Ensemble agent runner — executes a child ensemble (ADR-013)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import EnsembleAgentConfig

if TYPE_CHECKING:
    from llm_orc.core.execution.ensemble_execution import (
        EnsembleExecutor,
    )

# Callable that resolves an ensemble name to its config
ResolveEnsembleFn = Callable[[str], EnsembleConfig]


class EnsembleAgentRunner:
    """Runs ensemble agents by recursively executing child ensembles.

    A thin adapter: resolve the ensemble reference, create a child
    executor, execute, and return the result as JSON.
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
        agent_config: EnsembleAgentConfig,
        input_data: str,
    ) -> tuple[str, ModelInterface | None, bool]:
        """Execute a child ensemble and return its result as JSON.

        Returns:
            Tuple of (JSON-serialized result dict, None, False).
            Model instance and model_substituted are always None/False
            for ensemble agents.
        """
        child_depth = self._current_depth + 1
        if child_depth > self._depth_limit:
            msg = (
                f"Ensemble nesting depth limit exceeded: "
                f"depth {child_depth} > limit {self._depth_limit}"
            )
            raise RuntimeError(msg)

        if self._resolve is None or self._parent is None:
            msg = "EnsembleAgentRunner not configured"
            raise RuntimeError(msg)

        # Resolve ensemble reference to config
        child_config = self._resolve(agent_config.ensemble)

        # Create child executor (shares immutable, isolates mutable)
        child_executor = self._parent.create_child_executor(depth=child_depth)

        # Execute child ensemble
        child_result = await child_executor.execute(child_config, input_data)

        # Return full result dict as JSON
        return json.dumps(child_result), None, False
