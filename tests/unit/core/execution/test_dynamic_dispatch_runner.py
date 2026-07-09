"""Tests for the dynamic-dispatch runner.

The runner is the execution half of the dynamic-dispatch primitive: it reads the
target ensemble name resolved at the phase layer (``dispatch_resolved``), loads
that ensemble, and executes it through a child executor. Mirrors
EnsembleAgentRunner, differing only in that the target is runtime-resolved.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.runners.dynamic_dispatch_runner import (
    DynamicDispatchRunner,
)
from llm_orc.schemas.agent_config import DynamicDispatchAgentConfig


class _FakeChildExecutor:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.executed_with: tuple[Any, str] | None = None

    async def execute(self, config: Any, input_data: str) -> dict[str, Any]:
        self.executed_with = (config, input_data)
        return self._result


class _FakeParent:
    def __init__(self, child: _FakeChildExecutor) -> None:
        self._child = child
        self.child_depth: int | None = None

    def create_child_executor(self, depth: int) -> _FakeChildExecutor:
        self.child_depth = depth
        return self._child


def _dispatch_config(resolved: str | None) -> DynamicDispatchAgentConfig:
    config = DynamicDispatchAgentConfig(name="seat", dispatch="${classify.target}")
    return config.model_copy(update={"dispatch_resolved": resolved})


def _loader(result: object) -> Callable[[str], EnsembleConfig]:
    """A fake ensemble resolver returning a sentinel config for any name."""
    return lambda name: cast(EnsembleConfig, result)


class TestDynamicDispatchRunner:
    async def test_executes_resolved_ensemble(self) -> None:
        child = _FakeChildExecutor({"result": "ok"})
        parent = _FakeParent(child)
        sentinel_config = object()
        runner = DynamicDispatchRunner(
            ensemble_loader=_loader(sentinel_config),
            parent_executor=parent,  # type: ignore[arg-type]
        )

        response, model, substituted = await runner.execute(
            _dispatch_config("seat-a"), "the input"
        )

        assert response == json.dumps({"result": "ok"})
        assert model is None
        assert substituted is False
        assert parent.child_depth == 1
        assert child.executed_with == (sentinel_config, "the input")

    async def test_unresolved_target_raises(self) -> None:
        runner = DynamicDispatchRunner(
            ensemble_loader=_loader(object()),
            parent_executor=_FakeParent(_FakeChildExecutor({})),  # type: ignore[arg-type]
        )
        with pytest.raises(RuntimeError, match="not resolved"):
            await runner.execute(_dispatch_config(None), "x")

    async def test_depth_limit_exceeded_raises(self) -> None:
        runner = DynamicDispatchRunner(
            ensemble_loader=_loader(object()),
            parent_executor=_FakeParent(_FakeChildExecutor({})),  # type: ignore[arg-type]
            current_depth=5,
            depth_limit=5,
        )
        with pytest.raises(RuntimeError, match="depth limit"):
            await runner.execute(_dispatch_config("seat-a"), "x")
