"""Tests for the loop runner's compilation and body-output extraction."""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.runners.loop_runner import LoopAgentRunner
from llm_orc.schemas.agent_config import LoopAgentConfig, LoopSpec, ScriptAgentConfig


class TestTerminalOutput:
    def test_parses_json_deliverable(self) -> None:
        out = LoopAgentRunner._terminal_output({"deliverable": '{"ok": true}'})
        assert out == {"ok": True}

    def test_missing_deliverable_is_empty(self) -> None:
        assert LoopAgentRunner._terminal_output({}) == {}

    def test_non_json_deliverable_is_wrapped(self) -> None:
        out = LoopAgentRunner._terminal_output({"deliverable": "plain text"})
        assert out == {"value": "plain text"}


class TestUntilCompilation:
    def test_truthiness(self) -> None:
        until = LoopAgentRunner()._compile_until("${ok}")
        assert until({"ok": True}) is True
        assert until({"ok": False}) is False

    def test_equality(self) -> None:
        until = LoopAgentRunner()._compile_until('${choice} == "code"')
        assert until({"choice": "code"}) is True
        assert until({"choice": "prose"}) is False


class TestCarryCompilation:
    def test_none_carry_compiles_to_none(self) -> None:
        assert LoopAgentRunner()._compile_carry(None) is None

    def test_extracts_string_field(self) -> None:
        carry = LoopAgentRunner()._compile_carry("${reasons}")
        assert carry is not None
        assert carry({"reasons": "fix the import"}) == "fix the import"

    def test_stringifies_structured_field(self) -> None:
        carry = LoopAgentRunner()._compile_carry("${data}")
        assert carry is not None
        assert carry({"data": {"x": 1}}) == '{"x": 1}'


class _FakeChildExecutor:
    """Returns each of ``results`` in turn, one per ``execute`` call; the
    last entry repeats once exhausted."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def execute(self, _config: Any, input_data: str) -> dict[str, Any]:
        self.calls.append(input_data)
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]


class _FakeParent:
    def __init__(self, child: _FakeChildExecutor) -> None:
        self._child = child

    def create_child_executor(self, depth: int) -> _FakeChildExecutor:
        return self._child


def _body_config() -> EnsembleConfig:
    return EnsembleConfig(
        name="body",
        description="loop body",
        agents=[ScriptAgentConfig(name="worker", script="echo x")],
    )


def _loop_config(max_iterations: int) -> LoopAgentConfig:
    return LoopAgentConfig(
        name="looper",
        loop=LoopSpec(
            body="body", until="${ok}", max_iterations=max_iterations, carry=None
        ),
    )


class TestLoopBodyFailureIsAgentFailure:
    """A loop body whose terminal never succeeds on the final iteration is
    a loop-agent failure (fail-closed-composition B1, extended to
    ``loop:`` — ``LoopAgentRunner`` used to hand the last iteration's
    output upward as an unconditional "success" the same way
    ``EnsembleAgentRunner``/``DynamicDispatchRunner`` did before B1)."""

    @pytest.mark.asyncio
    async def test_every_iteration_failing_fails_the_agent(self) -> None:
        failing_result = {
            "status": "completed_with_errors",
            "results": {"worker": {"status": "failed", "error": "boom"}},
            "metadata": {},
            "deliverable": None,
        }
        child = _FakeChildExecutor([failing_result])
        runner = LoopAgentRunner(
            ensemble_loader=lambda _n: _body_config(),
            parent_executor=_FakeParent(child),  # type: ignore[arg-type]
            current_depth=0,
            depth_limit=5,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await runner.execute(_loop_config(max_iterations=3), "start")

        assert "worker" in str(exc_info.value)
        assert "boom" in str(exc_info.value)
        # every iteration ran to the bound — a failing body does not
        # short-circuit the loop early, only the final check does
        assert len(child.calls) == 3

    @pytest.mark.asyncio
    async def test_final_iteration_failing_fails_the_agent_even_after_earlier_success(
        self,
    ) -> None:
        succeeding_result = {
            "status": "completed",
            "results": {"worker": {"status": "success", "response": '{"ok": false}'}},
            "metadata": {},
            "deliverable": '{"ok": false}',
        }
        failing_result = {
            "status": "completed_with_errors",
            "results": {"worker": {"status": "failed", "error": "boom on last"}},
            "metadata": {},
            "deliverable": None,
        }
        child = _FakeChildExecutor([succeeding_result, failing_result])
        runner = LoopAgentRunner(
            ensemble_loader=lambda _n: _body_config(),
            parent_executor=_FakeParent(child),  # type: ignore[arg-type]
            current_depth=0,
            depth_limit=5,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await runner.execute(_loop_config(max_iterations=2), "start")

        assert "worker" in str(exc_info.value)
        assert "boom on last" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_final_iteration_succeeding_is_agent_success(self) -> None:
        succeeding_result = {
            "status": "completed",
            "results": {"worker": {"status": "success", "response": '{"ok": true}'}},
            "metadata": {},
            "deliverable": '{"ok": true}',
        }
        child = _FakeChildExecutor([succeeding_result])
        runner = LoopAgentRunner(
            ensemble_loader=lambda _n: _body_config(),
            parent_executor=_FakeParent(child),  # type: ignore[arg-type]
            current_depth=0,
            depth_limit=5,
        )

        response, model, substituted = await runner.execute(
            _loop_config(max_iterations=3), "start"
        )

        assert model is None
        assert substituted is False
        assert "until" in response
