"""Tests for the bounded loop combinator orchestration."""

from __future__ import annotations

from typing import Any

from llm_orc.core.execution.phases.loop_controller import LoopController


class TestLoopTermination:
    """The loop stops on `until` or on the iteration bound."""

    async def test_stops_when_until_true_on_first_iteration(self) -> None:
        controller = LoopController()
        seen: list[str] = []

        async def body(inp: str) -> dict[str, Any]:
            seen.append(inp)
            return {"ok": True}

        outcome = await controller.run(
            body,
            until=lambda o: bool(o["ok"]),
            max_iterations=3,
            base_input="go",
        )
        assert outcome.terminated == "until"
        assert outcome.iterations == 1
        assert seen == ["go"]

    async def test_exhausts_and_returns_last_output(self) -> None:
        controller = LoopController()
        count = 0

        async def body(_inp: str) -> dict[str, Any]:
            nonlocal count
            count += 1
            return {"ok": False, "n": count}

        outcome = await controller.run(
            body, until=lambda o: bool(o["ok"]), max_iterations=2
        )
        assert outcome.terminated == "exhausted"
        assert outcome.iterations == 2
        assert outcome.output["n"] == 2

    async def test_carries_state_and_stops_on_later_iteration(self) -> None:
        controller = LoopController()
        seen: list[str] = []

        async def body(inp: str) -> dict[str, Any]:
            seen.append(inp)
            return {"ok": inp == "fix", "reasons": "fix"}

        outcome = await controller.run(
            body,
            until=lambda o: bool(o["ok"]),
            max_iterations=3,
            carry=lambda o: str(o["reasons"]),
            base_input="start",
        )
        # iteration 1 gets the base input, iteration 2 gets the carried value
        assert seen == ["start", "fix"]
        assert outcome.iterations == 2
        assert outcome.terminated == "until"
