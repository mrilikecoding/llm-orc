"""Bounded loop combinator for the ensemble engine (control-flow primitive).

Re-runs a body executor under a termination policy: stop when the `until`
predicate holds over the body output, or when the mandatory iteration bound
trips. The top-level graph stays acyclic; iteration is scoped here, so the
termination guarantee replaces the acyclicity prohibition.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

BodyExecutor = Callable[[str], Awaitable[dict[str, Any]]]
Predicate = Callable[[dict[str, Any]], bool]
Carry = Callable[[dict[str, Any]], str]


@dataclass
class LoopOutcome:
    """Result of a bounded loop run."""

    output: dict[str, Any]
    iterations: int
    terminated: str  # "until" | "exhausted"


class LoopController:
    """Runs a body executor until a predicate holds or the bound trips."""

    async def run(
        self,
        body_executor: BodyExecutor,
        until: Predicate,
        max_iterations: int,
        carry: Carry | None = None,
        base_input: str = "",
    ) -> LoopOutcome:
        current_input = base_input
        output: dict[str, Any] = {}
        for iteration in range(max_iterations):
            output = await body_executor(current_input)
            if until(output):
                return LoopOutcome(output, iteration + 1, "until")
            if carry is not None:
                current_input = carry(output)
        return LoopOutcome(output, max_iterations, "exhausted")
