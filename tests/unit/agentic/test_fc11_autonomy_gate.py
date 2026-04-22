"""Static inspection tests for Fitness Criterion FC-11.

Per ``docs/agentic-serving/system-design.md`` §Fitness Criteria, FC-11:

    Autonomy Policy check executes before every Orchestrator Tool
    Dispatch | Integration test | 100% of dispatches | ADR-008.

FC-11's structural property is dispatch-side: every route to a tool
method must flow through :class:`AutonomyPolicy.decide`. The
TOOL_NAMES unknown-tool short-circuit in ``dispatch`` returns
``unknown_tool`` without consulting Autonomy — by design, AS-6 closure
lives in the closed set, not in the gate. The tests below enforce the
check for the *routed* path: any ``await self._route(...)`` call in
``dispatch`` must be lexically preceded by
``self._autonomy_policy.decide(...)``.

Mirrors the AST-dominance pattern from ``test_fc8_summarizer_bypass.py``:

1. ``dispatch`` calls ``self._autonomy_policy.decide`` at least once —
   without a call site the ordering check is vacuous.
2. Every ``await self._route(...)`` in ``dispatch`` is lexically after
   the decide call — a fast-path bypass (e.g. a cached-result
   early-return that skips the gate) trips this check.
3. Adversarial self-test — the detector catches a synthetic bypass.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import TypeGuard

from llm_orc.agentic import orchestrator_tool_dispatch

_DISPATCH_MODULE_PATH = Path(orchestrator_tool_dispatch.__file__)

_DISPATCH_CLASS = "OrchestratorToolDispatch"
_DISPATCH_METHOD = "dispatch"
_ROUTE_METHOD = "_route"
_POLICY_ATTR = "_autonomy_policy"
_DECIDE_METHOD = "decide"


def _parse_dispatch() -> ast.Module:
    return ast.parse(_DISPATCH_MODULE_PATH.read_text())


def _find_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(
        f"{class_name}.{method_name} not found in "
        f"{_DISPATCH_MODULE_PATH.name} — FC-11's subject has moved."
    )


def _is_policy_decide_call(node: ast.AST) -> TypeGuard[ast.expr]:
    """Match ``self._autonomy_policy.decide(...)`` — the gate consultation."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != _DECIDE_METHOD:
        return False
    inner = func.value
    if not isinstance(inner, ast.Attribute) or inner.attr != _POLICY_ATTR:
        return False
    return isinstance(inner.value, ast.Name) and inner.value.id == "self"


def _is_self_route_await(node: ast.AST) -> TypeGuard[ast.expr]:
    """Match ``await self._route(...)`` — the dispatch-to-tool-method hop."""
    if not isinstance(node, ast.Await):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != _ROUTE_METHOD:
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "self"


class TestFc11DecidePrecedesRoute:
    """``dispatch`` must gate every routed path through Autonomy Policy."""

    def test_dispatch_calls_autonomy_policy_decide(self) -> None:
        """Without a call site the ordering check is vacuously true.

        Keeping the gate call load-bearing means a regression that
        removed the ``self._autonomy_policy.decide`` call outright is
        caught here before the ordering check weakens into a no-op.
        """
        tree = _parse_dispatch()
        method = _find_method(tree, _DISPATCH_CLASS, _DISPATCH_METHOD)
        decide_calls = [
            node for node in ast.walk(method) if _is_policy_decide_call(node)
        ]
        assert decide_calls, (
            f"{_DISPATCH_CLASS}.{_DISPATCH_METHOD} no longer calls "
            "`self._autonomy_policy.decide(...)`. FC-11 (ADR-008) requires "
            "the gate to fire before routing — removing the call means "
            "dispatch routes to tool methods without consulting Autonomy."
        )

    def test_dispatch_routes_exactly_via_self_route(self) -> None:
        """Routing hops go through ``_route``, never directly to tool methods.

        The ``_route`` indirection is what makes the ordering check
        tractable — one call site to reason about. A regression that
        inlined ``_route`` back into ``dispatch`` (or added a parallel
        direct-call path) would leave the decide/route ordering
        untestable by line number. This test catches that.
        """
        tree = _parse_dispatch()
        method = _find_method(tree, _DISPATCH_CLASS, _DISPATCH_METHOD)
        route_awaits = [node for node in ast.walk(method) if _is_self_route_await(node)]
        assert route_awaits, (
            f"{_DISPATCH_CLASS}.{_DISPATCH_METHOD} does not call "
            "`await self._route(...)`. FC-11 expects the dispatch method to "
            "route through _route so the decide→route ordering is a single "
            "lexical check; re-anchor the test if the routing hop has moved."
        )

    def test_decide_precedes_every_route_await(self) -> None:
        """Lexical ordering: every ``await self._route(...)`` is after decide.

        Catches the class of regressions a loose ``decide is called
        somewhere`` check would miss: an early-return fast path that
        hits ``_route`` before reaching the gate, or a branch that
        short-circuits the gate for a specific tool name.
        """
        tree = _parse_dispatch()
        method = _find_method(tree, _DISPATCH_CLASS, _DISPATCH_METHOD)

        decide_lines = [
            node.lineno for node in ast.walk(method) if _is_policy_decide_call(node)
        ]
        route_lines = [
            node.lineno for node in ast.walk(method) if _is_self_route_await(node)
        ]
        assert decide_lines, "guarded by test_dispatch_calls_autonomy_policy_decide"
        assert route_lines, "guarded by test_dispatch_routes_exactly_via_self_route"

        first_decide = min(decide_lines)
        early_routes = [line for line in route_lines if line < first_decide]
        assert early_routes == [], (
            f"{_DISPATCH_CLASS}.{_DISPATCH_METHOD} calls "
            f"`await self._route(...)` at lines {early_routes} before "
            f"the first `self._autonomy_policy.decide(...)` at line "
            f"{first_decide}. FC-11 requires the gate to fire on every "
            "routed path — a decide-after-route ordering lets a tool run "
            "before Autonomy has a chance to Deny it."
        )

    def test_detection_logic_rejects_synthetic_bypass(self) -> None:
        """Adversarial check: the AST detector catches a simulated regression.

        Without this test, a bug in the detection logic (e.g. walking
        only top-level statements and missing nested awaits) would let
        real regressions through. Here a fast-path ``await self._route``
        lives inside an ``if`` before the gate fires; the detector must
        still flag it.
        """
        bypass_source = textwrap.dedent(
            """
            class OrchestratorToolDispatch:
                async def dispatch(self, call):
                    if call.name == "list_ensembles":
                        return await self._route(call)
                    decision = self._autonomy_policy.decide(
                        tool_name=call.name, arguments=call.arguments
                    )
                    if isinstance(decision, Deny):
                        return ToolCallError(
                            id=call.id,
                            name=call.name,
                            kind="denied_by_autonomy",
                            reason=decision.reason,
                        )
                    return await self._route(call)
            """
        )
        tree = ast.parse(bypass_source)
        method = _find_method(tree, _DISPATCH_CLASS, _DISPATCH_METHOD)

        decide_lines = [
            node.lineno for node in ast.walk(method) if _is_policy_decide_call(node)
        ]
        route_lines = [
            node.lineno for node in ast.walk(method) if _is_self_route_await(node)
        ]
        first_decide = min(decide_lines)
        early_routes = [line for line in route_lines if line < first_decide]

        assert early_routes, (
            "Detection logic failed to flag a synthetic fast-path that "
            "routes before consulting the Autonomy gate. The production "
            "FC-11 check would also fail to catch this regression."
        )
