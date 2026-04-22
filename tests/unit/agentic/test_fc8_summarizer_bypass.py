"""Static inspection tests for Fitness Criterion FC-8.

Per ``docs/agentic-serving/system-design.md`` §Fitness Criteria, FC-8
(as amended by Design Amendment #3):

    ``unsummarized-result`` cannot reach the Orchestrator Runtime's
    context | Static check: Runtime imports ``ToolCallResult``; no path
    from ``EnsembleExecutor`` to Runtime bypasses the Harness | 0 bypass
    paths | AS-7; ADR-004.

FC-8 is the structural enforcement of AS-7 (result summarization is a
correctness requirement) and ADR-004 (raw ensemble output is the
opt-in escape hatch, not the default). The three-part proof:

1. **Runtime's reasoning surface is isolated** — FC-4 (see
   ``test_fc4_runtime_import_surface``) already enforces that the
   Runtime cannot import ``EnsembleExecutor``, ``OrchestraService``,
   or any other surface that would let it reach raw ensemble output.
2. **Tool Dispatch is the only producer of ``ToolCallSuccess`` for
   ``invoke_ensemble``** — the Runtime receives ``ToolCallResult``
   values exclusively from ``ToolDispatcher.dispatch``. The ``match``
   arm for ``invoke_ensemble`` routes to
   ``OrchestratorToolDispatch.invoke_ensemble``.
3. **The ``invoke_ensemble`` method cannot construct a successful
   ``ToolCallSuccess`` without passing through the Harness** — the
   AST checks in this file prove the summarize-dominance property:
   every ``ToolCallSuccess`` constructor inside ``invoke_ensemble``
   lives inside a ``match`` block whose subject is the summarize
   result.

Combined, a regression that introduced a bypass (e.g. an
``if fast_path: return ToolCallSuccess(...)`` early-exit before the
harness call, or a second ``ToolCallSuccess`` branch that skipped
summarization) would trip one of these tests before it reached
review. The strict formulation catches the class of regressions the
loose "harness is mentioned somewhere" formulation would miss.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import TypeGuard

from llm_orc.agentic import orchestrator_tool_dispatch

_DISPATCH_MODULE_PATH = Path(orchestrator_tool_dispatch.__file__)

_DISPATCH_CLASS = "OrchestratorToolDispatch"
_INVOKE_METHOD = "invoke_ensemble"
_HARNESS_ATTR = "_harness"
_SUMMARIZE_METHOD = "summarize"
_SUCCESS_CONSTRUCTOR = "ToolCallSuccess"


def _parse_dispatch() -> ast.Module:
    return ast.parse(_DISPATCH_MODULE_PATH.read_text())


def _find_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.AsyncFunctionDef:
    """Return the named async method on the named class, or fail the test."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(
        f"{class_name}.{method_name} not found in "
        f"{_DISPATCH_MODULE_PATH.name} — FC-8's subject has moved."
    )


def _is_harness_summarize_call(node: ast.AST) -> TypeGuard[ast.expr]:
    """Match ``await self._harness.summarize(...)`` (or the bare ``Call``)."""
    call = node.value if isinstance(node, ast.Await) else node
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != _SUMMARIZE_METHOD:
        return False
    inner = func.value
    if not isinstance(inner, ast.Attribute) or inner.attr != _HARNESS_ATTR:
        return False
    return isinstance(inner.value, ast.Name) and inner.value.id == "self"


def _is_tool_call_success_construction(node: ast.AST) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _SUCCESS_CONSTRUCTOR
    )


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Map ``id(child) -> parent`` across the whole tree.

    Used to walk ancestors from a ``ToolCallSuccess`` call up to the
    enclosing ``match`` statement (if any).
    """
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _find_ancestor_match_on_name(
    node: ast.AST, parents: dict[int, ast.AST], stop: ast.AST, subject_name: str
) -> ast.Match | None:
    """Walk parents from ``node`` up to ``stop`` looking for a ``match subject_name:``.

    Returns the first such ``ast.Match`` node found, or ``None``. The
    walk halts at ``stop`` so a ``match`` statement outside the method
    body cannot satisfy the check.
    """
    current: ast.AST | None = parents.get(id(node))
    while current is not None and current is not stop:
        if isinstance(current, ast.Match):
            subject = current.subject
            if isinstance(subject, ast.Name) and subject.id == subject_name:
                return current
        current = parents.get(id(current))
    return None


def _summarize_binding_name(method: ast.AsyncFunctionDef) -> str:
    """Return the local name bound to the summarize call's result.

    Walks the method for ``<name> = await self._harness.summarize(...)``.
    Fails the test if the binding pattern changes — the AST check
    depends on the summarize result being a named local that the match
    statement can reference.
    """
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_harness_summarize_call(node.value):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    raise AssertionError(
        f"Could not find `<name> = await self._harness.summarize(...)` binding "
        f"in {_DISPATCH_CLASS}.{_INVOKE_METHOD}. FC-8's AST dominance check "
        "depends on the summarize result being bound to a local name that a "
        "subsequent `match` statement dispatches on."
    )


class TestFc8SummarizeDominatesInvokeEnsemble:
    """Amendment #3: Tool Dispatch cannot return a success for
    ``invoke_ensemble`` without routing the raw ensemble result through
    :class:`ResultSummarizerHarness`.
    """

    def test_invoke_ensemble_calls_harness_summarize(self) -> None:
        """The method must invoke the harness at least once.

        Without a call site, no summary path exists — the dominance
        check below is vacuously true and FC-8 has no teeth. This test
        keeps the harness call site load-bearing.
        """
        tree = _parse_dispatch()
        method = _find_method(tree, _DISPATCH_CLASS, _INVOKE_METHOD)
        harness_calls = [
            node for node in ast.walk(method) if _is_harness_summarize_call(node)
        ]
        assert harness_calls, (
            f"{_DISPATCH_CLASS}.{_INVOKE_METHOD} no longer calls "
            "`await self._harness.summarize(...)`. Per Amendment #3, "
            "Tool Dispatch must interpose the Harness on every invoke_ensemble "
            "return; removing the call breaks AS-7's structural enforcement."
        )

    def test_every_tool_call_success_lives_inside_summarize_match(self) -> None:
        """Strict dominance: every ``ToolCallSuccess`` constructor in
        ``invoke_ensemble`` must be nested inside a ``match`` statement
        whose subject is the summarize result's bound name.

        This catches the regression a loose check would miss: adding a
        new ``return ToolCallSuccess(id=..., name="invoke_ensemble", ...)``
        branch that does not flow from the summarization — for example,
        an early-return fast path, a short-circuit on a cached result,
        or a second ``match`` arm that wraps the raw result.
        """
        tree = _parse_dispatch()
        method = _find_method(tree, _DISPATCH_CLASS, _INVOKE_METHOD)
        parents = _build_parent_map(tree)
        subject_name = _summarize_binding_name(method)

        constructors = [
            node
            for node in ast.walk(method)
            if _is_tool_call_success_construction(node)
        ]
        assert constructors, (
            f"{_DISPATCH_CLASS}.{_INVOKE_METHOD} constructs no "
            f"{_SUCCESS_CONSTRUCTOR}. FC-8 has nothing to enforce — the "
            "invoke_ensemble surface has moved. Re-anchor this test."
        )

        unguarded: list[int] = []
        for constructor in constructors:
            match_node = _find_ancestor_match_on_name(
                constructor, parents, stop=method, subject_name=subject_name
            )
            if match_node is None:
                unguarded.append(constructor.lineno)

        assert unguarded == [], (
            f"{_DISPATCH_CLASS}.{_INVOKE_METHOD} constructs "
            f"{_SUCCESS_CONSTRUCTOR} at lines {unguarded} outside any "
            f"`match {subject_name}:` block. FC-8 (AS-7, ADR-004) requires "
            "every successful invoke_ensemble result to flow from the "
            "Result Summarizer Harness. A ToolCallSuccess constructed "
            "outside the summarize-result match is a bypass path — the raw "
            "ensemble result would reach the Orchestrator Runtime's "
            "context unsummarized."
        )

    def test_detection_logic_rejects_synthetic_bypass(self) -> None:
        """Adversarial check: the AST detector catches a simulated regression.

        Constructs a synthetic ``invoke_ensemble`` that adds a fast-path
        early return bypassing the harness, parses it with the same
        machinery as the production check, and asserts the detector
        flags the fast-path ``ToolCallSuccess`` as unguarded. Without
        this test, a bug in the detection logic would silently let real
        regressions through.
        """
        bypass_source = textwrap.dedent(
            """
            class OrchestratorToolDispatch:
                async def invoke_ensemble(self, id_, arguments):
                    if arguments.get("fast_path"):
                        return ToolCallSuccess(
                            id=id_, name="invoke_ensemble", content={}
                        )
                    result = await self._operations.invoke(arguments)
                    summarization = await self._harness.summarize(
                        result, raw_output=False
                    )
                    match summarization:
                        case SummarizationSuccess(summary=summary):
                            return ToolCallSuccess(
                                id=id_,
                                name="invoke_ensemble",
                                content={"summary": summary},
                            )
                        case _:
                            return ToolCallError(
                                id=id_,
                                name="invoke_ensemble",
                                kind="summarization_failed",
                                reason="",
                            )
            """
        )
        tree = ast.parse(bypass_source)
        method = _find_method(tree, _DISPATCH_CLASS, _INVOKE_METHOD)
        parents = _build_parent_map(tree)
        subject_name = _summarize_binding_name(method)

        constructors = [
            node
            for node in ast.walk(method)
            if _is_tool_call_success_construction(node)
        ]
        unguarded = [
            c.lineno
            for c in constructors
            if _find_ancestor_match_on_name(
                c, parents, stop=method, subject_name=subject_name
            )
            is None
        ]

        assert unguarded, (
            "Detection logic failed to flag the fast-path bypass. The "
            "production FC-8 check would also fail to catch this regression."
        )

    def test_harness_summarize_precedes_summarize_match(self) -> None:
        """Lexical ordering: the summarize call runs before its match.

        AST dominance is structural; this line-order check is a
        defense-in-depth cross-check. A regression that moved the match
        *above* the summarize call (e.g. by extracting the match into a
        helper and re-matching a stale binding) would still fail the
        primary dominance check — but this line-order assertion surfaces
        the regression with a clearer error message.
        """
        tree = _parse_dispatch()
        method = _find_method(tree, _DISPATCH_CLASS, _INVOKE_METHOD)
        subject_name = _summarize_binding_name(method)

        summarize_lines = [
            node.lineno for node in ast.walk(method) if _is_harness_summarize_call(node)
        ]
        match_lines = [
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Match)
            and isinstance(node.subject, ast.Name)
            and node.subject.id == subject_name
        ]

        assert summarize_lines, (
            "No harness summarize call found — covered by "
            "test_invoke_ensemble_calls_harness_summarize, surfaced here "
            "for clarity when ordering breaks."
        )
        assert match_lines, (
            f"No `match {subject_name}:` statement found in "
            f"{_INVOKE_METHOD}. FC-8 expects summarize's result to be "
            "dispatched via a match block."
        )
        assert min(summarize_lines) < min(match_lines), (
            f"`match {subject_name}:` (line {min(match_lines)}) precedes "
            f"`await self._harness.summarize(...)` (line "
            f"{min(summarize_lines)}). The match must consume the "
            "summarize result, not a stale or pre-summarize value."
        )
