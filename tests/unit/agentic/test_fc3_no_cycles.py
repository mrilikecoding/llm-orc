"""Static inspection tests for Fitness Criterion FC-3 (WP-B4).

Per ``docs/agentic-serving/system-design.agents.md`` §Fitness Criteria:

    FC-3 | No cycles in the dependency graph | Static cycle detection
    over the dependency edge list | 0 cycles | ARCHITECT principle

The system-design records 28 edges across the agentic-serving system
(21 from v2.0 + 7 Cycle 4 additions); the ADR-016 upward exception
(Ensemble Engine → Calibration Signal Channel) is one annotated edge
in that set. None of the 28 form a cycle; FC-3 is the mechanical
guarantee that none will be introduced under future changes.

This test reconstructs the import graph from the same scope as FC-2
— every ``*.py`` in ``src/llm_orc/agentic/`` plus the Ensemble Engine
at ``src/llm_orc/core/execution/ensemble_execution.py`` — and runs
Tarjan-flavored DFS cycle detection. Contract modules
(``orchestrator_chunk``, ``session_start``) participate in the graph
as ordinary nodes; their layer-neutrality is irrelevant to cycle
detection.
"""

from __future__ import annotations

import ast
from pathlib import Path

_AGENTIC_DIR = Path(__file__).resolve().parents[3] / "src" / "llm_orc" / "agentic"
_ENSEMBLE_ENGINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "llm_orc"
    / "core"
    / "execution"
    / "ensemble_execution.py"
)


def _imports_in(module_path: Path) -> set[str]:
    """Return runtime ``llm_orc.agentic.*`` or
    ``llm_orc.core.execution.ensemble_execution`` imports in this file —
    the FC-3 graph nodes.

    Imports inside ``if TYPE_CHECKING:`` blocks are excluded; they are
    a Python idiom for breaking circular type-annotation imports and
    do not execute at runtime, so they cannot form a runtime cycle.
    """
    tree = ast.parse(module_path.read_text())
    skip_ids = _type_checking_node_ids(tree)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in skip_ids:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_in_scope(alias.name):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_in_scope(node.module):
                imports.add(node.module)
    return imports


def _type_checking_node_ids(tree: ast.Module) -> set[int]:
    """Collect ``id()`` of every AST node nested under an ``if TYPE_CHECKING:``."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for sub in ast.walk(node):
                if sub is not node:
                    skip.add(id(sub))
    return skip


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _is_in_scope(module_name: str) -> bool:
    return module_name.startswith("llm_orc.agentic") or module_name == (
        "llm_orc.core.execution.ensemble_execution"
    )


def _path_to_module(module_path: Path) -> str:
    if module_path == _ENSEMBLE_ENGINE_PATH:
        return "llm_orc.core.execution.ensemble_execution"
    return f"llm_orc.agentic.{module_path.stem}"


def _build_dependency_graph() -> dict[str, set[str]]:
    """Build the directed import graph FC-3 verifies acyclic."""
    graph: dict[str, set[str]] = {}
    scan_paths = [
        *(p for p in sorted(_AGENTIC_DIR.glob("*.py")) if p.name != "__init__.py"),
        _ENSEMBLE_ENGINE_PATH,
    ]
    for path in scan_paths:
        source = _path_to_module(path)
        graph[source] = _imports_in(path)
    return graph


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return the first cycle found, or ``None`` if the graph is acyclic.

    Iterative DFS with three coloring (white/gray/black). When a gray
    node is re-encountered along a path, the cycle is the slice of the
    path from that node onward — returned for actionable error
    reporting rather than a bare "cycle exists".
    """
    color: dict[str, str] = dict.fromkeys(graph, "white")
    parent: dict[str, str | None] = dict.fromkeys(graph, None)

    for start in graph:
        if color[start] != "white":
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, post = stack.pop()
            if post:
                color[node] = "black"
                continue
            if color[node] == "black":
                continue
            color[node] = "gray"
            stack.append((node, True))
            for neighbor in graph.get(node, set()):
                if neighbor not in graph:
                    # Out-of-scope target (e.g., not-yet-landed module) —
                    # cannot participate in a cycle from this scan.
                    continue
                if color[neighbor] == "white":
                    parent[neighbor] = node
                    stack.append((neighbor, False))
                elif color[neighbor] == "gray":
                    return _reconstruct_cycle(parent, node, neighbor)
    return None


def _reconstruct_cycle(
    parent: dict[str, str | None], end: str, start: str
) -> list[str]:
    """Walk parent links from ``end`` back to ``start`` to produce the cycle."""
    path = [end]
    cur: str | None = end
    while cur is not None and cur != start:
        cur = parent[cur]
        if cur is not None:
            path.append(cur)
    path.reverse()
    path.append(end)  # close the loop visually
    return path


class TestNoCycles:
    """FC-3 — the agentic dependency graph is acyclic."""

    def test_no_cycles_in_agentic_import_graph(self) -> None:
        graph = _build_dependency_graph()
        cycle = _find_cycle(graph)

        assert cycle is None, (
            "FC-3 cycle detected in agentic-serving dependency graph:\n  "
            + " → ".join(cycle or [])
            + "\n\nPer ARCHITECT principle the graph must remain acyclic. "
            "The cycle indicates two or more modules mutually depend on "
            "each other; resolve by extracting the shared contract to a "
            "value-type module (see orchestrator_chunk / session_start) "
            "or by inverting the dependency at the higher layer."
        )

    def test_graph_is_non_empty(self) -> None:
        """A vacuous empty graph would make the no-cycles assertion meaningless."""
        graph = _build_dependency_graph()
        assert graph, "FC-3 graph scan returned no modules — scan logic is broken."
        assert any(graph.values()), (
            "FC-3 graph scan found no edges across the agentic package — "
            "either the scan is misconfigured or every agentic module is "
            "isolated (which would itself signal a structural problem)."
        )
