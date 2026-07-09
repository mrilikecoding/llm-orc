"""Score one cell run — ``(workspace, serve-log slice, cell) → MetricRecord``.

Pure function of the artifacts (produced files + the per-session serve-log slice);
no live calls. Per ``docs/agentic-serving/benchmark-design.md`` §4. Deterministic +
unit-tested; this module is CI-safe.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from benchmarks.agentic_serving.model import Cell, MetricRecord

_STRUCTURAL_EXT = {".py", ".json"}
_TURN_DECISION = re.compile(r"turn decision:\s*(.+)")


@dataclass(frozen=True)
class _FileSignals:
    """The file-derived signals both arms share (§4): the three hard signals
    that are pure functions of the produced files, plus the produced set and the
    form/coherence notes. Only the log-derived fields differ between arms."""

    form_valid: bool
    converged: bool
    content_coherent: bool
    produced: tuple[str, ...]
    notes: tuple[str, ...]


def _file_signals(workspace: Path, cell: Cell) -> _FileSignals:
    """Compute the file-derived signals from a workspace (shared by both arms)."""
    produced = _produced_files(workspace)
    form_valid, form_notes = _form_valid(workspace, produced)
    coherent, coh_notes = _content_coherent(workspace, produced)
    return _FileSignals(
        form_valid=form_valid,
        converged=_converged(produced, cell),
        content_coherent=coherent,
        produced=tuple(sorted(produced)),
        notes=tuple(form_notes + coh_notes),
    )


def score(workspace: Path, log_slice: str, cell: Cell) -> MetricRecord:
    """The deterministic metric record for one cheap-arm cell run (§4)."""
    sig = _file_signals(workspace, cell)
    return MetricRecord(
        form_valid=sig.form_valid,
        converged=sig.converged,
        content_coherent=sig.content_coherent,
        terminated_clean=_terminated_clean(log_slice),
        delegation_rate=_delegation_rate(log_slice),
        escalated=_escalated(log_slice),
        churn=_churn(log_slice, list(sig.produced)),
        produced=sig.produced,
        notes=sig.notes,
    )


def score_frontier(workspace: Path, cell: Cell) -> MetricRecord:
    """Score a frontier-arm cell — a one-shot subagent workspace, no serve-log (§7).

    The three file-derived hard signals (form / convergence / coherence) score
    identically to the cheap arm; loop-termination is N/A (``None``) because a
    one-shot model has no loop to terminate, and the log-only diagnostics
    (delegation rate, escalation, churn) are absent.
    """
    sig = _file_signals(workspace, cell)
    return MetricRecord(
        form_valid=sig.form_valid,
        converged=sig.converged,
        content_coherent=sig.content_coherent,
        terminated_clean=None,
        delegation_rate=None,
        escalated=False,
        churn=None,
        produced=sig.produced,
        notes=sig.notes,
    )


def _produced_files(workspace: Path) -> list[str]:
    """Files the session produced — excludes the opencode config + dotfiles."""
    return [
        p.name
        for p in workspace.iterdir()
        if p.is_file() and p.name != "opencode.json" and not p.name.startswith(".")
    ]


def _read(workspace: Path, name: str) -> str:
    return (workspace / name).read_text(encoding="utf-8", errors="replace")


# --- Hard-pass metrics -------------------------------------------------------


def _form_valid(workspace: Path, produced: list[str]) -> tuple[bool, list[str]]:
    """Every produced ``.py`` parses, every ``.json`` loads (ADR-041)."""
    ok = True
    notes: list[str] = []
    for name in produced:
        ext = os.path.splitext(name)[1].lower()
        if ext not in _STRUCTURAL_EXT:
            continue  # prose / other — not structurally checkable (§4 boundary)
        text = _read(workspace, name)
        try:
            if ext == ".py":
                ast.parse(text)
            else:
                json.loads(text)
        except (SyntaxError, ValueError) as exc:
            ok = False
            notes.append(f"form: {name} not valid {ext} ({exc})")
    return ok, notes


def _converged(produced: list[str], cell: Cell) -> bool:
    """All requested deliverables produced (ADR-040 completeness)."""
    return set(cell.expected_deliverables).issubset(set(produced))


def _content_coherent(workspace: Path, produced: list[str]) -> tuple[bool, list[str]]:
    """Dependent files reference real sibling APIs, not invented ones (ADR-039).

    For each produced ``.py``, find references to *other produced* modules
    (``from sib import X`` / ``import sib`` + ``sib.attr``) and check the
    referenced names are defined at the sibling's top level. An undefined
    reference = an invented API = incoherent. ``import *`` and aliased module
    imports are un-checkable and recorded as notes, not failures.
    """
    py = [n for n in produced if n.endswith(".py")]
    stem_to_file = {n[:-3]: n for n in py}
    defined: dict[str, set[str]] = {}
    trees: dict[str, ast.Module] = {}
    for name in py:
        try:
            trees[name] = ast.parse(_read(workspace, name))
        except SyntaxError:
            continue  # form check already penalizes this; don't double-count
        defined[name] = _top_level_names(trees[name])

    ok = True
    notes: list[str] = []
    for name, tree in trees.items():
        refs, ref_notes = _sibling_references(tree, set(stem_to_file))
        notes.extend(f"coherence: {name} {n}" for n in ref_notes)
        for module, used in refs.items():
            missing = used - defined.get(stem_to_file[module], set())
            if missing:
                ok = False
                notes.append(
                    f"coherence: {name} references undefined {module}.{sorted(missing)}"
                )
    return ok, notes


def _top_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module top level — defs, classes, simple assignments."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _sibling_references(
    tree: ast.Module, sibling_modules: set[str]
) -> tuple[dict[str, set[str]], list[str]]:
    """Names used from each produced-sibling module + un-checkable notes."""
    refs: dict[str, set[str]] = {}
    notes: list[str] = []
    plain_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            notes.extend(_from_import_refs(node, sibling_modules, refs))
        elif isinstance(node, ast.Import):
            notes.extend(_plain_import_modules(node, sibling_modules, plain_imports))
    _module_attr_refs(tree, plain_imports, refs)
    return refs, notes


def _from_import_refs(
    node: ast.ImportFrom, siblings: set[str], refs: dict[str, set[str]]
) -> list[str]:
    """``from sib import X`` → record X; ``import *`` is un-checkable."""
    if node.module not in siblings:
        return []
    notes: list[str] = []
    for alias in node.names:
        if alias.name == "*":
            notes.append(f"uses `from {node.module} import *` (un-checkable)")
        else:
            refs.setdefault(node.module, set()).add(alias.name)
    return notes


def _plain_import_modules(
    node: ast.Import, siblings: set[str], plain_imports: set[str]
) -> list[str]:
    """``import sib`` → track for attr-access scan; aliased imports are un-checkable."""
    notes: list[str] = []
    for alias in node.names:
        if alias.name not in siblings:
            continue
        if alias.asname:
            notes.append(f"aliases `import {alias.name}` (un-checkable)")
        else:
            plain_imports.add(alias.name)
    return notes


def _module_attr_refs(
    tree: ast.Module, plain_imports: set[str], refs: dict[str, set[str]]
) -> None:
    """``import sib`` then ``sib.attr`` → record attr against sib."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in plain_imports
        ):
            refs.setdefault(node.value.id, set()).add(node.attr)


def _terminated_clean(log_slice: str) -> bool:
    """The task reached a clean COMPLETE finish (ADR-037/038).

    True iff some turn-decision is a COMPLETE finish. This is robust to a
    trailing toolless aux request (e.g. OpenCode's title-generator) whose own
    ``judgment_verdict=?`` decision lands last in the slice — that must not mask
    the task's COMPLETE finish. A zombie/capped loop or a never-finished session
    never reaches a COMPLETE finish, so it stays False.
    """
    return any(
        "action=finish" in d and "judgment_verdict=COMPLETE" in d
        for d in _TURN_DECISION.findall(log_slice)
    )


# --- Reported (not pass-gating) ---------------------------------------------


def _delegation_rate(log_slice: str) -> float | None:
    """Delegated generation turns / generation turns (ADR-036; reported per §4 P3-A).

    ``None`` when the session had no generation-shaped turns.
    """
    gens = [d for d in _TURN_DECISION.findall(log_slice) if "shape=generation" in d]
    if not gens:
        return None
    delegated = sum(1 for d in gens if _delegated_value(d) not in (None, "-"))
    return delegated / len(gens)


def _delegated_value(decision: str) -> str | None:
    match = re.search(r"delegated=(\S+)", decision)
    return match.group(1) if match else None


def _escalated(log_slice: str) -> bool:
    """Did coder-tier escalation fire this session (ADR-041 §5)?"""
    return "form escalation:" in log_slice


def _churn(log_slice: str, produced: list[str]) -> int | None:
    """Write turns beyond distinct produced files — re-revision churn (Finding G).

    ``None`` when no turn decisions were captured.
    """
    decisions = _TURN_DECISION.findall(log_slice)
    if not decisions:
        return None
    write_turns = sum(1 for d in decisions if "action=write" in d)
    return max(0, write_turns - len(produced))
