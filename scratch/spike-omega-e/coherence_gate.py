#!/usr/bin/env python3
"""Architect-coherence gate (spike Ω, item 6.2a).

A deterministic, domain-free check on the architect's contract BEFORE any
building happens. It reads the contract as a typed dependency graph and checks
graph invariants only — never what a module *means*. See
`docs/agentic-serving/proposals/ensemble-agent-state-and-next-steps.md` §6.2a.

Invariants enforced:
  - referential closure: every import edge resolves to a contract sibling or a
    stdlib module (no dangling / hallucinated modules like `models` in a calc);
  - symbol resolution: a sibling import names a symbol that sibling defines;
  - no self-naming: a module does not define a symbol named after itself.

Scope: self-contained packages whose cross-module imports are bare module
names (the spike's domain). Third-party deps are out of scope — they would read
as unresolved here and need an explicit declaration to allow.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

Deliverable = dict[str, Any]
ArchitectFn = Callable[[str], Awaitable[list[Deliverable]]]


def _module_names(contract: list[Deliverable]) -> set[str]:
    return {Path(d["file"]).stem for d in contract if d["file"].endswith(".py")}


def _defined_symbols(contract: list[Deliverable]) -> dict[str, set[str]]:
    return {
        Path(d["file"]).stem: {
            s["name"] for s in d.get("defines", []) if s.get("name")
        }
        for d in contract
        if d["file"].endswith(".py")
    }


def _is_external(mod: str) -> bool:
    root = mod.split(".")[0]
    return root == "__future__" or root in sys.stdlib_module_names


def _check_self_naming(d: Deliverable) -> list[str]:
    stem = Path(d["file"]).stem
    return [
        f"{d['file']}: defines a symbol named after its own module ('{stem}')"
        for s in d.get("defines", [])
        if s.get("name") == stem
    ]


def _check_import(
    d: Deliverable, imp: str, siblings: set[str], defined: dict[str, set[str]]
) -> list[str]:
    f = d["file"]
    try:
        node = ast.parse(imp).body[0]
    except SyntaxError:
        return [f"{f}: unparseable import statement: {imp!r}"]
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if mod in siblings:
            return [
                f"{f}: imports '{a.name}' from '{mod}', which does not define it"
                for a in node.names
                if a.name not in defined[mod]
            ]
        if not _is_external(mod):
            return [f"{f}: imports from '{mod}', not a contract module or stdlib"]
    elif isinstance(node, ast.Import):
        return [
            f"{f}: imports module '{a.name}', not a contract module or stdlib"
            for a in node.names
            if a.name not in siblings and not _is_external(a.name)
        ]
    return []


def coherence_gate(contract: list[Deliverable]) -> tuple[bool, list[str]]:
    siblings = _module_names(contract)
    defined = _defined_symbols(contract)
    reasons: list[str] = []
    for d in contract:
        if not d["file"].endswith(".py"):
            continue
        reasons += _check_self_naming(d)
        for imp in d.get("imports", []):
            if imp.strip():
                reasons += _check_import(d, imp, siblings, defined)
    return not reasons, reasons


def _format_feedback(reasons: list[str]) -> str:
    bullets = "\n".join(f"- {r}" for r in reasons)
    return (
        "The previous contract was REJECTED by the coherence gate:\n"
        f"{bullets}\n"
        "Fix exactly these problems and re-emit the full contract JSON. Every "
        "cross-module import must reference a sibling module that exists in the "
        "contract (or the stdlib), and no module may define a symbol named "
        "after itself."
    )


async def resolve_contract(
    architect: ArchitectFn, max_repairs: int = 2
) -> tuple[list[Deliverable], list[str], int]:
    feedback = ""
    contract: list[Deliverable] = []
    reasons: list[str] = []
    for attempt in range(max_repairs + 1):
        contract = await architect(feedback)
        ok, reasons = coherence_gate(contract)
        if ok:
            return contract, [], attempt + 1
        feedback = _format_feedback(reasons)
    return contract, reasons, max_repairs + 1
