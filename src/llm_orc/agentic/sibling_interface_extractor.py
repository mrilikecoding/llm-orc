"""Sibling Interface Extractor (Cycle 7 loop-back #7, ADR-039) — leaf.

The content anchor's builder. Given the session's already-produced sibling
deliverables, produce the anchor text routed into the callee dispatch so a
dependent deliverable references real sibling APIs instead of inventing them
(Finding H). **Content-agnostic by construction:** full content is the
type-blind universal baseline (it sources an anchor from any sibling — code,
config, doc); where the framework has a structural extractor for the sibling's
type (Python AST now), it compacts to the public API surface (signatures). A
wrong anchor is worse than none (Spike ξ: decoy 0/10, below the unanchored
baseline), so the caller sources content from the real produced file; this
module never guesses.

A leaf — imports only the standard library (``ast``).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

__all__ = ["build_content_anchor", "extract_signatures"]

_AnyFunc = ast.FunctionDef | ast.AsyncFunctionDef


def extract_signatures(source: str) -> str | None:
    """The Python compaction path: a module's public API surface, bodies omitted.

    Returns ``None`` when the source does not parse or exposes nothing — the
    caller then falls back to the full-content baseline (the content-agnostic
    guarantee), so an unparseable sibling never breaks the mechanism.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, _AnyFunc) and not node.name.startswith("_"):
            lines.extend(_function_lines(node, indent=""))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            lines.extend(_class_lines(node))
    return "\n".join(lines) if lines else None


def _function_lines(node: _AnyFunc, *, indent: str) -> list[str]:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    out = [f"{indent}{prefix} {node.name}({args}){returns}: ..."]
    doc = ast.get_docstring(node)
    if doc:
        out.append(f'{indent}    """{doc.splitlines()[0]}"""')
    return out


def _class_lines(node: ast.ClassDef) -> list[str]:
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    out = [header]
    for sub in node.body:
        if isinstance(sub, _AnyFunc) and (
            not sub.name.startswith("_") or sub.name == "__init__"
        ):
            out.extend(_function_lines(sub, indent="    "))
    return out


def build_content_anchor(
    siblings: Sequence[tuple[str, str]], *, max_siblings: int | None = None
) -> str:
    """Compose the content anchor from the produced siblings.

    Signatures where the framework has an extractor for the type (``.py`` now);
    full content otherwise — so no sibling content type breaks the mechanism
    (content-agnostic by construction). Returns ``""`` when there are no
    siblings, so the caller injects nothing on a first file or a no-dependency
    write.

    ``max_siblings`` bounds the anchor to the most recent K siblings (Spike τ,
    2026-06-17, ADR-039 amendment): the unbounded all-prior anchor degrades the
    coder at scale — a form bleed escalation cannot fix, since every tier is fed
    the same bloated anchor. ``None`` keeps all (pre-amendment default); ``0``
    keeps none. Recency is the dependency heuristic; a dependency-scoped subset
    is the deferred more-correct option (Spike ξ).
    """
    if max_siblings is not None:
        siblings = siblings[-max_siblings:] if max_siblings > 0 else []
    if not siblings:
        return ""
    blocks: list[str] = []
    for path, content in siblings:
        surface = extract_signatures(content) if path.endswith(".py") else None
        body = surface if surface is not None else content
        blocks.append(f"--- {path} ---\n{body}")
    return (
        "These files already exist in this session. Use only the names, keys, "
        "and functions they actually define; do not invent others.\n\n"
        + "\n\n".join(blocks)
    )
