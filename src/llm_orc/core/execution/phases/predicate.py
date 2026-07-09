"""Shared predicate evaluation for control-flow primitives.

A tiny grammar — `${ref}` truthiness, or `${ref} == <literal>` — over a
caller-supplied reference resolver. The guard (`when:`) and the loop (`until:`)
both use it; they differ only in how a `${ref}` token resolves to a value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Resolver = Callable[[str], Any]


def evaluate(expr: str, resolve: Resolver) -> bool:
    """Evaluate a predicate string against a reference resolver."""
    expr = expr.strip()
    if "==" in expr:
        left, right = expr.split("==", 1)
        return bool(resolve(left.strip()) == parse_literal(right.strip()))
    return bool(resolve(expr))


def parse_literal(token: str) -> Any:
    """Parse a literal: true/false, a number, or a quoted string."""
    if token == "true":
        return True
    if token == "false":
        return False
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token
