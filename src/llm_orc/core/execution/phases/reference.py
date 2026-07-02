"""Shared ``${dep.field}`` reference resolution over accumulated results.

Both the guard predicate (``when:``) and dynamic dispatch (``dispatch:``) resolve
a ``${dep.field}`` token against the phase-layer ``results_dict``: look up the
dependency, parse its ``response`` as JSON, and walk the dotted field path. A
token that is not a ``${dep.field}`` reference is returned unchanged (guard
passes literals through the predicate evaluator this way).
"""

from __future__ import annotations

import json
import re
from typing import Any

_REF = re.compile(r"^\$\{([^.}]+)\.([^}]+)\}$")


def resolve_reference(token: str, results_dict: dict[str, Any]) -> Any:
    """Resolve a ``${dep.field}`` token against accumulated upstream results.

    Returns ``token`` unchanged when it is not a ``${dep.field}`` reference.
    """
    match = _REF.match(token)
    if not match:
        return token
    dep, field = match.group(1), match.group(2)
    result = results_dict.get(dep, {})
    parsed = json.loads(result.get("response", ""))
    value: Any = parsed
    for part in field.split("."):
        value = value[part]
    return value
