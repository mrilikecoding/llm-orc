#!/usr/bin/env python3
"""Serving resolve node — merge classify + the guarded decider into routing.

The final routing decision the ``seat`` dispatches on. When ``classify`` resolved
the turn structurally (``needs_decider: false``) its decision passes through
unchanged and the model-backed ``decide`` node never ran. When classify deferred
(``needs_decider: true``), resolve reads the decider's bounded target and derives
build/kind deterministically — the model only classifies into the closed seat
set, so the control decision stays deterministic (ADR-046 §1; determinism-over-
carve-outs). An out-of-set decider output leaves ``target`` empty so the dispatch
node fails deterministically rather than guessing a default seat.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from llm_orc.core.serving.shape_catalog import shape_catalog

# The closed seat set the decider chooses from, and the build/kind each implies.
# These are the semantic intents (code vs prose), not the serving shape.
_DERIVED = {
    "code-seat": ("python_module", True),
    "explainer": ("explanation", False),
    "tests-seat": ("python_tests", True),
}
# The intent -> serving shape mapping is now operator-curated (WP-C8): each shape
# declares the intent it ``serves`` and the Shape Catalog derives the map from the
# library, replacing the hardcoded default WP-D8 left here. ``build-gated`` serves
# ``code-seat`` (the accept gate is default-on for build turns; WP-D8 + ADR-048),
# ``explainer`` serves ``explainer``. An intent with no catalog shape passes
# through unchanged, so an unresolved target still fails deterministically at
# dispatch (determinism-over-carve-outs).
_CATALOG_DIR = Path(__file__).resolve().parents[2] / "ensembles" / "agentic-serving"
_JSON_RE = re.compile(r"\{[^{}]*\}")


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _decider_target(response: str) -> str:
    """A known seat target from the decider's output, or "" when none resolves.

    Strict first: the first JSON object's ``target`` if it is a known seat.
    Fallback: exactly one known token present in the raw text. No token, or an
    ambiguous both-tokens output, resolves to "" (deterministic dispatch fail).
    """
    match = _JSON_RE.search(response or "")
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            target = obj.get("target")
            if isinstance(target, str) and target in _DERIVED:
                return target
    present = [t for t in _DERIVED if t in (response or "")]
    return present[0] if len(present) == 1 else ""


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        classify = json.loads(_response(deps.get("classify", {})))
    except json.JSONDecodeError:
        classify = {}
    if not isinstance(classify, dict):
        classify = {}

    file = classify.get("file", "solution.py")
    dispatch_input = classify.get("dispatch_input", "")
    needs_files = classify.get("needs_files", [])
    if not isinstance(needs_files, list):
        needs_files = []
    read_failed = str(classify.get("read_failed", ""))
    needs_run = str(classify.get("needs_run", ""))
    needs_glob = str(classify.get("needs_glob", ""))
    glob_failed = str(classify.get("glob_failed", ""))

    if classify.get("needs_decider"):
        target = _decider_target(_response(deps.get("decide", {})))
        kind, build = _DERIVED.get(target, ("", False))
    else:
        target = classify.get("target", "")
        kind = classify.get("kind", "")
        build = bool(classify.get("build", False))

    # Map the semantic intent to the serving shape the seat dispatches, from the
    # operator-curated Shape Catalog (WP-C8). An intent with no registered shape
    # passes through unchanged (empty/unknown still fails deterministically).
    catalog = shape_catalog(_CATALOG_DIR)
    shape_target = catalog.get(target, target)

    print(
        json.dumps(
            {
                "target": shape_target,
                "kind": kind,
                "file": file,
                "dispatch_input": dispatch_input,
                "build": build,
                "needs_files": needs_files,
                "read_failed": read_failed,
                "needs_run": needs_run,
                "needs_glob": needs_glob,
                "glob_failed": glob_failed,
            }
        )
    )


if __name__ == "__main__":
    main()
