#!/usr/bin/env python3
"""Ω-P3 marshal (live): clean the fired generator's output into the file.

Terminal of the live build-one sub-ensemble. Reads the file name/kind from
classify and the raw content from whichever generator branch fired (an ensemble
node, so its response is a child-result whose `deliverable` is the model output),
strips any markdown fence (Ω-E clean_content), and emits {file, kind, content,
tier} — the per-file deliverable the fan-out gathers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scratch" / "spike-omega-p3"))
from buildlib import clean_content  # noqa: E402


def _generator_content(deps: dict) -> tuple[str, str]:
    """(raw content, tier) from whichever build branch fired."""
    for name, tier in (("build-code", "code"), ("build-prose", "prose")):
        node = deps.get(name)
        if isinstance(node, dict) and node.get("status") == "success" and node.get("response"):
            try:
                child = json.loads(node["response"])
                content = child.get("deliverable") or ""
            except (json.JSONDecodeError, ValueError):
                content = node["response"]
            return content if isinstance(content, str) else "", tier
    return "", ""


def main() -> None:
    try:
        deps = json.loads(sys.stdin.read()).get("dependencies", {})
    except (json.JSONDecodeError, ValueError, AttributeError):
        deps = {}
    try:
        cls = json.loads(deps.get("classify", {}).get("response", "{}"))
    except (json.JSONDecodeError, ValueError):
        cls = {}
    raw, tier = _generator_content(deps)
    print(
        json.dumps(
            {
                "file": cls.get("file", ""),
                "kind": cls.get("kind", ""),
                "tier": tier,
                "content": clean_content(raw, tier == "code"),
            }
        )
    )


if __name__ == "__main__":
    main()
