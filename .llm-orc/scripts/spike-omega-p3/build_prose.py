#!/usr/bin/env python3
"""Ω-P3 build-prose: the PROSE-tier branch of the per-file build.

Fires only when classify says is_doc (guarded). Stand-in for a real cheap prose
generator: emits Markdown content built against the deliverable.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        cls = json.loads(payload["dependencies"]["classify"]["response"])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        cls = {}
    file = cls.get("file", "")
    print(
        json.dumps(
            {
                "file": file,
                "kind": cls.get("kind"),
                "tier": "prose",
                "content": f"# {file}\n\nBuilt by the PROSE tier.\n",
            }
        )
    )


if __name__ == "__main__":
    main()
