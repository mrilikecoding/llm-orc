#!/usr/bin/env python3
"""Ω-P3 build-code: the CODE-tier branch of the per-file build.

Fires only when classify says is_code (guarded). Stand-in for a real cheap code
generator: emits the file content built against the deliverable. Reads the
deliverable from the classify dependency.
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
                "tier": "code",
                "content": f"# {file}\n# (built by the CODE tier)\n",
            }
        )
    )


if __name__ == "__main__":
    main()
