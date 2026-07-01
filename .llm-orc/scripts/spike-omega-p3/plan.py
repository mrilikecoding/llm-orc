#!/usr/bin/env python3
"""Ω-P3 plan stage: project a contract into per-file build tasks.

Emits a BARE JSON array (one element per deliverable) so a downstream
`fan_out: true` node expands over it. Each element is a COMPOSITE that carries
both the per-file deliverable AND the full contract, so each fanned build
instance has its file's spec plus the sibling APIs it must integrate with — the
fan-out mechanism only hands an instance its own chunk, so the sibling context
must ride inside the chunk.

For the free structural spike this emits a fixed contract (2 code files + 1 doc)
instead of reading an upstream loop, to isolate the build fan-out structure.
"""

from __future__ import annotations

import json

CONTRACT = [
    {"file": "tokenizer.py", "kind": "python_module", "brief": "tokenize()"},
    {"file": "evaluator.py", "kind": "python_module", "brief": "evaluate()"},
    {"file": "README.md", "kind": "markdown_doc", "brief": "usage docs"},
]


def main() -> None:
    tasks = [{"deliverable": d, "contract": CONTRACT} for d in CONTRACT]
    # Dict (not bare array) so the build node selects via input_key (routing-demo
    # convention; a top-level array trips the script request-processor).
    print(json.dumps({"tasks": tasks}))


if __name__ == "__main__":
    main()
