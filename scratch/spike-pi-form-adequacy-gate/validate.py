"""Spike π live-arm validator — per-session all-files-valid.

Validates each produced file against its destination form (the same gate
logic as gates.py / artifact_bridge._spike_pi_form_check):
  .py   -> ast.parse
  .json -> json.loads
  .md   -> pass-through (prose form is not structurally checkable)
  other -> pass-through

Prints one line per produced file, then a SUMMARY line the runner parses.
all_valid is over the structurally-checkable files only (.py/.json) — the
pre-registered Fork-3 measure. present5 reports convergence (all 5 target
files exist) for interpretation alongside.
"""

from __future__ import annotations

import ast
import json
import os
import sys

TARGET = {"converters.py", "test_converters.py", "cli.py", "test_cli.py", "README.md"}


def main(ws: str) -> None:
    files = [
        f
        for f in os.listdir(ws)
        if f != "opencode.json"
        and not f.startswith(".")
        and os.path.isfile(os.path.join(ws, f))
    ]
    results: list[tuple[str, str]] = []
    n_invalid = 0
    for f in sorted(files):
        ext = os.path.splitext(f)[1].lower()
        try:
            content = open(
                os.path.join(ws, f), encoding="utf-8", errors="replace"
            ).read()
        except OSError as exc:
            results.append((f, f"READERR:{exc}"))
            n_invalid += 1
            continue
        if ext == ".py":
            try:
                ast.parse(content)
                results.append((f, "PASS"))
            except SyntaxError as exc:
                results.append((f, f"FAIL:{exc.msg}"))
                n_invalid += 1
        elif ext == ".json":
            try:
                json.loads(content)
                results.append((f, "PASS"))
            except json.JSONDecodeError as exc:
                results.append((f, f"FAIL:{exc}"))
                n_invalid += 1
        elif ext == ".md":
            results.append((f, "SKIP_md"))
        else:
            results.append((f, "SKIP_other"))

    checkable = sum(
        1 for f, _ in results if os.path.splitext(f)[1].lower() in (".py", ".json")
    )
    present5 = TARGET.issubset(set(files))
    all_valid = n_invalid == 0
    for f, r in results:
        print(f"  {f}\t{r}")
    print(
        f"SUMMARY\tfiles={len(files)}\tpresent5={present5}\t"
        f"checkable={checkable}\tinvalid={n_invalid}\tall_valid={all_valid}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
