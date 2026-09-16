#!/usr/bin/env python3
"""Fail when a design doc names a test that exists nowhere (loop-protocol 19).

Doc and code drift silently: no test fails when prose describes deleted code.
Measured cost of that: one arc's instrument section was wrong in four
consecutive review rounds — a test named twice under different numbers, a
test listed as modified whose body was byte-identical to main, a count that
came out right only because two errors cancelled. Every one of those is a
name that does not resolve, or a name that is missing, and both are
mechanical.

Scope, deliberately narrow. This resolves `test_*` identifiers mentioned in
`docs/plans/*.md` against the names pytest can actually collect. It does NOT
try to check prose for truth — "the guard uses os.path.isfile" cannot be
checked this way, and pretending otherwise would make the check a denylist.
It catches the drift that is checkable, which is the half that kept costing
rounds.

Usage: python scripts/check_doc_drift.py [docs/plans/foo.md ...]
Exit 0 when every named test resolves, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs" / "plans"

# A test name as a doc writes it: inside backticks, optionally qualified by a
# class. Bare prose mentions are not matched — a name has to be marked up as
# code for this to treat it as a claim about a test that exists.
_NAMED = re.compile(r"`(?:([A-Za-z_][\w]*)::)?(test_[\w]+)`")
_TOKEN = re.compile(r"\btest_\w+\b")


_CODE = ("*.py", "*.yaml", "*.yml")
_SKIP = {
    ".git",
    ".venv",
    "node_modules",
    "htmlcov",
    "htmlcov_viz",
    "dist",
    "worktrees",
    "__pycache__",
}


def _known_names() -> set[str]:
    """Every ``test_*`` token appearing anywhere in code.

    Deliberately broader than "test functions defined under tests/". Measured:
    that narrower set flagged ``test_writer`` (a serving ensemble's AGENT
    name) and ``test_fns`` (a local variable), because a doc may legitimately
    name either in backticks. What is never legitimate is a name that
    resolves to nothing at all, which is what a renamed or deleted test
    leaves behind — the three real hits in this repo's corpus when the check
    was first run.

    Tokens rather than definitions, for the same reason: a doc naming a test
    by its call site or its parametrised id is still naming something real.

    Bound: a name mentioned ANYWHERE in code resolves, so this cannot catch
    a doc that names a real test which no longer covers what the doc claims.
    It catches names that resolve to nothing, which is the half that kept
    costing review rounds. Judging whether a pin still means what the prose
    says is a reader's job, and pretending otherwise would make this a
    denylist.
    """
    names: set[str] = set()
    for pattern in _CODE:
        for path in REPO.rglob(pattern):
            # Relative parts, not absolute: a delegated agent's checkout
            # lives AT .claude/worktrees/<agent>/, and matching the absolute
            # path skipped every file it owned — zero known names, so any
            # wrong doc passed while every right name reported as drift.
            if _SKIP & set(path.relative_to(REPO).parts):
                continue
            try:
                names.update(_TOKEN.findall(path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                continue
    return names


def check(docs: list[Path]) -> list[str]:
    """Unresolved `test_*` names, as reportable lines."""
    defined = _known_names()
    problems: list[str] = []
    for doc in sorted(docs):
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            continue
        seen: set[tuple[int, str]] = set()
        for number, line in enumerate(text.splitlines(), start=1):
            for _cls, name in _NAMED.findall(line):
                if name not in defined and (number, name) not in seen:
                    seen.add((number, name))
                    # A doc passed by absolute path from outside the repo
                    # (a test fixture) has no relative form.
                    rel = doc.relative_to(REPO) if doc.is_relative_to(REPO) else doc
                    problems.append(f"{rel}:{number}: names nothing: {name}")
    return problems


def main(argv: list[str]) -> int:
    docs = [Path(a).resolve() for a in argv[1:]] or sorted(DOCS.glob("*.md"))
    problems = check(docs)
    for problem in problems:
        print(problem)
    if problems:
        print(
            f"\n{len(problems)} doc claim(s) name a test_* identifier that "
            "exists nowhere in code (loop-protocol rule 19). Fix the doc, or "
            "the name."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
