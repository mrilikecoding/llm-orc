#!/usr/bin/env python3
"""build-gated gather node — assemble the accept-gate contract (WP-D8).

Reads the test_writer and code_writer sub-ensemble outputs plus the turn's
criteria (the node's base input), peels the nested ensemble envelopes, strips
code fences, and emits the flat ``{requirement, code, tests}`` the accept
executor verifies. Tests-first: the tests come from test_writer (authored from
the criteria) and the code from code_writer (built against those tests).

Emits JSON: {requirement, code, tests}
"""

from __future__ import annotations

import ast
import json
import re
import sys

import _helpers
from _helpers import HELD_TESTS_MARKER as _HELD_MARKER
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal










def _trim_to_parse(code: str, max_drops: int = 10) -> str:
    """Drop trailing non-parsing lines (bounded) — seat models sometimes leave
    a prose usage line inside the fence. Valid code returns byte-identical;
    if nothing parses within the bound, return the original unchanged."""
    lines = code.splitlines()
    for drop in range(min(max_drops, len(lines)) + 1):
        candidate = "\n".join(lines[: len(lines) - drop]).rstrip()
        if not candidate:
            break
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate if drop else code
    return code


def _extract_code(text: str) -> str:
    return _trim_to_parse(_helpers.extract_code(text, drop_test_blocks=True))


def _extract_tests(text: str) -> str:
    fenced = _helpers.extract_code(text)
    if fenced != text.strip():
        return fenced
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def test_", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return text.strip()


# The rung-1 context marker (classify composes it): everything before the
# marker is conversation context for generation seats; the requirement the
# verifier chain echoes is the clean turn after it (ADR-048 isolation).
_REQUEST_MARKER = "\n\nCurrent request: "

# The deliverable's destination named in the turn (mirrors classify's file
# extraction) — the executor shadows a stale workspace copy at this name.
_FILE_RE = re.compile(
    r"\b([\w./-]+\.(?:py|js|ts|jsx|tsx|json|md|txt|ya?ml|sh|go|rs|java|c|cpp|h))\b"
)

# A file block in the rendered context: conversation-written ([wrote ...])
# or client-read ([read ...], issue #83). '(truncated)' / '(failed)' /
# '(oversize)' variants are never materialized; a failed read line carries
# trailing reason text after ']' and so never matches the anchored $.
_FILE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)( \((?:truncated|failed|oversize)\))?\]$"
)


def _workspace(context: str) -> dict[str, str]:
    """Conversation-written and client-read files as {basename: body} for the
    sandbox.

    Fenced block grammar (2026-07-10): body lines carry a two-space indent
    the renderer added; the indent is stripped on materialization and ANY
    other non-empty line ends the body. Headers live only at column 0, so a
    header lookalike inside untrusted file content strips back to plain
    content and can never materialize a phantom file.
    """
    files: dict[str, str] = {}
    lines = context.splitlines()
    index = 0
    while index < len(lines):
        header = _FILE_HEADER_RE.match(lines[index])
        index += 1
        if not header:
            continue
        body_lines = []
        while index < len(lines):
            line = lines[index]
            if line.startswith("  "):
                body_lines.append(line[2:])
            elif not line.strip():
                body_lines.append("")
            else:
                break
            index += 1
        if not header.group(2):
            name = header.group(1).rsplit("/", 1)[-1]
            files[name] = "\n".join(body_lines).strip()
    return files


def _inject_workspace_imports(text: str, workspace: dict[str, str]) -> str:
    """Prepend imports for workspace-module names a deliverable uses but never
    imports (a common small-model omission caught by the accept gate)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    defined = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    prelude: list[str] = []
    for filename, body in workspace.items():
        module = filename.rsplit(".", 1)[0]
        already_imported = (
            f"import {module}" in text or f"from {module} import" in text
        )
        if already_imported or not module.isidentifier():
            continue
        try:
            exported = {
                n.name
                for n in ast.parse(body).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        except SyntaxError:
            continue
        missing = sorted((used - defined) & exported)
        if missing:
            prelude.append(f"from {module} import {', '.join(missing)}")
    return "\n".join(prelude) + "\n" + text if prelude else text


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    requirement = str(payload.get("input_data", ""))
    workspace: dict[str, str] = {}
    if _REQUEST_MARKER in requirement:
        context, requirement = requirement.rsplit(_REQUEST_MARKER, 1)
        workspace = _workspace(context)
    deps = payload.get("dependencies", {})
    if not isinstance(deps, dict):
        deps = {}

    # held round (issue #100): no test_writer seat — the carry's sentinel
    # block IS the spec; strip it from the requirement the verifiers echo.
    # A fresh round with test_writer output never takes this path, so a
    # sentinel in user text worst-cases into a reject, never a wrong accept.
    tests_terminal = _terminal(_response(deps.get("test_writer", {})))
    held = not tests_terminal.strip() and _HELD_MARKER in requirement
    if held:
        requirement, _, held_block = requirement.partition(_HELD_MARKER)
        requirement = requirement.strip()
        tests = _extract_tests(held_block)
    else:
        tests = _extract_tests(tests_terminal)
    code = _extract_code(_terminal(_response(deps.get("code_writer", {}))))
    tests = _inject_workspace_imports(tests, workspace)
    code = _inject_workspace_imports(code, workspace)

    file_match = _FILE_RE.search(requirement)
    target_file = file_match.group(1).rsplit("/", 1)[-1] if file_match else ""

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": code,
                "tests": tests,
                "held": held,
                "workspace": workspace,
                "target_file": target_file,
            }
        )
    )


if __name__ == "__main__":
    main()
