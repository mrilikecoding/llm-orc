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
from _helpers import public_top_level_names as _public_top_level_names
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
# '(oversize)' / '(over-budget)' (C1, #145) variants are never
# materialized; a failed read line carries trailing reason text after ']'
# and so never matches the anchored $. MAJOR 1 (review round 1): a variant
# missing from this alternation isn't rejected — the non-greedy name-group
# absorbs " (variant)" into the "path" instead, and the header still
# matches as if unvariant, materializing a corrupted phantom file (this
# grammar now has four consumers across the codebase — classify.py,
# refix_gather.py, and the caller that produces it; a shared vocabulary
# constant is noted as follow-up, not built here).
_FILE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)"
    r"( \((?:truncated|failed|oversize|over-budget)\))?\]$"
)


def _workspace_entries(context: str) -> list[tuple[str, str]]:
    """Ordered (full path, body) for every valid (non-variant) read/write
    block in ``context`` — the FULL path exactly as rendered, not
    truncated to a basename. ``_workspace`` derives its {basename: body}
    mapping from this (last write for a given basename wins, preserving
    its existing behavior); ``_prior_surface`` (#182 slice B-1, review F3
    fix) uses the full paths directly, so two files sharing a basename in
    different directories are never conflated.

    Fenced block grammar (2026-07-10): body lines carry a two-space indent
    the renderer added; the indent is stripped on materialization and ANY
    other non-empty line ends the body. Headers live only at column 0, so a
    header lookalike inside untrusted file content strips back to plain
    content and can never materialize a phantom file.
    """
    entries: list[tuple[str, str]] = []
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
            entries.append((header.group(1), "\n".join(body_lines).strip()))
    return entries


def _fold_basenames(entries: list[tuple[str, str]]) -> dict[str, str]:
    """{basename: body}, folding ``entries`` in order — the last block for
    a given basename wins."""
    files: dict[str, str] = {}
    for path, body in entries:
        files[path.rsplit("/", 1)[-1]] = body
    return files


def _workspace(context: str) -> dict[str, str]:
    """Conversation-written and client-read files as {basename: body} for
    the sandbox — the last block for a given basename wins (unchanged
    behavior; see ``_workspace_entries`` for the full-path-preserving
    parse this derives from)."""
    return _fold_basenames(_workspace_entries(context))


def _top_level_defs(text: str) -> frozenset[str]:
    """Function/class names bound at MODULE level of ``text``, or empty when
    it doesn't parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    return frozenset(
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _inject_workspace_imports(
    text: str,
    workspace: dict[str, str],
    candidate_defined: frozenset[str] = frozenset(),
) -> str:
    """Prepend imports for workspace-module names a deliverable uses but never
    imports (a common small-model omission caught by the accept gate).

    ``candidate_defined`` (#171, WA-2b): names the CANDIDATE deliverable
    already binds at module level — never a re-injection target, on either
    call site. Without this, a bare-name test reference that would have
    bound the deliverable got rebound to a workspace module by the injected
    import (the runner execs code THEN tests into one shared namespace, so
    the import always wins), retargeting the gate at the workspace's copy
    regardless of whether the candidate was right or wrong.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    defined = _top_level_defs(text)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    prelude: list[str] = []
    for filename, body in workspace.items():
        module = filename.rsplit(".", 1)[0]
        already_imported = f"import {module}" in text or f"from {module} import" in text
        if already_imported or not module.isidentifier():
            continue
        exported = _top_level_defs(body)
        missing = sorted((used - defined - candidate_defined) & exported)
        if missing:
            prelude.append(f"from {module} import {', '.join(missing)}")
    return "\n".join(prelude) + "\n" + text if prelude else text


def _prior_body(
    entries: list[tuple[str, str]], target_file: str, target_path: str
) -> str:
    """The prior body to check the deliverable's surface against (#182
    slice B-1), or "" when it cannot be determined without guessing.

    Review F3 fix (wrong-reject): two files can share a basename in
    different directories (``todo/storage.py`` and ``lib/storage.py``) —
    ``_workspace``'s {basename: body} mapping keys by basename alone, so
    whichever block was rendered LAST silently won, regardless of which
    file the ask actually named. When the ask names a real directory
    component (``target_path`` contains ``/``), the matching block is the
    one whose FULL rendered path equals it exactly — never conflated with
    a same-basename file elsewhere. Otherwise (a bare-basename ask) fall
    back to basename matching, but ONLY when exactly one block shares that
    basename; two (or more) same-named files in different directories
    then yield "" — a named bound (recorded in the design brief) rather
    than a guess at which one the ask means."""
    if not target_file:
        return ""
    if "/" in target_path:
        path_matches = [body for path, body in entries if path == target_path]
        return path_matches[-1] if path_matches else ""
    name_matches = [
        body for path, body in entries if path.rsplit("/", 1)[-1] == target_file
    ]
    return name_matches[0] if len(name_matches) == 1 else ""


def _prior_surface(
    entries: list[tuple[str, str]], target_file: str, target_path: str
) -> list[str]:
    """The target file's PRIOR public top-level names (#182 slice B-1),
    when its prior body is visible in the rendered context — a
    conversation-written ``[wrote <path>]`` block or a client ``[read
    <path>]`` block (see ``_prior_body`` for how the matching block is
    chosen). Empty when no prior is visible, the match is ambiguous, or
    the prior does not parse — the gate must never refuse over
    indeterminate surface."""
    prior_body = _prior_body(entries, target_file, target_path)
    if not prior_body:
        return []
    try:
        tree = ast.parse(prior_body)
    except SyntaxError:
        return []
    return _public_top_level_names(tree)


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    requirement = str(payload.get("input_data", ""))
    entries: list[tuple[str, str]] = []
    if _REQUEST_MARKER in requirement:
        context, requirement = requirement.rsplit(_REQUEST_MARKER, 1)
        entries = _workspace_entries(context)
    workspace = _fold_basenames(entries)
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
    candidate_defined = _top_level_defs(code)
    tests = _inject_workspace_imports(tests, workspace, candidate_defined)
    code = _inject_workspace_imports(code, workspace, candidate_defined)

    file_match = _FILE_RE.search(requirement)
    target_path = file_match.group(1) if file_match else ""
    target_file = target_path.rsplit("/", 1)[-1] if target_path else ""
    prior_surface = _prior_surface(entries, target_file, target_path)

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": code,
                "tests": tests,
                "held": held,
                "workspace": workspace,
                "target_file": target_file,
                "prior_surface": prior_surface,
            }
        )
    )


if __name__ == "__main__":
    main()
