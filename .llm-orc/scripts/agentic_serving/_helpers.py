#!/usr/bin/env python3
"""Shared helpers for the agentic_serving script nodes (issue #92).

Python sets ``sys.path[0]`` to the script's directory, so sibling scripts
import this module directly. One implementation each for the three things
every node re-implemented: the script-node payload contract, the
sub-ensemble envelope peel, and fenced-code extraction — the extractors had
drifted into three divergent copies, so the accept gate could judge
different code than emit shipped.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from typing import Any

# Fences tagged with a shell-ish language are usage examples, not code.
SHELL_LANGS = {"bash", "sh", "shell", "console", "zsh", "text"}

# The TDD retry sentinel (issue #100): a rejected round whose tests collected
# and were judged adequate carries them under this marker; route dispatches
# the held round on it, and gather reads the tests back out. One constant —
# the envelope writes it, route and gather read it.
HELD_TESTS_MARKER = "[HELD TESTS: round 1 spec; regenerate ONLY the code]"

# The convergent-fix sentinel (rung 2, docs/plans/2026-07-12-convergent-fix-
# design.md): classify composes the fix-led write's content under this
# marker in the re-fix dispatch_input; refix_gather reads it back out. One
# constant — classify writes it, refix_gather reads it.
PRIOR_CODE_MARKER = "[PRIOR CODE: this turn's write, before the re-fix]"

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

# A ``[ran <command>]`` block header, with its optional failed/truncated
# variant and any inline trailing detail — shared by run_verdict (verdict
# prose) and classify (rung 2's failure-shape routing signal), issue #83 /
# convergent-fix rung 2.
_RAN_HEADER_RE = re.compile(r"^assistant: \[ran (.+?)( \((failed|truncated)\))?\](.*)$")

# A file block in the rendered context: conversation-written ([wrote ...])
# or client-read ([read ...], issue #83). '(truncated)' / '(failed)' /
# '(oversize)' / '(over-budget)' (C1, #145) variants are never
# materialized; a failed read line carries trailing reason text after ']'
# and so never matches the anchored $. A variant missing from this
# alternation isn't rejected — the non-greedy name-group absorbs "
# (variant)" into the "path" instead, and the header still matches as if
# unvariant, materializing a corrupted phantom file. Shared by every
# reader of the conversation workspace (#184: accept_gather, tests_gather,
# refix_gather) — one grammar, so a variant added for one consumer cannot
# drift from the others.
_FILE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)"
    r"( \((?:truncated|failed|oversize|over-budget)\))?\]$"
)


def raw_workspace_entries(context: str) -> list[tuple[str, str]]:
    """Ordered (path exactly as rendered, body) for every valid
    (non-variant) read/write block in ``context`` — the client's OWN path,
    absolute or relative, exactly as its tool call carried it; never
    truncated to a basename. ``workspace_entries`` (below) is what every
    consumer actually reads — it additionally strips a shared absolute
    prefix when one is present (#184 mechanism 1).

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


def _common_absolute_root(paths: list[str]) -> str:
    """The longest shared absolute DIRECTORY among ``paths`` (#184
    mechanism 1) — "" when fewer than one path is absolute, or when the
    only thing they share is the filesystem root itself (a header with no
    real common project root: recorded fresh-input bound, two clients'
    absolute trees in one render leave every path exactly as rendered,
    and downstream materialization refuses an absolute path rather than
    guess which root it belongs to).

    A single absolute path's own directory counts (the flat-workspace
    subcase: one file at the client's repo root strips to its bare
    basename, exactly as basename-keying already did)."""
    abs_paths = [p for p in paths if p.startswith("/")]
    if not abs_paths:
        return ""
    if len(abs_paths) == 1:
        root = abs_paths[0].rsplit("/", 1)[0]
    else:
        root = posixpath.commonpath(abs_paths)
    return root if root not in ("", "/") else ""


def _relative_to_root(path: str, root: str) -> str:
    """``path`` with ``root`` stripped, when it actually shares it —
    unchanged otherwise (a relative header, or an absolute header sharing
    no common root with the others: left absolute on purpose, #184's
    path-safety check refuses to materialize it rather than guess)."""
    if not root:
        return path
    if path == root:
        return path.rsplit("/", 1)[-1]
    prefix = root + "/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def workspace_entries(context: str) -> list[tuple[str, str]]:
    """Ordered (RELATIVE path, body) for every read/write block in
    ``context`` (#184 mechanism 1) — the client's own absolute prefix
    stripped once (see ``_common_absolute_root``), so ``todo/storage.py``
    and ``lib/storage.py`` stay two distinct files instead of folding to
    one ``storage.py`` by basename. A header already relative (the common
    live shape — the server's own echo mechanism names files by their
    relative path already) passes through unchanged."""
    raw = raw_workspace_entries(context)
    root = _common_absolute_root([path for path, _ in raw])
    return [(_relative_to_root(path, root), body) for path, body in raw]


def fold_workspace(entries: list[tuple[str, str]]) -> dict[str, str]:
    """{relative path: body}, folding ``entries`` in order — the last
    block for a given path wins. Keyed by the FULL relative path (#184
    mechanism 1), not a basename — two files sharing a basename in
    different directories no longer conflate."""
    files: dict[str, str] = {}
    for path, body in entries:
        files[path] = body
    return files


def workspace(context: str) -> dict[str, str]:
    """{relative path: body} for the sandbox — the one workspace reader
    (#184): shared by ``accept_gather`` (build routes), ``tests_gather``
    (write-tests, #98), and ``refix_gather`` (re-fix, #184)."""
    return fold_workspace(workspace_entries(context))


def payload(raw: str) -> dict[str, Any]:
    """The script-node stdin payload as a dict ({} on anything malformed)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def deps(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """The ``dependencies`` mapping from a script-node payload."""
    value = payload_dict.get("dependencies", {})
    return value if isinstance(value, dict) else {}


def response(dep: Any) -> str:
    """A dependency node's response string ('' when absent or non-string)."""
    if isinstance(dep, dict):
        resp = dep.get("response", "")
        return resp if isinstance(resp, str) else json.dumps(resp)
    return ""


def terminal(text: str) -> str:
    """Peel sub-ensemble envelope layers (deliverable / output / results) to
    the terminal node's raw output."""
    current = text
    for _ in range(6):
        try:
            obj = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return current
        if not isinstance(obj, dict):
            return current
        if isinstance(obj.get("deliverable"), str):
            current = obj["deliverable"]
            continue
        if isinstance(obj.get("output"), str):
            current = obj["output"]
            continue
        results = obj.get("results")
        if isinstance(results, dict) and results:
            node = results[list(results.keys())[-1]]
            current = node.get("response", "") if isinstance(node, dict) else str(node)
            continue
        return current
    return current


def extract_code(text: str, *, drop_test_blocks: bool = False) -> str:
    """The code deliverable from a (possibly chatty) seat response.

    Non-shell fenced blocks joined; falls back to all fences, then to the
    raw text. The ONE set of semantics for the gate, the envelope, and
    emit — divergent copies meant the gate could approve code that was not
    the code shipped.

    ``drop_test_blocks`` is for the CODE consumers only: seat models
    sometimes emit the code and a copy of the tests as two fences, and
    joining them ships a file with the test suite embedded. Pure-test
    blocks are dropped when a non-test block exists; the tests extraction
    never sets this (test blocks are its point).
    """
    tagged = _FENCE_RE.findall(text)
    blocks = [
        body for lang, body in tagged if lang.lower() not in SHELL_LANGS
    ] or [body for _, body in tagged]
    if drop_test_blocks and len(blocks) > 1:
        non_test = [block for block in blocks if not _is_pure_test_block(block)]
        blocks = non_test or blocks
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return text.strip()


def latest_ran_block(text: str) -> tuple[str, str, str, str] | None:
    """(command, variant, inline detail, body) of the LAST ``[ran ...]``
    block in ``text``, or ``None`` when no block is present.

    The body is the block's two-space-indented lines, de-indented (a blank
    line inside the body stays blank); untrusted output text can never be
    confused with block headers since headers live only at column 0 (fenced
    block grammar). Shared by run_verdict's verdict-prose parse and
    classify's rung-2 failure-shape classification — moved out of
    run_verdict so both read the identical block shape (no duplicate parser
    to drift).
    """
    lines = text.splitlines()
    found: tuple[int, re.Match[str]] | None = None
    for index, line in enumerate(lines):
        match = _RAN_HEADER_RE.match(line)
        if match:
            found = (index, match)
    if found is None:
        return None
    index, match = found
    body_lines: list[str] = []
    for line in lines[index + 1 :]:
        if line.startswith("  "):
            body_lines.append(line[2:])
        elif not line.strip():
            body_lines.append("")
        else:
            break
    command = match.group(1)
    variant = match.group(3) or ""
    detail = (match.group(4) or "").strip()
    return command, variant, detail, "\n".join(body_lines).strip()


def assign_target_names(target: ast.expr) -> set[str]:
    """Public plain-``Name`` targets an assignment target binds: a bare
    ``Name`` (excluding a leading underscore, same rule as def/class), or a
    ``Tuple``/``List`` of them for simple unpacking (``A, B = 1, 2``).
    Anything else — ``Attribute``, ``Subscript``, ``Starred`` — contributes
    no name, same as it always has for def/class (this function only ever
    ADDS names, never removes one the old rule already caught).

    Shared by ``refix_select`` (the re-fix smoke test's surface, #171) and
    ``accept_gather`` (an edit's prior-surface contract, #182 slice B-1) —
    moved here so both read one definition rather than two that could drift.
    """
    if isinstance(target, ast.Name):
        return set() if target.id.startswith("_") else {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names |= assign_target_names(elt)
        return names
    return set()


def public_top_level_names(tree: ast.Module) -> list[str]:
    """Every PUBLIC top-level binding a module makes: def/class names
    (#171 F-2), plus (#171 F-1 round 3, widened rather than skipped)
    plain-assignment targets — ``Assign`` and ``AnnAssign`` with a
    ``Name`` target, simple tuple/list unpacking included. A settings
    module's ``PORT = 8080`` is a public binding exactly the way a
    function name is; an edit that drops it must refuse for the same
    reason an edit that drops a function refuses. Import statements bind
    names too but are deliberately NOT surface here (never were) — an
    import-only module (an ``__init__`` re-export, say) has zero public
    bindings by this function's own definition.

    Round 2b fix 2 (independent confirmation review, #171): an
    ``AnnAssign`` with ``value is None`` — a bare annotation, ``PORT: int``
    with no ``= ...`` — binds NOTHING at module level; executing that
    statement alone creates no attribute. Including it put an
    unsatisfiable name in the surface (the prior module's OWN bare
    annotation could never satisfy ``hasattr(solution, "PORT")`` either),
    refusing legitimate fixes to sibling names for "dropping" something
    the prior never bound in the first place. Only an ``AnnAssign`` that
    actually assigns a value counts."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for assign_target in node.targets:
                names |= assign_target_names(assign_target)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            names |= assign_target_names(node.target)
    return sorted(names)


def _is_pure_test_block(block: str) -> bool:
    """A parsed block whose top-level defs/classes are all test-named
    (imports and docstrings allowed). Non-parsing blocks are kept."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return False
    named = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not named:
        return False
    return all(
        node.name.startswith("test_") or node.name.startswith("Test")
        for node in named
    )
