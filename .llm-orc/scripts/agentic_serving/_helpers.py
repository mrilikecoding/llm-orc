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


# A '[globbed ...]' block (issue #83 discovery grammar, serving_ensemble_
# caller._render_glob_block) — never itself materialized as workspace
# content (its header is a distinct shape from _FILE_HEADER_RE and its body
# is one path per line, not a file body), but read here PURELY as a
# structural signal for the workspace's true project root (#184 A1 rework,
# rule a): a repo-wide glob routinely spans several directories, which one
# or two read/write headers alone cannot.
_GLOB_HEADER_RE = re.compile(
    r"^assistant: \[globbed [^\]]+?( \((?:truncated|failed)\))?\]$"
)


def _glob_listing_paths(context: str) -> list[str]:
    """Path lines from the LATEST usable ``[globbed ...]`` block in
    ``context`` — "" (failed) blocks carry no real path lines and are
    skipped; a "(truncated)" listing's paths are still real and kept."""
    lines = context.splitlines()
    paths: list[str] = []
    index = 0
    while index < len(lines):
        header = _GLOB_HEADER_RE.match(lines[index])
        index += 1
        if not header:
            continue
        failed = header.group(1) == " (failed)"
        body: list[str] = []
        while index < len(lines) and lines[index].startswith("  "):
            body.append(lines[index][2:])
            index += 1
        if not failed:
            paths = body  # the LATEST block wins
    return paths


def _glob_listing_root(context: str) -> str:
    """Rule (a): the longest common directory of the client's own glob
    listing, when at least two of its paths are absolute — "" when no
    usable listing is visible, or it does not resolve past the filesystem
    root itself."""
    paths = [p for p in _glob_listing_paths(context) if p.startswith("/")]
    if len(paths) < 2:
        return ""
    root = posixpath.commonpath(paths)
    return root if root not in ("", "/") else ""


def _suffix_matched_root(abs_path: str, candidates: list[str]) -> str:
    """Rule (b): the prefix obtained by matching a KNOWN relative path (the
    ask's own named destination, or another header already relative in the
    same render) as a SUFFIX of ``abs_path`` — the longest candidate that
    matches wins (most specific). "" when none matches."""
    best = ""
    for candidate in candidates:
        if not candidate or candidate.startswith("/"):
            continue
        if abs_path.endswith("/" + candidate) and len(candidate) > len(best):
            best = candidate
    return abs_path[: -(len(best) + 1)] if best else ""


def _resolve_root(
    raw: list[tuple[str, str]], abs_paths: list[str], target_path: str, context: str
) -> tuple[str, str]:
    """(root, rule) — the priority ladder #184's A1 rework replaces the
    single "longest common directory" computation with:

    (a) the client's own ``[globbed ...]`` listing's common directory, when
        a useful one is visible;
    (b) for a SINGLE absolute header, the ask's own named destination or
        another already-relative header in the same render, matched as a
        SUFFIX of that absolute path — recovers the true root exactly when
        the client read the file it is about to edit (the arc's own
        motivating shape);
    (c) the longest common directory of 2+ absolute headers (unchanged);
    (d) none of the above resolves anything: "" — a lone header falls back
        to its bare basename (a recorded bound, not a silent guess: the
        caller reports this as the "basename" rule); 2+ headers with no
        common root stay absolute and UNRESOLVED — #184 A2's executor
        refusal is what makes that honest rather than a silent drop.
    """
    glob_root = _glob_listing_root(context)
    if glob_root:
        return glob_root, "glob"
    if len(abs_paths) == 1:
        already_relative = [p for p, _ in raw if not p.startswith("/")]
        candidates = ([target_path] if target_path else []) + already_relative
        matched = _suffix_matched_root(abs_paths[0], candidates)
        if matched:
            return matched, "suffix-match"
        # (d) basename fallback: the dirname is the root to strip, which
        # always reduces `_relative_to_root` to the bare basename — the
        # pre-#184-A1-rework single-header behavior, just reported by name
        # now instead of silently reached.
        return abs_paths[0].rsplit("/", 1)[0], "basename"
    root = posixpath.commonpath(abs_paths)
    if root not in ("", "/"):
        return root, "common-prefix"
    return "", "unresolved"


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


def resolve_workspace_entries(
    context: str, target_path: str = ""
) -> tuple[list[tuple[str, str]], str]:
    """(relativized entries, root_rule) — #184 A1's one workspace reader,
    with which rule resolved the root reported alongside (never silent):
    "none" (no absolute headers at all — every header was already
    relative), "glob", "suffix-match", "common-prefix", "basename" (a lone
    header, unresolved by (a)/(b)), or "unresolved" (2+ headers sharing no
    common root — left absolute; #184 A2 is what refuses on this rather
    than silently running an incomplete workspace)."""
    raw = raw_workspace_entries(context)
    abs_paths = [path for path, _ in raw if path.startswith("/")]
    if not abs_paths:
        return raw, "none"
    root, rule = _resolve_root(raw, abs_paths, target_path, context)
    return [(_relative_to_root(path, root), body) for path, body in raw], rule


def workspace_entries(context: str, target_path: str = "") -> list[tuple[str, str]]:
    """Ordered (RELATIVE path, body) for every read/write block in
    ``context`` (#184 mechanism 1) — see ``resolve_workspace_entries`` for
    the rule the root resolved by; this is the entries alone, for callers
    that do not need the rule reported."""
    return resolve_workspace_entries(context, target_path)[0]


def fold_workspace(entries: list[tuple[str, str]]) -> dict[str, str]:
    """{relative path: body}, folding ``entries`` in order — the last
    block for a given path wins. Keyed by the FULL relative path (#184
    mechanism 1), not a basename — two files sharing a basename in
    different directories no longer conflate."""
    files: dict[str, str] = {}
    for path, body in entries:
        files[path] = body
    return files


def workspace(context: str, target_path: str = "") -> dict[str, str]:
    """{relative path: body} for the sandbox — the one workspace reader
    (#184): shared by ``accept_gather`` (build routes), ``tests_gather``
    (write-tests, #98), and ``refix_gather`` (re-fix, #184)."""
    return fold_workspace(workspace_entries(context, target_path))


def workspace_unplaced_reason(names: list[str]) -> str:
    """The path-free refusal sentence for #184 A2: a workspace entry that
    survived root resolution still absolute (or escaping) never got
    materialized, and the brief's "refused, nothing materialized" bound
    was only half true — nothing turned the drop into a verdict, so the
    turn ran (and could ship) against a partial or empty workspace.
    ``names`` are bare basenames only, never the unplaceable path itself.
    Shared by ``accept_gate`` (build routes) and ``refix_envelope``
    (re-fix) so neither drifts from the other's wording."""
    listed = names[:3]
    joined = ", ".join(listed)
    remaining = len(names) - len(listed)
    if remaining > 0:
        joined += f", and {remaining} more"
    return (
        f"a workspace file could not be placed in the sandbox ({joined}); "
        "the original is unchanged"
    )


def payload(raw: str) -> dict[str, Any]:
    """The script-node stdin payload as a dict ({} on anything malformed)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


class MultipleQueriesError(ValueError):
    """A selected child input carries more than one query (issue #202).

    One search runs per invocation; N queries compose via ``fan_out: true``
    (ADR-014), one child execution per query. Silent truncation would run
    one search and discard the rest with no signal.
    """

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(f"child input carries {count} queries")


def child_input_value(data: dict[str, Any]) -> Any:
    """The child input under the script-node payload contract (issue #202).

    A root (no-dependency) script receives ``{"input": <value>, ...}``;
    both payload builders emit that shape (``agents/script_agent.py`` and
    ``core/execution/scripting/agent_runner.py``). A dependent script
    receives ScriptAgentInput (``{"input_data": "<value>",
    "dependencies": {...}}``, ADR-001), with input_data typed str. Returns
    the value under either key (input_data wins when both are present;
    the engine never ships both), or None when neither is present.
    """
    value = data.get("input_data")
    if value is None:
        value = data.get("input")
    return value


def query_from_value(value: Any) -> str:
    """The search query from a child-input value (issue #202).

    A str unwraps via _query_from_text (a JSON-encoded list is the
    input_key-selected array; empty means no query); a list applies the
    one-query-per-invocation rule (more than one item raises
    MultipleQueriesError); a dict recurses for its query key; anything
    else (None, bool, number) has no usable query.
    """
    if isinstance(value, str):
        return _query_from_text(value)
    if isinstance(value, list):
        return _query_from_list(value)
    if isinstance(value, dict):
        return extract_query(value)
    return ""


def _query_from_list(items: list[Any]) -> str:
    """The query from an input_key-selected array (issue #202)."""
    if not items:
        return ""
    if len(items) > 1:
        raise MultipleQueriesError(len(items))
    return _query_from_item(items[0])


def _query_from_item(value: Any) -> str:
    """The query from one array item or a direct query key."""
    if isinstance(value, str):
        return _query_from_text(value)
    if isinstance(value, dict):
        return extract_query(value)
    if isinstance(value, list):
        return _query_from_list(value)
    return ""


def _query_from_text(text: str) -> str:
    """The query from a child-input string (issue #202).

    The text may be the query itself, a JSON-encoded array (the
    input_key-selected array; empty means no query), or a JSON-encoded
    dict with its own query key. Anything else is used verbatim.
    """
    stripped = text.strip()
    if not stripped.startswith(("[", "{")):
        return stripped
    try:
        parsed: Any = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(parsed, list):
        return _query_from_list(parsed)
    if isinstance(parsed, dict):
        return extract_query(parsed)
    return stripped


def extract_query(payload: dict[str, Any]) -> str:
    """The search query from a script-node payload (issue #202).

    Precedence: a direct ``query`` key (flat or nested under
    ``parameters``, the orchestrator's dispatch convention), then a
    ``data`` prompt, then the child input under the payload contract
    (child_input_value).
    """
    if "query" in payload:
        return _query_from_item(payload["query"])
    parameters = payload.get("parameters") or {}
    if isinstance(parameters, dict) and "query" in parameters:
        return _query_from_item(parameters["query"])
    if isinstance(payload.get("data"), str):
        return _query_from_text(payload["data"])
    return query_from_value(child_input_value(payload))


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
    blocks = [body for lang, body in tagged if lang.lower() not in SHELL_LANGS] or [
        body for _, body in tagged
    ]
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
        node.name.startswith("test_") or node.name.startswith("Test") for node in named
    )
