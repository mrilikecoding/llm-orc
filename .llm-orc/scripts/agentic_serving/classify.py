#!/usr/bin/env python3
"""Serving Ensemble — classify (decider) node.

Emits the routing decision the dispatch seat resolves:

    {"target", "kind", "file", "dispatch_input", "build"}

Routing is deterministic where the signal is structural: an explain-shaped turn
is non-build prose; a turn with a build verb or a named target file is a build
routed to the default code-generation seat. The build-vs-non-build (executable-
deliverable) determination is classify's own responsibility (ADR-046 §1
responsibility matrix) — ``build`` gates marshal's file-vs-prose shaping and, at
WP-D8, the gated accept shape.

When neither structural signal resolves the turn, classify does NOT guess a
default seat: it emits ``needs_decider: true`` and leaves ``target`` empty, so a
guarded model-backed ``decide`` node reads the turn intent and a ``resolve`` node
merges the two (scenarios.md "classify reads intent with a model-backed decider
when the signal is not structural"; ADR-046 §1, classify is the decider seat).
Determinism is preserved: the model runs only on the guarded ambiguous path, its
output is a closed target set, and an unresolved target fails at dispatch.

The seat is filled by dynamic dispatch on ``${resolve.target}``, so swapping a
seat strategy is a change to this decision or the operator default, never to the
skeleton (AS-11).
"""

from __future__ import annotations

import json
import re
import sys

_EXPLAIN_MARKERS = (
    "explain",
    "what does",
    "how does",
    "describe",
    "summarize",
    "why does",
    "what is",
    "tell me",
)
# An interrogative-shaped turn asks for understanding; it outranks the
# named-file build signal ("What approach does palindrome.py use?" is an
# explain turn, not a build). The yes/no forms are deliberately narrow —
# only memory-shaped questions addressed to the assistant ("did you…",
# "have you…"); "can/could/will you write X" are polite imperatives and
# must stay on the build path (ladder turn 5 mis-route, 2026-07-09).
_INTERROGATIVE_RE = re.compile(
    r"^(?:what|why|how|when|where|which|who)\b|^(?:did|have) you\b",
    re.IGNORECASE,
)
_DEFAULT_CODE_SEAT = "code-seat"
_EXPLAIN_SEAT = "explainer"
_TESTS_SEAT = "tests-seat"
# Tests as the OBJECT of the request (issue #98): a build verb directly
# asking for tests, or "tests for/of/against <target>". A trailing "with
# tests" mention stays a code turn — routing it here would ship only tests.
_TESTS_PRIMARY_RE = re.compile(
    r"\b(?:write|add|create|generate|implement|build)\s+"
    r"(?:some\s+|unit\s+|more\s+|the\s+)?tests?\b"
    r"|\btests?\s+(?:for|of|against)\b",
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r"\b([\w./-]+\.(?:py|js|ts|jsx|tsx|json|md|txt|ya?ml|sh|go|rs|java|c|cpp|h))\b"
)
# A structural build signal: an imperative verb that asks for code to be
# produced or changed. Word-boundaried so "add" does not fire on "address".
_BUILD_RE = re.compile(
    r"\b(write|implement|create|build|generate|refactor|fix|add|code)\b",
    re.IGNORECASE,
)
# issue #83: a build verb that implies the named file already exists in the
# client workspace. "write"/"create" stay fresh-create — requesting a read
# for a file that does not exist yet would refuse a valid build.
_EXISTING_RE = re.compile(
    r"\b(fix|update|modify|refactor|edit|change|existing)\b", re.IGNORECASE
)
# Chained fix-execution trigger: the task must be LED by a fix imperative —
# mid-sentence "existing"/"change" are ordinary build prose (PR #115
# review). Mirrors the caller's _FIX_CHAIN_RE; a regression test pins
# pattern and flags equal.
_FIX_VERB_RE = re.compile(
    r"^\s*(?:fix|update|modify|refactor|edit|change)\b", re.IGNORECASE
)
# Context-block headers (the caller's render grammar). Visible = untruncated
# wrote block or successful read block; attempted = any read header. The
# optional variant group keeps a "(truncated)" suffix out of the path.
_VISIBLE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)"
    r"( \((?:truncated|failed|oversize)\))?\]$",
    re.MULTILINE,
)
_READ_ATTEMPT_RE = re.compile(
    r"^assistant: \[read ([^\]]+?)( \((failed|oversize)\))?\]", re.MULTILINE
)
_READ_CAP_KB = 24
# issue #83 run half: an imperative run verb with a tests object later in
# the same sentence fragment ("run the unit tests", "rerun pytest", "run
# every single one of the unit tests"). A named test_*.py file with a run
# verb also qualifies ("run test_calc.py"). Composite turns are kept off
# the run path by verb suppression, not the window: any build or edit verb
# in the turn ("write tests ... and run them", "fix ... and rerun the
# tests") routes build-first — the follow-on run is the user's next turn
# (review finding 2026-07-09: the run route must never swallow a build).
_RUN_VERB_RE = re.compile(r"\b(?:re-?run|run|execute)\b", re.IGNORECASE)
_RUN_TESTS_RE = re.compile(
    r"\b(?:re-?run|run|execute)\b[^.!?\n]{0,60}?\b(?:tests?|pytest|suite)\b",
    re.IGNORECASE,
)
_RAN_HEADER_RE = re.compile(r"^assistant: \[ran ", re.MULTILINE)
# Defense in depth on top of _FILE_RE's already-safe charset: an argument
# that could carry shell metacharacters never reaches the command template.
_SAFE_ARG_RE = re.compile(r"^[\w./-]+$")
# issue #83 discovery: the exact rung-1 module-stem phrasings ("<stem>
# module", "module <stem>", "tests for <stem>"). The captured stem is
# identifier-ish — a strict charset subset of _SAFE_ARG_RE, so the glob
# pattern template downstream stays metacharacter-free (the run-command
# discipline).
_STEM_RES = (
    re.compile(r"\b([A-Za-z_]\w*)\s+modules?\b", re.IGNORECASE),
    re.compile(r"\bmodules?\s+([A-Za-z_]\w*)\b", re.IGNORECASE),
    re.compile(r"\btests?\s+for\s+(?:the\s+)?([A-Za-z_]\w*)\b", re.IGNORECASE),
)
# Anaphora, filler, and imperative verbs the phrasings can capture ("tests
# for it", "fix module storage" capturing "fix") — these stay with today's
# routing (design bounds).
_STEM_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "it",
        "them",
        "this",
        "that",
        "these",
        "those",
        "all",
        "each",
        "every",
        "some",
        "any",
        "both",
        "one",
        "same",
        "my",
        "your",
        "our",
        "his",
        "her",
        "its",
        "their",
        "me",
        "us",
        "and",
        "or",
        "in",
        "of",
        "for",
        "with",
        "to",
        "so",
        "then",
        "please",
        "now",
        "named",
        "called",
        "which",
        "whole",
        "module",
        "modules",
        "existing",
        "new",
        "test",
        "tests",
        "python",
        "write",
        "implement",
        "create",
        "build",
        "generate",
        "refactor",
        "fix",
        "add",
        "code",
        "update",
        "modify",
        "edit",
        "change",
        "run",
    }
)


def _extract_file(task: str) -> str:
    """A structural filename signal from the turn (e.g. 'in add.py')."""
    match = _FILE_RE.search(task)
    return match.group(1) if match else ""


def _named_source_files(task: str) -> list[str]:
    """Every named non-test source file, first-mention order, deduped."""
    files: list[str] = []
    for match in _FILE_RE.finditer(task):
        path = match.group(1)
        if path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if path not in files:
            files.append(path)
    return files


def _named_test_files(task: str) -> list[str]:
    """Every named test_*.py file, first-mention order, deduped."""
    files: list[str] = []
    for match in _FILE_RE.finditer(task):
        path = match.group(1)
        if not path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if path.endswith(".py") and path not in files:
            files.append(path)
    return files


def _run_test_command(task: str) -> str:
    """The closed run template: ``pytest -q`` + regex-safe named test files.

    Never model text (deterministic control) — the only variable part is
    filenames already restricted to ``_FILE_RE``'s metacharacter-free
    charset, re-asserted here.
    """
    named = [path for path in _named_test_files(task) if _SAFE_ARG_RE.match(path)]
    return " ".join(["pytest", "-q", *named]).strip()


def _visibility(context: str) -> tuple[set[str], dict[str, str]]:
    """(visible basenames, attempted basename -> failure detail)."""
    visible = {
        path.rsplit("/", 1)[-1]
        for path, variant in _VISIBLE_HEADER_RE.findall(context)
        if not variant
    }
    attempted: dict[str, str] = {}
    for path, _, variant in _READ_ATTEMPT_RE.findall(context):
        basename = path.rsplit("/", 1)[-1]
        if variant == "oversize":
            attempted[basename] = f"file exceeds the {_READ_CAP_KB} KB read cap"
        elif variant == "failed":
            attempted[basename] = "client read failed"
    return visible, attempted


def _files_to_request(
    task: str,
    context: str,
    tests_primary: bool,
    has_build_signal: bool,
    glob_file: str = "",
) -> tuple[list[str], str]:
    """(paths to request, refusal reason) — at most one is non-empty.

    Deterministic one-round control (issue #83): a named source file that is
    neither conversation-written nor client-read triggers ONE read request;
    a file whose read was already attempted and still is not visible refuses.
    ``glob_file`` is the discovery match feeding the same seam — a
    discovering turn names no source file itself, so it is the only entry.
    """
    wants_existing = tests_primary or (
        has_build_signal and bool(_EXISTING_RE.search(task))
    )
    if not wants_existing:
        return [], ""
    named = _named_source_files(task)
    if glob_file:
        named = [glob_file, *named]
    visible, attempted = _visibility(context)
    to_request: list[str] = []
    for path in named:
        basename = path.rsplit("/", 1)[-1]
        if basename in visible:
            continue
        if basename in attempted:
            return [], f"could not read {path}: {attempted[basename]}"
        to_request.append(path)
    return to_request, ""


def _module_stem(task: str) -> str:
    """The turn's single module stem, or "" (no stem, or multi-stem).

    Exact rung-1 phrasings only (discovery design 2026-07-10). Multi-stem
    turns are out of scope — they fall back to today's routing rather than
    guessing which stem the user meant.
    """
    stems: list[str] = []
    for pattern in _STEM_RES:
        for match in pattern.finditer(task):
            stem = match.group(1).lower()
            if stem not in _STEM_STOPWORDS and stem not in stems:
                stems.append(stem)
    return stems[0] if len(stems) == 1 else ""


def _globbed_candidates(context: str, stem: str) -> list[str] | None:
    """Candidate paths from the turn's ``[globbed ...]`` block, or ``None``
    when no listing exists yet (pass 1 fires).

    Matching is deterministic on the rendered block only (design bounds):
    basename contains the stem, ``.py``, not ``test_*``-named. The header
    scan is column-0 anchored, so an indented lookalike inside a read or
    run body never counts as a listing (fenced block grammar).
    """
    lines = context.splitlines()
    start = -1
    for index, line in enumerate(lines):
        if line.startswith("assistant: [globbed "):
            start = index
    if start < 0:
        return None
    candidates: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        basename = line.strip().rsplit("/", 1)[-1]
        if (
            stem in basename.lower()
            and basename.endswith(".py")
            and not basename.startswith("test_")
        ):
            candidates.append(line.strip())
    return candidates


def _discovery(
    task: str, context: str, tests_primary: bool, has_build_signal: bool
) -> tuple[str, str, str]:
    """(glob stem to request, matched path, refusal reason) — at most one is
    non-empty (issue #83 discovery, design 2026-07-10).

    One glob round per turn: a workspace-needing turn naming a module stem
    but no source file requests ONE listing; once a ``[globbed]`` block
    exists the deterministic MATCH step takes over — exactly one candidate
    becomes the turn's named file (the existing read seam fires next); zero
    or several candidates refuse honestly, never re-glob.
    """
    wants_existing = tests_primary or (
        has_build_signal and bool(_EXISTING_RE.search(task))
    )
    # A turn that names ANY file has nothing to discover — including
    # test_*-named files, which _named_source_files deliberately excludes
    # (review blocker 2026-07-10: "tests for test_storage.py" stemmed
    # "test_storage" and burned a doomed glob round).
    if not wants_existing or _extract_file(task):
        return "", "", ""
    stem = _module_stem(task)
    if not stem:
        return "", "", ""
    candidates = _globbed_candidates(context, stem)
    if candidates is None:
        visible, _ = _visibility(context)
        # The same candidate discipline as the globbed MATCH step (review
        # blocker 2026-07-10: any-extension + set-order pick shipped a
        # test_storage.json deliverable): .py only, not test_*, sorted for
        # determinism; one match names the file, several refuse, none
        # falls through to the glob request — the listing decides.
        matches = sorted(
            name
            for name in visible
            if name.rsplit(".", 1)[0].lower() == stem
            and name.endswith(".py")
            and not name.startswith("test_")
        )
        if len(matches) == 1:
            # the stem IS a visible file — nothing to discover, but it is
            # still the turn's named file (live finding 2026-07-10: without
            # this a retried module turn shipped to test_solution.py)
            return "", matches[0], ""
        if len(matches) > 1:
            listed = ", ".join(matches)
            return (
                "",
                "",
                f"multiple visible files match '{stem}': {listed} — please name one",
            )
        return stem, "", ""
    if len(candidates) == 1:
        return "", candidates[0], ""
    if not candidates:
        return "", "", f"no file matching '{stem}' in the workspace listing"
    listed = ", ".join(candidates)
    return (
        "",
        "",
        (
            f"multiple files match '{stem}' in the workspace listing: {listed}"
            " — please name one"
        ),
    )


def _turn(raw: str) -> dict:
    """Recover the turn dict from the ScriptAgent wrapper or a bare task.

    A no-dependency phase-0 script receives ``{"input": "<turn>", ...}``; a
    dependent script receives ``{"input_data": "<turn>", "dependencies": {...}}``.
    Handle both keys plus a bare turn dict for direct use.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"task": raw}
    if not isinstance(data, dict):
        return {"task": str(data)}
    inner = data.get("input_data")
    if inner is None:
        inner = data.get("input")
    if inner is None:
        inner = data
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        try:
            parsed = json.loads(inner)
            return parsed if isinstance(parsed, dict) else {"task": inner}
        except json.JSONDecodeError:
            return {"task": inner}
    return {"task": ""}


def _route(
    *,
    is_explain: bool,
    run_signal: bool,
    fix_chain: bool,
    has_run_block: bool,
    needs_glob: str,
    glob_failed: str,
    needs_files: list[str],
    read_failed: str,
    tests_primary: bool,
    has_build_signal: bool,
    kind_hint: str,
) -> tuple[str, str, bool, bool]:
    """(target, kind, build, needs_decider) — the deterministic routing chain.

    Run outranks marker-based explain (run_signal is already false on
    interrogative or marker-led turns), so "run the tests and tell me what
    failed" delegates the run instead of narrating one.
    """
    if fix_chain and has_run_block:
        # fix-execution verdict leg: the chained run came back — parse it.
        return "run-verdict", "run_verdict", False, False
    if fix_chain:
        # fix-execution run leg: this turn's fix already shipped its write
        # (wrote_path is structural, from the caller's post-boundary
        # tool_calls); delegate ONE closed-template run to verify it
        # client-side. Never re-enters the build — the branches below are
        # unreachable while fix_chain holds.
        return "need-run", "need_run", False, False
    if run_signal and has_run_block:
        # issue #83 run half: the client ran the command — the deliverable
        # is the deterministic verdict, one run round per turn.
        return "run-verdict", "run_verdict", False, False
    if run_signal:
        # issue #83 run half: delegate one closed-template test run.
        return "need-run", "need_run", False, False
    if is_explain:
        return _EXPLAIN_SEAT, "explanation", False, False
    if needs_glob or glob_failed:
        # issue #83 discovery: one glob round (or its honest refusal) before
        # the read seam. Exclusive with needs_files/read_failed by
        # construction — a discovering turn names no source file, a reading
        # turn does — so the order here only mirrors the seam chain
        # (discover -> read -> build).
        return "need-glob", "need_glob", False, False
    if needs_files or read_failed:
        # issue #83: request the client files (or refuse a failed request)
        # before any seat runs — the need-files shape is a cheap script echo.
        return "need-files", "need_files", False, False
    if tests_primary:
        # the deliverable IS a test file, run against the workspace alone
        # (issue #98) — never build-gated's code/tests duality
        return _TESTS_SEAT, "python_tests", True, False
    if has_build_signal:
        return _DEFAULT_CODE_SEAT, kind_hint, True, False
    # No structural signal — hand the routing to the guarded model decider.
    return "", "", False, True


def main() -> None:
    turn = _turn(sys.stdin.read().strip())
    task = str(turn.get("task", "")).strip()
    is_explain = any(marker in task.lower() for marker in _EXPLAIN_MARKERS) or bool(
        _INTERROGATIVE_RE.match(task)
    )
    named_file = turn.get("file") or _extract_file(task)
    has_build_signal = bool(named_file) or bool(_BUILD_RE.search(task))

    named_basename = named_file.rsplit("/", 1)[-1] if named_file else ""
    tests_primary = bool(_TESTS_PRIMARY_RE.search(task)) or named_basename.startswith(
        "test_"
    )

    # Interrogatives and turns LED by an explain marker stay explain turns;
    # a trailing marker ("run the tests and tell me what failed") does not
    # suppress the imperative run — the verdict IS the telling.
    is_interrogative = bool(_INTERROGATIVE_RE.match(task))
    leading_explain = task.lower().startswith(_EXPLAIN_MARKERS)
    run_signal = (
        not is_interrogative
        and not leading_explain
        and not _BUILD_RE.search(task)
        and not _EXISTING_RE.search(task)
        and (
            bool(_RUN_TESTS_RE.search(task))
            or (bool(_RUN_VERB_RE.search(task)) and bool(_named_test_files(task)))
        )
    )
    conversation_raw = str(turn.get("context", ""))
    has_run_block = bool(_RAN_HEADER_RE.search(conversation_raw))

    # Chained fix-execution: a fix-intent turn whose gated build already
    # shipped its write THIS turn chains into the run seam. wrote_path is
    # structural (the caller derives it from post-boundary write tool_calls,
    # never from context text — forged [wrote] lines cannot set it).
    wrote_path = str(turn.get("wrote_path", ""))
    fix_chain = bool(wrote_path) and bool(_FIX_VERB_RE.match(task))

    needs_glob = glob_file = glob_failed = ""
    needs_files: list[str] = []
    read_failed = ""
    if not is_explain and not run_signal and not fix_chain:
        needs_glob, glob_file, glob_failed = _discovery(
            task, conversation_raw, tests_primary, has_build_signal
        )
        if glob_file:
            # issue #83 discovery MATCH step: the single candidate is the
            # turn's named file — the EXISTING read seam takes over
            # (invisible -> one read request fires; visible -> the seat
            # builds against it with the right destination)
            named_file = glob_file
            named_basename = glob_file.rsplit("/", 1)[-1]
        needs_files, read_failed = _files_to_request(
            task, conversation_raw, tests_primary, has_build_signal, glob_file
        )
    wants_run = run_signal or fix_chain
    needs_run = _run_test_command(task) if wants_run and not has_run_block else ""

    target, kind, build, needs_decider = _route(
        is_explain=is_explain,
        run_signal=run_signal,
        fix_chain=fix_chain,
        has_run_block=has_run_block,
        needs_glob=needs_glob,
        glob_failed=glob_failed,
        needs_files=needs_files,
        read_failed=read_failed,
        tests_primary=tests_primary,
        has_build_signal=has_build_signal,
        kind_hint=str(turn.get("kind", "python_module")),
    )

    if target == _TESTS_SEAT:
        if named_basename.startswith("test_"):
            file = named_file
        elif named_basename:
            file = f"test_{named_basename}"
        else:
            file = "test_solution.py"
    else:
        file = named_file or "solution.py"

    # Rung-1 conversation memory: context composes into dispatch_input behind
    # the deterministic marker (generation seats resolve referents; verifier
    # seats strip back to the clean turn at the marker). Routing above reads
    # the task ALONE — a past build request must not re-trigger a build.
    dispatch_input = task or str(turn.get("dispatch_input", ""))
    conversation = str(turn.get("context", "")).strip()
    if conversation:
        dispatch_input = (
            f"Conversation so far:\n{conversation}\n\nCurrent request: {dispatch_input}"
        )
    if target == "run-verdict":
        # The verdict derives from the run block alone. The raw task is
        # multiline user text appended AFTER the context — a forged
        # column-0 [ran ...] block in it would win the latest-block scan
        # and fabricate a verdict (independent review, 2026-07-10).
        dispatch_input = conversation

    print(
        json.dumps(
            {
                "target": target,
                "kind": kind,
                "file": file,
                "task": task,
                "dispatch_input": dispatch_input,
                "build": build,
                "needs_decider": needs_decider,
                "needs_files": needs_files,
                "read_failed": read_failed,
                "needs_run": needs_run,
                "needs_glob": needs_glob,
                "glob_failed": glob_failed,
            }
        )
    )


if __name__ == "__main__":
    main()
