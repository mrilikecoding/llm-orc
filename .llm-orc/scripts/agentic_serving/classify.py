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

from _helpers import PRIOR_CODE_MARKER as _PRIOR_CODE_MARKER
from _helpers import latest_ran_block as _latest_ran_block
from chain_plan import _EXPLAIN_SEAT, _TESTS_SEAT
from chain_plan import SignalBundle as _SignalBundle
from chain_plan import advance as _advance

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


def _visible_target_body(context: str, basename: str) -> str:
    """The LATEST visible ``[wrote <path>]``/``[read <path>]`` block's body
    for ``basename`` (grounded-explain design, docs/plans/2026-07-12-
    grounded-explain-design.md): the real material a grounded explain
    dispatch quotes. Fenced block grammar — the header lives at column 0
    and the body is two-space indented (the same shape ``latest_ran_block``
    reads), so a forged header inside another block's indented body can
    never be selected; "last wins" mirrors ``_globbed_candidates``.
    """
    lines = context.splitlines()
    start = -1
    for index, line in enumerate(lines):
        match = _VISIBLE_HEADER_RE.match(line)
        if (
            match
            and not match.group(2)
            and match.group(1).rsplit("/", 1)[-1] == basename
        ):
            start = index
    if start < 0:
        return ""
    body_lines: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("  "):
            body_lines.append(line[2:])
        elif not line.strip():
            body_lines.append("")
        else:
            break
    return "\n".join(body_lines).strip()


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


# Rung 1.5, convergent-fix design (docs/plans/2026-07-12-convergent-fix-
# design.md): a closed template, never model text — the stem is charset-
# checked before it may enter the "test_<stem>.py" read request.
_TARGET_STEM_RE = re.compile(r"^[A-Za-z_]\w*$")


def _target_test_file(
    task: str, named_basename: str, context: str, tests_primary: bool
) -> str:
    """The ``test_<stem>.py`` to read once before a fix-led turn's gated
    build (rung 1.5): reuses the need-files read seam, skips (never
    refuses) when absent or already attempted — unlike a named source file,
    a missing test costs nothing but today's behavior.
    """
    if tests_primary or not _FIX_VERB_RE.match(task):
        return ""
    if not named_basename.endswith(".py") or named_basename.startswith("test_"):
        return ""
    stem = named_basename[: -len(".py")]
    if not _TARGET_STEM_RE.match(stem):
        return ""
    test_name = f"test_{stem}.py"
    visible, attempted = _visibility(context)
    if test_name in visible or test_name in attempted:
        return ""
    return test_name


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


# Rung 2, convergent-fix design: a deterministic failure-shape signal over
# the LATEST [ran ...] block, read via the shared block parser (_helpers,
# the same one run_verdict reads) so a forged block in user text can never
# feed the classifier — the selector is column-0-anchored block structure,
# never raw text (spoof-probe requirement).
# Includes the pytest-printed SUBCLASS names, not just the bases: an in-test
# ``import missing`` reports FAILED (not a collection ERROR) with
# ``ModuleNotFoundError`` — the ImportError subclass, the most common import
# failure — and a bad indent reports IndentationError/TabError (SyntaxError
# subclasses). Matching only the bases classified those localized and burned
# a re-fix round, against the fail-closed-to-structural intent.
_STRUCTURAL_ERROR_RE = re.compile(
    r"^E\s+(?:NameError|ModuleNotFoundError|ImportError"
    r"|IndentationError|TabError|SyntaxError)\b",
    re.MULTILINE,
)
_FAILSHAPE_SUMMARY_RE = re.compile(
    r"\b(?:\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed"
    r"|warnings?)|no tests ran)\b.*\bin [\d.]+s\b",
    re.IGNORECASE,
)
_FAILSHAPE_TAIL_LINES = 3
# The "small threshold" the design names without pinning a number; kept
# local and named so ladder evidence can retune it without touching the
# routing shape.
_LOCALIZED_MAX_FAILED = 3


def _failshape_count(pattern: str, summary: str) -> int:
    match = re.search(pattern, summary)
    return int(match.group(1)) if match else 0


def _failure_shape(context: str) -> str:
    """"structural" or "localized" for the LATEST ``[ran ...]`` block.

    Fails CLOSED to structural: a collection ERROR, a NameError/ImportError/
    SyntaxError in the traceback, zero tests collected, every test failing,
    more than the small threshold failing, or an unparseable summary all
    stay structural — only a summary with at least one pass and a small,
    non-error failure count is localized.
    """
    run = _latest_ran_block(context)
    if run is None:
        return "structural"
    _, variant, _, body = run
    if variant == "failed":
        return "structural"  # the run command itself never executed
    if _STRUCTURAL_ERROR_RE.search(body):
        return "structural"
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    summary = ""
    for line in reversed(lines[-_FAILSHAPE_TAIL_LINES:]):
        if _FAILSHAPE_SUMMARY_RE.search(line):
            summary = line
            break
    if not summary:
        return "structural"
    failed = _failshape_count(r"\b(\d+) failed\b", summary)
    passed = _failshape_count(r"\b(\d+) passed\b", summary)
    errors = _failshape_count(r"\b(\d+) errors?\b", summary)
    if errors > 0 or passed == 0:
        return "structural"
    if not (1 <= failed <= _LOCALIZED_MAX_FAILED):
        return "structural"
    return "localized"


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


def _discover_and_read(
    task: str,
    context: str,
    tests_primary: bool,
    has_build_signal: bool,
    named_file: str,
    named_basename: str,
) -> tuple[str, str, str, str, list[str], str]:
    """The discover -> read seam (issue #83) plus rung 1.5's target-test read
    batched into the same round: (named_file, named_basename, needs_glob,
    glob_failed, needs_files, read_failed). A glob MATCH renames the turn's
    file; the target-test read never causes a refusal on its own."""
    needs_glob, glob_file, glob_failed = _discovery(
        task, context, tests_primary, has_build_signal
    )
    if glob_file:
        # issue #83 discovery MATCH step: the single candidate is the turn's
        # named file — the EXISTING read seam takes over (invisible -> one
        # read request; visible -> the seat builds against the right dest).
        named_file = glob_file
        named_basename = glob_file.rsplit("/", 1)[-1]
    needs_files, read_failed = _files_to_request(
        task, context, tests_primary, has_build_signal, glob_file
    )
    if not read_failed:
        # rung 1.5 (convergent-fix design): batched into the same read round
        # as the target-file request above, never refusing on its own.
        target_test = _target_test_file(task, named_basename, context, tests_primary)
        if target_test and target_test not in needs_files:
            needs_files = [*needs_files, target_test]
    return (
        named_file,
        named_basename,
        needs_glob,
        glob_failed,
        needs_files,
        read_failed,
    )


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

    # Grounded explain (docs/plans/2026-07-12-grounded-explain-design.md): a
    # real named-file target gates on _visibility of the SAME wire the
    # read/run seams already trust — never free text, so a forged [wrote
    # ...] line in the user's own task prose cannot flip the gate (spoof-
    # probe requirement). Conceptual explains (no named_file) never gate.
    explain_ungrounded = False
    if is_explain and named_file:
        explain_visible, _ = _visibility(conversation_raw)
        explain_ungrounded = named_basename not in explain_visible

    # Chained fix-execution: a fix-intent turn whose gated build already
    # shipped its write THIS turn chains into the run seam. wrote_path is
    # structural (the caller derives it from post-boundary write tool_calls,
    # never from context text — forged [wrote] lines cannot set it).
    wrote_path = str(turn.get("wrote_path", ""))
    fix_chain = bool(wrote_path) and bool(_FIX_VERB_RE.match(task))

    # Rung 2 (convergent-fix design): write_count is structural (the
    # caller's post-boundary write tool_call count); run_count is read from
    # the rendered [ran ...] blocks the SAME way has_run_block is — never
    # from raw text. needs_another_run means a write this turn (the fix's
    # own, or the re-fix's) has no run of its own yet; has_refixed is the
    # one-round bound (the re-fix already shipped its write this turn).
    write_count = int(turn.get("write_count", 0) or 0)
    run_count = len(_RAN_HEADER_RE.findall(conversation_raw))
    needs_another_run = fix_chain and run_count < write_count
    has_refixed = write_count >= 2
    failure_shape = (
        _failure_shape(conversation_raw)
        if fix_chain and not needs_another_run and run_count >= 1
        else ""
    )

    needs_glob = glob_failed = ""
    needs_files: list[str] = []
    read_failed = ""
    if not is_explain and not run_signal and not fix_chain:
        (
            named_file,
            named_basename,
            needs_glob,
            glob_failed,
            needs_files,
            read_failed,
        ) = _discover_and_read(
            task,
            conversation_raw,
            tests_primary,
            has_build_signal,
            named_file,
            named_basename,
        )
    bundle = _SignalBundle(
        is_explain=is_explain,
        explain_ungrounded=explain_ungrounded,
        run_signal=run_signal,
        fix_chain=fix_chain,
        has_run_block=has_run_block,
        needs_another_run=needs_another_run,
        has_refixed=has_refixed,
        failure_shape=failure_shape,
        needs_glob=needs_glob,
        glob_failed=glob_failed,
        needs_files=needs_files,
        read_failed=read_failed,
        tests_primary=tests_primary,
        has_build_signal=has_build_signal,
        kind_hint=str(turn.get("kind", "python_module")),
    )
    decision = _advance(bundle)
    target, kind, build, needs_decider = (
        decision.target,
        decision.kind,
        decision.build,
        decision.needs_decider,
    )
    # needs_run mirrors the routing decision itself (rather than the old
    # pre-route "wants_run and not has_run_block" guess) so a SECOND
    # need-run round — the re-fix's write awaiting its own run — reissues
    # the same closed-template command instead of going silently empty.
    needs_run = _run_test_command(task) if target == "need-run" else ""
    # grounded-explain design: the target named in an explain turn with no
    # visible build or read on the wire — emit.py composes the honest
    # message from it; empty for every other routing decision.
    not_grounded = named_file if target == "not-grounded" else ""

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
    elif target == "re-fix":
        # Rung 2 (convergent-fix design): the re-fix producer needs the
        # fix pass's prior code alongside the conversation (which already
        # carries the failure report and, when rung 1.5 fired, the visible
        # test) — composed under the shared PRIOR_CODE_MARKER sentinel
        # (mirrors the existing HELD_TESTS_MARKER convention) so
        # refix_gather can split it back out deterministically.
        wrote_content = str(turn.get("wrote_content", ""))
        dispatch_input = (
            f"Conversation so far:\n{conversation}\n\n"
            f"{_PRIOR_CODE_MARKER}\n{wrote_content}\n\n"
            f"Current request: {task}"
        )
    elif target == _EXPLAIN_SEAT and named_file:
        # grounded-explain design: named_file present here always means
        # grounded (the ungrounded case routes to "not-grounded" above) —
        # point the seat AT the target's real wire content and instruct it
        # to explain that, not to recall or guess.
        block_body = _visible_target_body(conversation_raw, named_basename)
        dispatch_input = (
            f"Conversation so far:\n{conversation}\n\n"
            f"The actual current content of {named_file}:\n{block_body}\n\n"
            f"Current request: {task}\n\n"
            f"Explain {named_file}'s ACTUAL content shown above — do not "
            "guess or invent behavior it does not have."
        )

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
                "not_grounded": not_grounded,
            }
        )
    )


if __name__ == "__main__":
    main()
