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
from typing import NamedTuple

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
# glob->read grounded-explain (WS-3 slice 1): the memory-shaped clause of
# _INTERROGATIVE_RE above, isolated. "did you.../have you..." are questions
# about the assistant's own past actions, not a bare-symbol code question —
# explain-discovery must never glob the workspace for one (a category
# mismatch, not an honesty gap the mechanism is meant to close).
_MEMORY_INTERROGATIVE_RE = re.compile(r"^(?:did|have) you\b", re.IGNORECASE)
# review round 1 blocker 1: the memory-interrogative sub-shape where seeing
# (receiving) the previous message is STRUCTURALLY certain — it's on the
# wire. Every other "did/have you..." turn asks about a proposition the
# ledger cannot confirm or deny wholesale ("did you delete my files?", "did
# you run the tests?"); only THIS shape earns the affirmative "Yes -" lead
# (_memory_interrogative_message). Round 2 major 1: anchored at the END too
# — the noun phrase must terminate the question (optional "?" only), so a
# trailing qualifier ("...about the auth bug?") falls to the non-affirming
# record-only template instead of affirming a claim about a DIFFERENT
# question than the one actually on the wire.
_SAW_QUERY_RE = re.compile(
    r"^(?:did|have) you (?:see|get|receive|read) my (?:previous|last) "
    r"(?:query|message|question)\??$",
    re.IGNORECASE,
)
# review round 1 blocker 3 (widened round 2, new blocker 1): a tight
# structural floor for recap questions — deliberately NOT keyed on the loose
# _MAYBE_RECALL_RE/defer_recall extension, which fires on any incidental
# ordinal word (including inside a genuine concept question like "explain
# how first-class functions work"). Anchored at BOTH ends (optional "?"
# only, round 2 major 2): "what did we build?" resolves here, but "what did
# we build this with?" — a trailing prepositional continuation asking about
# something else entirely (the tool, not the artifact) — falls through
# instead of being answered as if the object were "everything".
_RECAP_RE = re.compile(
    r"^what (?:have|'ve) (?:we|you) (?:built|made|done|written) so far\??$"
    r"|^what did (?:we|you) build\??$"
    r"|^summarize what (?:we|you)(?:'ve| have) built\??$"
    r"|^list everything (?:we|you) (?:made|created|built|wrote)\??$"
    r"|^list everything (?:we|you)(?:'ve| have) (?:made|created|built|wrote)\??$"
    r"|^recap what (?:we|you)(?:'ve| have) done\??$"
    r"|^what files have (?:we|you) (?:created|made|written)\??$"
    r"|^give me a summary of the work\??$"
    r"|^summarize the (?:work|session)\??$"
    r"|^so what do we have now\??$"
    r"|^where did we end up\??$",
    re.IGNORECASE,
)
# Review round 2 new blocker 1 (fuzzy recap unguarded) + new blocker 3 (the
# now-deleted phantom-symbol backstop was dead code): one architectural
# resolution mirroring #82's shipped two-layer pattern (_RECALL_RE tight
# floor + _MAYBE_RECALL_RE loose decider extension). A loose recap-flavored
# turn the tight floor above did not resolve — first/second person plus an
# artifact-verb flavor — defers to the guarded decider with a "recap"
# option, instead of either answering deterministically (too risky — this
# is a genuine free-text judgment call, "is this really asking me to list
# what I've built") or falling silently through to the free explainer (the
# unguarded gap new blocker 1 demonstrated).
#
# Review round 3 blocker: the pattern above was unanchored, so "you made" /
# "we built" / etc. matched ANYWHERE in a sentence — including as a MODIFIER
# clause on a build/action object ("tests for the storage module you
# wrote", "delete the file you created"). Anchored at BOTH ends now, same
# discipline as _RECAP_RE and _SAW_QUERY_RE: an interrogative-shaped OPENER
# (what/how/can/could/would/so/where), a we/you + past-tense make-verb
# core, terminating at an optional trailing "so far" then "?"/end-of-string
# — no trailing object or action clause survives. This is deliberately
# narrower than the floor's exact literal phrasings: it exists only to
# catch a recap QUESTION the floor misses by inflection or an inserted
# adverb ("what exactly have you built so far?", "can you list everything
# you've made?"), never a recap-flavored clause attached to something else.
#
# Review round 4 MAJOR: the both-ends anchor above tests the VERB's
# POSITION only — but an English relative clause ends on its verb too ("the
# helper you made", "the module you created"). A specific-artifact question
# ("can you explain the helper you made?", "where is the test file you
# created?") satisfies the same shape up through the final verb, so it
# matched too — stealing a glob->read grounding round (need-glob on
# origin/main) for a decider deferral with no glob round at all. The
# reviewer's discriminator, now binding: a recap's OBJECT is universal
# (everything/anything/all, or the what/which...so far frame) — never a
# determiner + specific noun immediately before the final make-verb. Two
# independent, both-required guards:
#
# (a) _UNIVERSAL_RECAP_RE — a universal-object token (everything/anything/
#     all) in the object position, OR the narrower what/which...so far
#     frame (the two round-2/round-3 positives: "what exactly have you
#     built so far?" has no universal token but IS the so-far frame; "can
#     you list everything you've made?" has the token but not "so far").
# (b) _SPECIFIC_ARTIFACT_RE — a VETO: a determiner (the/that/this/my/your/
#     our) + noun phrase immediately preceding "you/we + verb" marks a
#     specific-artifact relative clause, never a recap object, even when a
#     universal token happens to appear elsewhere in the same sentence.
#
# Tightened in the conservative direction: an under-matched recap question
# falls through to routing byte-identical to origin/main (a decider-judgment
# residual already accepted by round 2); an over-match steals a grounding
# round, the failure round 4 exists to close.
_UNIVERSAL_RECAP_RE = re.compile(
    r"^(?:what|how|can|could|would|so|where)\b[^.!?]*"
    r"\b(?:everything|anything|all)\b[^.!?]*"
    r"\b(?:we|you)\b[^.!?]*"
    r"\b(?:built|made|created|wrote|written|done)\b"
    r"(?: so far)?\??$"
    r"|^(?:what|which)\b[^.!?]*"
    r"\b(?:we|you)\b[^.!?]*"
    r"\b(?:built|made|created|wrote|written|done)\b"
    r"\s+so far\??$",
    re.IGNORECASE,
)
_SPECIFIC_ARTIFACT_RE = re.compile(
    r"\b(?:the|that|this|my|your|our)\b[^.!?]*?"
    r"\b(?:you|we)\s+(?:built|made|created|wrote|written|done)\b",
    re.IGNORECASE,
)


def _is_maybe_recap(task: str) -> bool:
    """The recap decider extension's loose match (review round 4): a
    universal-object recap signal present, AND no specific-artifact
    relative clause vetoing it. See the guard regexes' own comments."""
    return bool(_UNIVERSAL_RECAP_RE.search(task)) and not bool(
        _SPECIFIC_ARTIFACT_RE.search(task)
    )
# #82 deep recall: a FIRST-anchored query about the "first thing" I/you
# built ("what did the first thing I asked you to build do?"). This is an
# INTERIM structural detector, deliberately tight to avoid the review's
# false-positive hijacks ("the first argument to build()", "the first class
# created in models.py"): it requires the exact "first thing" anchor plus a
# first-person agent (I/you) bound to a build verb. The model-decider replaces
# it next (WS-2), where a loose pre-filter gates a model that classifies recall
# robustly; write-history SELECTION keeps the answer honest either way. Only
# "first" matches (the selector answers ledger[0]); last/Nth ladder later.
_RECALL_RE = re.compile(
    r"\bfirst thing\b[^?.!]*\b(?:i|you)\b[^?.!]*"
    r"\b(?:ask(?:ed)?|build|built|wrote|created?|made|implement(?:ed)?)\b"
    r"|\b(?:i|you)\b[^?.!]*"
    r"\b(?:build|built|wrote|created?|made)\b[^?.!]*\bfirst thing\b",
    re.IGNORECASE,
)
# #82 detection layer 2: a LOOSE first-ordinal pre-filter for the fuzzy
# phrasings the tight _RECALL_RE cannot safely anchor ("the earliest thing you
# built", "what did you build originally"). It only gates DEFERRAL to the
# guarded model-decider — the model discriminates genuine recall from
# incidental ordinal use ("first-class functions", "the original design"), and
# structural selection keeps the answer honest either way. First-semantic
# terms only (adversarial-review finding 3 widened the set); last/Nth are
# named-forward (the selector answers ledger[0]).
_MAYBE_RECALL_RE = re.compile(
    r"\b(?:first|1st|earliest|initial|originally|original|beginning)\b"
    r"|\bthe start\b",
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
# optional variant group keeps a "(truncated)"/"(over-budget)" suffix out
# of the path.
_VISIBLE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)"
    r"( \((?:truncated|failed|oversize|over-budget)\))?\]$",
    re.MULTILINE,
)
_READ_ATTEMPT_RE = re.compile(
    r"^assistant: \[read ([^\]]+?)( \((failed|oversize|over-budget)\))?\]",
    re.MULTILINE,
)
_READ_CAP_KB = 96
# C1 (#145; re-denominated in tokens, BLOCKER 1 review round 1): mirrors
# the caller's _READ_TOKEN_BUDGET exactly (same name, same unit — no KB
# conversion needed this time) — classify.py runs standalone with no
# cross-boundary import (the same reason _READ_CAP_KB above duplicates its
# caller's per-file cap), so the number is repeated here for the
# over-budget refusal's wording. The corpus pins the mirror with a drift
# assert (MAJOR 2, review round 1).
_READ_TOKEN_BUDGET = 34000
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
        elif variant == "over-budget":
            # C1 (#145): the caller already refused to render this read's
            # body (it would have pushed the total held projected-token
            # count over budget) — name the budget and the files already
            # holding it, and state the remedy as its own plain sentence
            # (minor 5, review round 1) so the refusal is actionable, not
            # just honest.
            held = ", ".join(sorted(visible)) if visible else "other files"
            attempted[basename] = (
                f"the {_READ_TOKEN_BUDGET} projected-token read budget is "
                f"already held by {held}. Start a fresh session, or ask "
                "about one file at a time."
            )
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


def _ledger_recap(turn: dict) -> str:
    """The deterministic ledger recap: shipped paths plus a not-shipped
    count. Used both as the recap-floor's own answer (review round 1
    blocker 3, "what have we built so far?") and as the phantom-symbol
    backstop's fail-closed fallback (#133/#134 §4) — never more than the
    ledger holds, the same no-claims-beyond-ledger rule as
    ``_recall_message``. The count combines every non-shipped kind
    (rejected_contract/rejected_gate/refused, review round 1 blocker 2)
    rather than naming a gate for each — an aggregate tally has no
    gate-specific claim to get wrong.
    """
    ledger = turn.get("recall_ledger") or []
    valid = [entry for entry in ledger if isinstance(entry, dict)]
    shipped = [
        str(entry["path"])
        for entry in valid
        if entry.get("outcome", "shipped") == "shipped" and entry.get("path")
    ]
    not_shipped = sum(
        1 for entry in valid if entry.get("outcome", "shipped") != "shipped"
    )
    if not shipped and not not_shipped:
        return "Nothing has been built in this session yet."
    sentences = []
    if shipped:
        listed = ", ".join(f"`{path}`" for path in shipped)
        sentences.append(f"Shipped so far: {listed}.")
    if not_shipped:
        plural = "s" if not_shipped != 1 else ""
        sentences.append(f"{not_shipped} build{plural} did not ship.")
    return " ".join(sentences)


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


# glob->read grounded-explain (WS-3 slice 1, docs/plans/2026-07-14-glob-
# read-grounded-explain-design.md): general English function words only —
# NOT code terms (docs/plans/2026-07-14-grep-read-spikes/spike1-bare-
# mentions.py, the ``STOP`` set, reused verbatim). Deliberately excludes
# code-flavored words so the extractor cannot be hand-tuned to a passing
# gate; a bare-symbol turn's real code tokens (module/function names, and
# ordinary English content words like "decorator") always survive.
_EXPLAIN_STOPWORDS = frozenset(
    {
        "how",
        "what",
        "where",
        "when",
        "why",
        "which",
        "who",
        "whose",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "doing",
        "has",
        "have",
        "had",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "you",
        "i",
        "my",
        "your",
        "our",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "by",
        "at",
        "as",
        "into",
        "about",
        "over",
        "under",
        "between",
        "through",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "because",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "get",
        "gets",
        "got",
        "make",
        "makes",
        "made",
        "use",
        "uses",
        "used",
        "using",
        "there",
        "here",
        "not",
        "no",
        "yes",
        "any",
        "all",
        "some",
        "each",
    }
)
_EXPLAIN_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*")
_EXPLAIN_STEM_MIN_LEN = 3
# classify's OWN single-word explain markers (the rest of _EXPLAIN_MARKERS is
# either a multi-word phrase whose head word the general stopword set above
# already covers, or an auxiliary already in it) are routing vocabulary, not
# content — excluding them from stem candidacy mirrors _STEM_STOPWORDS's
# existing exclusion of build-verb markers from module-stem extraction. Kept
# separate from _EXPLAIN_STOPWORDS so that set stays a verbatim copy of the
# spike's STOP.
_EXPLAIN_MARKER_WORDS = frozenset({"explain", "describe", "summarize"})


def _explain_stems(task: str) -> list[str]:
    """Candidate glob stems for a bare-symbol explain turn (glob->read
    grounded-explain, WS-3 slice 1): every identifier-shaped token, len >=
    3, minus the general-English stopword set above and classify's own
    explain-marker words, first-mention order, deduped. Tokens are
    charset-checked by construction (the token regex itself), so the result
    is safe to comma-join into a glob pattern template downstream (the
    run-command discipline)."""
    seen: set[str] = set()
    stems: list[str] = []
    for match in _EXPLAIN_TOKEN_RE.finditer(task.lower()):
        token = match.group(0)
        if (
            len(token) >= _EXPLAIN_STEM_MIN_LEN
            and token not in _EXPLAIN_STOPWORDS
            and token not in _EXPLAIN_MARKER_WORDS
            and token not in seen
        ):
            seen.add(token)
            stems.append(token)
    return stems


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


# Mirrors serving_ensemble_caller._GLOB_MAX_PATHS (issue #148) — classify.py
# runs as a standalone script with no import across the src/.llm-orc
# boundary (the same reason _READ_CAP_KB above duplicates its caller's read
# cap), so the number is repeated here for the truncation refusal's
# wording. The M1 render-through tests catch drift between the two.
_GLOB_MAX_PATHS = 50


class _GlobListing(NamedTuple):
    """The turn's latest ``[globbed ...]`` block (issue #148 M3): its raw
    path lines, plus whether the caller's ``_GLOB_MAX_PATHS`` cap cut it.
    A grounding decision is only made over the COMPLETE candidate set, so
    ``truncated`` is a third state distinct from "no listing yet" (which
    ``_latest_glob_listing`` signals with ``None``, not this type) and
    "complete, no match" (``truncated=False`` with ``paths`` empty or non-
    matching). Collapsing truncated into either of those was the bug: into
    "no match" it produced a confidently false "no file matching" claim the
    wire itself contradicts (BLOCKER 1); into a bare ``[]`` with no signal
    at all it killed the visible-file fallback for a turn where a PRIOR
    read already grounds the file even though this round's listing is
    unusable (M3)."""

    paths: list[str]
    truncated: bool


def _latest_glob_listing(context: str) -> _GlobListing | None:
    """The turn's LATEST ``[globbed ...]`` block, or ``None`` when no
    listing exists yet (pass 1 fires). Column-0 anchored header scan
    (fenced block grammar) — an indented lookalike inside a read or run
    body never counts as a listing. Shared by ``_globbed_candidates`` (the
    single-stem build seam) and ``_explain_glob_candidates`` (the
    multi-stem explain-discovery seam, glob->read grounded-explain design).

    A ``(truncated)`` header (issue #148) means the caller's
    ``_GLOB_MAX_PATHS`` cap cut the listing — real stem families bust it,
    and the underlying glob order is mtime-based, so which paths survive is
    nondeterministic. The paths are still returned (a caller may want them
    for diagnostics), but ``truncated=True`` tells every caller a grounding
    decision must not be made over them — see ``_GlobListing``.
    """
    lines = context.splitlines()
    start = -1
    for index, line in enumerate(lines):
        if line.startswith("assistant: [globbed "):
            start = index
    if start < 0:
        return None
    truncated = lines[start].endswith(" (truncated)]")
    paths: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        paths.append(line.strip())
    return _GlobListing(paths=paths, truncated=truncated)


def _globbed_candidates(context: str, stem: str) -> tuple[list[str], bool] | None:
    """(candidates, truncated) from the turn's ``[globbed ...]`` block, or
    ``None`` when no listing exists yet (pass 1 fires).

    Matching is deterministic on the rendered block only (design bounds):
    basename contains the stem, ``.py``, not ``test_*``-named. A truncated
    listing (issue #148 M3) never contributes a candidate — the listing is
    not the complete candidate set, so matching against it could ground on
    a coin-flip subset; ``truncated=True`` lets ``_discovery`` fall back to
    visibility instead of silently treating this as "complete, zero
    matches".
    """
    listing = _latest_glob_listing(context)
    if listing is None:
        return None
    if listing.truncated:
        return [], True
    candidates: list[str] = []
    for path in listing.paths:
        basename = path.rsplit("/", 1)[-1]
        if (
            stem in basename.lower()
            and basename.endswith(".py")
            and not basename.startswith("test_")
        ):
            candidates.append(path)
    return candidates, False


def _explain_glob_candidates(context: str, stems: list[str]) -> list[str] | None:
    """Candidate paths from the turn's ``[globbed ...]`` block whose basename
    is NAMED-AFTER-the-symbol by the question, or ``None`` when no listing
    exists yet — the multi-stem sibling of ``_globbed_candidates`` (glob->read
    grounded-explain, WS-3 slice 1). Shares the listing scan
    (``_latest_glob_listing``) and candidate discipline (``.py``, not
    ``test_*``).

    A file matches only when the question names EVERY significant word-
    component of its basename-stem — the component set (words len >= 3, non-
    digit, split on non-alphanumerics) is a subset of the question stems.
    Substring-union was the blocker (adversarial review): a broad stopword-
    surviving word like "context" is a substring of ``project_context.py``,
    so "how does context management work?" grounded a confident answer on an
    unrelated file. The subset rule is the design's promised "named-after-the-
    symbol" bound: ``classify.py`` ({classify}) and ``accept_gate.py``
    ({accept, gate}) still match when fully named; ``project_context.py``
    ({project, context}) does not, because "project" is not named.

    A truncated listing (issue #148 M3) never contributes a candidate,
    exactly like ``_globbed_candidates`` — ``_explain_discover``'s existing
    zero-candidate fall-through (answer conceptually, never a false refusal
    claim) already covers this case correctly, so truncation collapses into
    it rather than needing its own branch here.
    """
    listing = _latest_glob_listing(context)
    if listing is None:
        return None
    if listing.truncated:
        return []
    stem_set = set(stems)
    candidates: list[str] = []
    for path in listing.paths:
        basename = path.rsplit("/", 1)[-1]
        if not basename.endswith(".py") or basename.startswith("test_"):
            continue
        name = basename[: -len(".py")].lower()
        components = {
            component
            for component in re.split(r"[^a-z0-9]+", name)
            if len(component) >= 3 and not component.isdigit()
        }
        if components and components <= stem_set:
            candidates.append(path)
    return candidates


def _explain_read_request(path: str, context: str) -> tuple[list[str], str]:
    """(paths to request, refusal reason) for the explain-discovery matched
    candidate (glob->read grounded-explain, WS-3 slice 1) — the same
    visibility/attempted discipline as ``_files_to_request``, without its
    build-only ``wants_existing`` gate: a discovery match is always wanted.
    Visible -> nothing to request; a prior failed attempt refuses instead of
    re-requesting; otherwise ONE read request.
    """
    basename = path.rsplit("/", 1)[-1]
    visible, attempted = _visibility(context)
    if basename in visible:
        return [], ""
    if basename in attempted:
        return [], f"could not read {path}: {attempted[basename]}"
    return [path], ""


def _explain_discover(
    context: str, stems: list[str]
) -> tuple[str, str, str, str, list[str], str]:
    """(named_file, named_basename, needs_glob, glob_failed, needs_files,
    read_failed) for a bare-symbol explain turn (glob->read grounded-
    explain, WS-3 slice 1): one glob round (comma-joined stems, brace-
    alternation pattern), then the shared candidate discipline (``.py``, not
    ``test_*``, one-or-refuse) once the listing returns. Mirrors the build
    discover -> read seam's two-pass shape without touching it — never
    several files read, never a guess.
    """
    candidates = _explain_glob_candidates(context, stems)
    if candidates is None:
        return "", "", ",".join(stems), "", [], ""
    stem_list = ",".join(stems)
    if len(candidates) == 1:
        candidate = candidates[0]
        needs_files, read_failed = _explain_read_request(candidate, context)
        basename = candidate.rsplit("/", 1)[-1]
        return candidate, basename, "", "", needs_files, read_failed
    if not candidates:
        # No repo file matched the stems — fall through to the conceptual
        # explainer (general-knowledge answer, today's behavior) rather than
        # refusing: the slice only ADDS grounding, it never removes the
        # general-answer capability. Loop-safe — the [globbed] listing
        # already exists in context, so _explain_glob_candidates returns []
        # (not None), and these all-empty signals never re-set needs_glob.
        return "", "", "", "", [], ""
    listed = ", ".join(candidates)
    refusal = (
        f"multiple files match '{stem_list}' in the workspace listing: {listed}"
        " — please name one"
    )
    return "", "", "", refusal, [], ""


def _visible_stem_paths(context: str, stem: str) -> list[str]:
    """DISTINCT visible (already read/written) full paths whose basename's
    stem exactly matches ``stem``, ``.py``, not ``test_*``-named, sorted
    for determinism (the same candidate discipline as the globbed MATCH
    step — review blocker 2026-07-10: any-extension + set-order pick
    shipped a test_storage.json deliverable). Scans full paths directly off
    ``_VISIBLE_HEADER_RE`` rather than reusing ``_visibility``'s basename-
    collapsed set (issue #148 round 2 MAJOR A): two DISTINCT paths can
    share one basename (``/srv/app/telemetry.py`` and
    ``/vendor/telemetry.py`` both look like "telemetry.py" to a
    basename-only check) — a basename-only scan collapses them into
    "exactly one match" and grounds ambiguously, where the complete-listing
    MATCH step would have refused, naming both. Shared by ``_discovery``'s
    "no listing yet" and "truncated listing" (issue #148 M3) fallback
    branches — a truncated glob listing disables LISTING-based grounding
    only, so a file already visible from a prior read/write still grounds
    either way.
    """
    paths = {
        path for path, variant in _VISIBLE_HEADER_RE.findall(context) if not variant
    }
    return sorted(
        path
        for path in paths
        if path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() == stem
        and path.endswith(".py")
        and not path.rsplit("/", 1)[-1].startswith("test_")
    )


def _visible_stem_result(context: str, stem: str) -> tuple[str, str, str] | None:
    """The visible-file fallback shared by ``_discovery``'s "no listing yet"
    and "truncated listing" (issue #148 M3) branches: exactly one DISTINCT
    visible path names the file (nothing left to discover), several refuse
    rather than guess — counted over paths, not basenames (round 2 MAJOR
    A) — and no match at all returns ``None`` so the caller supplies its
    own next step (issue a glob request, or the truncation-specific
    refusal). The multi-match refusal mirrors the glob-listing MATCH
    step's own wording, naming the distinct paths."""
    paths = _visible_stem_paths(context, stem)
    if len(paths) == 1:
        # the stem IS a visible file — nothing to discover, but it is
        # still the turn's named file (live finding 2026-07-10: without
        # this a retried module turn shipped to test_solution.py)
        return "", paths[0].rsplit("/", 1)[-1], ""
    if len(paths) > 1:
        listed = ", ".join(paths)
        return (
            "",
            "",
            f"multiple visible files match '{stem}': {listed} — please name one",
        )
    return None


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

    A truncated listing (issue #148 M3) is a third state: LISTING-based
    matching is disabled, but the visible-file fallback still runs (a prior
    read grounds the file regardless of this round's glob), and the
    honest-refusal wording says so truthfully (BLOCKER 1) — never the "no
    file matching" claim, which the truncated block itself may contradict.
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
    result = _globbed_candidates(context, stem)
    if result is None:
        visible_result = _visible_stem_result(context, stem)
        return visible_result if visible_result is not None else (stem, "", "")
    candidates, truncated = result
    if truncated:
        visible_result = _visible_stem_result(context, stem)
        if visible_result is not None:
            return visible_result
        return (
            "",
            "",
            (
                f"the workspace listing for '{stem}' was cut at "
                f"{_GLOB_MAX_PATHS} paths, so I can't tell which files "
                "match — please name the file"
            ),
        )
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


def _outcome_clause(kind: str, reason: str) -> str:
    """The honest, kind-specific outcome phrase for a non-shipped ledger
    entry (review round 1 blocker 2 / major 2) — never claims a SPECIFIC
    gate the record doesn't support: a seat-contract miss is never called
    an accept-gate rejection, and a read/glob/build-invalid refusal is
    never attributed to either gate. "" for an unrecognized kind — the
    caller fails CLOSED to disclosing uncertainty instead of guessing."""
    if kind == _REJECTED_CONTRACT:
        return "did not clear the seat contract"
    if kind == _REJECTED_GATE:
        return "was rejected by the accept gate"
    if kind == _REFUSED:
        return f"was refused: {reason}" if reason else "was refused"
    return ""


# Ledger outcome kinds (review round 1 blocker 2), mirrored from the caller
# (serving_ensemble_caller._SHIPPED/_REJECTED_CONTRACT/_REJECTED_GATE/
# _REFUSED) — classify never mints a ledger entry itself, only reads the
# kind the caller already recorded, so these are read-side constants only.
_SHIPPED = "shipped"
_REJECTED_CONTRACT = "rejected_contract"
_REJECTED_GATE = "rejected_gate"
_REFUSED = "refused"


class _Disclosure(NamedTuple):
    """The first-ask disclosure clause's ingredients (review round 1
    blocker 2c): the verbatim first ask, its outcome KIND, and (for
    "refused") the wire reason. Empty fields mean nothing to disclose."""

    ask: str = ""
    kind: str = ""
    reason: str = ""


_NO_DISCLOSURE = _Disclosure()


def _recall_select(turn: dict, context: str) -> tuple[str, str, str, _Disclosure]:
    """(case, ask, path, disclosure) — deterministic ordinal SELECTION over
    the caller's ask-outcome ledger (#82 deep recall; disclosure extension
    #133, docs/plans/2026-07-17-recap-grounding-design.md; review round 1
    blocker 2c anchors disclosure on the EARLIEST LEDGERED entry), independent
    of detection.

    ``case``: "grounded" (the first SHIPPED build is visible -> ride the
    grounded explainer via a named_file injection), "built_deep" (shipped but
    windowed out of the context -> name it, defer the body to a read), or
    "none" (nothing shipped this session). Selection over "shipped" entries
    stays #82's exact behavior — an entry with no ``outcome`` key (the
    pre-#133/#134 ledger shape) is treated as shipped, so the existing
    ordinal-recall test suite is unaffected.

    ``disclosure`` carries the ledger's FIRST entry's ask/kind/reason when
    that entry is not itself shipped — i.e. when the very first thing the
    user asked for is not the same as the first thing that shipped. Every
    build-reachable emit terminal mints a ledger entry now (blocker 2a/2b:
    "Refused:" is recognized alongside the seat-contract/accept-gate
    prefixes), so the earliest entry IS the first build-outcome ask, no
    prose inference required. Empty when nothing was ever rejected/refused,
    or when nothing ever shipped (the "none" case already says so honestly;
    there is no shipped fact to disclose alongside).
    """
    ledger = turn.get("recall_ledger") or []
    valid = [entry for entry in ledger if isinstance(entry, dict)]
    shipped = [entry for entry in valid if entry.get("outcome", "shipped") == _SHIPPED]
    first = shipped[0] if shipped else {}
    path = str(first.get("path", ""))
    ask = str(first.get("ask", ""))
    if not path:
        return "none", "", "", _NO_DISCLOSURE
    disclosure = _NO_DISCLOSURE
    first_entry = valid[0]
    first_outcome = str(first_entry.get("outcome", "shipped"))
    if first_outcome != _SHIPPED:
        disclosure = _Disclosure(
            ask=str(first_entry.get("ask", "")),
            kind=first_outcome,
            reason=str(first_entry.get("reason", "")),
        )
    visible, _ = _visibility(context)
    if path.rsplit("/", 1)[-1] in visible:
        return "grounded", ask, path, disclosure
    return "built_deep", ask, path, disclosure


def _recall_message(
    case: str, ask: str, path: str, disclosure: _Disclosure = _NO_DISCLOSURE
) -> str:
    """The honest, deterministic recall answer. Framed as what SHIPPED
    (structural), never as an unverifiable "asked" — except the disclosure
    clause (#133), which states the first ask's recorded outcome as a
    ledger FACT, kind-specific (review round 1 blocker 2 / major 2), never
    a guess at what it "was". An unrecognized disclosure kind fails CLOSED
    to disclosing uncertainty rather than claiming a gate."""
    clause_text = ""
    if disclosure.ask:
        clause = _outcome_clause(disclosure.kind, disclosure.reason)
        if clause:
            clause_text = (
                f'The first thing you asked me to build ("{disclosure.ask}") '
                f"{clause} — nothing shipped for it. "
            )
        else:
            clause_text = (
                "I can't confirm the outcome of your first ask from the "
                "record. "
            )
    if case == "none":
        return "Nothing has been built in this session yet."
    if case == "built_deep":
        return (
            f"{clause_text}The first thing that actually shipped was `{path}` "
            f"(from your request '{ask}'). Ask me to read `{path}` and I'll "
            "explain what it does."
        )
    if case == "grounded" and clause_text:
        return (
            f"{clause_text}The first thing that actually shipped was `{path}` "
            f"(from your request '{ask}'). Ask me to explain `{path}` and "
            "I'll walk through what it does."
        )
    return ""


def _recall_route(
    task: str,
    turn: dict,
    context: str,
    is_explain: bool,
    named_file: str,
    named_basename: str,
) -> tuple[str, str, bool, str, bool]:
    """Resolve an ordinal-recall turn to routing effects: (named_file,
    named_basename, is_recall_answer, recall_answer, defer_recall).

    Two detection layers over one structural selector (#82 design doc):
    - STRUCTURAL floor (``_RECALL_RE``): a tight-anchored recall resolves here
      with NO decider — grounded rewrites named_file for inline grounded-
      explain; a non-grounded case sets the honest recall_answer + the routing
      flag.
    - MODEL extension (``_MAYBE_RECALL_RE``): a loose first-ordinal explain the
      tight regex missed, with no named file, defers to the guarded decider
      (``defer_recall``). The honest answer is pre-computed for resolve to
      apply on a recall vote; grounded collapses into the "ask me to read"
      message on this path so resolve stays a thin merge.
    A non-recall turn returns no effects.

    #133 disclosure: when the FIRST ask in the ledger was rejected/refused,
    the grounded case ALSO answers deterministically (never rides the
    explainer seat) — doctrine 9, no model judgment on an honesty-critical
    path. The plain (no disclosure owed) grounded case keeps its routing
    byte-for-byte.
    """
    if not is_explain:
        return named_file, named_basename, False, "", False
    if _RECALL_RE.search(task):
        case, ask, path, disclosure = _recall_select(turn, context)
        if case == "grounded" and not disclosure.ask:
            return path, path.rsplit("/", 1)[-1], False, "", False
        message = _recall_message(case, ask, path, disclosure)
        return named_file, named_basename, True, message, False
    if not named_file and _MAYBE_RECALL_RE.search(task):
        case, ask, path, disclosure = _recall_select(turn, context)
        if case == "built_deep" or (case == "grounded" and disclosure.ask):
            # Finding 2 fail-closed: the first build is windowed out of context
            # (or a disclosure is owed), so an ungrounded explainer could only
            # GUESS or omit a fact the ledger already knows. Answer
            # structurally (no decider) — deterministic honesty on the
            # deep-history case #82 exists for. Over-fire on a concept
            # question is irrelevant-but-true, never a lie.
            message = _recall_message(case, ask, path, disclosure)
            return named_file, named_basename, True, message, False
        # grounded (first build visible, nothing to disclose) or none: the
        # explainer can answer honestly (from the visible wire, or a concept
        # explanation), so let the decider judge recall-vs-concept — a
        # mis-vote here is not a deep-history guess. grounded collapses into
        # the built_deep "ask me to read" message.
        msg_case = "built_deep" if case == "grounded" else case
        message = _recall_message(msg_case, ask, path, disclosure)
        return named_file, named_basename, False, message, True
    return named_file, named_basename, False, "", False


def _memory_interrogative_message(previous_ask: dict, affirm: bool) -> str:
    """The deterministic memory-interrogative answer (#134, docs/plans/
    2026-07-17-recap-grounding-design.md; review round 1 blocker 1 splits
    the template): quote the previous query verbatim from the wire
    (structurally present — the caller supplies it), state its outcome from
    the ask-outcome ledger — shipped (with path), a kind-specific
    rejected/refused clause, or no outcome claim at all when the ask itself
    carried no build outcome (a question, a read).

    ``affirm`` (from ``_SAW_QUERY_RE``) gates the ONLY honest "Yes -" lead:
    seeing/receiving the previous message is structurally certain. Every
    other memory interrogative ("did you run the tests?", "did you delete
    my files?") asks about a proposition the ledger cannot confirm or deny
    wholesale — it only reports the record, never leading with Yes/No.
    Never enumerates beyond what ``previous_ask`` holds.
    """
    ask = str(previous_ask.get("ask", ""))
    if not ask:
        return "I don't have a previous message in this session to confirm."
    outcome = str(previous_ask.get("outcome", ""))
    path = str(previous_ask.get("path", ""))
    reason = str(previous_ask.get("reason", ""))
    # review round 2 minor 2: "your" stays lowercase regardless of the
    # affirmative lead — "Yes — Your..." reads as two capitalized sentence
    # starts stitched together.
    lead = "Yes — your" if affirm else "Your"
    report = f'{lead} previous message was: "{ask}".'
    if outcome == _SHIPPED:
        return f"{report} It shipped as `{path}`."
    clause = _outcome_clause(outcome, reason)
    if clause:
        return f"{report} That build {clause}."
    return report


def _memory_interrogative_route(
    task: str, turn: dict, is_explain: bool, named_file: str
) -> tuple[bool, str]:
    """(is_recall_answer, recall_answer) for a bare memory-interrogative turn
    ("did you.../have you...", #134) — answered on the recall-answer emit
    path, the free explain seat bypassed entirely.

    Never fires on a named-file turn ("did you see storage.py?") — that
    stays on grounded/not-grounded explain, unchanged (what deliberately
    does not change). The caller checks this only after the #82 ordinal-
    recall detectors have had their turn, so a phrasing that matches both
    ("did you build the first thing I asked?") resolves via the more
    specific, already-structural #82 mechanism.
    """
    if not is_explain or named_file or not _MEMORY_INTERROGATIVE_RE.match(task):
        return False, ""
    previous_ask = turn.get("previous_ask")
    if not isinstance(previous_ask, dict):
        previous_ask = {}
    affirm = bool(_SAW_QUERY_RE.match(task))
    return True, _memory_interrogative_message(previous_ask, affirm)


def _recap_route(task: str, turn: dict, named_file: str) -> tuple[bool, str, bool]:
    """(is_recall_answer, recall_answer, defer_recap) for a recap-shaped
    turn ("what have we/you built so far?", "list everything you made",
    review round 1 blocker 3, widened round 2, anchored round 3): mirrors
    #82's two-layer structural-floor + decider-extension pattern.

    - STRUCTURAL floor (``_RECAP_RE``): a tight, anchored recap phrasing
      resolves here with NO decider — the deterministic ledger recap on the
      SAME recall-answer emit path as ordinal recall and memory
      interrogatives, never a model seat.
    - DECIDER extension (``_is_maybe_recap``): a loose recap-flavored turn
      the tight floor did not resolve defers to the guarded decider (a
      "recap" vote alongside #82's "recall"). CRITICAL WIRING (round 2 new
      blocker 3): the ledger recap is PRECOMPUTED here into
      ``recall_answer``, exactly like ``defer_recall`` precomputes its
      message — a decider recap vote always has a structural answer to
      resolve to (``resolve.py``), never falling through to the explainer
      with nothing to say. ``_ledger_recap`` always returns non-empty text,
      so this is impossible by construction, not by convention.

    Review round 3 blocker: this used to also require the caller's global
    ``is_explain`` flag, which forced classify to WIDEN is_explain so a
    bare "list everything you made" (no explain marker, no interrogative
    lead) could even reach this branch — and that widening was itself the
    blocker, since the (then-unanchored) loose match matched build/action
    turns anywhere in the sentence and dragged them onto the explain path
    too (36 routing changes over a 65-input diff against origin/main, only
    7 intended). ``_RECAP_RE``/``_is_maybe_recap`` are now anchored and
    themselves sufficient recap-question signals — the same self-
    sufficiency ``_MEMORY_INTERROGATIVE_RE`` already had (its pattern is a
    strict subset of the interrogative check that feeds ``is_explain``, so
    it never needed is_explain widened for its own sake either).
    ``is_explain`` reverts to origin/main's exact two-clause form; this
    route no longer takes or needs it.

    Review round 4 MAJOR: ``_is_maybe_recap`` additionally guards against a
    specific-artifact relative clause ("the helper you made") that ends on
    its verb exactly like a recap question does — see its own docstring.

    Never fires on a named-file turn.
    """
    if named_file:
        return False, "", False
    if _RECAP_RE.search(task):
        return True, _ledger_recap(turn), False
    if _is_maybe_recap(task):
        return False, _ledger_recap(turn), True
    return False, "", False


def _valid_recall_answer(recall_answer: str, target: str, needs_decider: bool) -> str:
    """recall_answer survives ONLY when the turn's outcome is a recall answer:
    the structural recall-answer step, or a deferred turn heading to the decider
    (needs_decider, where resolve applies or drops it on the vote). A higher-
    priority chain (run/fix) can preempt a recall-phrased turn (e.g. "run the
    tests and tell me the first thing you made" -> run-verdict); clearing the
    stale message keeps emit — which fires on recall_answer PRESENCE — from
    shadowing the real seat/verdict output (adversarial review finding 1)."""
    if target == "recall-answer" or needs_decider:
        return recall_answer
    return ""


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
    explain_stems: list[str] | None = None,
) -> tuple[str, str, str, str, list[str], str]:
    """The discover -> read seam (issue #83) plus rung 1.5's target-test read
    batched into the same round: (named_file, named_basename, needs_glob,
    glob_failed, needs_files, read_failed). A glob MATCH renames the turn's
    file; the target-test read never causes a refusal on its own.

    ``explain_stems``, when non-empty, takes the explain-discovery branch
    (glob->read grounded-explain, WS-3 slice 1) instead — the caller only
    passes stems for an is_explain turn with no named file, so this and the
    build discover -> read seam below never cross (isolation)."""
    if explain_stems:
        return _explain_discover(context, explain_stems)
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


def _strip_truncated_glob_block(conversation: str) -> str:
    """Remove EVERY truncated ``[globbed ...]`` block from the conversation
    text before dispatch_input composes it into a seat prompt (issue #148
    BLOCKER 2; round 2 review minor 1: every block, not only the latest).

    ``_discovery``/``_explain_discover`` already refuse to GROUND a routing
    decision on a truncated listing (via ``_latest_glob_listing``), but on
    the bare-symbol explain-discovery fall-through (zero candidates, target
    ``explainer``, no named file) the raw block was still riding along in
    dispatch_input's generic conversation composition. ``explainer.yaml``
    carries no grounding instruction, so handing it a truncated listing —
    often containing the fully-named file — is exactly the coin-flip ground
    the routing refusal exists to prevent; the invariant has to hold at the
    seat's actual prompt, not just at the router. A stale EARLIER truncated
    block (round 2 review minor 1) is the same hazard even when the LATEST
    block happens to be complete — routing only ever reads the latest
    block, but dispatch_input still carries the whole conversation, so
    every truncated block gets stripped, not just the one routing looked
    at.

    Mirrors ``_latest_glob_listing``'s own scan exactly (column-0
    ``"assistant: [globbed "`` header, two-space-indented body) so the
    strip is deterministic and header-anchored, never a text-search
    heuristic. A complete (non-truncated) listing — a genuine no-match — is
    left untouched; only the truncated marker triggers a strip.
    """
    lines = conversation.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("assistant: [globbed ") or not lines[
            index
        ].endswith(" (truncated)]"):
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].startswith("  "):
            end += 1
        del lines[index:end]
    return "\n".join(lines)


def main() -> None:
    turn = _turn(sys.stdin.read().strip())
    task = str(turn.get("task", "")).strip()
    # Review round 3 blocker: this used to be widened with a third clause for
    # recap phrasings, guarded only by _BUILD_RE — which misses past-tense
    # build verbs by construction, so an unanchored _MAYBE_RECAP_RE dragged
    # 29 non-recap turns (tests-seat builds, bare-symbol explain discovery,
    # eleven action turns) onto the explain path. is_explain now reverts to
    # origin/main's exact behavior; _recap_route no longer needs it widened
    # (see its docstring) because the recap regexes are anchored and
    # self-sufficient.
    is_explain = any(marker in task.lower() for marker in _EXPLAIN_MARKERS) or bool(
        _INTERROGATIVE_RE.match(task)
    )
    named_file = turn.get("file") or _extract_file(task)
    has_build_signal = bool(named_file) or bool(_BUILD_RE.search(task))

    named_basename = named_file.rsplit("/", 1)[-1] if named_file else ""
    tests_primary = bool(_TESTS_PRIMARY_RE.search(task)) or named_basename.startswith(
        "test_"
    )
    # Review round 2 new blocker 2: whether THIS turn carried a build ask at
    # all — threaded to emit so a read/glob refusal can mint a build-scoped
    # ledger entry only when it actually answers one. has_build_signal alone
    # under-counts: "tests for the storage module" is tests_primary (a build
    # ask) but names no file and contains no _BUILD_RE verb.
    #
    # Review round 3 minor: has_build_signal alone OVER-counts too — a named
    # file on an explain turn ("explain what foo.py does") or an incidental
    # _BUILD_RE token used as an ordinary noun ("explain the code you
    # wrote", "code" matching \bcode\b) sets has_build_signal without the
    # turn ever asking for a build. Narrow to False on an explain turn
    # unless it's ALSO led by a fix verb (a fix-verb-led turn is never
    # is_explain by construction — the two vocabularies don't overlap — so
    # this only documents the invariant rather than changing behavior).
    is_build_ask = (has_build_signal or tests_primary) and (
        not is_explain or bool(_FIX_VERB_RE.match(task))
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

    # #82 deep recall: an ordinal-recall query resolves deterministically over
    # the caller's chronological ledger. A non-grounded case (nothing shipped,
    # etc.) routes to the honest recall-answer instead of the guessing seat.
    named_file, named_basename, is_recall_answer, recall_answer, defer_recall = (
        _recall_route(
            task, turn, conversation_raw, is_explain, named_file, named_basename
        )
    )

    # #134 memory interrogative: a bare "did you.../have you..." turn the
    # ordinal-recall detectors above did not already resolve is answered
    # deterministically from the immediately preceding ask's outcome, on the
    # SAME recall-answer emit path — never the free explain seat.
    if not is_recall_answer and not defer_recall:
        is_memory_answer, memory_answer = _memory_interrogative_route(
            task, turn, is_explain, named_file
        )
        if is_memory_answer:
            is_recall_answer, recall_answer = True, memory_answer

    # Review round 1 blocker 3 (widened round 2, anchored round 3): a
    # recap-shaped turn ("what have we built so far?", "list everything you
    # made") is answered from the deterministic ledger recap, on its own
    # tight structural floor (never the loose defer_recall extension). A
    # fuzzy recap phrasing the floor doesn't resolve defers to the guarded
    # decider (defer_recap), with the answer precomputed so the decider's
    # recap vote always has something to route to. Gates on the anchored
    # regexes alone, not the global is_explain (round 3 blocker — see
    # _recap_route's docstring).
    defer_recap = False
    if not is_recall_answer and not defer_recall:
        is_recap_answer, recap_answer, defer_recap = _recap_route(
            task, turn, named_file
        )
        if is_recap_answer:
            is_recall_answer, recall_answer = True, recap_answer
        elif defer_recap:
            recall_answer = recap_answer

    # Grounded explain (docs/plans/2026-07-12-grounded-explain-design.md): a
    # real named-file target gates on _visibility of the SAME wire the
    # read/run seams already trust — never free text, so a forged [wrote
    # ...] line in the user's own task prose cannot flip the gate (spoof-
    # probe requirement). Conceptual explains (no named_file) never gate.
    explain_ungrounded = False
    explain_attempted: dict[str, str] = {}
    if is_explain and named_file:
        explain_visible, explain_attempted = _visibility(conversation_raw)
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

    # glob->read grounded-explain (WS-3 slice 1): a bare-symbol explain turn
    # (is_explain, no named file) resolves candidate stems for one glob->read
    # discovery round — the mechanism this intercepts is exactly today's
    # conceptual-explain speculation, so it is gated OFF the higher-priority
    # signals that already resolve the turn honestly or structurally
    # (run_signal, fix_chain, is_recall_answer, defer_recall — each already
    # produces its own correct outcome; letting explain-discovery ALSO set
    # needs_glob there would leak a stray glob request ahead of that outcome
    # at emit's seam-priority check) and off memory-shaped "did/have you"
    # questions, a distinct category classify already recognizes.
    explain_stems = (
        _explain_stems(task)
        if (
            is_explain
            and not named_file
            and not run_signal
            and not fix_chain
            and not is_recall_answer
            and not defer_recall
            and not defer_recap
            and not _MEMORY_INTERROGATIVE_RE.match(task)
        )
        else []
    )

    needs_glob = glob_failed = ""
    needs_files: list[str] = []
    read_failed = ""
    if (not is_explain and not run_signal and not fix_chain) or explain_stems:
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
            explain_stems=explain_stems,
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
        is_recall_answer=is_recall_answer,
        defer_recall=defer_recall,
        defer_recap=defer_recap,
    )
    decision = _advance(bundle)
    target, kind, build, needs_decider = (
        decision.target,
        decision.kind,
        decision.build,
        decision.needs_decider,
    )
    chain, step_index = decision.chain, decision.step_index
    recall_answer = _valid_recall_answer(recall_answer, target, needs_decider)
    # needs_run mirrors the routing decision itself (rather than the old
    # pre-route "wants_run and not has_run_block" guess) so a SECOND
    # need-run round — the re-fix's write awaiting its own run — reissues
    # the same closed-template command instead of going silently empty.
    needs_run = _run_test_command(task) if target == "need-run" else ""
    # grounded-explain design: the target named in an explain turn with no
    # visible build or read on the wire — emit.py composes the honest
    # message from it; empty for every other routing decision.
    not_grounded = named_file if target == "not-grounded" else ""
    # minor 3 (review round 1): when a PRIOR turn already attempted (and
    # failed) to read this exact target — recorded in explain_attempted
    # from the SAME _visibility scan the gate above used — emit composes a
    # message that states the recorded reason instead of suggesting the
    # exact action that just failed ("ask me to read it").
    not_grounded_reason = (
        explain_attempted.get(named_basename, "") if target == "not-grounded" else ""
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
    if target == _EXPLAIN_SEAT and not named_file:
        # issue #148 BLOCKER 2: the bare-symbol explain-discovery fall-
        # through (candidates == [], often because the listing was
        # truncated) dispatches to explainer.yaml, a cheap seat with no
        # grounding instruction — strip a truncated glob block out of the
        # conversation before it becomes that seat's prompt. A complete
        # listing (a genuine no-match) is left exactly as before. The
        # grounded case (named_file set) is untouched — it already carries
        # its own explicit "do not guess" instruction, and its own glob
        # round can never itself be truncated (a truncated listing never
        # yields a single named-after candidate, see
        # _explain_glob_candidates).
        conversation = _strip_truncated_glob_block(conversation)
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
                "not_grounded_reason": not_grounded_reason,
                "recall_answer": recall_answer,
                "is_build_ask": is_build_ask,
                "chain": chain,
                "step_index": step_index,
            }
        )
    )


if __name__ == "__main__":
    main()
