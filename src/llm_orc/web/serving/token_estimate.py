"""Projected-token estimation for read-body budgeting (#145 BLOCKER 1).

STATUS (review round 2, 2026-08-13): this is v2 of the estimator —
validated against a real tokenizer (qwen3:8b) and NOT YET WIRED into the
live read-accumulator budget. ``serving_ensemble_caller.py`` still runs its
own v1 estimator (``_projected_tokens``) for ``_budget_read_blocks``. Wiring
v2 in is BLOCKED on a design fork the lead needs to resolve — see the
"Sanity constraint" section below and the round-2 note in
``docs/plans/2026-08-13-repo-scale-reads-design.md``. Do not delete v1 or
swap this in without that resolution; doing either silently would be
exactly the "silent scope change" the review explicitly ruled out.

v1's failure (round 2 review): v1 counted a WHOLE ASCII word-run as one
token regardless of length. Real BPE tokenizers split long, high-entropy
runs (base64, hex digests, PEM bodies, long random identifiers) into many
subword tokens — a 94KB PEM cert passed v1's budget check at "21%
utilization" while a real tokenizer's prompt_eval_count showed it had
actually silently overflowed the window (the discard signature:
prompt_eval_count == roughly half the window on every over-window prompt,
HTTP 200, no error). v1 also didn't count newlines or multi-space
indentation runs, under-counting YAML-shaped content.

v2's terms (round 2 review, adjudicated):
    (a) ASCII word-runs (``[A-Za-z0-9_]+``) of length <= 30  -> 1 unit each
    (b) ASCII word-runs of length > 30                       -> ceil(len/1.3)
        units each (high-entropy runs measured near 1.34 chars/token)
    (c) non-space punctuation characters (ASCII or not)      -> 1 unit each
    (d) non-ASCII word characters (CJK, etc.)                -> 1 unit each
    (e) newline characters                                   -> 1 unit each
    (f) runs of >= 2 consecutive space characters            -> 1 unit each
        (indentation tokens — what v1 missed on YAML)
    total * SAFETY_FACTOR

(c) and (d) collapse to the same "1 unit per remaining non-whitespace
character" rule in the implementation below, mirroring v1's own
simplification — the distinction only matters for exposition.

SAFETY_FACTOR derivation (measured ground truth, not asserted): the
reviewer's ten round-2 fixture classes plus five real repo files were
measured against qwen3:8b's real tokenizer via ``/api/chat``
(``think: false``, ``num_predict: 1`` — bounds generation to keep runs
fast; ``prompt_eval_count`` reflects the full prompt regardless of
generation length) minus the verified chat-template overhead (17 tokens
for a bare single-character user message — confirmed independently,
matches the reviewer's own cited figure). See
``tests/unit/web/serving/test_token_estimate_ground_truth.py`` for the
full frozen table, dated 2026-08-13, with the exact generation command.

The worst (smallest) v2-before-factor ratio against real tokens was the
PEM certificate fixture at 0.6630 (v2 UNDER-counts real by ~34% before any
safety factor — the base64 alphabet's ``+``/``/``/``=`` characters chop an
otherwise-long high-entropy run into pieces <= 30 chars each, so the
length-scaling rule (b) rarely engages; each chopped piece then only
costs 1 unit under rule (a), which is NOT enough for genuinely
high-entropy content). Solving for the smallest factor F such that
``v2(text) * F >= real(text) * 1.05`` (5% margin) for every measured
fixture gives F = 1.5837, which rounds up to 1.59.

SANITY CONSTRAINT (review round 2): classify.py must still project under
``_READ_TOKEN_BUDGET`` (34,000) at the shipped factor. It does NOT at
1.59: classify.py's v2-before-factor count is 21,598, so
``ceil(21598 * 1.59) == 34,341 > 34,000`` — a narrow miss (the max factor
that keeps classify.py under budget is 1.5742; PEM's 5% margin needs at
least 1.5837 — a gap of about 1%). This is the exact conflict the review
anticipated and explicitly said must go back to the lead rather than be
resolved silently (e.g. by quietly raising the budget or quietly
accepting less than 5% margin for the worst fixture). ``SAFETY_FACTOR``
below is set to the mathematically-derived value (1.59); it is deliberately
NOT yet imported anywhere live.
"""

from __future__ import annotations

import math
import re

_ASCII_WORD_RUN_RE = re.compile(r"[A-Za-z0-9_]+")
# Alternation order matters: at each position the engine tries the ASCII
# word-run branch first, so any character that IS an ASCII word character
# is always consumed as part of its run (never falls through to ``\S``).
# ``\S`` therefore only ever matches non-ASCII-word-run material: newline
# is peeled off first (rule e), a 2+ space run next (rule f), and any
# other single non-whitespace character last (rules c/d combined — see
# module docstring).
_TOKEN_UNIT_RE = re.compile(r"[A-Za-z0-9_]+" r"|\n" r"| {2,}" r"|\S")

# Derived from measured ground truth — see module docstring and
# tests/unit/web/serving/test_token_estimate_ground_truth.py. NOT YET
# WIRED into the live budget (see STATUS above).
SAFETY_FACTOR = 1.59

# The length above which an ASCII word-run is treated as high-entropy and
# scaled by measured chars-per-token instead of counted as one token.
_LONG_RUN_THRESHOLD = 30
# Measured chars/token for high-entropy runs (base64, hex digests, long
# random identifiers) — see module docstring.
_HIGH_ENTROPY_CHARS_PER_TOKEN = 1.3


def projected_tokens_v2(text: str, *, safety_factor: float = SAFETY_FACTOR) -> int:
    """A conservative (over-counting, once ``safety_factor`` is applied)
    projected token count for ``text`` — see the module docstring for the
    term-by-term derivation and the measured ground truth backing
    ``safety_factor``'s default.
    """
    total = 0
    for match in _TOKEN_UNIT_RE.finditer(text):
        unit = match.group(0)
        if _ASCII_WORD_RUN_RE.fullmatch(unit):
            if len(unit) <= _LONG_RUN_THRESHOLD:
                total += 1
            else:
                total += math.ceil(len(unit) / _HIGH_ENTROPY_CHARS_PER_TOKEN)
        else:
            total += 1
    return math.ceil(total * safety_factor)
