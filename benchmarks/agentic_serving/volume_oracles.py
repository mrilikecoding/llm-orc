"""Hidden correctness oracles for the #138 volume ladder.

One probe per fixture module (design:
docs/plans/2026-08-15-138-volume-instrument-design.md). They answer the
question the seeded tests cannot: the seeded tests are VISIBLE to the
arm, so a fix that merely satisfies the assertion it can read would
otherwise score as correct. Each probe re-asks the same contract with
NONCE-DERIVED inputs, so a hardcoded expectation fails.

Everything from ``oracles.py``'s doctrine applies and is not restated
here: positive proof via the ``PROBE-OK-<nonce>`` token (exit 0 is what
a module that killed the interpreter at import also produces), probes
run against a throwaway COPY (they CALL arm-authored code), and an
instrument failure escapes rather than becoming a ``passed=False``
verdict. The sandbox contract has one home: ``oracles.run_probe``.

TOLERANCE IS BOUNDED. Each probe exercises only behavior the fixture's
own seeded tests pin: sums and the empty-entries raise, integer-valued
decimal strings, a zero window and ordinary tails, minute-to-hour
conversion, lowercase hyphenation. Deliberately NOT probed, because the
seeded tests pin none of them and a false reject scores a CORRECT fix as
shipped-broken — the direction that FABRICATES the hypothesis under
measurement: negative or oversized windows (``n > len(items)``), zero
minutes, fractional hour answers, punctuation or digits in titles,
exponent notation, and the returned numeric TYPE (``float`` vs ``int``
vs ``Decimal``).

Two probe-design rules learned from review round 1, both of which had
made a verdict depend on the draw rather than the code:

1. Nonce-derived inputs must land where every reasonable implementation
   AGREES. ``rate`` asked for ``count / 2`` on a nonce-drawn count, so an
   implementation that rounds disagreed on odd draws and the verdict was
   a coin flip — injected only at L5, where ``rate`` first appears.
2. Nonce-derived inputs must look like real data. ``label`` built titles
   with hex digits inside words, a shape no title has, which made a
   digit-stripping slugify disagree for a reason the contract never
   pinned.

KNOWN NON-DISCRIMINATION (pre-flight finding 9): for ``ledger`` and
``window`` the seeded expectation IS the correct general fix, so those
two probes cannot distinguish teaching-to-the-test from a real fix; they
only catch an unfixed or broken module. The oracle layer earns its keep
on ``qty``, ``rate``, and ``label``. Records must not over-credit it.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.agentic_serving.oracles import OracleResult, run_probe

_PREAMBLE = """
import sys
NONCE = sys.argv[1]
TOKEN = "PROBE-OK-" + NONCE
SEED = int(NONCE[:6], 16)

def fail(why):
    print("FAIL:", why)
    raise SystemExit(1)
"""

_LEDGER_PROBE = (
    _PREAMBLE
    + """
try:
    from ledger import balance
except BaseException as exc:
    fail("import ledger: %r" % (exc,))

base = SEED % 997 + 3
amounts = [base, base + 11, base + 101]
want = sum(amounts)
try:
    got = balance(list(amounts))
except BaseException as exc:
    fail("balance(%r) raised %r" % (amounts, exc))
if got != want:
    fail("balance(%r) == %r, want %r" % (amounts, got, want))

try:
    balance([])
except ValueError:
    pass
except BaseException as exc:
    fail("balance([]) raised %r, want ValueError" % (exc,))
else:
    fail("balance([]) returned instead of raising")

print(TOKEN)
"""
)

_QTY_PROBE = (
    _PREAMBLE
    + """
try:
    from qty import parse_qty
except BaseException as exc:
    fail("import qty: %r" % (exc,))

n = SEED % 9000 + 13
# The decimal FORM varies, not just the numeral: textual surgery like
# text.replace(".0", "") satisfies the seeded test and every ".0" case
# while mangling any other decimal form.
for text, want in ((str(n), n), (str(n) + ".0", n), (str(n) + ".000", n)):
    try:
        got = parse_qty(text)
    except BaseException as exc:
        fail("parse_qty(%r) raised %r" % (text, exc))
    if got != want:
        fail("parse_qty(%r) == %r, want %r" % (text, got, want))

print(TOKEN)
"""
)

_WINDOW_PROBE = (
    _PREAMBLE
    + """
try:
    from window import last_n
except BaseException as exc:
    fail("import window: %r" % (exc,))

tag = NONCE[:8]
items = ["a-" + tag, "b-" + tag, "c-" + tag, "d-" + tag]
for n, want in ((0, []), (2, items[-2:]), (len(items), list(items))):
    try:
        got = last_n(list(items), n)
    except BaseException as exc:
        fail("last_n(items, %r) raised %r" % (n, exc))
    if list(got) != want:
        fail("last_n(items, %r) == %r, want %r" % (n, got, want))

print(TOKEN)
"""
)

_RATE_PROBE = (
    _PREAMBLE
    + """
try:
    from rate import per_hour
except BaseException as exc:
    fail("import rate: %r" % (exc,))

# EVEN, so every probed answer is a whole number. An odd count made the
# minutes=120 answer a half, where exact division and a rounding
# implementation legitimately disagree — the seeded test pins neither, so
# the probe was adjudicating an unpinned contract by coin flip, and `rate`
# appears only at L5.
count = (SEED % 250 + 7) * 2
for minutes, want in ((60, count), (120, count // 2), (30, count * 2)):
    try:
        got = per_hour(count, minutes)
    except BaseException as exc:
        fail("per_hour(%r, %r) raised %r" % (count, minutes, exc))
    # float() rather than arithmetic on the returned object: Decimal and
    # float do not mix, and rejecting a Decimal implementation would be a
    # false reject.
    try:
        close = abs(float(got) - want) < 1e-9
    except BaseException as exc:
        fail("per_hour(%r, %r) == %r, not a number (%r)"
             % (count, minutes, got, exc))
    if not close:
        fail("per_hour(%r, %r) == %r, want %r" % (count, minutes, got, want))

print(TOKEN)
"""
)

_LABEL_PROBE = (
    _PREAMBLE
    + """
try:
    from label import slug
except BaseException as exc:
    fail("import label: %r" % (exc,))

# LETTERS only. The hex nonce embedded digits inside words, which no real
# title looks like, and a slugify that strips non-letters then disagreed
# with a plain lowercase-and-hyphenate — a false reject manufactured by
# the probe's own unrealistic input.
tag = "".join(chr(ord("a") + int(digit, 16)) for digit in NONCE[:6])
title = "Alpha" + tag.capitalize() + " Beta" + tag.upper()
want = title.lower().replace(" ", "-")
try:
    got = slug(title)
except BaseException as exc:
    fail("slug(%r) raised %r" % (title, exc))
if got != want:
    fail("slug(%r) == %r, want %r" % (title, got, want))

print(TOKEN)
"""
)

VOLUME_PROBES: dict[str, str] = {
    "ledger": _LEDGER_PROBE,
    "qty": _QTY_PROBE,
    "window": _WINDOW_PROBE,
    "rate": _RATE_PROBE,
    "label": _LABEL_PROBE,
}


def run_volume_oracle(workspace: Path, module: str) -> OracleResult:
    """The hidden verdict for one fixture module in ``workspace``.

    Raises ``KeyError`` for a module with no probe: an unoracled module
    is an instrument gap, and silently returning "passed" would score
    every arm correct on it.
    """
    return run_probe(workspace, VOLUME_PROBES[module])
