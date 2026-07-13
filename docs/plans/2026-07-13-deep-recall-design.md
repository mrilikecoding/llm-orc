# Deep recall: deterministic ordinal selection, never a guess

**Status:** Designed 2026-07-13, reconciled 2026-07-13 after the
adversarial review. Selection (write-history ledger) implemented and
live-validated (11/13, zero-dishonest, pre-review code). Detection is
being reworked to the two-layer design below. This is #82's WS-2 item 2
(the recall half) — a separate deliverable from grounded-explain (WS-2
item 1, shipped v0.18.13), whose machinery it reuses.

## Problem

The standing dishonest miss: battery turn 10
(`benchmarks/agentic_serving/ladder_battery.sh`), "what did the first
thing I asked you to build do?", ships a confidently-WRONG file — it
names `calc.py` (a seeded read) or `storage.py` (a later build) as
"the first thing built". Every recorded run since the rung existed
carried this miss until the selection ledger landed. Honesty is the
product's differentiator; a confidently-wrong recall is worse than a
low score.

It is NOT a routing miss. Turn 10 correctly reaches the `explainer`
seat (`_INTERROGATIVE_RE` fires on "what…", `is_explain=True`,
`named_file=""` so the grounded-explain gate is skipped, `advance()`
lands `CHAIN_EXPLAIN`'s `_explain_explainer` step). The miss is that
the material the seat receives has **no chronological order** to select
from:

- The transcript render is windowed to the last N conversational
  messages, so by turn 10 the first ask is gone.
- The written-file blocks that DO come from full history are re-sorted
  relevance-first, not chronologically.
- The render-grammar headers (`[wrote <path>]`, `[read <path>]`) carry
  no index, sequence, or timestamp.

So an LLM asked for the *first* build answers by salience/recency and
names a later or seeded file. No deterministic component ever selects
by turn sequence.

## Recall has two jobs; honesty lives in only one

The load-bearing split (decided with the practitioner):

- **SELECTION** — "which build was first?" This is where honesty lives.
  A model choosing "which was first" IS the original turn-10 miss, so
  selection must be **deterministic-structural over an ordered record**.
- **DETECTION** — "is this an ordinal-recall query?" This is fuzzy NL.
  Because selection is structural, a detection error over-firing recall
  is irrelevant-but-true, never a lie. But a detection error that
  UNDER-fires (a genuine recall turn missed) routes to the guessing
  explainer seat — the original dishonest miss reintroduced. So
  detection cannot be a pure model vote either.

The doctrine applied: determinism for the answer, model judgment only
for bounded routing, with a structural floor under the routing so the
known-hard phrasing never depends on the model.

## Selection (implemented): the write-history ledger

Two facts scope where selection runs:

- **Files retrieve from deep history; prose asks do not.** classify only
  ever sees the caller's windowed, relevance-sorted `context` string.
  The chronological full history lives ONLY in the caller (the
  `messages` sequence). So the ordinal SELECTION cannot happen in
  classify from the context it is handed — it is computed in the caller.
- **The lossless server-side record stays unwired here.**
  `core/session/artifacts.py` is reserved for #82's OTHER half (the
  server-side record + divergence classifier), entry-gated on WS-5's
  compaction observation (WS-6). This design needs none of it: the
  append-only wire (`messages`) is a sufficient ordered record.

**The ledger is shipped writes only.** On every turn the caller
(`serving_ensemble_caller._recall_ledger`) scans full `messages` in wire
order and builds a bounded `recall_ledger`: one `{ask, path}` entry per
shipped write, in chronological order — `ask` a short excerpt of the
user request that preceded the write, `path` the write's structural
destination (from an assistant `[wrote <path>]` tool_call). There is no
`shipped`/`rejected` field: the ledger records what SHIPPED, so there is
no prose-inferred "rejected" case to fabricate (adversarial-review
blocker 1/2). Spoof-safe by construction: built from message ROLES and
write TOOL_CALLS, never parsed from free text, delivered as a structured
turn field classify never has to parse out of a string.

**classify selects deterministically over the ledger**
(`_recall_answer`): the first entry is the first build. Three cases,
reusing grounded-explain's visible/not-visible split:

1. **grounded** — the first shipped build's basename is visible in the
   current context (`_visibility`). Ride the grounded explainer on that
   real body (inject the selected `path` as `named_file`; the existing
   `elif target == _EXPLAIN_SEAT and named_file` dispatch grounds the
   seat). The seat describes "what it does" from real content.
2. **built_deep** — shipped but windowed out of context. A deterministic
   honest message names the artifact and defers the body to a read:
   "The first thing built in this session was `<path>` (from your
   request '<ask>'). Ask me to read `<path>` and I'll explain what it
   does." The read arrives via the WS-3 chain executor later, not a
   hand-rolled chain here.
3. **none** — nothing shipped this session: "Nothing has been built in
   this session yet." No seat is called; there is no speculation path.

Only "first" is selected today (the selector answers `ledger[0]`);
last/Nth ladder later (Named forward directions).

## Detection (this rework): structural floor + model extension

Two detection layers feed the one selector. Selection is unchanged.

**Layer 1 — structural floor (`_RECALL_RE`, classify only).** A tight
regex requiring the "first thing" anchor bound to a first-person agent
(I/you) and a build verb. On a match (with `is_explain`), recall
resolves structurally the way it does today: grounded → `named_file`
injection → inline grounded-explain; built_deep/none → `is_recall_answer`
+ message, `CHAIN_EXPLAIN`'s recall step. **No decider.** This is
turn-10's measured phrasing; its honesty is deterministic and never
depends on the model. The regex is demoted from "the detection brain"
to a structural fast-path (the role `_INTERROGATIVE_RE` already plays).
It lives in classify only — the caller no longer carries a copy, so
there is no regex-parity coupling to test.

**Layer 2 — model extension (`maybe_recall` + the `decide` node).** A
loose pre-filter — `is_explain` + an ordinal word (first/last/earliest/…)
and **no named file** (a named file already belongs to grounded-explain)
— that did NOT match the tight regex sets a new
`SignalBundle.defer_recall`. `_explain_explainer`'s guard becomes
`is_explain and not defer_recall`, so a deferred turn matches no
`CHAIN_EXPLAIN` step, falls through to `CHAIN_DECIDER`, and emits
`needs_decider=true`. classify pre-computes the honest `recall_answer`
from the ledger and passes it through without applying it. The `decide`
node (`serving.yaml`) gains `recall` in its closed target set with
examples. `resolve` applies the vote: `decide=="recall"` →
`target="recall-answer"`, `build=False`, `kind="recall"` (the
pre-computed `recall_answer` already passes through); any other vote →
the existing `_DERIVED` path (a file-less conceptual explain routes to
the plain explainer, which is correct — there is nothing to be
dishonest about).

**Deliberate simplification vs the first sketch.** On the DEFERRED path,
grounded collapses into `recall-answer` ("the first thing built was
`<path>`; ask me to read it") rather than reconstructing the inline
grounded-explainer routing in `resolve`. The tight-regex path keeps full
inline grounded-explain for the measured turn-10 phrasing; only
*fuzzy*-phrased grounded recalls take one extra read round instead of an
inline answer. Honest either way, and `resolve` stays a thin merge
instead of a second routing brain (preserves the WS-3 "chain_plan is the
routing table" line).

## Turn flow

1. Caller renders context as today; additionally computes `recall_ledger`
   from full `messages` every turn (no gate) and adds it to the executor
   input dict.
2. classify computes `is_explain`, the ledger selection, and the two
   detection layers:
   - `_RECALL_RE` match → structural recall (grounded / built_deep /
     none) resolved in `CHAIN_EXPLAIN`. No decider.
   - else `maybe_recall` (loose, file-less) → `defer_recall=true`,
     pre-compute `recall_answer`, fall through to `CHAIN_DECIDER`
     (`needs_decider=true`).
   - else → today's explain/build/run/fix routing, untouched.
3. On the deferred path the `decide` node votes; `resolve` merges:
   `recall` → `recall-answer` shape with the pre-computed message; any
   seat vote → `_DERIVED`.
4. Non-recall explain turns are untouched (both detection layers are
   false), so conceptual and grounded/ungrounded explains behave exactly
   as today.

## Bounds and error handling

- **Precise about intent, not just keywords.** The tight `_RECALL_RE`
  requires the anchored "first thing … I/you … build" shape, so a plain
  "what did you do?" or "explain the first function in foo.py" does not
  trip it. The loose `maybe_recall` requires `is_explain` + an ordinal
  word + no named file, and the model backstops it, so over-firing costs
  an extra decider call, never a wrong answer.
- **Fail closed to honesty.** An empty/malformed `recall_ledger`, or a
  recall match with no shipped build → the honest "nothing built yet"
  message, never the guessing seat. A deferred turn the decider does not
  vote `recall` for is a file-less conceptual explain — correct on the
  plain explainer.
- **Spoof-safe.** The ledger is structured data from roles + write
  tool_calls; `path` derives from the same structural tool_call signal
  `wrote_path` uses, never from context text. A forged `[wrote
  secret.py]` line in user prose cannot enter the ledger.
- **Flat per-turn cost.** The caller builds the ledger every turn (no
  gate): the full-history scan is the same order as the written-file
  scan the caller already runs, so a gate would not change the cost
  class. An inert ledger on a non-recall turn is ignored. `ask` excerpts
  are capped.

## Testing and validation

- **Hermetic classify + chain_plan units:** structural detection (the
  tight regex fires on the turn-10 phrasing, stays off "first function
  in foo.py" / turn 5 / conceptual explain), the three selection cases
  (grounded-visible, built-deep, none) over a fixture `recall_ledger`,
  and the deferred path (`maybe_recall` sets `defer_recall`,
  `_explain_explainer` suppressed, `needs_decider=true`).
- **Decider-vote units (stubbed decide):** a fixture swaps the decide
  model for an echo returning `{"target":"recall"}`; assert `resolve`
  routes a deferred turn to `recall-answer` with the pre-computed
  message on a `recall` vote, and to the plain explainer on an
  `explainer` vote.
- **Caller unit:** `recall_ledger` built from a full-history fixture
  where the first build is past the context window and a later build
  outranks it by relevance — the ledger still lists the first build
  first.
- **Spoof probe:** a forged `[wrote todo.py]` line in the user's task
  prose cannot enter the ledger.
- **Live real-OpenCode at the earliest runnable point:** the exact
  turn-10 conversation returns the correct first-built file (grounded or
  ask-me-to-read), never a guess; and a fuzzy-phrased variant
  ("what was the earliest thing you built here?") exercises the deferred
  decider path.
- **Ladder:** the exit gate is one full run with ZERO dishonest outcomes
  (turn 10 converts), no regression on the other rungs, then variance
  measured over three same-seed runs (WS-2 exit: median ≥ 10/13).

## Named forward directions (not built here)

- **Richer recall grammar:** last/latest/most-recent (the immediate
  sibling of "first"; needs `ledger[-1]` selection), then Nth ("the
  third thing"), "before/after X", and absolute-time forms, laddered
  from battery evidence (minimal-gate-first).
- **Deep-history body via the WS-3 read chain:** the built_deep "ask me
  to read `<path>`" message becomes an automatic read→explain chain once
  the WS-3 chain executor carries explain→read (grounded-explain's named
  integration point). The same deterministic selection feeds it.
- **Store-derived recall:** when `core/session/artifacts.py` is wired
  (#82's other half, WS-6, compaction-gated), the ledger moves from
  wire-derived to store-derived with the same observable behavior and
  cross-session reach.
