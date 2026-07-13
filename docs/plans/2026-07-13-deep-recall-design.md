# Deep recall: deterministic ordinal selection, never a guess

**Status:** Designed 2026-07-13 (brainstormed with the practitioner;
approved, Approach B). Not yet implemented. This is #82's WS-2 item 2
(the recall half) — a separate deliverable from grounded-explain
(WS-2 item 1, shipped v0.18.13), whose machinery it reuses.

## Problem

The standing dishonest miss: battery turn 10
(`benchmarks/agentic_serving/ladder_battery.sh`), "what did the first
thing I asked you to build do?", ships a confidently-WRONG file — it
names `calc.py` (a seeded read) or `storage.py` (a later build) as
"the first thing built". Every recorded run since the rung existed
carries this miss. Honesty is the product's differentiator; a
confidently-wrong recall is worse than a low score.

It is NOT a routing miss. Turn 10 correctly reaches the `explainer`
seat (`_INTERROGATIVE_RE` fires on "what…", `is_explain=True`,
`named_file=""` so the grounded-explain gate is skipped, `advance()`
lands `CHAIN_EXPLAIN`'s `_explain_explainer` step). The miss is that
the material the seat receives has **no chronological order** to select
from:

- The transcript render is windowed to the last 8 conversational
  messages (`_CTX_MAX_MESSAGES = 8`,
  `serving_ensemble_caller.py:197`), so by turn 10 the first ask is
  gone.
- The written-file blocks that DO come from full history are re-sorted
  **relevance-first, not chronologically** (`_select_written_files`
  sort key, `serving_ensemble_caller.py:305`).
- The render-grammar headers (`[wrote <path>]`, `[read <path>]`) carry
  no index, sequence, or timestamp.

So an LLM asked for the *first* build answers by salience/recency and
names a later or seeded file. No deterministic component ever selects
by turn sequence.

## Key finding (selection must be caller-side; the record stays unwired)

Two facts scope the fix:

- **Files retrieve from deep history; prose asks do not.**
  `_select_written_files` already selects file bodies from the FULL
  `messages` sequence — but classify only ever sees the caller's
  windowed, relevance-sorted `context` string
  (`classify.py:633,738`). The chronological full history lives ONLY in
  the caller (`serving_ensemble_caller.py`, the `messages` sequence).
  So the ordinal SELECTION cannot happen in classify from the context
  it is handed — it must be computed where the ordered history is.
- **The lossless server-side record is deliberately unwired and stays
  that way here.** `core/session/artifacts.py` is "reserved for
  roadmap Stage 2 (issue #82)" but not invoked on the serving path.
  Wiring it is #82's OTHER half (the server-side record + divergence
  classifier), entry-gated on WS-5's compaction observation (WS-6).
  This design needs none of it: the append-only wire (`messages`) is a
  sufficient ordered record. No new store, no new seat.

## Decisions

**Approach B — deterministic structural selection, then grounded
explain.** The model never does ordinal reasoning (the thing that is
failing). A deterministic component selects the anchored turn from the
ordered record; the seat only PHRASES a description of one selected,
grounded artifact — exactly grounded-explain's role. This is the
roadmap's "selection, never summarization". (Rejected: Approach A,
handing the seat an ordered ledger to reason over — it keeps model
ordinal reasoning and re-opens the miss on a longer history.)

**The seam split mirrors how the system already works:**

- **Caller provides the ordered selection data as a STRUCTURED turn
  field**, not new context text. On a recall-shaped turn the caller
  scans full `messages` in wire order and builds a bounded
  `recall_ledger`: one entry per build-ask turn, in chronological
  order, `{ask, path, shipped}` — `ask` a short excerpt of the user's
  request, `path`/`shipped` its structural outcome (an assistant
  `[wrote <path>]` tool_call after that ask → shipped; none → rejected).
  Passed alongside `wrote_path`/`write_count` in the executor input
  dict (`serving_ensemble_caller.py:869-877`). Spoof-safe by
  construction: built from message ROLES and write TOOL_CALLS, never
  parsed from free text, and delivered as structured data classify
  never has to parse out of a string.

- **classify detects the recall query, picks the anchor, and routes.**
  A new `_RECALL_RE` (regex on the task alone) detects the ordinal-
  recall shape and its anchor; classify reads `recall_ledger`, picks
  the anchored entry deterministically, and sets a new
  `SignalBundle.is_recall`. A new `Step` in `CHAIN_EXPLAIN` **ahead of**
  `_explain_explainer` fires on it, so a recall turn never falls
  through to the ungrounded seat.

**Anchor keyed off the query verb** (the roadmap's stated-correct
reading — "the todo ask came first"):

- "the first thing I **asked** (to build)" → ask-anchored: the FIRST
  `recall_ledger` entry, shipped or not.
- "the first thing **built/wrote**" → build-anchored: the first entry
  with `shipped=True`.
- "first" → earliest; "last"/"latest"/"most recent" → newest. (Nth,
  "before X", and absolute-time forms are named-forward, not built.)

**Three answer cases, reusing grounded-explain's visible/not-visible
split:**

1. **Selected entry shipped AND its body is visible** (`_visibility`
   over the current context has the basename) → route to the
   `explainer` seat grounded on that file's real body (the existing
   `elif target == _EXPLAIN_SEAT and named_file` dispatch at
   `classify.py:762-774`, pointed at the recall-selected path). The
   seat describes "what it does" from real content.
2. **Selected entry shipped but its body is NOT visible** (deep
   history, windowed out) → a deterministic honest message that names
   the real artifact and defers the body to a read: "The first thing
   you asked me to build was `<ask>` — I wrote `<path>`. Ask me to
   read `<path>` and I'll explain what it does." This is the same
   not-visible → read integration point grounded-explain names; the
   read capability arrives via the WS-3 chain executor, NOT a
   hand-rolled chain here.
3. **Selected entry rejected, or no build ask exists** → a
   deterministic honest message: "The first thing you asked me to
   build was `<ask>`, but that build was rejected — nothing shipped, so
   I can't tell you what it does." / "You haven't asked me to build
   anything yet." No seat is called; there is no speculation path.

Cases 2 and 3 emit via a new deterministic field composed in
`emit.py`, exactly as `not_grounded` does (`emit.py:77-83`).

## Turn flow

1. Caller renders context as today; additionally computes `recall_ledger`
   from full `messages` every turn (no gate) and adds it to the executor
   input dict.
2. classify: `_RECALL_RE` matches → extract anchor (first/last, verb) →
   read `recall_ledger` → pick the anchored entry.
3. `is_recall` true → `CHAIN_EXPLAIN`'s new recall step fires (ahead of
   the plain explainer step).
4. Answer case selected deterministically:
   a. shipped + visible → grounded explainer seat on the picked path.
   b. shipped + not visible → honest "ask me to read `<path>`" message.
   c. rejected / none → honest "rejected, nothing shipped" / "nothing
      asked yet" message.
5. Non-recall explain turns are untouched (the recall step's guard is
   false), so conceptual and grounded/ungrounded explains behave
   exactly as today.

## Bounds and error handling

- **The recall gate is precise about intent, not just keywords.**
  `_RECALL_RE` requires an ordinal/temporal anchor bound to an
  ask/build referent ("first/last … asked/built/wrote/created"), so a
  plain "what did you do?" or "explain the first function in foo.py"
  does not trip it. Turn 5 ("did you see my previous query?") is a
  memory question with no ordinal-over-builds anchor → stays on today's
  explainer path.
- **Fail closed to honesty.** An empty or ambiguous `recall_ledger`, a
  recall match whose anchor cannot be resolved, or a malformed field →
  the honest "nothing asked yet" message, never the guessing seat.
- **Spoof-safe.** The ledger is structured data from roles + write
  tool_calls; `shipped`/`path` derive from the same structural
  tool_call signal `wrote_path` uses, never from context text. A forged
  `[wrote secret.py]` line in user prose cannot enter the ledger.
- **Flat per-turn cost (no caller gate — revised in build).** The caller
  builds the ledger every turn rather than gating on `_RECALL_RE`: the
  full-history scan is the same order as the `_select_written_files`
  scan the caller already runs each turn, so a gate would not change the
  cost class, and dropping it removes a duplicated regex and its parity
  test. classify's `_RECALL_RE` is the sole recall detector; an inert
  ledger on a non-recall turn is ignored. The ledger's `ask` excerpts
  are capped (`_RECALL_ASK_CAP`).

## Testing and validation

- **Hermetic classify + chain_plan units:** ordinal detection (first
  vs last, asked vs built), anchor pick over a fixture `recall_ledger`,
  the three answer cases (shipped-visible, shipped-not-visible,
  rejected/none), and the negative guards (turn 5, conceptual explain,
  "first function in foo.py" all stay off the recall path).
- **Caller unit:** `recall_ledger` built from a full-history fixture
  where the first ask is past the 8-message window and a later build
  outranks it by relevance — the ledger still lists the first ask
  first, with the correct `shipped` outcome.
- **Spoof probe:** a forged `[wrote todo.py]` line in the user's task
  prose cannot make a rejected first-ask read as shipped.
- **Regex parity regression:** caller and classify `_RECALL_RE` equal.
- **Live real-OpenCode at the earliest runnable point:** the exact
  turn-10 conversation — first build rejected, then "what did the first
  thing I asked you to build do?" returns the honest "rejected, nothing
  shipped" message (not a guessed file); and a variant where the first
  ask shipped returns a grounded (or ask-me-to-read) answer naming the
  correct path.
- **Ladder:** the exit gate is one full run with ZERO dishonest
  outcomes (turn 10 converts), no regression on the other rungs, then
  variance measured over three same-seed runs (WS-2 exit: median ≥
  10/13).

## Named forward directions (not built here)

- **Richer recall grammar:** Nth ("the third thing"), "before/after
  X", and absolute-time forms, laddered from battery evidence
  (minimal-gate-first).
- **Deep-history body via the WS-3 read chain:** case 2's "ask me to
  read `<path>`" becomes an automatic read→explain chain once the WS-3
  chain executor carries explain→read (grounded-explain's named
  integration point). Same deterministic selection feeds it.
- **Store-derived recall:** when `SessionArtifactStore` /
  `core/session/artifacts.py` is wired (#82's other half, WS-6,
  compaction-gated), the ledger moves from wire-derived to
  store-derived with the same observable behavior and cross-session
  reach.
