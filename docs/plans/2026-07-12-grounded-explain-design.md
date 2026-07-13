# Grounded explain: answer from the wire, refuse to speculate

**Status:** Designed 2026-07-12 (brainstormed with the practitioner;
approved). Not yet implemented. Queued BEHIND the convergent-fix branch:
both edit `.llm-orc/scripts/agentic_serving/classify.py`, so this lands
after convergent-fix merges rather than fighting a merge conflict.

## Problem

The 2026-07-10 fresh-rig battery produced the series' first DISHONEST
miss on the explain path: turn 3, "explain how todo.py stores its
state", shipped hedged speculation about a `todo.py` whose build (turn
1) had been rejected, so nothing was ever written. Honesty is the
product's differentiator; a confidently-wrong explanation of a
non-existent artifact is worse than a low score. The explain seat today
answers purely from the model plus the rendered wire context, with no
deterministic grounding, so when the wire does not actually contain the
artifact the seat fills the gap by guessing.

## Key finding (grounding is wire-derived, not record-derived)

The roadmap framed this as "the explain seat consults the session
record." There is no structured record to consult: `SessionArtifactStore`
is not wired into the serving path, has no lookup-by-stem, and keeps no
reject log. Accept/reject lives on the append-only wire, an accepted
build leaves a `[wrote <path>]` block (content included), a rejected
build leaves only refusal prose. classify already parses this:
`_visibility` (classify.py:239-253) computes visible-vs-attempted
basenames from those render-grammar headers. So grounding derives from
the wire, the same source of truth everything else uses. No new store is
built here (that is WS-6 / #82 territory).

Two facts that scope the gate:

- A file target is extracted deterministically only for EXPLICIT
  filenames: `_extract_file` / `_FILE_RE` require an extension, so
  "explain how todo.py stores its state" yields `named_file = "todo.py"`
  at classify.py:499, but a bare stem ("the todo module") does not
  resolve on explain turns (stem/glob discovery sits inside the
  `if not is_explain ...` block).
- A conceptual explain with no filename gets a FAKE `file = "solution.py"`
  default (classify.py:573-574). The gate must therefore key off "did
  `_FILE_RE` actually match" (the `named_file` value), NEVER off the
  `file` field, or every "what is a decorator" question would trip it.

## Decisions

- **The grounding gate fires only when an explain turn has a real
  extracted filename target** (`is_explain` and `named_file is not
  None`). Conceptual explains (no filename) are untouched, the seat
  answers exactly as today.

- **Grounding status comes from `_visibility` on the target basename:**

  - **Target visible** (a `[wrote <path>]` or `[read <path>]` block for
    it is on the wire) -> grounded. Compose the explainer seat's
    `dispatch_input` (classify.py:580-585) to point AT that block's
    content and instruct it to explain the file's actual content. The
    seat answers from real material, not memory.

  - **Target not visible** -> NOT grounded. Emit a deterministic,
    non-speculative honest message (via emit.py, alongside today's
    refusal prose): "No `<target>` in this session (no successful build
    or read of it), so I can't explain its internals without guessing.
    If it's in your workspace, ask me to read it." The seat is NOT
    called, so no speculation path exists.

- **No bespoke explain->read chain.** When the not-visible target is a
  real client file, the honest message names the path but the serve does
  NOT hand-roll a read chain here. Reading-then-explaining a real file is
  the general capability, and it arrives via the WS-3 chain executor (the
  primitive that retires the ad-hoc chains), not a fourth one-off. This
  is the practitioner's generality decision (2026-07-12): the honesty
  GATE is a permanent invariant and ships now; the read CAPABILITY comes
  from the right primitive. The not-visible case is the named
  integration point.

- **No new store, no new seat.** Reuses `_visibility`, the existing
  explainer seat, and emit.py. `SessionArtifactStore` stays unwired.

## Turn flow

1. Explain turn arrives -> classify computes `is_explain` and
   `named_file`.
2. `is_explain` false -> other routing (unchanged).
3. `is_explain` and `named_file is None` (conceptual) -> explainer seat,
   today's behavior.
4. `is_explain` and `named_file` present -> grounding gate:
   a. target basename visible (`_visibility`) -> explainer seat with
      `dispatch_input` grounded on the target's `[wrote]`/`[read]`
      content.
   b. else -> deterministic honest non-speculative message, no seat.

## Bounds and error handling

- The gate keys off the real `_FILE_RE` match (`named_file`), never the
  `"solution.py"` default, so conceptual explains are never gated.
- Deterministic and spoof-resistant: `_visibility` reads the wire's
  render-grammar headers (block structure via post-boundary selection),
  not free text, so a forged `[wrote secret.py]` line in user prose
  cannot make the gate treat `secret.py` as grounded. Same protection
  the fenced-block grammar (v0.18.9) gives the read/run seams.
- Fail closed: if grounding status is ambiguous, treat as not-grounded
  (honest message) rather than let the seat speculate.

## Testing and validation

- Battery turn 3 conversion (the exact 2026-07-10 miss): "explain how
  todo.py stores its state" with no `[wrote todo.py]` on the wire ->
  honest non-speculative message, not hedged speculation.
- Visible target: after a successful `[wrote foo.py]`, "explain foo.py"
  -> grounded explanation that references the real content.
- Conceptual guard: "what is a decorator" (no filename) -> seat answers
  normally; the gate does NOT fire despite the `"solution.py"` default.
- Spoof probe: a forged `[wrote secret.py]` in user prose cannot flip
  the gate to grounded.
- Regression: existing conceptual-explain behavior unchanged.
- Live real-OpenCode at the earliest runnable point: a session where a
  build is rejected, then an explain of that file returns the honest
  message; and a session where a build succeeds, then an explain returns
  a grounded answer.
- Ladder: the exit gate is one full run with ZERO dishonest outcomes on
  the explain path (turn 3 honest), no regression on the other rungs.

## Named forward directions (not built here)

- **Explain->read via the WS-3 chain executor:** the general
  read-then-explain capability for real client files. grounded-explain's
  not-visible case is the named integration point; do NOT hand-roll the
  chain (practitioner generality decision, 2026-07-12).
- **#82 deep recall (WS-2 item 2, a separate deliverable):** ordinal /
  temporal prose queries ("the first thing I asked") over the record.
  This fixes battery turn 10's confidently-wrong recall, a DIFFERENT
  dishonest miss with a different mechanism (deterministic ordinal
  selection). Not this design.
- **Rejected-build enrichment:** distinguish "that build was rejected"
  from "never seen" for a more precise message, if the wire /
  attempted-set supports it cheaply.
- **Structured record:** when `SessionArtifactStore` is wired with a
  stem lookup and a reject log (WS-6 / #82), grounding moves from
  wire-derived to store-derived, same observable behavior.
