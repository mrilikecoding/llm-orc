# Repo-scale reads: raise the read cap on measurement (#145)

Meta-task rung. Dogfood entry 1: "explain the subagent adapter" routed
correctly and refused honestly at the 24KB read cap (`_READ_FILE_CAP =
24576` in `serving_ensemble_caller.py`, mirrored as `_READ_CAP_KB = 24`
in `classify.py`). Real repo files routinely exceed 24KB. Exit (#145):
a >24KB real file grounds an explain without speculation; refusal
remains for genuinely unreadable targets.

## Measured basis (both committed, rerunnable)

`docs/plans/2026-08-13-context-curve-results.md` (+ extension):

- qwen3:8b accuracy is FLAT at 100% through 32K tokens of serve-shaped
  context (192 + 48 calls, zero drift, no placement effect); two-fact
  synthesis holds at 100% through 16K. Accuracy never votes against big
  chunks inside the window (40,960 architectural, verified untruncated
  via `prompt_eval_count`).
- The binding constraint is COLD LATENCY (prompt processing): ~17s at
  8K tok, ~67s at 16K, ~171s at 30K; warm ~2s regardless. Real files:
  `subagent_adapter.py` (25.8KB, 6,262 tok) 24.7s; `classify.py`
  (~80KB, 19,452 tok) 92.9s — both 100% accurate.

## Decision

Raise the cap to **96KB (98,304 bytes ≈ ~23K tokens)** on both sides of
the mirror, preserving whole-file-or-refuse semantics and the honest
refusal beyond it.

Why 96KB and not less: 64KB would exclude `classify.py` (~80KB), the
repo's own central module and the natural target of repo-scale
explains. Why not more: ~23K tokens costs ~2 min cold — comparable to
the serve's normal per-turn wall (run 5: ≈157s/turn) so relative
latency does not degrade, and it leaves ≥17K tokens (>40% of the
window) for conversation history and instructions; 128KB (~31K tok)
crowds the window against long histories and crosses into ~3 min turns.
This is a latency/window POLICY bound, recorded as such — the accuracy
curve supports the whole band.

Not built here (deferred until a real need appears, per YAGNI): chunked
or windowed reads for >96KB files. The refusal message names the actual
bound, so the next dogfood hit documents itself.

## Invariants and instruments

1. Whole-file-or-refuse unchanged: no truncated module body ever enters
   a context render. (Existing corpus pins; unchanged.)
2. The two mirrored constants cannot drift: a test asserts
   `_READ_FILE_CAP == _READ_CAP_KB * 1024` by importing both sides
   (same drift family as the `_GLOB_MAX_PATHS` render-through tests).
3. The refusal message states the true bound (no stale "24 KB" text
   anywhere after the change); corpus test pins message ↔ constant.
4. A file in (24KB, 96KB] now renders whole and grounds; a file >96KB
   refuses with the same honest shape as today. Corpus tests both sides
   of the new boundary.
5. Routing corpus green; no other seam changes.

## Exit gate (live, RIG)

Dogfood entry 1 converts: `opencode run -m llm-orc/agentic "explain the
subagent adapter"` grounds a read of `subagent_adapter.py` (25.8KB) and
answers from its actual content without speculation — logged as a new
dogfood entry. A >96KB target still refuses honestly.

## Sequencing

Reviewer pre-flight (one exchange: the 96KB bound, the mirror-drift
instrument, anything the 4x raise breaks downstream — render caps,
context composition, `_CTX_TAIL_CAP` interactions) → TDD on a branch →
hermetic green → live exit gate on the rig → adversarial review with
wrong-accept hunt → merge.

## Pre-flight verdict (2026-08-13): PROCEED-WITH-CHANGES

Measured findings (all against the real renderer + real ollama, no
repo changes):

1. **BLOCKING — the read accumulator is unbounded and cap-exempt.**
   `_select_read_blocks` re-renders the latest read per path from the
   FULL history into every turn (the anti-read-loop exemption). At
   96KB/file, 2–3 held reads cross the 40,960-token window, and the
   runtime discards context SILENTLY: three real files (~58,100
   projected tokens) returned HTTP 200 with `prompt_eval_count` 20,482
   — a third of what was sent, reproduced with a cache-busting nonce.
   Even two files sit at 94% of the window before any history. The
   raise converts an unreachable failure into a routine one.
2. The single-file path is mechanically sound at 4x (renders whole
   through every seam; strip scan 0.3ms; sandbox materialization
   byte-identical) — but a PRE-EXISTING silent mid-body cut exists:
   `_normalize_read`'s non-greedy `</content>` extraction truncates 58
   repo files today, unmarked, into the accept sandbox. Filed as #150;
   invariant 1 below is corrected (it was false as written).
3. No seat sets `num_ctx`; the serve inherits the loaded server's
   window. Silent runtime truncation is unobservable today.
4. The 96KB bound was self-confirming (sized to fit classify.py, which
   sits 31% past the highest verified synthesis level; the marginal
   (64,96]KB band admits 8 test files + classify.py). The 16K fast-call
   anomaly was CLOSED from the raw log (prefix-cache hit, not
   truncation — prompt_tokens identical to slow siblings).

**Adopted changes:**

- **C1 (blocking): bound the read accumulator with VISIBLE overflow.**
  A total rendered-read-bytes budget, window-derived (keep the full
  dispatch_input under ~36K projected tokens with generation margin;
  ≈128KB of rendered read bodies). When the NEXT read would cross it,
  classify REFUSES the read honestly at request time, naming the held
  files — never eviction (re-request = the read loop the exemption
  exists to prevent), never silence. Corpus-pinned with a
  multi-file-accumulation case.
- **C2: record `prompt_eval_count`** from every ollama seat call into
  the turn trace; the live exit gate asserts it tracks the projected
  prompt size (the only direct detector for runtime truncation).
- **C3: #150 fixed in this arc** (own commit): extraction anchored (or
  ambiguous extraction refused with a variant marker), pinned by a
  fixture whose body contains `</content>`.
- **C4 RESOLVED: cap = 96KB (98,304).** The synthesis extension ran
  clean: 100% at 20K and 24K tokens two-fact, 100% at 20K three-fact
  (18/18, ~91–118s cold). The 96KB band is synthesis-covered, not just
  recall-covered; classify.py (19,452 tok) sits inside with headroom.
  The self-confirming objection is answered by measurement, not policy.
- **C5: instruments strengthened**: mirror assert
  (`_READ_FILE_CAP == _READ_CAP_KB * 1024`, both imports already
  available to the corpus), boundary render-through pair (cap-byte
  whole / cap+1 oversize), refusal text pinned to `str(_READ_CAP_KB)`.

Invariant 1 is RESTATED: a rendered read body is byte-identical to the
file the client read, or the read refuses with a variant marker —
never a silent fragment, never a silent drop from context (C1's budget
refusal is the visible form). Exit gate gains: the accumulation case
(two large reads then a third ask → budget refusal, wire-visible) and
`prompt_eval_count` tracking on the grounded-explain turn.

## Review round 1 (2026-08-13): BLOCKER — the budget's unit was wrong

The adversarial review found the char-denominated budget open to charset
density: a 97KB JSON file (measured 2.07 chars/token; JSON is ASCII —
density comes from punctuation structure) passes BOTH caps and silently
overflows the window (the runtime's discard signature: prompt_eval_count
exactly 20,482 = window/2 on every over-window prompt, HTTP 200).
Measured density table: ASCII Python 4.0 chars/tok, JSON 2.07, CJK 1.99,
emoji 1.0.

Adjudicated fix: the budget re-denominates in **projected tokens** via a
deterministic uniformly-conservative estimator at the same render seam —
`ASCII word-runs + non-space punctuation chars + non-ASCII word chars`
(over-counts every measured class; still admits classify.py at ~25K
projected vs 24.6K real) — `_READ_TOKEN_BUDGET = 34000` (window 40,960
− ancillary − generation margin), mirrored with a drift assert. The
96KB per-file byte cap remains as the coarse whole-file-or-refuse gate.
The general backstop no estimator can evade — refusing any answer whose
recorded prompt_eval_count shows runtime truncation — is **#151**.

Also from the round: the `(over-budget)` variant joins the grammar in
FOUR consumers (accept_gather and refix_gather were missed — the
recurring grammar-coupling class; a shared variant vocabulary is noted
as follow-up); every mirrored constant pair now carries a drift assert;
`_not_grounded` reflects a recorded attempt reason instead of
recommending the action that just failed; the greedy single-wrapper
precondition is pinned against the 81/81 wire evidence; and budget
order-dependence is DECIDED: first-read-wins stands (never-evict is the
rule), pinned and named in the refusal remedy.
