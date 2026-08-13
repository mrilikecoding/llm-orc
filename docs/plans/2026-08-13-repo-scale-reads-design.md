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

## Review round 2 (2026-08-13): BLOCKER 1 still open — v1 self-confirmed

Round 1's conservativeness test validated `_projected_tokens` only
against the round-1 reality-check TABLE on synthetic fixtures the
formula itself was tuned against — self-confirming, not independent
evidence. Round 2 measured against qwen3:8b's real tokenizer (fresh
fixtures the round-1 formula never saw) and found v1 under-counts on 8
of 10 classes: base64/PEM/hex as low as 7-12% of real. Root cause: v1
counts a whole ASCII word-run as ONE token regardless of length, but
BPE splits long high-entropy runs (base64, digests, long identifiers)
into many subword tokens. A 94KB PEM cert passed both guards at "21%
utilization" while its real prompt_eval_count showed the window had
silently overflowed (the same discard signature: ~half the window,
HTTP 200).

Estimator v2 (adjudicated term list): ASCII word-runs <=30 chars cost 1
unit, >30 chars scale as `ceil(len/1.3)` (measured high-entropy
density), non-space punctuation and non-ASCII word characters cost 1
unit each, newlines cost 1 unit each, runs of >=2 spaces (indentation)
cost 1 unit each — total times a safety factor derived from measurement,
not asserted.

Ground truth (rig-measured, qwen3:8b, `/api/chat`, `think:false`,
`num_predict:1`, minus a verified 17-token chat-template overhead):
the reviewer's ten round-2 fixture classes plus five real repo files
(classify.py, subagent_adapter.py, serving_ensemble_caller.py, emit.py,
accept_gather.py). Frozen table, dated, generation command documented:
`tests/unit/web/serving/test_token_estimate_ground_truth.py`. Worst
v2-before-factor ratio: PEM certificate at 0.6630 (the base64 alphabet's
`+`/`/`/`=` chop an otherwise-long entropy run into pieces <=30 chars
each, so the length-scaling rule rarely engages). Smallest factor
clearing every fixture with >=5% margin: **1.5837, rounds to 1.59**.

**Open fork (not resolved here — reported per instruction, not silently
decided):** at factor 1.59, classify.py's own projected count is 34,341
> `_READ_TOKEN_BUDGET` (34,000) — a real repo file failing to admit is
the exact regression #145 exists to prevent. The max factor keeping
classify.py under budget is ~1.574; PEM's 5% margin needs ~1.584 — about
a 1% gap. `src/llm_orc/web/serving/token_estimate.py` implements and
validates v2 as a standalone module (green, tested, dated ground truth)
but it is **not wired into the live budget** — `serving_ensemble_caller.py`
still runs v1. `test_classify_py_sanity_constraint_conflict_is_open` pins
the conflict as a known, open fact so it fails loudly (not silently) the
moment either number changes. Options for the lead: raise the budget
past 34,341 (eats into the generation-margin reasoning the 34,000 figure
was itself derived from); accept less than 5% margin specifically for
the least-repo-realistic fixture class (PEM/high-entropy binary-ish
content); or refine the v2 formula's entropy-run detection to not be
defeated by base64's punctuation characters. Runtime backstop (Part 2 of
the round-2 ask, implementing #151's core) is deferred alongside this —
its 0.5 detection threshold's safety margin is derived assuming factor
<=1.5, so it depends on this fork's resolution too.

## Review round 2 resolution (2026-08-13): fork closed, v2 wired, backstop shipped

The lead resolved the fork with numbers, not preference: safety factor
stays **1.59** (no fixture class sacrificed — PEM keeps its full >=5%
conservativeness margin); `_READ_TOKEN_BUDGET` raised **34,000 -> 35,000**.
Honest window arithmetic, documented at the constant: 40,960 − 35,000 =
5,960 reserve >= measured ancillary render (~1,800) + explain-generation
headroom (~1,000-2,000) + margin. classify.py now projects to 34,341 at
measurement time (34,515 against the file's current, slightly larger
size) — admitted with real (>1%) margin, not a hairline pass.
`token_estimate.projected_tokens_v2` is wired in as
`serving_ensemble_caller._projected_tokens` (v1 retired); classify.py's
mirror constant follows to 35,000, drift-asserted; the over-budget
refusal now speaks in plain terms ("the session read budget") instead of
surfacing the raw figure. `test_classify_py_sanity_constraint_conflict_is_open`
is re-pointed to `test_classify_py_projects_under_budget_with_real_margin`,
asserting the RESOLVED fact so it fails loudly if either number regresses.

**Part 2 — the runtime backstop (#151's core) shipped**, with corrected
dual thresholds (a single 0.5 ratio would false-positive on estimator
v2's most over-projected density class — CJK+code projects ~2.15x real
at factor 1.59, putting a legitimate in-window call's ratio as low as
~0.465). `turn_trace._truncation_check` computes projected tokens over
classify's own `dispatch_input` and flags the turn when EITHER fires:
trigger 1 (`prompt_eval_count < projected * 0.42`, catching deep
overflow directly) or trigger 2 (`prompt_eval_count` within 64 of
`WINDOW // 2`, gated on `projected > WINDOW * 0.8` so a small legitimate
call can never trip the signature alone — both real captured over-window
prompts returned exactly `WINDOW // 2 + 2`). `WINDOW = 40960` is a
documented, hardcoded constant; #151 stays open for a server-queried
window and threshold re-measurement per model era.
`ServingEnsembleCaller._serve()` discards the pipeline's answer and
returns a hardcoded refusal when the trace flags it — pinned end-to-end
with a mocked executor (the withheld answer text never leaks into the
refusal).
