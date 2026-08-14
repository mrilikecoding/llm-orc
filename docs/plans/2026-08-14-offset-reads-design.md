# Offset-continuation reads (#153) — design

**Goal:** restore whole-file grounding for 50–96KB files. OpenCode's read
tool caps output at 50KB (and 2000 lines) with an explicit trailer naming
the continuation offset ("Use offset=1105 to continue."); the serve's
whole-file window is 96KB. Today a capped read refuses honestly (the #121
fix); this arc reassembles the parts instead. The captured schema
confirms `offset` (1-indexed line start) on the advertised read tool. The
demonstrating case: `serving_ensemble_caller.py` (80KB, ~32.3K projected
tokens post-extraction) — the #121 live gates' most-picked file — grounds
end to end.

## Mechanism (caller-side, deterministic, no classify changes)

1. **Pre-pipeline continuation.** In `run()`, before the ack/pipeline
   path: scan THIS turn's read results; if any path's LAST part carries
   the cap trailer and that path's part count is under
   `_READ_PART_BOUND = 3` (96/50 needs 2; one spare), emit the
   continuation read (`filePath` + the trailer's `offset`) directly as a
   ClientToolCall — no pipeline pass, no model call. The trailer parse
   extends `_CLIENT_READ_CAP_RE` to capture `Use offset=(\d+)`.
2. **Stitching at render.** `_read_blocks` groups read results per
   (path, offset) over the full history (latest result per key, the
   existing durability rule), normalizes EACH part (per-part gutter and
   trailer strip), and stitches ascending by offset with a CONTIGUITY
   check: each capped part's continuation offset must equal the next
   part's `offset` param. A contiguous, complete stitch (final part
   uncapped) renders through the normal read-block tail — the 96KB cap
   and the token budget apply to the WHOLE (budget parity: a stitched
   102KB classify.py still renders oversize; an over-budget stitch
   renders over-budget). Gaps, an over-bound part count, or a still-
   capped final part render today's refusing `(truncated)` variant —
   fail closed, never a silently partial whole.
3. **Bound honesty.** The part bound is the deterministic backstop
   (the #144/`_SELF_READ_MAX_ROUNDS` pattern); bound-exceeded refuses
   with the existing client-cap reason. Self reads are untouched
   (native, uncapped by the client).

## Invariants (rule 6) and instruments

- **Never a partial whole:** a rendered full read block either came from
  an uncapped result or a contiguous complete stitch. Instruments:
  crafted two-part and gap/incomplete/over-bound cases; the gap case
  must refuse, not concatenate.
- **Budget parity on the whole:** stitched content charges the same
  render/cap/budget path as a single read. Instrument: a stitched
  >96KB fixture renders oversize; projection parity against an uncapped
  equivalent.
- **Bounded continuations:** at most `_READ_PART_BOUND` parts per path
  per turn; the continuation loop can never spin (each continuation
  strictly increases the part count toward the bound). Instrument: a
  crafted always-capped result sequence stops at the bound with the
  honest refusal.
- **No behavior change for uncapped reads** (the entire existing corpus
  is the instrument).

## Exit gate & validation

Corpus/caller tests per the instruments; hermetic endpoint test with a
two-part continuation conversation; live (RIG): "where is the recall
ledger built?" grounds the 80KB caller end to end via a 2-part stitch
(converting the #121 recorded bound), plus a battery row and the
author-independent adversarial review.

## Not built here

The line-cap (2000-line) trailer variant's exact wording is unverified —
the parser keys on the captured KB-cap trailer; if the live gate meets a
different trailer for long-line-count files, it gets captured and added
(instrument-preconditions discipline). Multi-file read fans and
`max_rounds` generalization stay deferred.
