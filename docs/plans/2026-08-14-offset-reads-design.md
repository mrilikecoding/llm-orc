# Offset-continuation reads (#153) — design v1.1

**Status:** v1.1 after reviewer pre-flight (PROCEED-WITH-CHANGES: 2
blockers + 2 majors + minors, all folded in below; record in the session
transcript and issue #153).

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

1. **Pre-pipeline continuation.** In `run()`, AFTER the toolless gate
   and before the ack/pipeline path: scan THIS turn's read results; if
   any path's LAST part carries the cap trailer, emit the continuation
   read (`filePath` + the trailer's `offset`) directly as a
   ClientToolCall — no pipeline pass, no model call. **Termination
   (pre-flight blocker 1):** the bound counts READ CALLS for the path
   this turn (`_READ_PART_BOUND = 3`, from the wire's post-boundary
   tool_calls — never a dict keyed by offset, which freezes when a
   non-conforming client repeats the same offset), AND the trailer's
   continue-offset must be strictly greater than the part's own offset
   param, AND the trailer's "Showing lines X-Y" start must equal the
   requested offset — any violation stops continuing and the render
   refuses. Each continuation emission writes a model-free trace row
   (path/offset/part index — pre-flight major 4, the #144
   resolution-10 accounting class).
2. **Stitching at render, same-turn-segment only (pre-flight major
   3).** Parts are grouped per (turn segment, path) — a segment is the
   span between consecutive user messages — and only the path's LATEST
   segment stitches; parts from different segments never mix (a stale
   part 1 from an earlier turn can never be concatenated with a fresh
   part 2 after the file changed on disk, and an orphaned old key can
   never wedge future stitches). Per-part normalization strips the cap
   trailer BEFORE the gutter-uniformity check (pre-flight minor 5 —
   the ungutted trailer line otherwise defeats the `all()` check and
   leaves gutters on every line), then gutters, then the EOF trailer.
   Contiguity: each capped part's continue-offset must equal the next
   part's `offset` param. **Completeness is POSITIVE (pre-flight
   blocker 2):** the final part must carry the wire's own
   `(End of file - total N lines)` trailer AND N must equal the
   stitched line count — never "no cap trailer seen" (an unrecognized
   trailer variant, e.g. the unverified 2000-line-cap wording, or a
   per-continuation-call line-capped part, then fails the EOF check
   and refuses instead of rendering a corrupt whole). A complete
   stitch renders through the normal read-block tail — the 96KB cap
   and the token budget apply to the WHOLE. Gaps, over-bound,
   monotonicity violations, or a failed completeness check render the
   refusing `(truncated)` variant — fail closed.
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

## Module home (pre-flight minor 7)

The stitch/continuation code lands in a NEW module
(`web/serving/read_stitch.py`), not the caller — the caller's own
rendered block sits at ~32.3K projected tokens against the 35K budget
(7.7% headroom) and this arc's code would consume most of it; the
grep_render extraction is the precedent. The whale pin stays the
instrument.

## Not built here

The line-cap (2000-line) trailer variant's exact wording is unverified —
positive completeness makes that safe (an unrecognized cap fails the EOF
check and refuses); if the live gate captures the wording it gets added.
The schema's per-line 2000-char silent truncation carries NO trailer and
the EOF/total-N check cannot catch it — recorded as an open #149-family
flank, not solved here. Multi-file read fans and `max_rounds`
generalization stay deferred; the truncated-write-then-reread interplay
(pre-flight note 8b) is a pre-existing hazard to trace before any arc
multiplies it.
