# Chained fix-execution: write → run → verdict in one turn

**Status:** Implemented 2026-07-10, same session as the design (branch
worktree-fix-execution). Live-validated against real OpenCode: the full
chain fired (need-files → code-seat → need-run → run-verdict) with an
honest red verdict on a fix the seat failed to apply — the exact
server-gate blind spot this rung closes — and a plain build turn stayed
terminal. Rung-1 scope shipped as designed; no deviations from this text.
Seam map evidence: this session's exploration of the caller and classify
(file:line references below are v0.18.11).

## Problem

A fix/build turn ships its write and terminates. The artifact was verified
in the serve's sandbox against serve-authored tests — never against the
client repo's real suite in the client's real environment. The run seam
(v0.18.8) closed that gap for explicit "run the tests" turns; fix turns
can't reach it. Today's battery showed the cost concretely: a fix turn's
deliverable is a one-shot bet, and per-decision success doesn't compose to
per-session success.

The single structural blocker (mapped 2026-07-10): a write continuation is
terminal. `_resumes_turn` (serving_ensemble_caller.py:578) admits only
read/run/glob-shaped tool results; a write result gets a deterministic
"Wrote X." ack (`_tool_result_ack`:589) and the turn ends. Multi-hop
chaining through stateless classify re-routing already works everywhere
else (read→build, run→verdict).

## Decisions

- **Chain trigger (rung 1, vary one thing): turns LED by a fix
  imperative.** A task matching `_FIX_VERB_RE`
  (`^\s*(?:fix|update|modify|refactor|edit|change)\b`) whose gated build
  ACCEPTED chains into one delegated run. Anchored to the task start
  after PR #115 review: mid-sentence "existing"/"change" are ordinary
  build prose ("write add.py so the existing tests pass" must not
  chain), and `_EXISTING_RE`'s read-first role is untouched. Ordinary
  "write X" build turns keep today's terminal ack — no behavior change
  outside the fix-led class. Widening the trigger (all gated builds;
  presence of client-side tests) is a later rung, decided on ladder
  evidence.
- **The write continuation resumes; classify decides what happens.**
  `_resumes_turn` admits write-shaped results. On resume the whole
  pipeline re-runs statelessly from the wire (the mechanism read→build
  already uses). A new deterministic classify signal — `has_wrote_block`
  (this-turn `[wrote <path>]` present, selected from post-boundary
  messages exactly as `_run_blocks`/`_glob_blocks` are,
  serving_ensemble_caller.py:475/511) — routes:
  - fix-intent + wrote-block + no run block → `need-run` (existing shape)
  - run block present → `run-verdict` (existing shape, classify.py:446)
  - anything else (non-fix turn, or guards below) → terminal "Wrote X."
    ack, exactly today's behavior.
- **The run-verb suppression stays.** classify suppresses `run_signal`
  when build/fix verbs are present (classify.py:495) so a fix turn can
  never route to run on its FIRST pass. Unchanged. The chain's run leg is
  triggered by the wrote-block signal on the resume, not by the verb —
  the suppression and the chain compose instead of conflicting.
- **The run is the existing closed template.** `pytest -q` (plus named
  `test_` files when the turn names them) — the same deterministic
  builder and verdict parser as explicit run turns. Zero model calls in
  the chain's second and third legs.
- **Red verdict = honest report, no within-turn re-fix (rung 1).**
  "Wrote todo.py. Ran `pytest -q`: 2 failed, 5 passed." + failure lines.
  The bounded re-fix loop (feed the client-side failure report back as
  retry carry, like the #100 held round) is the natural rung 2 — it is
  where the compounding-reliability payoff lives, but it multiplies
  rounds and belongs behind a measured rung-1 baseline. Held open, not
  decided here.
- **No new shape, no new catalog entry.** The chain reuses `need-run` and
  `run-verdict` as-is; the #106 two-homes trap is not fed. The only
  touched surfaces are `_resumes_turn`/`_tool_result_ack`, classify, and
  (possibly) emit's finish-prose composition so the verdict message names
  the written file.

## Turn flow (three passes, all stateless from the wire)

1. **Fix pass (today's behavior):** "fix the divide bug in calc.py" →
   read preamble if calc.py is invisible (existing need-files seam) →
   gated build → emit write tool_call. Reject path unchanged (honest
   "Another round needed", no chain — nothing shipped, nothing to run).
2. **Run pass (new):** client applies the write, POSTs the appended wire
   ending in the write tool result → `_resumes_turn` admits it →
   classify: fix-intent + `has_wrote_block` + no run block → `need-run`
   → ONE bash tool_call (`pytest -q`).
3. **Verdict pass (existing):** run result arrives → classify sees the
   `[ran]` block → `run-verdict` parses pytest's summary → finish prose
   reporting write + verdict honestly, green or red.

## Bounds and error handling

- One write, one run, one verdict per turn. Loop-safety is the existing
  per-seam idempotency: `has_wrote_block` never re-dispatches the build
  on the run/verdict passes (the new guard, symmetric with
  `has_run_block` flipping need-run → run-verdict today), and the run
  seam's one-round rule is untouched.
- A failed client-side write must NOT chain: ack honestly ("Write
  failed for X."), terminal. Detection mirrors the read path's
  lowercased prefixes ("error", "file not found") plus the client
  permission-denial phrase and empty/absent result bodies — all
  fail-closed to the honest ack (PR #115 review blocker: the original
  case-sensitive match let denied writes chain, and the verdict framed
  an unapplied fix as verified).
- Forged `[wrote ...]` lines in user prose must not trigger runs — the
  fenced block grammar (v0.18.9) already anchors real blocks; the
  `has_wrote_block` selector reads the same post-boundary tool_call
  structure `_run_blocks` does, not text. Pin with a spoof probe anyway.
- Non-OpenCode clients that never re-POST after a write see exactly
  today's behavior (the chain only exists on the continuation).

## Testing and validation

- Hermetic e2e through the real engine: fix turn → write resume → run
  tool_call emitted → verdict prose; reject-path and non-fix-build
  regressions (terminal ack unchanged); write-failure no-chain; spoof
  probe (forged wrote block in user text).
- Live real-OpenCode at the earliest runnable point (the WP-A lesson):
  seed a repo with a failing test + buggy source; drive "fix the failing
  test" end to end; verify green verdict matches client ground truth,
  and a deliberately unfixable case reports red honestly.
- Ladder: add the fix-execution rung as a new battery turn against a
  seeded red repo. Exit gate: rung green on a clean-rig run with zero
  regressions on the existing 12.
- Trace: assert the chain is diagnosable from the turn trace (three
  passes visible) — the ~280-char node-response truncation found today
  makes post-hoc gate diagnosis impossible and should be fixed or
  bounded before this rung's live validation leans on the trace.

## Named forward directions (not built here)

- **Within-turn re-fix on red** (rung 2): bounded retry carrying the
  client-side failure report, composing with the #100 held-round
  machinery. The reliability-compounding rung.
- **Cross-turn repair** (sibling rung, first clean evidence in today's
  battery): "target file was never successfully built but the
  conversation describes what it should contain" is a deterministic,
  detectable condition; turn 2 of the ladder could rebuild todo.py fresh
  instead of cascading. Design separately; do not fold into this rung.
- **Runner generalization**: cargo test for the plexus half of the
  meta-task rung — the command builder and verdict parser are the only
  pytest-aware seams (unchanged claim from the run-half design).
