# Convergent fix: read the target, iterate toward it

**Status:** Designed 2026-07-12 (brainstormed with the practitioner;
approved). Not yet implemented. Extends the chained fix-execution rung
shipped in v0.18.12 (`docs/plans/2026-07-10-fix-execution-design.md`),
which added fix pass -> run pass -> verdict pass but stops at an honest
red verdict with no re-fix.

## Problem

A fix turn today reads nothing about expected behavior before building,
and a red client-side verdict terminates the turn. Two costs, both seen
in battery turn 13:

1. The fix invents its own contract. Turn 13's seat added the guard with
   its own exception message ("scale of empty sequence") where the
   seeded test expects "no values" — the fix never saw the test, so it
   guessed the spec and guessed wrong.
2. A red verdict is a dead end. The verdict already carries the exact
   expected-vs-actual mismatch (turn 13 proved the precision), but
   nothing consumes it. The turn ships a known-red artifact and stops.

The roadmap names the dominant hard-turn failure as spec-freedom
divergence: independently resampled tests and code disagree on choices
the spec leaves open. Whole-file regeneration on retry (the #100 held
round freezes the tests but regenerates the code wholesale) re-draws
those open choices every round, so a regen "fix" can introduce a new
disagreement with the held tests. Convergence beats resampling here:
keep round 1's choices, close only the measured gap.

## Decisions

- **Rung 1.5: target-read (deterministic, on the fix pass).** When a fix
  turn targets `<stem>.py` and `test_<stem>.py` is available
  (client-visible or already in the record), insert ONE deterministic
  read round for it before the gated build, so the build sees the
  expected behavior. Rides the existing need-files read seam. Stem ->
  test-name is a closed template (`test_` + a charset-checked stem),
  never model text. No `test_<stem>.py` present -> skip, exactly today's
  behavior. One read, bounded.

- **Rung 2: convergent re-fix (on a red verdict), routed on failure
  shape.** The `run-verdict` shape, on a red result for a fix-led turn,
  routes to a new re-fix step instead of terminating. What the re-fix
  does depends on a deterministic classification of the pytest output:

  - **Structural failure -> no re-fix, today's honest-red report.**
    Signals (deterministic, from pytest output): a collection `ERROR`, a
    `NameError`/`ImportError`/`SyntaxError` in the traceback, zero tests
    collected, or every test failing. Editing a structurally-broken base
    wastes a round; refuse honestly and let the cross-turn rebuild rung
    (named-forward) handle it. This path is also the latency guard.

  - **Localized failure -> one re-fix round.** Signals: at least one test
    passed, the failures are `AssertionError` / exception-type / message
    mismatches, and the failing count is at or below a small threshold.
    The re-fix producer is a two-rung ladder:

    - **Deterministic edit where the verdict pins the fix.** The narrow,
      safe case first: a `pytest.raises(..., match="X")` or
      expected-message assertion failed and the actual value is present
      in the captured output -> deterministic string-literal replacement
      in the source. This is the round-2 test-repair machinery
      (`docs/plans/2026-07-10-test-repair-round-2-design.md`) pointed at
      source instead of tests. The pinnable set starts at this single
      case and extends only from fixture evidence.
    - **Model edit otherwise.** The code-writer seat regenerates given
      its prior code, the specific failure report, and the visible test,
      instructed to change only what the failure requires. Framed as
      "produce the corrected full file; the gate judges it," not as
      `old_string`/`new_string` surgery — convergent in spirit, with no
      malformed-edit failure mode. Still a sample, but a
      narrowly-targeted one; the accept gate backstops it.

- **One re-fix round, hard bound.** Mirrors #100's held round: red
  verdict -> one re-fix attempt -> re-gate through the accept executor ->
  emit. Green -> honest green; still red -> honest red with the failure
  report. No unbounded loop.

- **Delivery is a full `write`.** The serve already holds the corrected
  file from its own sandbox, so it emits a whole-file write. The client
  `edit` tool (#122) is a later plumbing optimization (large files, not
  clobbering client-side edits) and is out of scope here. This keeps the
  change to the existing write/gate/emit seams.

- **No new catalog shape.** Rung 1.5 reuses the need-files read seam;
  rung 2 reuses the accept executor and the code-writer seat. The only
  new surfaces are the failure-shape classifier, the deterministic-edit
  inference, and the run-verdict -> re-fix route.

## Turn flow (fix-led turn, all passes stateless from the wire)

1. **Fix pass:** target-read (rung 1.5) reads `test_<stem>.py` if present
   -> gated build sees the expected behavior -> emit write. Reject path
   unchanged.
2. **Run pass (existing):** client applies the write, re-POSTs ->
   `need-run` -> one `pytest -q` bash tool_call.
3. **Verdict pass:** run result arrives -> `run-verdict` parses pytest's
   summary.
   - Green -> honest green, terminal.
   - Red + structural -> honest red, terminal (today's behavior).
   - Red + localized -> **re-fix pass**.
4. **Re-fix pass (new, localized only):** deterministic edit or model
   edit produces a corrected candidate -> re-gate through the accept
   executor -> emit write.
5. **Run + verdict again:** the re-fixed write runs and reports honestly,
   green or red. The one-round bound stops here.

## Bounds and error handling

- Exactly one re-fix round per turn; the re-fix pass never re-triggers
  itself (a `has_refixed` guard, symmetric with the existing
  `has_wrote_block`/`has_run_block` idempotency).
- Structural classification fails closed: anything the classifier cannot
  confidently call localized is treated as structural (no re-fix), so a
  misparse costs honesty-preserving caution, never a wasted or wrong
  edit.
- Deterministic-edit inference fails closed to the model-edit path: if
  the literal cannot be unambiguously located in the source, fall
  through to the model edit rather than guess.
- Latency: the re-fix pass adds one build + run + verdict cycle. The
  780s battery cap is already tight against the seat's 720s two-round
  budget; the structural -> no-refix path bounds the worst case, and the
  re-fix round budget is one.

## Testing and validation

- Hermetic e2e through the real engine: fix -> red localized verdict ->
  deterministic-edit path -> green; fix -> red localized verdict where
  the literal is not pinnable -> model-edit path -> green; fix -> red
  structural verdict -> honest reject, no re-fix (regression: today's
  behavior); the one-round bound (a still-red re-fix reports honestly and
  does not loop); a spoof probe (a forged red verdict in user prose
  cannot trigger a re-fix — the classifier reads the run-result block
  structure, not text).
- Rung 1.5 in isolation: fixing `<stem>.py` with a visible
  `test_<stem>.py` reads it before building; no `test_<stem>.py` present
  skips the read.
- Live real-OpenCode at the earliest runnable point (the WP-A lesson):
  the seeded `buggy.py` + `test_buggy.py` — drive "fix the bug in
  buggy.py" end to end; verify the seeded-red test goes green within the
  one re-fix round, and a deliberately unpinnable/unfixable case reports
  red honestly.
- Ladder: exit gate is battery turn 13 converting (seeded-red goes green
  inside the re-fix round) on a clean-rig full rerun, zero regressions on
  the other 12. Trajectory row appended.

## Named forward directions (not built here)

- **Widened deterministic-edit set:** more pinnable failure classes
  (off-by-one constants, a wrong comparison operator surfaced by an
  assert diff) as fixtures justify them.
- **Cross-turn rebuild** (sibling rung, already named in the
  fix-execution design): "target file was never successfully built but
  the conversation describes what it should contain" is a deterministic
  condition; a later turn rebuilds fresh instead of cascading. The
  structural-failure counterpart to rung 2's localized re-fix. Design
  separately.
- **`edit`-tool delivery (#122):** surgical client-side edits for large
  files and to avoid clobbering client changes, once WS-3's edit
  delegation lands.
- **More than one re-fix round:** only if ladder evidence shows a class
  that a single round consistently half-fixes.
