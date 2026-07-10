# Deterministic gate repairs, round 2 (test-quality residual)

**Status:** Approved design, 2026-07-10. Evidence:
`scratchpad test-quality spike, 11 live replays × qwen3:8b` — findings
mirrored below; raw dumps retained per spike-retention discipline at the
session scratchpad (`test-quality-spike/`).

## Evidence (what the spike measured)

11 replays of the battery's two failing build turns (turn1 ×6, turn6 ×5),
per-round capture inside the loop primitive. Accept base rate: turn1 4/6
round-1 (5/6 by round 2), turn6 2/5 round-1, 3/5 never. The held TDD
retry converted **0/4** — every held round carried an unsatisfiable test
or timed out. Across 8 captured rejected rounds: **code wrong 0,
spec-freedom divergence 0, over-specified assertions 0.** The classes,
with counts:

1. **Undefined helper name in tests ×3** — `assert file_exists(...)`
   (NameError; a call to a name bound nowhere — the bare-name sanitizer
   and import injection cannot reach it).
2. **Unconditional `os.remove` setup crash ×2** — setup deletes a file
   that does not exist in the fresh per-test sandbox.
3. **Inverted exception-expectation block ×2 (+1 co-cause)** — the test
   fails a correct implementation that raises as specified.
4. **Adequacy false-reject of mutation-style tests ×1 (+1 contributory)**
   — `add_todo(todos, "x"); assert todos == ["x"]` carries value but the
   value-bearing-assert rule does not count it; the false-reject
   cascaded into a worse fresh resample.
5. **Held-round code_writer timeout ×1** — `build-code-round.yaml`'s
   `code_writer` has no `timeout_seconds`; the 3-call code-generator ran
   under the 300s default on the system's longest input, shipped empty
   code, and turned the one genuinely convertible retry into a
   guaranteed reject.

## Decision

Extend the shipped deterministic repair layer (v0.18.7's isolation +
sanitizer + import injection) with the four repairs the evidence names,
plus the two non-repair fixes. No model judgment added anywhere
(structure beats tiers, measured twice).

1. **Config:** `timeout_seconds: 600` on `build-code-round.yaml`'s
   `code_writer` (bounded by the dispatch/loop budget above it).
2. **Adequacy fix:** `adequacy_check.py` counts a mutation-pattern
   assert as value-bearing — an assert comparing a name to a literal
   after that name was passed to a call in the same test body.
3. **New deterministic repairs** (AST-grounded, in the gather/sanitizer
   layer, echoed into the shipped artifact like the existing repairs):
   a. **Unbound-callable excision:** a test function that CALLS a name
      bound nowhere (not module, not import, not defined in tests or
      code, not a builtin) is excised per-test. No name mapping —
      guessing `file_exists → os.path.exists` risks changing intent;
      excision is honest. The executor already runs per-test isolated;
      3+ of 4–6 tests were good in every observed case.
   b. **Unguarded-removal guard:** a bare `os.remove(<literal>)` /
      `os.unlink(<literal>)` statement (not inside try/with) is wrapped
      in `contextlib.suppress(FileNotFoundError)` (import injected).
   c. **Inverted exception-expectation rewrite:** the observed
      anti-pattern (a test body that calls the target inside
      `try/except <E>` and asserts failure IN the except path, or
      asserts `False` after an expected-raise call) rewrites to the
      canonical `with pytest.raises(<E>):` form. Only exact AST
      signatures rewrite; anything ambiguous is left alone.
4. **Adequacy re-check after excision:** repairs run BEFORE the
   adequacy check; if excision leaves no value-bearing test, the round
   still rejects honestly (the gate never gets weaker than "at least
   one adequate test ran").

## Bounds

- Every repair has an exact static signature; no heuristic rewrites.
- Repairs are echoed into the shipped test artifact (the client sees
  what actually ran — the v0.18.7 convention).
- Excision is bounded: if it would drop ALL tests, reject unchanged.
- The wrong-accept class is the review target (v0.18.7 lesson): every
  repair must be provably behavior-preserving for correct tests — the
  reviewer brief names this explicitly.

## Ruled out by the evidence

Elicit-then-build (0/8 spec-freedom), code-first ordering swap (code was
right 7/7 non-timeout), N-sample test voting (the dominant defect is a
systematic verbatim tic voting would retain, at k× rig cost), model
escalation (still dead, consistent with the 4-arm spike).

## Follow-up filed separately

Accepted-artifact quality is ungated (an accepted round carried a
duplicated `def` with junk empty-string behavior) — issue to open, not
this PR.

## Validation

TDD per repair with the spike's verbatim exemplars as fixtures; full
suite; live replay of the two spike turns (expect turn6's never-accept
rate to drop); recorded ladder rerun + trajectory row; independent
adversarial review with the wrong-accept class as the named target.

## Validation round (2026-07-10, 7 replays on the repaired branch)

The repairs fired live: turn1_sv1 accepted round 1 WITH an in-flight
rewrite; turn6_sv5 converted end-to-end (2 excisions + 1 rewrite +
the fixed timeout → held round accepted a previously-fatal shape). Two
defects the validation surfaced, both fixed on-branch:

- **Runner crash on DID-NOT-RAISE:** pytest's `Failed` derives from
  `BaseException`, so a repaired `pytest.raises` test whose expected
  exception never raised crashed the runner child (fail-closed, but the
  report was a truncated traceback — no evidence for the retry round).
  The runner now catches `BaseException` (KeyboardInterrupt/SystemExit/
  GeneratorExit still propagate).
- **Attribute-form coverage gap:** every remaining turn6 reject
  (sv1/sv3/sv4) was `except json.JSONDecodeError:` — the dotted form the
  Name-only rewrite signature skipped. Widened; the declared-expectation
  check matches the dotted name's last segment.

Second review round (independent adversarial reviewer, wrong-accept as
the named target) found and we fixed three wrong-accept vectors before
merge: substring expectation-matching ("Unexpected ValueError" inverted a
correct guard; "Error" matched inside "KeyError"), anywhere-in-body
removal wrapping (neutered post-call existence assertions), and
lambda-param/except-alias blindness in the binding scan (excised a good
test). Post-fix seams, both fail-closed and accepted: negation tokens
anywhere in a message refuse the rewrite ("Expected TypeError not
raised" now costs a round instead of converting — tighten to
expectation-adjacent negation if ladder evidence demands), and a setup
removal after an `os.`-using fixture statement goes unwrapped when the
code also binds `os` (a missed repair, never a wrong accept).

Observed residual (1/7, not repaired here): import-guard boilerplate —
tests wrapping the deliverable import in try/except and asserting
`False, "<module> not found or function not implemented"` in the guard,
which fails both rounds identically (turn1_sv2). Deterministic signature
exists (strip the guard, let a real ImportError report honestly); left
for ladder evidence to justify.

One rewrite-ambiguity note for reviewers: a test whose handler declares
"Expected FileNotFoundError" but whose NAME says "returns empty list"
(self-contradictory authorship, turn6_sv5 r1) rewrites per the declared
message, fails against return-empty code, and converges via the held
round regenerating raising code. Honest both rounds; the ambiguity costs
one round, never a wrong accept.
