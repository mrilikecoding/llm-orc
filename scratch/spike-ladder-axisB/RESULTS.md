# Ladder axis B results — mixed read-then-write (2026-06-08)

$0 local, qwen3:14b via Ollama /v1, n=10/state. Real production composition
(judge call 1 → anchor → seat call 2) driven through the real `decide()`.
Task: **read config.py, then write settings_loader.py + test_settings_loader.py**
(one read [setup] + two write deliverables).

## Result

| State (done) | first/next action | advance | churn | delegated | judge verdict | turn_shape |
|--------------|-------------------|---------|-------|-----------|---------------|------------|
| R0 (nothing) | read 10/10 | — | 0 | 0 | (first turn) | generation 10/10 |
| R1 (read) | write 9/10, read 1/10 | 9/10 | 0 | 9/10 | REMAINING 10/10 | carry 10/10 |
| R2 (read+module) | write 9/10, finish 1/10 | 9/10 | 0 | 9/10 | REMAINING 10/10 | carry 10/10 |
| RC (read+2 writes) | finish 10/10 | — | 0 | 0 | COMPLETE 10/10 | carry 10/10 |

## Verdict — PASS (the mixed read-then-write flow works end to end)

1. **Read-first adherence: 10/10.** The seat-filler reads `config.py` before
   writing — it honors "read X, then write …" and never jumps to a premature write.
2. **FC-61 carry-side discharged.** The judge treats the read as context, not a
   deliverable: REMAINING 10/10 at R1 (read done, nothing written → writes remain),
   and converges COMPLETE 10/10 at RC (the read in the record neither blocks nor
   false-counts). This is the outstanding carry-side assertion from the WP-LB-K
   acceptance gate (the Run-2 leading-read incidental).
3. **Advance + converge hold with a read in the record.** Advance to each write
   9/10, churn 0/10, first-churn never; convergence COMPLETE 10/10.
4. no-tool-call ~10% (R2 1/10; the premature-finish risk — backstopped by
   re-judgment + AS-3 cap). One R1 re-read (1/10) — a minor inefficiency, not
   churn of a write deliverable.

## Meter finding — now confirmed BIDIRECTIONAL; the fix is a hybrid, not a one-liner

`turn_shape` is wrong in both directions, consistently (10/10 each):

| turn | action taken | stamped | correct | why wrong |
|------|-------------|---------|---------|-----------|
| R0 mixed first turn | read | generation | carry | instruction "…write module…" has the gen verb; action is a read |
| R1/R2 REMAINING | delegated write | carry | generation | judge's descriptive anchor ("X not produced") has no gen verb |
| RC | finish | carry | carry | (correct) |

An **action-based** signal corrects both errors (read→carry, write→generation,
finish→carry) and preserves the rate's purpose: a literal-write *carry* (the C1
"should've delegated but inline-wrote" failure) is still a `write` action →
generation denominator, no numerator → the rate drops. The premature-finish
failure is not a delegation failure — it is already instrumented by the
finish-policy fields (`judgment_verdict`), not the delegation rate.

**But a PURE action-based fix is insufficient.** A `write` action can be either:
(a) a delegated generation (C1 denominator — should count), or (b) a legitimate
grounded carry of an observed value ("write the test output to results.log" —
should stay `carry`). Telling these apart still needs the instruction signal
(the observed-carry check), and the repair/uncovered-domain `boundary_excluded`
cases are instruction-derived too. So the correct fix is a **hybrid**: action
(write vs read/finish) AND instruction (observed-carry, repair, uncovered-domain).

This is a deliberate meter redesign (its own TDD cycle, and the labeled-set
fixtures may need re-framing from "classify the instruction" to "classify the
turn given action + instruction"), not an inline patch. The `boundary_excluded`
branch wants axis-C (repair) evidence before it is finalized.

## Disposition

Axis B passes; the mixed flow and FC-61 carry-side are sound. Two open items:
- **Meter hybrid-classification redesign** — well-grounded across axes A+B
  (bidirectional error confirmed); finalize the `boundary_excluded` branch with
  axis-C evidence. A WP-LB-J follow-up with its own design.
- **Axis C (repair-shaped)** — needs the P1 repair-turn deliverable-scoring
  clarification (methods review) before it is pre-registered.
