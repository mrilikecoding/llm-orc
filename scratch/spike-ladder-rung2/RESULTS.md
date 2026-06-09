# Ladder rung 2 results — axis A, depth-3 write-only (2026-06-08)

$0 local, qwen3:14b via Ollama /v1, n=10 per state. The **real production
composition** (real judgment seat call 1 → `strip_verdict` → real
`_seat_filler_messages(remaining_anchor=...)` call 2), driven through the real
`LoopDriver.decide()` — both composing factors exercised at depth 3.
Task: `string_utils.py` + `test_string_utils.py` + `README.md` (3 deliverables).

## Result

| State (produced) | advance | churn | delegated | no-tool/finish | judge verdict |
|------------------|---------|-------|-----------|----------------|---------------|
| A (module) | **8/10** | 0/10 | 9/10 | 1/10 | REMAINING 10/10 |
| B (module+test) — the deep test | **9/10** | 0/10 | 9/10 | 1/10 | REMAINING 10/10 |
| C (module+test+readme) | — | 0/10 | 0/10 | finish 10/10 | **COMPLETE 10/10** |

## Verdict — PASS (axis A is not the binding constraint at depth 3)

1. **The anchor holds with depth.** State B (the deepest REMAINING point — 2
   prior deliverables in the action record) advances **9/10**, at/above the
   Spike ρ depth-2 statement-only reference (8/10). Advance does **not** degrade
   from depth 2 to depth 3.
2. **The Finding G failure mode does not reappear.** Churn = **0/10** at both
   REMAINING states; **first-churn = never** (∞). The seat-filler never
   re-targeted an already-produced deliverable — the WP-LB-L anchor fix scales
   to depth 3.
3. **Convergence is clean.** State C judged **COMPLETE 10/10** → text-only
   finish. No false-continue once all deliverables are produced.
4. **Delegation preserved** (9/10 both REMAINING states); no-tool-call ~10%
   (1/10 each — the premature-finish risk, consistent with rung-1's 2/10,
   backstopped by re-judgment on the next trailing turn + the AS-3 cap).
5. **Cloud-contrast trigger does NOT fire** (advance 8–9/10, both > 7/10). Axis
   A holds locally on qwen3:14b; no cloud spend. The pre-committed threshold
   (≤7/10) kept the call honest — it would have fired had advance degraded.

## Secondary finding (the instrument, not the mechanism) — METER COVERAGE GAP

All 30 rows stamped `turn_shape=carry`, including the State A/B generation turns
that delegated and produced a file (9/10). **The delegation rate (WP-LB-J)
under-instruments multi-file sessions.**

- **Root cause.** On a REMAINING turn, `LoopDriver.decide` passes
  `remaining_anchor or _user_task` to `classify_turn`. `remaining_anchor` is the
  judge's *descriptive* statement ("the README has not yet been produced"),
  which carries no generation verb → classified `carry`. The turn is a
  generation turn by construction (the framework is steering the seat-filler to
  produce the named remaining deliverable), but the classifier infers shape from
  the anchor's surface phrasing and misses it.
- **Impact.** In a multi-file session only the first turn (the user "Write…"
  ask) counts as `generation`; turns 2..n — the actual multi-file work — are
  excluded from the rate's denominator. A regression where the trailing
  REMAINING turns stop delegating would NOT show in the rate. The meter
  instruments single-turn delegation well and multi-file delegation poorly —
  the opposite of where ADR-038's risk lives.
- **Severity.** Does not affect rung-2's PASS (advance is measured from the
  seat-filler's target, not the meter). It is a hole in the regression-visibility
  net, not in the convergence/advance mechanism. Candidate WP-LB-J follow-up.
- **Fix sketch (interacts with axis B — do not land blind).** "REMAINING anchor
  present → generation" is correct for write-only (axis A) but would mis-stamp a
  *read* turn in a mixed read-then-write flow (axis B) as generation. The right
  fix needs axis-B evidence first: a REMAINING turn is a generation turn only
  when the steered action is a write. Options to weigh: (a) classify over the
  original user task on REMAINING turns (correct for axis A, wrong for axis B);
  (b) defer the shape to the action taken (conflates denominator with numerator
  — rejected); (c) a generation-framed anchor the classifier can read; (d)
  accept the gap as a documented limitation until axis B characterizes mixed
  flows. Recommend deciding after axis B, not before.

## Limitations

- The judge's remaining-naming accuracy at depth 3 is inferred from the advance
  rate (advancing to an unproduced file requires a correct name), not isolated
  as ρ did at depth 2 (20/20). The probe captured the verdict, not the statement
  text — a re-run with statement capture could isolate it, but the 8–9/10
  advance is strong composite evidence.
- States are reconstructed (pre-populated action record + conversation), not a
  single live multi-turn session. The WP-LB-L discharge already showed the live
  2-file session; this isolates the deeper decision points faithfully.
- n=10, single task family, qwen3:14b — same scope caveats as ρ/θ.

## Disposition

Axis A (deliverable count) holds at depth 3 — not the binding constraint;
the ladder proceeds to **axis B (mixed read-then-write)**. The meter coverage
gap is a recorded finding + candidate WP-LB-J follow-up, best decided after
axis B (the fix interacts with the `carry` read turn axis B introduces). The
P1 repair-turn scoring clarification remains required before axis C.
