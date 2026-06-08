# Progressive Task-Shape Ladder — design (Cycle 7, post-WP-LB-J)

**Status:** METHODS-REVIEWED (`housekeeping/audits/research-methods-progressive-ladder.md`) —
rung 2 sound to run; four P2 clarifications applied below; one P1 (repair-turn scoring)
held until axis C. Pending practitioner go/no-go + the cloud-escalation threshold (§6). No runs yet.
**Date:** 2026-06-08
**Instrument:** Delegation Rate Meter (WP-LB-J, `delegation_rate_meter.py`) — the rate
and the boundary-excluded share are now computable from emitted events alone.
**Grounding:** WP-LB-K Run 1 (multi-file weak link) + the rung-1 probe
(`scratch/spike-multifile-progress/`) + the WP-LB-L discharge (2-file advance+converge).

---

## 1. What the ladder is, and is not

The ladder is a **proactive, controlled escalation of task shape** to find where the
seat-filler's multi-deliverable progress and the digest's expressiveness break —
before production false-stops surface it (the FC-67 trailing trigger is the reactive
complement). It is the named "after the gate passes" work in roadmap §WP-LB-K.

**Under test:** the in-loop *progress mechanism* — does the session advance through all
deliverables and converge, under the two-call composition (ADR-037) + remaining-work
anchor (ADR-038) + framework-owned digest (Session Action Record)? This is axis-2
(long-horizon coherence), the recorded load-bearing risk (ADR-033 §6b, ADR-097).

**Not under test:** end-to-end code correctness (owned by the capability ensemble +
calibration gate + PLAY) — the judge's deliverable-accounting standard explicitly scopes
correctness out. The ladder measures *whether work advances and stops*, not whether the
generated code is right.

## 2. The passing baseline (rung 1)

WP-LB-L discharge: a **2-file write-only** task ("string_utils.py + test_string_utils.py"),
real OpenCode 1.15.5 + qwen3:14b, advanced (turn 1 → file 1, turn 2 REMAINING → file 2,
no churn) then converged (turn 3 COMPLETE → text-only finish). Scope: qwen3:14b,
file-write deliverables, two-deliverable depth. The weak link Run 1 exposed is qwen3:14b's
multi-file action-selection; the anchor fixed it at depth 2. The ladder asks: how far does
that hold?

## 3. Design principles

1. **Crawl before walk.** One escalation rung at a time; stop and read the evidence before
   designing the next. Do not pre-specify past the next learning point (per roadmap).
2. **Vary one axis at a time** from a characterized baseline, so a failure diagnoses the
   axis, not a confounded task shape (the WP-A scar / Spike θ vary-one-thing discipline).
3. **Free-first.** Local qwen3:14b ($0) characterizes each rung first; escalate to cloud
   models (**~$5 cap authorized**, report actuals, stop at cap) only to contrast a more
   capable seat where local hits a ceiling worth contrasting against.
4. **Events-alone measurement.** The meter (WP-LB-J) is the instrument — no log archaeology.

## 4. Escalation axes (mapped to the meter's turn shapes)

Each axis stresses a distinct part of the mechanism. They map onto the three turn shapes
the meter classifies, so the instrument reads each axis directly:

| Axis | Task shape | Stresses | Meter turn shape |
|------|-----------|----------|-----------------|
| A — deliverable count | 3 → 4–5 write-only files | digest length; anchor's hold as the action record grows | `generation` |
| B — mixed read-then-write | "read X, then write files consistent with it" | judge's deliverable-accounting (a read is context, not a deliverable); discharges FC-61's carry-side | `carry` (read) + `generation` (writes) |
| C — repair-shaped | read-modify-write ("fix the bug in X, then add a test") | a turn the meter *excludes* from the denominator — does the loop still advance through it? | `boundary_excluded` (repair) |
| D — multi-part / mid-session refinement | a new ask injected mid-session | the `new_user_task` tail (ADR-037's recorded boundary) | (re-classified per the new ask) |

## 5. Measurement (per rung)

- **Primary (pass/fail):** advance-through-all-deliverables (one `dispatch start` per distinct
  deliverable, no churn) AND converge (final COMPLETE, text-only finish, client loop ends) —
  the WP-LB-L joint criterion, scaled to the rung's deliverable set.
- **Secondary (from the meter + serve log, events-alone):**
  - delegation rate over `generation` turns (the WP-LB-J numerator/denominator);
  - boundary-excluded share (denominator-degradation signal — rises as axis C/D enter);
  - no-tool-call rate (the rung-1 limitation: 2/10 premature-finish risk — does it worsen deeper?);
  - judge remaining-naming accuracy (Spike ρ measured ≈1.0 at depth 2 — does specificity hold deeper?);
  - **first-churn turn** (P2-A — operationalizes the SlopCodeBench leverage): the turn index at
    which the first re-target of an already-produced deliverable occurs (∞ if none). Events-alone,
    zero cost, and directly comparable to SlopCodeBench's long-horizon degradation curves — the
    concrete form of the framing §7 borrows.

**Repair-turn scoring (P1 — held until axis C).** A `boundary_excluded` repair turn is excluded
from the delegation-rate denominator by design; the deliverable-advance accounting must separately
say whether such a turn counts as a deliverable produced. This does not affect axes A/B (write/read
only); it is a required clarification paragraph before any axis-C rung is pre-registered.

## 6. First rung to run (rung 2) — proposed

**Axis A, depth 3, write-only:** "Write a python module, a test file for it, and a short
README describing it" (the WP-LB-K Run 1 intended shape, deliverable count legible in the
task text). Varies *only* deliverable count from the passing baseline. n≈10 local
(qwen3:14b), then a single real-OpenCode confirmation run if the composition-layer probe
advances. Falsifier: if depth-3 churns (re-targets an already-produced file) the way depth-2
did pre-anchor, the anchor's hold is depth-bounded and axis A is the binding constraint
before B/C/D are worth running.

**Depth-2 production-form baseline (P2-D — dependency satisfied).** Spike ρ is complete: it
measured the *production* anchor form (the judge's real statement, not the rung-1 hardcoded
string) at depth 2 — ρ.2 statement-only advanced 8/10 (B2) + 9/10 (B3), remaining-naming 20/20.
Rung 2 runs the same production form at depth 3, so a below-baseline depth-3 result is
interpretable as depth degradation (not confounded with the production-vs-hardcoded-anchor gap
the rung-1 probe left open). The ρ P1-B distinction — rung-1's anchor carried a naming statement
*plus* an imperative, the shipped form carries the statement + the fixed "Produce that next."
imperative (ADR-038) — means rung 2 already runs the shipped composition; cite ρ.2 as the depth-2
reference when reading rung-2 results.

**Cloud-escalation trigger (P2-B — pre-committed before the run, not after).** Free-first means a
pre-registered threshold, else the local-vs-cloud call is made post-hoc against a known result
(the motivated-reasoning capture the ρ audit flagged). **Proposed (practitioner to confirm):** if
rung-2 local advance is **≤ 7/10** (at or below the rung-1 hardcoded-anchor 8/10 baseline, i.e. no
worse-than-baseline tolerance), a single ~$5-capped cloud-contrast run is authorized to separate
"qwen3:14b seat ceiling" from "mechanism ceiling." Above 7/10, axis A holds locally and the ladder
proceeds to axis B without spending.

**Axis sequencing rationale (P2-C).** All of axis A precedes axis B because an axis-A failure
(the anchor not holding at depth) would require an anchor/composition redesign that changes the
conditions axis B is tested under — characterizing depth first keeps axis B's read-then-write test
on a settled composition rather than a moving one.

## 7. Benchmark leverage (the practitioner's question — assessment, not a commitment)

**The major agentic benchmarks measure a different target.** SWE-bench Verified, Terminal-Bench
2.0, τ²-Bench, and Aider Polyglot score **end-to-end task success** (did the agent resolve the
issue / pass hidden tests). The ladder measures the **in-loop progress mechanism** (does the
seat-filler advance + converge under the framework composition), with code correctness
explicitly out of scope. So these benchmarks **do not drop in as the harness** — they would
test the whole agent's correctness, not the mechanism, and full runs are expensive (wrong
target + wrong cost profile for a $0-local-first ladder).

**What is genuinely leverageable:**

1. **Task-shape taxonomy as rung templates.** The benchmarks' task structures (multi-file
   feature-add, read-then-modify, fix-then-test) are a vetted source of rung shapes with
   external validity over synthetic constructs — borrow a *handful* of representative shapes,
   not the suites.
2. **SlopCodeBench** (arXiv 2603.24755 — "Benchmarking How Coding Agents Degrade Over
   Long-Horizon Iterative Tasks") is the closest methodological cousin: it measures the exact
   failure mode the ladder probes (degradation over iterative turns — the re-revise-file-1
   signature). Its degradation-measurement framing could inform the ladder's measurement design.
3. **Agent Psychometrics** (arXiv 2604.00594 — task-level performance prediction) for
   calibrating rung difficulty relative to known model capability.

**Recommendation (for practitioner decision):** borrow task *shapes* + SlopCodeBench's
degradation-measurement framing; keep the ladder's own advance+converge+meter metric as the
measurement (it is mechanism-specific; the benchmarks are not). Do not run benchmark suites.

## 8. Open questions (for the methods reviewer + practitioner)

- Is "vary one axis at a time" the right decomposition, or do real failures only surface when
  axes compound (e.g. depth-3 *and* mixed read-write)? The single-axis discipline buys clean
  diagnosis but may miss interaction effects — what is the cost of that bet here?
- Is the composition-layer probe (the rung-1 method — real `_seat_filler_messages` bytes, no
  client) sufficient for each rung, or does each rung need a real-OpenCode confirmation run (cost)?
- Where is the local→cloud boundary? Characterize qwen3:14b across all axes first, then one
  cloud contrast — or escalate to cloud the moment local hits a ceiling on axis A?
- Does borrowing benchmark task shapes import their assumptions (e.g. their notion of "a task")
  in a way that biases the ladder toward shapes the framework already handles?
