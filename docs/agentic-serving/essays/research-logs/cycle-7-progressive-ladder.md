# Progressive Task-Shape Ladder — design (Cycle 7, post-WP-LB-J)

**Status:** LADDER COMPLETE (axes A + B + C all PASS on qwen3:14b, $0 local). The mechanism
question is answered for the file-action subset: delegate + advance + converge holds across depth,
mixed read-write, and repair shapes (only a minor repair churn tail). The meter `turn_shape`
redesign (hybrid action+instruction) is now grounded across all three turn shapes — a WP-LB-J
follow-up with its own TDD cycle. Methods-reviewed
(`housekeeping/audits/research-methods-progressive-ladder.md`). Evidence:
`scratch/spike-ladder-rung2/RESULTS.md` + `…-axisB/RESULTS.md` + `…-axisC/RESULTS.md`.
**Date:** 2026-06-08

## Rung 2 outcome (axis A, depth-3 write-only) — PASS

n=10/state, $0 local qwen3:14b, **real production composition** (judge → anchor → call 2) at depth 3.

| State (produced) | advance | churn | delegated | judge verdict |
|------------------|---------|-------|-----------|---------------|
| A (module) | 8/10 | 0/10 | 9/10 | REMAINING 10/10 |
| B (module+test) — deep test | **9/10** | 0/10 | 9/10 | REMAINING 10/10 |
| C (all 3) | — | 0/10 | 0/10 | **COMPLETE 10/10** |

- The WP-LB-L anchor **scales to depth 3**: advance does not degrade (State B 9/10 ≥ ρ depth-2
  8/10), churn 0/10, first-churn never. Convergence clean (COMPLETE 10/10). Cloud trigger did not
  fire (>7/10) — axis A is not the binding constraint.
- **Secondary finding — meter coverage gap (the ladder caught the instrument).** All 30 turns
  stamped `turn_shape=carry`, including the generation turns that delegated: `classify_turn` reads
  the judge's *descriptive* remaining-statement (no generation verb) on REMAINING turns, so the
  delegation rate counts only first-turn generation and under-instruments the very multi-file
  sessions ADR-038 serves. Candidate WP-LB-J follow-up; the fix interacts with axis B's read turn
  (a REMAINING turn is generation only when the steered action is a write), so decide after axis B.
  Detail + fix options in `scratch/spike-ladder-rung2/RESULTS.md`.

## Axis B outcome (mixed read-then-write) — PASS

n=10/state, $0 local qwen3:14b, real production composition. Task: read config.py, then write
settings_loader.py + test_settings_loader.py.

| State (done) | result | judge verdict | turn_shape |
|--------------|--------|---------------|------------|
| R0 (nothing) | **read first 10/10** | (first turn) | generation ✗ |
| R1 (read) | advance to module 9/10, churn 0 | **REMAINING 10/10** | carry ✗ |
| R2 (read+module) | advance to test 9/10, churn 0 | REMAINING 10/10 | carry ✗ |
| RC (read+2 writes) | finish 10/10 | **COMPLETE 10/10** | carry ✓ |

- The mixed flow works end to end: read-first 10/10, advance 9/10 each write, converge 10/10.
- **FC-61 carry-side discharged:** the judge treats the read as context, not a deliverable
  (REMAINING at R1; COMPLETE at RC — read present, not false-counted).
- **Meter gap now confirmed BIDIRECTIONAL** (10/10 each): read→`generation` (R0), delegated
  write→`carry` (R1/R2). An action-based signal corrects both AND preserves the C1 detection
  (a literal-write carry still counts in the denominator), but a *pure* action signal can't tell
  a C1 inline-write from a legitimate observed-value carry-write — that needs the instruction
  signal too. So the fix is a **hybrid (action + instruction)** meter redesign with its own TDD
  cycle, the `boundary_excluded` branch finalized with axis-C evidence. Detail:
  `scratch/spike-ladder-axisB/RESULTS.md`.

## Axis C outcome (repair-shaped) — PASS

n=10/state, $0 local qwen3:14b, real production composition. Task: fix string_utils.py (buggy
module inlined) + write test_string_utils.py.

| State (done) | result | judge verdict | turn_shape |
|--------------|--------|---------------|------------|
| RC0 (nothing) | repair-write 10/10, delegated 10/10 | (first turn) | **boundary_excluded 10/10** |
| RC1 (module fixed) | advance to test 8/10, **churn 2/10** | **REMAINING 10/10** | carry 8 / boundary_excluded 2 |
| RCc (fixed+test) | finish 10/10 | **COMPLETE 10/10** | carry 10/10 |

- Repair flow engages + delivers (RC0 10/10), the **P1 accounting resolution holds** (the judge
  counts the repair-as-write as the deliverable done — REMAINING at RC1, COMPLETE at RCc), and it
  converges (10/10). `boundary_excluded` confirmed (the third turn shape).
- **Minor finding — first churn in the ladder:** RC1 re-targets the fixed module 2/10 (axes A/B were
  0/10). The "fix the bug" framing keeps the module salient; advance still 8/10, backstopped by
  re-judgment. The ADR-038 anchor is slightly less decisive on repair than on clean writes.
- RC1 also shows the anchor-phrasing fragility directly: the *same* state stamps `carry` 8/10 vs
  `boundary_excluded` 2/10 by whether the judge's statement contained "fix" — the shape must come
  from framework knowledge (action + instruction nature), not the anchor text. Detail:
  `scratch/spike-ladder-axisC/RESULTS.md`.

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

**Repair-turn scoring (P1 — RESOLVED for axis C).** The two accountings are **independent**:
a repair turn is `boundary_excluded` for the delegation-RATE denominator (the instruction is
repair-shaped — multi-step read-then-generate — not clean generation), AND it counts as the
requested deliverable produced for **advance**-accounting. Grounded in the current driver: generation
maps to `write` (the `edit`/`bash` mapping is the deferred LB-3), so a repair is delivered as a write
(full rewrite) of the target file, which the existing judge accounting standard ("a successful write
of a requested file counts as that deliverable being produced") already recognizes. Advance therefore
= the repaired file was (re)written; correctness of the fix is out of scope (owned by the ensemble +
calibration + PLAY), consistent with the ladder's standing not-under-test line. **Carry-forward for a
future cycle:** if LB-3 lands the `edit`/`bash` tool-mapping, the judge accounting standard must be
widened to recognize edits explicitly — flagged for that cycle, not this rung.

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

## 9. Compound rungs (interaction effects) — answering open question #1

The single-axis rungs (A depth, B mixed, C repair) all passed. Open question #1 above
(do failures surface only when axes COMPOUND?) is the compound ladder's target.

**Methods-review redirect (A×C → A×B).** The first compound rung was drafted as A×C
(depth × repair — pull axis C's 2/10 repair-churn thread). The research-methods review
(`housekeeping/audits/research-methods-compound-axc.md`) found a **P1**: at n=10 the
churn comparison (2/10 vs ~4/10) is statistically indistinguishable (z≈0.94, p≈0.17) —
the rung could not answer its own question — and the PASS criterion ("advance ≥8/10")
would pass regardless of churn, hiding the null. The review also flagged a README
content-type confound (P2-A), two unmeasured failure modes (no-tool-call + judge
remaining-naming, P2-B), an unfalsifiable interaction claim (P2-C), and **recommended
A×B over A×C as the sharper limit-finder (P3)**: A×C compounds two already-characterized
mechanisms, while A×B compounds depth with the carry-side read-then-write flow — the
most undercharacterized mechanism in the ladder. Practitioner chose A×B.

**A×B = read + 3 code writes** (axis B's read-then-write deepened from 2 writes to 3;
all-code, no prose confound). Pre-registration + decision boundaries:
`scratch/spike-ladder-compound-axb/probe.py`; the redesign was confirmed to discharge
the prior findings before running (`research-methods-compound-axb.md`, no P1). Powered
states R1/R2/R3 at n=15; tracked no-tool, read-churn, and judge remaining-naming.

**A×B PASSED — the carry-side holds under depth; no interaction-effect limit** ($0 local,
qwen3:14b; full detail `scratch/spike-ladder-compound-axb/RESULTS.md`). Read-first 9/10
(R0); write-advance R1 14/15, R2 12/15, R3 13/15 (all ≥ the 0.80 holds line; R2 sits at
the boundary — the pre-registered middle-band caveat); **carry-side integrity held across
all three remaining states** — read-churn ~0 (1/15 at R1, the same minor re-read axis B
saw), judge clean-naming 14/15 → 15/15 → 15/15 (never counts the read as a deliverable,
FC-61 under depth); convergence COMPLETE 10/10 at RC with zero false-continue. The one
elevated signal: premature-finish (no-tool) rose to ~13% on the deep REMAINING states
(R2/R3 2/15 each), consistent with axis-B's ~10% and non-monotonic in depth, backstopped
by re-judgment + the AS-3 cap. The cloud-contrast trigger (advance ≤ 0.73) did not fire.

**WP-LB-M real-model acceptance (folded in).** The harness dogfooded the just-committed
outcome-derived `turn_shape` stamping. Across all 65 decisions it was correct, validating
the bidirectional bug this log's axis-A/B/C meter findings documented: reads → `carry`
(was `generation`), delegated/inline writes → `generation` (was `carry`), finishes →
`carry`; zero anchor-phrasing fragility.

**Disposition.** First compound rung holds. Candidate next rungs: A×C (now with the
powered n + pre-registered boundary the review requires), deeper depth (4–5 writes),
axis D (mid-session intent refinement), or the live multi-turn trajectory run (the
state-injection limitation shared by all rungs — a BUILD-phase OpenCode validation).
