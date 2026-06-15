# Research Methods Review — Benchmark Design (Cycle 7 Agentic-Serving)

**Reviewed artifact:** Benchmark/measurement instrument design proposal (inline, provided for review)
**Constraint-removal response included:** n/a — this is not a RESEARCH-loop question-set review; it is a pre-build instrument review
**Date:** 2026-06-15

---

## Summary

- **Targets reviewed:** 6 (construct validity; sampling validity; embedded conclusions / premature narrowing; confound control; tier-comparison validity; additional flags)
- **Flags raised:** 10 (2 P1; 5 P2; 3 P3)
- **Criteria applied:** adapted from the measurement-instrument framework — construct validity, sampling validity, confound identification, embedded-conclusion detection, scope/purpose coherence

---

## Review

### Target 1: Construct Validity

**Question being probed:** Do the five hard-pass metrics (form validity, convergence, content coherence, termination, delegation rate) actually measure "structural reliability" / the north-star claim? Is anything load-bearing unmeasured?

**What the metrics cover well.** Each metric maps cleanly to a named ADR and to a causal mechanism the cycle validated empirically. Form validity (ADR-041) tests the destination-validity gate. Convergence (ADR-040) tests the deterministic completeness gate. Content coherence (ADR-039) tests the content anchor via AST cross-reference. Termination (ADR-037/038) tests the two-call judgment-first composition and the remaining-work anchor. Delegation rate (ADR-036) tests the V3 user-turn guidance composition. Each of these is a structural property the framework is supposed to guarantee deterministically or near-deterministically on the validated stack. The metrics are therefore a reasonable operational read of "did the framework deliver its structural guarantees in this run?"

**The honest scope gap — and it is scope, not a hidden conclusion.**

The briefing explicitly carves out semantic correctness ("the metrics are structural; is that gap honestly scoped or a hidden conclusion?"). The answer is: honestly scoped, and the corpus supports that scoping. ADR-037/040's deliverable-accounting standard explicitly excludes code correctness. ADR-039's construct is cross-file *reference resolution*, not code *correctness*. The north star, as the corpus records it, is "all that remains is ensemble iteration and improvement" — meaning structural delivery is what the framework guarantees, and ensemble quality is explicitly deferred to PLAY and future cycles. So the benchmark measuring structural metrics is consistent with the north-star claim the cycle is actually making.

The risk here is not a hidden conclusion but a labeling risk: if the benchmark's findings are ever reported as "the system works on this task class," a reader not familiar with the "structural only" scope will overread the result. The benchmark should carry an explicit scope statement on every output it produces — something like "structural delivery only; semantic correctness of generated code is out of scope and not measured here." This is a presentation-layer concern, not a measurement-validity concern, but it is worth making part of the harness output format.

**P3-A: The delegation-rate metric is informative for axis-2 ceiling-finding but noisy as a regression gate at the cell level.** ADR-036's refutation threshold is ≥0.9 over a 24-hour rolling window of generation-shaped turns. A single cell in the benchmark grid may have only 2–4 generation-shaped turns, making the per-cell delegation-rate metric statistically uninformative as a pass/fail gate. If a cell at horizon=5 produces 4 generation turns and 3 delegate, is 0.75 a failure signal or sampling noise? The metric was designed for a soak window, not per-cell evaluation. **Design change:** treat delegation rate as a reported-not-pass-gating metric at the per-cell level (consistent with the briefing's "Reported" column) and note in the harness output that the regression-guard use requires a soak window reading, not a per-cell threshold. Do not elevate it to a hard-pass signal at this scope.

**P3-B: The content-coherence metric (AST cross-reference check) carries a prose-exclusion boundary that should be stated explicitly.** ADR-039 validated content coherence for both code and prose (README) deliverables. The AST-checkable resolution applies to code; prose deliverables get a regex-heuristic identifier check. Both are in scope per the discharge gate. The benchmark design should specify: for `.py` files, AST cross-reference (binary, objective); for `.md` files, identifier heuristic (soft, adjudicator-required). Conflating the two in a single binary pass/fail risks the regression-guard read being correct for code while quietly passing a prose coherence failure that would need adjudication.

---

### Target 2: Sampling Validity

**Question being probed:** Does adaptive n=1 coarse → n=3-5 boundary concentrating bias the ceiling estimate? Is "boundary cell" well-defined on a 2D stochastic grid? Is k/n defensible?

**P1-A: n=1 on the coarse pass is insufficient to establish a ceiling frontier — the pass/fail topology is unknown, and a single lucky or unlucky sample can mislabel an entire region of the grid.**

This is a P1 because it affects all three reads simultaneously. The regression guard is safe (targeted cells pass/fail is what matters, not a ceiling estimate), but axis-2 ceiling-finding and tier comparison both depend on correctly locating the pass/fail frontier, which n=1 cannot do reliably.

The core problem: the pass/fail outcome at each cell is stochastic (the briefing acknowledges this). On a 2D stochastic grid, a single sample per cell means each cell's label is a Bernoulli draw with unknown p. If a cell at (horizon=5, complexity=argparse) has a true pass rate of 0.6, it will be labeled FAIL with probability 0.4 and PASS with probability 0.6. The ceiling frontier drawn from n=1 passes can easily be drawn one rung lower than reality (pessimistic), or one rung higher (optimistic), depending on sampling luck. This is not a boundary-sharpening problem that concentrate-at-boundary sampling can fix, because the coarse pass determines where the boundary is — if the boundary is mis-estimated in pass 1, pass 2 concentrates at the wrong location.

The briefing's rationale is cost (5–25 min per cell). That is correct and should govern the design. The right fix is not to raise n in pass 1 across the whole grid but to add a confirmation step: **after pass 1 identifies the apparent ceiling cell C, run n=3 at C itself before declaring C the ceiling and moving to boundary concentration.** If the apparent ceiling C passes 2+/3 at n=3, the ceiling label is credible; if it passes 0-1/3, the ceiling was a lucky sample and the true ceiling is one rung lower. This adds at most 2 extra runs (if the ceiling cell passes 1/3, you need to re-confirm the next lower cell too) and resolves the P1 without changing the adaptive design.

**Design change:** after pass-1 ceiling identification, run n=3 on the apparent ceiling cell. Apply a ≥2/3 threshold to confirm it as the ceiling before concentrating at the boundary. If it fails ≤1/3, call the next lower cell the provisional ceiling and confirm it the same way. This is inexpensive (at most 4–6 extra runs above the n=1 baseline) and removes the mislabeled-ceiling risk.

**P2-A: "Boundary cell" is not well-defined on a 2D grid with stochastic pass/fail, and the concentration strategy needs a definition before it can be executed.**

The briefing says "concentrate samples where they decide the ceiling." In 1D this is straightforward. In a (horizon × complexity) 2D grid, the boundary is a line, not a point, and cells along the boundary may have very different stochasticity profiles (a high-horizon/low-complexity cell may be less reliable than a low-horizon/high-complexity cell). Without a defined criterion for which cells are "adjacent to the frontier," the concentrating-at-boundary strategy is an instruction the harness cannot follow without human judgment at run time.

Proposal: define the boundary as the set of cells where pass-1 labeled the cell PASS but the cell immediately above (on either the horizon or complexity axis) labeled FAIL. In a 2D grid this gives a set of up to N boundary cells (one per row and per column). Run n=3 on this set, apply a ≥2/3 pass threshold, and report the highest-passing cell on each axis as the ceiling on that axis. This makes "boundary" a deterministic function of pass-1 results, executable by the harness without mid-run human direction.

**Design change:** define boundary-cell selection as a computable criterion — specifically, PASS cells with at least one FAIL neighbor on a higher-difficulty dimension — and hard-code that definition into the harness rather than leaving it for run-time human judgment.

**P2-B: The k/n threshold at n=3-5 is not pre-specified, and without a pre-committed threshold, boundary-cell results will be interpreted post-hoc.**

The briefing mentions "k/n threshold at n>1" without specifying k or n. Running n=3 and observing 2/3 passes is not interpretable as a PASS or FAIL without a pre-committed rule. This is the motivated-reasoning capture the prior spike-ρ review flagged (the cloud-escalation trigger must be pre-committed before the run, not after). The same principle applies here.

**Design change:** pre-register the confirmation threshold before any pass-2 runs. Recommended: ≥2/3 for a PASS label (consistent with the cycle's general near-threshold discipline). For the tier-comparison purpose specifically, note whether the threshold should be the same for cheap and frontier configs, or whether "match" means the cheap ceiling reaches the same ≥2/3-labeled cell as the frontier ceiling.

---

### Target 3: Embedded Conclusions / Premature Narrowing

**Question being probed:** Does the grid or metric set bake in assumptions about where the system fails? Does greenfield-first narrow prematurely? Is "three reads from one ladder" sound?

**P2-C: Starting greenfield-only is not premature narrowing given the instrument's scope, but the scope restriction should be stated explicitly as a design choice, not left implicit.**

The axis-2 claim the benchmark is designed to test is "does the framework's structural mechanism hold at increasing horizon and complexity?" That claim is scope-compatible with greenfield tasks. The edit and repair paths are tested but have already been validated by the progressive ladder (axis-B mixed read-write, axis-C repair-shaped — both PASS on qwen3:14b), and the benchmark's role is regression-guarding the structural ceiling, not re-exploring all task shapes.

However: the briefing describes the task type axis as "start greenfield" without stating explicitly that edit/bash/repair shapes are excluded from the initial grid. A downstream reader of the benchmark report may not realize the ceiling finding applies only to greenfield tasks. The repair-churn minor finding (axis-C's 2/10 churn) suggests repair shapes might have a lower ceiling — and the benchmark would miss that if it only runs greenfield.

**Design change:** add a scope statement to the benchmark spec and harness output: "This grid runs greenfield tasks only. Edit/repair shape ceilings are not characterized by this instrument. If edit/bash tool-mapping lands in a future cycle, re-run the repair axis against the same ladder to characterize its ceiling separately."

**P3-C: "Three reads from one ladder" is sound in principle but requires that the ladder design not be over-specified for one read at the expense of another.**

Regression guard needs targeted cells on the structural grid that exercise the known failure modes — the briefing is right that these are well-characterized. Axis-2 ceiling-finding needs the ceiling to be genuinely high enough to find a failure (if the hardest cell in the grid always passes, the instrument only gives a lower bound on the ceiling, not the ceiling itself). Tier comparison needs a frontier config that can be run on the same ladder without modification.

These three reads do not have conflicting design requirements if the grid is wide enough to include cells above the cheap stack's ceiling. If the cheap-local stack passes every cell in the grid, axis-2 ceiling-finding has failed (you know the ceiling is above the grid, not where it is). The grid should include at least one cell that is expected to fail under the cheap config. The complexity column "cross-importing module" at horizon 8-10 seems like a reasonable candidate for this, but the design should state explicitly: "the grid is designed to include cells above the cheap-local ceiling, so that the ceiling is locatable within the grid, not merely above it."

**Design change:** add a design requirement to the harness spec: the grid must include cells at which the cheap-local stack is expected to fail, to ensure the ceiling is locatable within the grid. If pre-run analysis suggests the cheap stack will pass all cells, extend the grid before running.

---

### Target 4: Confound Control

**Question being probed:** Will the form gate / escalation be exercised by the complexity axis? Are there confounds the robustness requirements miss?

**P1-B: The form-directive-masks-bleeds finding makes form validity (ADR-041 metric) near-constant across the grid — it will be 1 (PASS) on virtually all cells, making it uninformative as a ceiling-differentiator and making escalation behavior (ADR-041 recovery + coder-tier escalation) nearly impossible to observe.**

The briefing directly names this: "deliberately undersizing the coder (2B, 0.6B) did NOT reliably reproduce form bleeds — the bare-output form directive kept even a 0.6B model parseable on converters/Calculator/cli.py (0/4 sessions escalated). So form bleeds are narrower/more stochastic than 'small model on a hard file,' and escalation firing is rare/hard to trigger on demand."

This is a serious measurement problem for the benchmark's purpose. The form-validity metric, if it passes on nearly all cells, tells you the gate is not breaking — but it does not tell you whether the gate is being exercised. The difference matters: a metric that is "PASS because the mechanism isn't triggered" is not evidence the mechanism works under load; it is absence of a stress event.

For the regression-guard read, this is acceptable — the regression guard needs to know the mechanism is not *broken*, not that it is actively *exercised*. But for axis-2 ceiling-finding, a near-constant metric is uninformative: it does not differentiate the ceiling.

The escalation-behavior reported metric (did recovery / escalation fire and close on bleeds) will be near-constant-at-zero for the same reason. The benchmark will produce lots of data saying "the gate was never triggered" on most cells, which is neither a failure nor an informative ceiling signal.

**Design change (two options):**

Option A (preferred, $0-local cost): Separate what the benchmark measures from what it exercises. Keep form validity as a regression-guard metric (it should pass on nearly all cells — if it fails, that is a regression). Remove escalation-behavior from the axis-2-ceiling-finding analysis entirely, since it will be near-zero. Instead, identify a synthetic stress condition — specifically, a task at (horizon=5–8, complexity=argparse or cross-importing) run with the cheap coder but with an *intentional* form-bleed injection (a parallel arm where the coder system prompt is adversarially extended to elicit bleed) — to confirm the gate catches bleeds when they occur. This is a separate, targeted probe, not a grid cell.

Option B (adds cost): Add a cheap-coder-downgrade arm (the 0.6B or 2B coder that Spike π established bleeds more) as a secondary arm for a small number of hardest-grid cells only, to create observable bleed events without running the whole grid on the downgraded coder. The briefing already knows 0.6B bleeds more stochastically than 2B; using it on targeted cells gives the benchmark a chance to observe recovery and escalation behavior.

Either option should be decided pre-run and stated in the harness spec.

**P2-D: The unique-session-id harness requirement correctly names the sha256(first-message) collision from Spike η, but does not specify how uniqueness is achieved.**

The Spike η research log identifies the collision as a sha256 of the first user message bleeding action records across headless runs when the same prompt is reused. The fix is any approach that makes the session ID unique per run. Concrete options: append a UUID to the first message, use a combination of timestamp + message hash, or pass an explicit session ID to the serve layer. The harness spec should pin one of these rather than leaving "unique per-session prompt marker" as an underspecified requirement. If the wrong approach is used (e.g., a timestamp with second precision on a machine that runs two cells within the same second), the collision risk re-occurs.

**Design change:** specify the session-ID uniqueness mechanism precisely in the harness spec. Recommended: prepend `[BENCHMARK RUN {uuid4()}]` to the first user message, which is observable in the serve log, verifiable, and cheap.

**P2-E: Model warmup / caching across cells is not addressed by the robustness requirements.**

The fresh-ollama discipline and degradation pre-flight smoke are named in the briefing. But the briefing does not address the ordering of cells across the grid. If cells are run sequentially from lowest to highest difficulty (the natural order), the model's context / KV cache state may differ between early and late cells in ways that are not controlled by a fresh-ollama reset between sessions but by how ollama handles inter-session state. Additionally, the form-directive-masks-bleeds finding showed that marathon-session degradation appeared but was "dismissed" in Spike π because fresh-8b bled at the same rate — but that was a single-file comparison. A 16-cell grid run sequentially without intermediate fresh-ollama resets could accumulate degradation effects not present in a single-file probe.

The briefing's "floor-and-report on marathon degradation" policy is correct directionally. The missing piece is a concrete criterion: if the pre-flight smoke (or a mid-grid smoke) detects degradation, what does the harness do? Options: pause and alert, abort the run, or tag the remaining cells as "run under degraded conditions" and continue. Without a pre-specified response, the harness will either silently continue (producing confounded data) or block on human judgment at an inconvenient moment mid-grid.

**Design change:** add a degradation-response protocol to the harness spec: define what the smoke test measures (a latency threshold? a probe task known to complete in <N seconds under fresh conditions?), and specify the response (recommended: tag-and-continue with a `degraded=true` flag on cells run under degraded conditions, and exclude those cells from the ceiling-finding analysis while preserving them for regression-guard if they pass).

---

### Target 5: Tier-Comparison Validity

**Question being probed:** Is comparing cheap-local vs. frontier on this grid a fair test of the north-star claim? Are there apples-to-oranges issues?

**P2-F: "Match" in the tier comparison is undefined, which makes the comparison's conclusion uninterpretable.**

The north-star claim is that "a cheap-local stack + framework structural reliability matches a frontier single model on long-horizon agentic work." The benchmark proposes to test this by running the same ladder under both configs and comparing ceilings. But what does "match" mean quantitatively?

Options that would be natural conclusions:
- The cheap-local ceiling equals the frontier ceiling on the ladder (same highest PASS cell).
- The cheap-local ceiling is within one rung of the frontier ceiling on both grid axes.
- The cheap-local pass rate at the frontier's ceiling cell is not statistically distinguishable from the frontier's pass rate at that cell.

These are substantially different claims. If the frontier config passes cells at horizon=10 complexity=cross-importing and the cheap config passes cells at horizon=8 complexity=argparse, is that a "match" or not? The benchmark cannot answer this question without a pre-specified definition.

This also interacts with the five hard metrics: the frontier config may pass all five metrics on every cell in the grid while the cheap config fails content coherence on the hardest cells. Is "one metric fails" a mismatch? The briefing describes the pass criterion as "a cell passes when all hard signals hold" — so that would be a formal mismatch. But the cycle's claim is specifically about the *framework's structural reliability*, not about semantic correctness, so a content-coherence failure in the cheap config on the hardest cells might be expected and not a refutation of the claim.

**Design change:** pre-specify the match criterion before running the tier comparison. Recommended: "match" = the cheap-local stack's ceiling on the grid (highest PASS cell on both axes) is within one rung of the frontier ceiling, where "within one rung" is defined as: same or lower on one axis, same on the other. This gives a concrete, falsifiable criterion. Also specify whether content-coherence failures specifically are disqualifying for the "structural reliability" match claim or whether they are expected and excluded from that claim.

**P3-D: The frontier config's own stochasticity is uncontrolled in the tier comparison.**

If the frontier config also has stochastic pass/fail outcomes (which it will, being an LLM-based system), then running n=1 per cell under both configs does not establish whether a cell-level difference is model-tier-attributable or sampling noise. For the tier comparison to be meaningful, the frontier config needs at least n=3 on the boundary cells, the same as the cheap config under pass-2 sampling. Running n=3 on boundary cells for both configs is already implied by the adaptive design, but the briefing should make this explicit for the tier comparison arm specifically, since running n=1 frontier vs n=3 cheap at the boundary would produce an unequal comparison.

---

### Target 6: Additional Flags

**P2-G: The dev-traffic isolation requirement (benchmark sessions tagged to never contaminate a production delegation-rate soak window) is underspecified.**

The briefing names this as a robustness requirement but does not specify the tagging mechanism or how the soak window consumer excludes tagged sessions. The delegation-rate meter (WP-LB-J) reads from emitted events. If benchmark sessions emit `TurnDecision` events to the same sink as production sessions, the soak window computation would need to filter on a session tag. The harness spec should specify: (a) what tag is applied (e.g., a `BENCHMARK=true` label in the serve config or session metadata), (b) which event fields carry the tag, and (c) how the soak-window consumer (the 24-hour rolling window from ADR-036 Decision 3) excludes tagged sessions. If the consumer does not yet have this filter, it is a BUILD dependency the benchmark harness cannot assume is ready.

**Design change:** before the benchmark is built, verify whether the delegation-rate meter and its consumer support session-level tagging. If not, either add the filter to the meter (a BUILD item gating the benchmark) or run benchmark sessions on a separate serve instance with a separate event sink, so contamination is impossible by construction rather than by filter.

---

## Summary Table

| ID | Severity | Target | Finding | Design Change |
|----|----------|--------|---------|---------------|
| P1-A | P1 | Sampling | n=1 coarse pass is insufficient to locate the ceiling on a stochastic 2D grid | Run n=3 on apparent ceiling cell before pass-2 concentration; apply ≥2/3 confirm threshold |
| P1-B | P1 | Confound | Form-validity metric is near-constant; escalation behavior is near-unobservable on the real coder stack | Separate regression-guard from ceiling-finding for form metrics; add a targeted bleed-injection probe or a downgraded-coder arm on hard cells |
| P2-A | P2 | Sampling | "Boundary cell" is undefined on a 2D grid and not computable without a pre-specified criterion | Define boundary cells as PASS cells with a FAIL neighbor on a higher-difficulty dimension; hard-code into harness |
| P2-B | P2 | Sampling | k/n threshold at n=3-5 is not pre-committed | Pre-register ≥2/3 threshold before any pass-2 runs |
| P2-C | P2 | Embedded conclusions | "Three reads / one ladder" requires the grid to contain cells above the cheap ceiling; if all cells pass, ceiling-finding fails silently | Require at least one above-expected-ceiling cell in the grid; if not present, extend the grid before running |
| P2-D | P2 | Confound | Session-ID uniqueness mechanism is underspecified; collision risk from Spike η is known but not fully addressed | Specify the uniqueness mechanism in the harness spec (recommended: prepend `[BENCHMARK RUN {uuid4()}]` to first user message) |
| P2-E | P2 | Confound | No degradation-response protocol; sequential cell ordering can accumulate degradation silently | Add degradation-response policy: tag-and-exclude-from-ceiling vs abort; define the smoke criterion concretely |
| P2-F | P2 | Tier comparison | "Match" is undefined; the comparison conclusion is uninterpretable without a pre-committed criterion | Pre-specify match criterion: cheap ceiling within one rung of frontier ceiling on both grid axes |
| P3-A | P3 | Construct | Per-cell delegation-rate is statistically uninformative as a hard-pass gate at n=1-5 | Keep as reported-not-gating at cell level; note regression use requires a soak-window reading |
| P3-C | P3 | Embedded conclusions | Greenfield-only scope not stated explicitly in the design | Add explicit scope statement to harness output and benchmark spec |
| P3-D | P3 | Tier comparison | Frontier config stochasticity uncontrolled; n=1 frontier vs n=3 cheap is an unequal comparison | Run n=3 on boundary cells for both configs in the tier comparison |

---

## Recommendations in Priority Order

**Before building the harness:**

1. (P1-A) Add the ceiling-confirmation step to the pass-1 / pass-2 protocol. This costs at most 4–6 extra runs and removes the primary sampling-validity threat. Pin the ≥2/3 confirmation threshold in the spec.

2. (P1-B) Decide on the form-metrics strategy: either accept that form validity is a regression-guard metric only (not a ceiling-differentiator) and remove escalation behavior from the ceiling-finding analysis, or add a targeted bleed-injection probe on a small number of hard cells. Do not leave this implicit — the harness should not collect escalation-behavior data it cannot interpret.

3. (P2-A + P2-B) Define boundary cell computably and pre-register k/n. Both can be written into the harness spec in a single paragraph.

4. (P2-F) Pre-specify the match criterion for the tier comparison and write it into the benchmark spec. The criterion determines what claim the benchmark's tier-comparison read can make.

**Before running the benchmark:**

5. (P2-D) Specify the session-ID uniqueness mechanism precisely. One line in the harness.

6. (P2-E) Add a degradation-response protocol. Define the smoke criterion (e.g., "a known 2-deliverable task at horizon=2 must complete in under N minutes as a pre-flight check") and the response (tag-and-continue vs abort).

7. (P2-G) Verify the delegation-rate meter's event sink supports session-level filtering before running benchmark sessions concurrently with a production soak window. If not, use a separate serve instance.

**In the benchmark spec document:**

8. (P3-A) Clarify that delegation rate is reported-not-gating at the cell level, and note the soak-window interpretation requirement.

9. (P3-C) Add the explicit greenfield-only scope statement to the benchmark spec and harness output format.

10. (P2-C) Verify the grid design includes cells expected to fail under the cheap config. If the preliminary design does not, add a horizon=10 / cross-importing cell before running.

11. (P3-D) For the tier-comparison arm, specify that both configs run n=3 at boundary cells.

---

## What is well-designed (do not change)

The harness robustness requirements — unique session ID, per-session serve-log slice, fresh-ollama discipline, degradation pre-flight, dev-traffic isolation — are the right set of controls given the prior harness failures. They address the known failure modes and are grounded in evidence.

The $0-checkable metric derivation (produced files + serve-log slice) is sound and preserves the cycle's free-first discipline. AST cross-reference as the content-coherence check is the right instrument — it is what Spike ξ validated, it is objective, and it does not require adjudication for the code case.

The adaptive sampling design (coarse → concentrate at boundary) is the right architecture for an expensive stochastic grid. The P1-A finding addresses its current under-specification, not its structure.

The three-read framing (regression guard / axis-2 ceiling / tier comparison) is coherent and avoids the trap of designing three separate harnesses. The findings above address the pre-conditions each read requires, not the framing itself.
