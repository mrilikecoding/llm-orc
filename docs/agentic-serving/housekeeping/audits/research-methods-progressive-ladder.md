# Research Design Review — Progressive Task-Shape Ladder (Cycle 7, post-WP-LB-J)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-progressive-ladder.md` — full draft (design principles, escalation axes, measurement, rung 2 proposal, benchmark-leverage assessment, open questions)
**Constraint-removal response included:** n/a (ladder design; no ADR-082 constraint-removal artifact in scope)
**Date:** 2026-06-08
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4, ρ-audit calibration bar applied)

---

## Summary

- **Design elements reviewed:** 5 (pass/fail criterion operationalization, axis-isolation assumption, rung-2 first-escalation choice, benchmark-leverage assessment, measurement instruments)
- **Flags raised:** 7 (1 P1, 4 P2, 2 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Per-Element Review

### Element 1: Pass/fail criterion — "advance through all deliverables AND converge"

**Belief-mapping.** The criterion carries forward the WP-LB-L joint condition verbatim: one `dispatch start` per distinct deliverable (no churn) AND a final COMPLETE with text-only finish. This is appropriate for axes A and B. What would the researcher need to believe for a different criterion to be more productive? They would need to believe that the advance/converge criterion is sufficient across all four axes — which it is not cleanly for axis C (repair-shaped). A repair turn is meter-classified as `boundary_excluded` and explicitly excluded from the delegation-rate denominator; but the pass/fail criterion as written ("advance through all deliverables") does not define how the repair turn is scored. If the repair is the turn that blocks advance — the model fixes the bug but does not proceed to write the test — does the criterion fire as "failed to advance" or as "boundary_excluded turn, advance continues on the next non-repair turn"? The design does not specify.

**Embedded conclusions.** The "advance through all deliverables" framing presupposes that deliverables are enumerable and distinguishable at the start of the task. For axes A and B this holds: the task text names the files. For axis C (read-modify-write: "fix the bug in X, then add a test"), the repair turn modifies an existing deliverable rather than producing a new one. Whether the repair counts as "a deliverable advanced" or as a context-setting turn before the test-file deliverable is a classification judgment the criterion does not resolve. This is not fatal for rung 2 (axis A, write-only), but must be resolved before an axis C rung runs.

**Scope.** Appropriate for rungs on axis A. Needs pre-specification before axis C.

---

### Element 2: "Vary one axis at a time" — the interaction-effect assumption

**Belief-mapping.** §3 principle 2 and §8 open question 1 both name the interact-effect concern: "real failures may only surface when axes compound." The design names this as an open question but does not specify what evidence would close it — that is, it acknowledges the concern without creating a decision rule for when to run a compound rung. What would the researcher need to believe for a different decomposition to be more productive? They would need to believe that axis interactions produce failure at lower depths than single-axis escalation would expose — specifically, that depth-3 write-only (rung 2) passes while depth-3 mixed-read-write (a compound of axes A+B) would fail. If the mechanism fails at depth-3 only in the compound condition, single-axis escalation through A then B in sequence would miss the failure mode until much later in the ladder.

The design's "crawl before walk" principle (§3 principle 1) provides the practical justification: stop and read evidence before the next rung, so if rung 2 passes cleanly the evidence informs whether a compound rung is worth inserting before the full axis B rung. This is a defensible adaptive strategy. But the design does not say this explicitly, leaving the interaction-effect concern as a named open question without a resolution path. The absence of a stated resolution path means the open question will arrive in the build-the-next-rung decision without a pre-specified trigger for when compound testing is warranted.

**Embedded conclusions.** The single-axis decomposition embeds the assumption that axis-specific failures are diagnosable from single-axis runs alone. The ρ spike review raised an analogous concern (P1-B: the spike could not distinguish remaining-work content from trailing-token perturbation as the causal mechanism). Here the risk is symmetric: a pass on axis A (rung 2) does not tell you whether axis A at depth 3 is genuinely clean or whether the anchor's hold at depth 3 is compensating for a weakness that axis B would expose. The decomposition buys diagnosis clarity at the cost of potentially underestimating compound failure risk.

**Scope.** The decision to start single-axis is defensible at this stage of the ladder (there is only one characterized rung; compound designs require two characterized axes to compound). The gap is not the decomposition choice itself but the missing trigger condition for introducing compound rungs.

---

### Element 3: Rung 2 proposal — axis A depth-3, write-only as the first escalation

**Belief-mapping.** The rung-2 choice varies only deliverable count (depth 2 → 3), holding task type (write-only) and model (qwen3:14b) constant. What would the researcher need to believe for a different escalation to be more productive as the first probe beyond rung 1? They would need to believe either: (a) depth-3 write-only does not expose the anchor's hold limit and axis B (mixed read-then-write) would be more informative, or (b) the local→cloud boundary is a more consequential variable than depth, and running rung 2 locally while deferring cloud contrast until axis A is fully characterized misses a capability contrast that matters earlier.

The case for starting on deliverable count is strong: it is the most direct extension of the rung-1 characterization (the WP-LB-K Run 1 failure was at depth 2, so depth-3 is the minimal escalation of the proven weak axis). The case against is that it sequences all of axis A (depth 3, 4, 5) before touching the judge's deliverable-accounting logic (axis B's `carry` turn classification), and axis B is where the WP-LB-L scope note explicitly flags "carry side not yet tested." If axis B's read-deliverable classification is broken, that failure is invisible through all of axis A.

**Embedded conclusions.** The rung-2 proposal does not embed a conclusion about the overall failure shape, but it does embed the ordering assumption that axis A failure (anchor's hold depth) is more likely and more consequential than axis B failure (judge's deliverable-accounting for read turns). This ordering assumption is not stated as such in the design, and the evidence does not clearly support it over the alternative. The WP-LB-L discharge explicitly called out the carry side as an uncharacterized risk; that risk is currently deferred until axis A is exhausted.

**Scope.** See P2-C finding. The rung-2 choice is reasonable but the ordering rationale needs to be made explicit and evaluated against the carry-side risk.

---

### Element 4: Benchmark-leverage assessment (§7)

**Belief-mapping.** The assessment concludes: do not run benchmark suites (wrong target + wrong cost); borrow task shapes and SlopCodeBench's degradation-measurement framing. What would the researcher need to believe for a stronger benchmark-leverage move to be more productive? They would need to believe one of: (a) an existing benchmark's task taxonomy is better-vetted than the four-axis decomposition the ladder uses, making it worth running even a subset; (b) SlopCodeBench's degradation-measurement framing is actually applicable to the in-loop mechanism, making quantitative comparison of the ladder's degradation curves against published results informative; or (c) Agent Psychometrics' difficulty calibration is robust enough to inform where on the task-difficulty axis qwen3:14b's capability ceiling sits, which would tell the ladder designer when to escalate to cloud.

The target-mismatch argument is sound as stated: SWE-bench, Terminal-Bench, τ²-Bench, and Aider Polyglot all score end-to-end task success — whether the final repository state passes hidden tests. The ladder scores whether the in-loop progress mechanism (the two-call composition + anchor) advances through deliverables and converges. These are not the same measurement. A session could advance and converge perfectly (the ladder passes) while producing buggy code (SWE-bench fails). Running SWE-bench on ladder tasks would produce results the ladder cannot interpret as mechanism evidence, and the cost ($50–$500 per benchmark suite run) is not justified.

**Embedded conclusions.** The target-mismatch argument as written presupposes that the "in-loop mechanism" and "end-to-end correctness" are independent enough that mechanism-level validation is meaningful without correctness validation. This is reasonable for the current stage (the ladder is a controlled probe, not a production readiness claim), but the design should make this independence assumption explicit: the ladder's passing rungs do not imply the agent is production-ready; they imply the composition mechanism works. This is already stated in §1 ("Not under test: end-to-end code correctness"), but the benchmark-leverage assessment does not cross-reference it, leaving the "we don't run benchmarks" conclusion potentially readable as "we don't need external validation" rather than "external validation is available but addresses a different question."

**Scope.** The assessment's core recommendation (borrow shapes + framing, do not run suites) is well-reasoned. See P2-A finding on SlopCodeBench leverage and P2-B on Agent Psychometrics.

---

### Element 5: Measurement instruments — secondary measures and events-alone claim

**Belief-mapping.** The four secondary measures are: delegation rate over `generation` turns, boundary-excluded share, no-tool-call rate, and judge remaining-naming accuracy. The events-alone claim is that the WP-LB-J meter can compute all of these from emitted events without log archaeology. What would the researcher need to believe for a different measurement design to be more productive? They would need to believe that at least one of the secondary measures cannot be computed from events alone, or that a secondary measure the design omits is more diagnostic for the failure modes the ladder is probing.

The events-alone claim depends on the meter correctly classifying every turn's `tail_kind` and `judgment_verdict` from the serve log's event stream. This claim was validated for the two-deliverable depth in WP-LB-J and WP-LB-L. Whether it extends cleanly to depth 3+ and to the `carry` / `boundary_excluded` turn shapes the meter classifies but has not yet seen in production is not confirmed.

**Embedded conclusions.** The "events-alone measurement" design principle (§3 principle 4) embeds the assumption that the meter's classification covers all the turn shapes the ladder will encounter. For axis A (all `generation` turns) this is safe. For axes B and C, the `carry` and `boundary_excluded` turn shapes are new in the production path — the meter classifies them by design, but the design does not specify a validation step that confirms the meter correctly classifies a real carry or repair turn before those axes run. If the meter misclassifies a read turn as a `generation` turn, the delegation-rate denominator is inflated and the secondary measure is wrong without the error being visible in the events stream.

**Scope.** The events-alone claim is appropriate and supported for axis A. The gap is a missing validation step for the meter against novel turn shapes before axes B and C run.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The design treats the rung-1 probe and the WP-LB-L discharge as prior art correctly — naming their scope, quoting their results, and using them to define the passing baseline. The four-axis decomposition is grounded in those measurements. No narrowing-without-prior-art-treatment detected.

One narrowing concern: the design sequences all of axis A before axis B, and the axis-B `carry` turn is the scope note explicitly carried from WP-LB-L. The narrowing is not premature in the sense that it skips prior-art examination — it is premature in the sense that it may defer the most undercharacterized risk (carry-side deliverable accounting) until after axis A is fully mapped. This is flagged as P2-C.

### Incongruity surfacing

One incongruity between the ladder's design and the ρ spike's findings that the ladder does not surface for examination.

The ρ spike's P1-B finding identified that the rung-1 probe's hardcoded anchor contained both a naming statement ("the remaining deliverable not yet produced is test_string_utils.py") and an imperative directive ("Produce test_string_utils.py next"). The spike's production form (judge's real REMAINING statement) is statement-only. The progressive ladder is designed to run with the production form — the anchor is always the judge's actual output, as the spike defined it.

But the ladder's depth escalation (axis A, rung 2: depth-3) will test the judge's remaining-naming accuracy at depth 3 for the first time. At depth 2, Spike ρ is measuring this (ρ.1 hypothesis). At depth 3, the judge must name one of two remaining deliverables correctly, and the naming accuracy at that depth is entirely uncharacterized. The rung-1 probe used a hardcoded anchor at depth 2; ρ will measure judge-produced anchor accuracy at depth 2. But rung 2 (the first ladder run) uses the judge-produced anchor at depth 3, before ρ has established the production-form naming accuracy at depth 2.

The incongruity: ρ characterizes depth-2 naming accuracy; rung 2 runs at depth-3 naming accuracy. If ρ and rung 2 run in parallel (or if rung 2 runs before ρ completes), rung 2 will produce depth-3 advance results but the depth-2 baseline for the production anchor will not yet be established, making it impossible to attribute a rung-2 failure to depth-3 naming degradation versus depth-3 anchor sensitivity.

The ladder does not surface this sequencing dependency. It refers to ρ as grounding but does not specify whether rung 2 depends on ρ completing first.

### Coverage gaps

The design does not address:
- The local→cloud transition trigger: §8 names this as an open question but does not specify what qwen3:14b rung-2 outcome would trigger cloud escalation. A design choice this consequential for the cost budget should be pre-specified.
- The stopping rule: the design says "stop and read evidence before the next rung" but does not specify what rung-2 outcome pattern would cause the ladder to terminate (vs. continue to rung 3) or change axis (vs. deepen on axis A).

### Recommendations

Prioritized by severity:

1. **(P1) Resolve the axis C pass/fail operationalization before any axis C rung runs.** The criterion "advance through all deliverables AND converge" is unspecified for repair turns. The minimum pre-specification: a `boundary_excluded` turn does not count against advance (it is excluded from the criterion's deliverable-accounting, consistent with its exclusion from the delegation-rate denominator); advance is assessed over the `generation` turns only. This requires a one-paragraph addition to the criterion definition.

2. **(P2-A) Strengthen the SlopCodeBench leverage.** The design recommends borrowing "degradation-measurement framing" but does not specify what that means in operational terms for the ladder. SlopCodeBench's core measurement — iterative task performance as a function of turn count — maps directly onto the ladder's secondary measure "no-tool-call rate: does it worsen deeper?" before that measure was characterized. The ladder would benefit from: identifying one specific SlopCodeBench-derived measurement protocol (e.g., tracking the first-deviation turn as a function of depth), adopting it as a secondary measure for rung 2 onwards, and noting the deviation from benchmark-suite-as-instrument. This does not require running the benchmark; it means naming the specific analytical move being borrowed.

3. **(P2-B) Pre-specify the cloud-escalation trigger.** The free-first principle is clear; the trigger for moving from local to cloud is not. The design says "escalate to cloud only to contrast a more capable seat where local hits a ceiling worth contrasting against." "Hits a ceiling" is not operationalized. Pre-register: if rung 2's advance rate on qwen3:14b is ≤ N/10, that constitutes a local ceiling and a single cloud-contrast run is authorized. If advance is ≥ N/10, axis A continues locally. The trigger N should be set before rung 2 runs so the cloud decision is not post-hoc.

4. **(P2-C) Make the axis-B sequencing rationale explicit and evaluate it against the carry-side risk.** The design sequences axis A before axis B without stating why axis A failure is more likely or more informative than axis B failure. The WP-LB-L discharge note explicitly flags the carry side as uncharacterized. One sentence in §6 stating the ordering rationale (e.g., "axis A is the minimal escalation of the characterized weak axis; axis B's carry classification is uncharacterized but orthogonal to anchor depth — axis A failure at depth 3 would require redesign that changes the test conditions for axis B anyway") would close this.

5. **(P2-D) Add a sequencing note on the ρ spike dependency.** Rung 2 runs the production anchor at depth 3. Spike ρ establishes the production-anchor baseline at depth 2. If rung 2 runs before ρ completes, a depth-3 failure cannot be attributed to naming degradation vs. anchor sensitivity (no depth-2 baseline for the production form). The design should note whether rung 2 depends on ρ's ρ.1 result, and if so, what minimum ρ.1 result is required before rung 2 is interpretable.

6. **(P3-A) Add the trigger condition for compound rung insertion.** §8 open question 1 names the interaction-effect concern. Close it with a stated trigger: "If rung N on axis A passes but by a margin ≤ X/10 over the pass threshold (suggesting the mechanism is near its hold limit), a compound rung (axis A + B simultaneously) will be inserted before continuing axis A escalation." Without this trigger, the interaction-effect concern will arrive at each rung as an unresolved design decision rather than a pre-specified adaptive rule.

7. **(P3-B) Pre-specify a stopping rule.** "Stop and read evidence before the next rung" is a crawl-before-walk discipline, not a stopping rule. Specify: what rung-2 outcome pattern (e.g., advance < 4/10, systematic churn at depth 3) would close axis A and route directly to the ADR-037 amendment loop-back rather than continuing the ladder? A stopping rule prevents the ladder from continuing to run after it has already answered its question.

---

## Findings

### P1 — Design flaw that would invalidate a conclusion before running

**P1: Axis C pass/fail criterion is unspecified for repair turns.**

As written, the criterion "advance through all deliverables AND converge" does not define how a `boundary_excluded` repair turn is scored. If the model completes the repair correctly but does not then produce the test-file deliverable in the same turn (which axis C's "fix the bug in X, then add a test" task shape requires), the criterion could fire as "failed to advance" when the actual failure is a separate deliverable-sequencing issue, not a mechanism failure. Conversely, a criterion that treats `boundary_excluded` turns as exempt could pass a session where the model produced the repair but the test-file advance never happened — which is also not a pass.

This is a P1 for axis C specifically. For rung 2 (axis A, write-only), the criterion is unambiguous. The P1 does not block rung 2 from running; it must be resolved before any axis C rung is pre-registered.

**Recommended design change:** Add a criterion clarification paragraph to §5 before axis C is pre-registered: "`boundary_excluded` turns (repair-shaped turns) are scored as follows: they do not count as 'deliverable produced' (the meter's exclusion from the delegation-rate denominator is mirrored in the pass/fail criterion); advance is assessed over the `generation` turns only. A session passes if every non-repair deliverable receives exactly one `dispatch start` (no churn) and the session converges. A session fails if any `generation` turn re-targets an already-produced deliverable OR if the repair turn is not followed by a `generation` turn that advances to the next deliverable within the session."

---

### P2 — Weaknesses that bound the claims

**P2-A: SlopCodeBench leverage is underspecified.**

The design recommends borrowing SlopCodeBench's "degradation-measurement framing" but does not specify what operational move that means. The reference remains aspirational rather than actionable. If the ladder finishes axis A and the practitioner wants to compare degradation curves against SlopCodeBench's published results, the comparison is undefined because the ladder has not pre-specified a measurement that maps onto SlopCodeBench's iterative-performance-vs-turn metric.

The specific leverageable move: SlopCodeBench tracks the first turn at which the agent deviates from the task specification (re-revises an already-correct file, produces output inconsistent with the prior state). The ladder's "churn" event (re-targets an already-produced file) is exactly this measure. Pre-specifying "for each rung, record the first-churn turn (if any) as a function of deliverable count" makes the ladder's data directly comparable to SlopCodeBench's degradation curves without running the suite.

**Recommended design change:** Add one secondary measure to §5: "first-churn turn (the turn index at which the first re-target of an already-produced deliverable occurs, if any; `none` for sessions that advance cleanly)." This measure is computable from events alone, costs nothing additional, and provides the SlopCodeBench-comparable degradation signal the §7 assessment says it wants to borrow.

---

**P2-B: Cloud-escalation trigger is unspecified.**

The design's free-first principle is clear and correct. The trigger for moving from local characterization to cloud contrast is not pre-specified, which means the escalation decision will be made post-run without a pre-committed standard. This is the same gap that the ρ spike's P2-B finding identified for the Conditional-Acceptance band motivation: a post-run decision rule can be captured by motivated reasoning in either direction (escalate too early because the result is interesting; defer escalation to protect the free-model characterization).

The cloud-escalation decision is consequential for the $5 cap. Pre-specifying the trigger before rung 2 runs removes the post-hoc ambiguity.

---

**P2-C: Axis-B sequencing rationale is unstated.**

The design defers axis B (mixed read-then-write) until axis A is characterized. The WP-LB-L discharge note explicitly carried the `carry` turn (read-then-write) as an uncharacterized risk. No rationale is given for why axis A depth escalation is sequenced before the carry-side characterization. The rationale may be correct (axis A failure would require anchor redesign that changes axis B's test conditions) but it should be stated explicitly rather than left implicit. Without a stated rationale, the sequencing looks like premature narrowing to the write-only case.

---

**P2-D: Rung 2 / Spike ρ sequencing dependency is unstated.**

Rung 2 tests the production anchor at depth 3. Spike ρ will establish the production-anchor baseline at depth 2. If rung 2 produces a depth-3 advance rate below the rung-1 baseline (8/10 hardcoded), the natural interpretation is "anchor degrades at depth 3." But that interpretation requires knowing the production-anchor rate at depth 2, which ρ is measuring. If rung 2 runs before ρ completes, a depth-3 result below 8/10 is uninterpretable: it could be that the production anchor is weaker than the hardcoded anchor at depth 2 (ρ's question) and the depth-3 escalation compounds that, or it could be that depth-2 production rate matches 8/10 but depth-3 genuinely degrades it.

The design does not name this dependency. If ρ and rung 2 are intended to run sequentially (ρ first), that should be stated. If they can run in parallel, the rung-2 interpretation protocol should include a clause: "If ρ.2's depth-2 advance rate is not yet known, rung 2 depth-3 results are held pending ρ completion before the axis A characterization is written."

---

### P3 — Improvements

**P3-A: No stated trigger for compound rung insertion.**

§8 open question 1 names the interaction-effect concern correctly. The cost of the single-axis discipline is that compound failures are invisible until they are run. The compound-rung insertion trigger turns this from an unanswered open question into a pre-specified adaptive rule. Without it, the interaction-effect concern must be re-litigated at each rung as a fresh decision.

**P3-B: No stopping rule.**

"Stop and read evidence before the next rung" describes the inter-rung review process, not a stopping rule. A stopping rule specifies what evidence closes the ladder (either "mechanism holds across all four axes" or "mechanism fails at axis X depth N, route to ADR loop-back"). Without a pre-specified stopping rule, the ladder could run indefinitely without reaching a conclusion the roadmap can act on.

---

## Overall Verdict

**Sound to proceed on rung 2 — with four pre-run clarifications before rung 2 is pre-registered.**

The ladder's core design is well-grounded. The four-axis decomposition is correctly derived from the WP-LB-L scope note and the rung-1 probe. The passing baseline (WP-LB-L: depth-2 advance-then-converge) is correctly characterized. The free-first principle, events-alone measurement discipline, and composition-layer probe method all carry forward from the ρ spike with appropriate fidelity.

The P1 finding (axis C criterion unspecified for repair turns) does not block rung 2, which is write-only. It must be resolved before axis C is pre-registered.

The four P2 findings each require one or two sentences added to the design before rung 2 runs:

- **P2-B** (cloud-escalation trigger): pre-specify the advance-rate threshold that authorizes a cloud-contrast run — this protects the $5 cap from post-hoc expansion.
- **P2-C** (axis-B sequencing rationale): one sentence explaining why axis A is sequenced before the carry-side characterization — this prevents the sequencing from looking like narrowing at the post-run review.
- **P2-D** (ρ spike dependency): specify whether rung 2 requires ρ.1's depth-2 advance-rate result before rung 2's depth-3 results are interpreted — this protects the attribution of any depth-3 degradation.
- **P2-A** (SlopCodeBench leverage): add "first-churn turn" as a secondary measure — this is the specific operational move that makes the §7 degradation-framing aspiration concrete, adds no cost, and produces benchmark-comparable data.

The two P3 findings (compound-rung trigger, stopping rule) improve the ladder's usability as a decision instrument and should be added when the design is finalized, but they do not block rung 2 execution.

The benchmark-leverage assessment (§7) is sound. The target-mismatch argument is valid and the recommendation (borrow shapes + framing, do not run suites) is well-reasoned. The only gap is that "borrow the degradation-measurement framing" is aspirational without P2-A's operationalization. With P2-A applied, the assessment's recommendation becomes actionable.
