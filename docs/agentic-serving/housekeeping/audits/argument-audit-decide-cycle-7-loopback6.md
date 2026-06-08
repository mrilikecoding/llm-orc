# Argument Audit Report — ADR-038 (Remaining-Work Anchor)

**Audited document:** `docs/agentic-serving/decisions/adr-038-remaining-work-anchor.md`
**Source material:**
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-rho-remaining-work-anchor.md`
- `docs/agentic-serving/housekeeping/audits/research-methods-spike-rho.md`
- `scratch/spike-multifile-progress/RESULTS.md`
**Genre:** ADR
**Date:** 2026-06-08

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (composed ~0.9 estimate; strict-dominance claim; causal isolation; framework-checklist rejection; routing-planner rejection)
- **Issues found:** 6 (P1: 1, P2: 3, P3: 2)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

#### P1-1: The "strictly dominates" claim is not supported by the B3 evidence and the hedging in the Provenance section does not reach the Decision section

**Location:** §Decision / §Rejected alternatives ("Statement-only anchor") / §Provenance check

**Claim:** The rejected-alternatives section states "the imperative is never worse and strictly dominates." The Consequences section lists "delegation preserved (9-10/10 across arms)." Both are framed as endorsements that support the Decision section's adoption of the statement + imperative form.

**Evidence gap:** The research log is explicit: B3 ρ.2-imp produced 9/10 and ρ.2 (statement-only) also produced 9/10 — a tie on the deeper-tail base. The imperative adds value on B2 (10/10 vs 8/10) but is flat on B3. The Provenance section correctly labels "strictly dominates" as "a drafting-time characterization of a modest measured edge, not a population claim," and the rejected-alternatives section itself acknowledges "B3: 9 vs 9 — equal." So the hedge exists, but it is buried in the Provenance section rather than integrated into the claim. A reader who stops at the rejected-alternatives section or the Decision section takes away a stronger assertion than the evidence supports.

More precisely: "strictly dominates" in formal terms means the imperative is at least as good on every base and strictly better on at least one. The B3 tie does not violate that reading — but at n=10/cell, 9/10 vs 9/10 is statistically indistinguishable from 10/10 vs 9/10 (the P2-C caveat in the methods review). The claim of strict dominance implies the imperative's superiority is a reliable property, not a one-base artifact; the single-base advantage on B2 at n=10 cannot bear that weight.

The Provenance section's label does not propagate into the Decision or Rejected alternatives sections, leaving the document internally inconsistent: the claim is hedged in one place and stated without hedge in another.

**Recommendation:** Either move the Provenance hedge into the rejected-alternatives paragraph — "the imperative is never worse and shows a measured edge on B2 (10/10 vs 8/10), with B3 equal (9/10 vs 9/10), meeting the pre-registered ≥ 0.2 gap threshold on one base but not two" — or soften the Decision section's characterization of the imperative choice to "adopted on the measured B2 margin and the zero-cost reasoning, not on population superiority." The current structure gives two inconsistent signals; the Provenance hedge must be visible where the claim is made.

---

### P2 — Should Fix

#### P2-1: The composed ~0.9 production estimate conflates two measurement designs without marking the key measurement-design asymmetry

**Location:** §Consequences (Positive), first bullet / §Empirical grounding

**Claim:** The Consequences section presents "composed ~0.9 progress rate at qwen3:14b, n=20 across two bases; A_current baseline 0/10 advance." The Empirical grounding section repeats this without additional qualification.

**Evidence gap and arithmetic legitimacy concern:** The research log's Results section carries the composed estimate as: "Factor 1 (judge names correct) ≈ 20/20 × Factor 2 (imperative-anchored advance) ≈ 19/20 → the end-to-end multi-file progress rate is ~0.9." The log then marks this: "Labeled composed (the two factors measured on the same trials for ρ.2-imp; ρ.1 measured separately confirms Factor 1 is not the bottleneck)."

This is the asymmetry that requires explicit marking in the ADR: ρ.1 and ρ.2-imp were run as separate arms on the same trials (i.e., ρ.1 runs the judgment call, then ρ.2-imp runs the action call anchored by that same judgment output). Because ρ.2-imp used the real judge output from ρ.1, the "Factor 1 × Factor 2" framing is not independent-measurement multiplication — it is a sequential composition on the same n=10 trials per base. Factor 1 is not a separate draw; it is the upstream of the same pipeline run ρ.2-imp depends on.

This matters because the research log itself acknowledges "ρ.1 measured separately" — but that "separately" refers to ρ.1 having its own arm, not to ρ.1 being an independent draw from a different sample. Every ρ.2-imp trial that advanced did so on a trial where ρ.1 also named correctly (the pipeline ran sequentially). The composed estimate is therefore essentially the end-to-end ρ.2-imp advance rate over trials where ρ.1 was also measured — which at 19/20 is a valid direct reading, not a Factor 1 × Factor 2 multiplication from independent arms. Presenting it as "Factor 1 × Factor 2" invites a misread: a reader may assume the factors were measured on independent samples and that the estimate extends to future production runs where Factor 1 might vary independently of ρ.2-imp's sample. Given that ρ.1 came in at 20/20, the composed estimate is ceiling-dominated by Factor 2, and the multiplication form obscures this.

The ADR already carries a Provenance caveat — "Factor 1 × Factor 2 arithmetic from same-and-separate-trial measurements, not an end-to-end multi-file measurement" — but "same-and-separate-trial measurements" is not a standard phrase and does not make the measurement design sufficiently transparent for a downstream reader.

**Recommendation:** Add a parenthetical clarification in the Consequences section: "(The ~0.9 is the ρ.2-imp observed advance rate; the 'composed' label reflects that ρ.1 ran upstream on the same trials, confirming Factor 1 is not the bottleneck — not that the factors were multiplied from independent samples.)" This matches what the evidence actually shows without overstating the independence of the two arms.

#### P2-2: The causal-isolation argument has a confound the ADR does not acknowledge — the control's content differed from the judge's statement in more than remaining-work information

**Location:** §Rejected alternatives ("Mere trailing-token perturbation") / §Consequences (Positive, third bullet)

**Claim:** "The remaining-work content is causally responsible" because control B2 (content-neutral trailing addition) advanced 0/10 vs ρ.2's 8/10. The Consequences section draws the stronger inference that "the remaining-work content is causally isolated."

**Evidence gap:** The control arm's text, as described in the methods review and the research log, was a paraphrase of the delegation standard — "Remember: delegate generation to a capability ensemble rather than writing inline." The judge's actual statement named a specific file by filename (e.g., "The test file test_string_utils.py has not been created yet.").

These two texts differ in two respects: (1) remaining-work content (file-specific vs. none) and (2) semantic domain (task-progress vs. meta-instruction about delegation style). The control's 0/10 advance rate is consistent with "remaining-work content is the mechanism," but it is also consistent with "the control's delegation-reminder content actively reinforced the stuck pattern by redirecting attention to delegation quality rather than target selection." A control that is semantically about delegation style is not a pure length-and-format match for the judge's task-progress statement; it has a competing positive content (delegation reminder) that may suppress target-switching. A neutral filler of the same length with no positive content at all (e.g., repeated periods, or a content-neutral sentence like "This session continues.") would be a cleaner causal isolation.

The ADR states the control was "a content-neutral trailing addition of the same length and format as the judge's statement, carrying no remaining-work content." This characterization is accurate as far as it goes, but the control's actual content (delegation reminder) is not semantically inert — it may actively pull the model away from target-switching. The 0/10 vs 8/10 gap is large and convincing for the practical go/no-go decision, but the ADR's claim that "remaining-work content is causally responsible" is slightly stronger than the control design strictly supports.

**Recommendation:** The Consequences paragraph on causal isolation should add: "The control's content was a delegation-style reminder rather than a semantically inert filler; the control design therefore refutes 'any trailing text' more cleanly than 'any non-remaining-work trailing text,' and the gap (0/10 vs 8/10) is large enough to be practically decisive even with this qualification." This neither weakens the decision nor misrepresents the strength of the causal claim — it matches what the evidence actually rules out.

#### P2-3: The framework-checklist rejection argument commits a scope equivocation between task-decomposition and deliverable-tracking

**Location:** §Rejected alternatives ("Framework-tracked deliverable checklist")

**Claim:** The framework-checklist alternative is rejected because "'requested deliverables' is exactly the semantic task-decomposition ADR-037 established the framework cannot compute" (quoting ADR-037 §Context's argument that task-text parsing is semantic judgment in disguise).

**Evidence gap — equivocation:** ADR-037's argument in §Context is specifically about determining task-completeness — i.e., whether the current work-product satisfies the intent of the user's task. That is the judgment needed to terminate a session. The framework-checklist alternative proposed in ADR-038 is a different (and potentially easier) operation: once files have been written, diffing the written paths against the originally-mentioned filenames in the task text. These are different operations:

- ADR-037's "semantic judgment" argument: the framework cannot know if a written file is adequate/correct relative to the user's intent.
- Framework-checklist's operation: the framework notes filenames mentioned in the task text and checks them against the written file list.

The second operation is still semantic in the sense that filenames must be extracted from natural-language task text, but it is closer to structured extraction than to completeness-quality judgment. The ADR conflates the two under "semantic task-decomposition" without distinguishing them. This is not fatal to the rejection — the ADR's actual practical argument (the judge already produces this reliably at 20/20, so a checklist would duplicate a trustworthy signal) is stronger and stands independently — but the "semantic in disguise" framing borrows more authority from ADR-037 than is strictly warranted. The equivocation should be corrected so the rejection rests on the stronger ground (redundancy of the checklist given ρ.1's result) rather than the weaker ground (the checklist cannot be computed, which is not the same claim as ADR-037's termination argument).

**Recommendation:** Revise the framework-checklist rejection to separate the two arguments: "The checklist's deliverable extraction is lighter than ADR-037's completeness judgment (it is closer to structured filename extraction than task-quality assessment), but ρ.1's 20/20 result means the judge already does this at ceiling reliability. A deterministic checklist would duplicate a signal that is already trustworthy and would require a separate filename-extraction step — adding complexity the judge dissolves. The semantic-decomposition argument from ADR-037 applies to completeness quality; the checklist's task-parsing is a distinct and lighter operation, but redundancy is sufficient to reject it here."

---

### P3 — Consider

#### P3-1: The Conditional Acceptance discharge condition conflates two separate convergence properties without stating explicitly that both must hold in a single run

**Location:** §Empirical grounding, Conditional Acceptance paragraph

**Claim:** "A real-OpenCode session on a multi-deliverable task in which the session both advances through all deliverables (no churn on file 1) and converges (the COMPLETE finish ADR-037 validated)."

**Observation:** ADR-037's own Conditional Acceptance gate also requires convergence and delegation. ADR-038's gate adds the advance-through-deliverables requirement. The ADR states "it folds into ADR-037's existing Conditional Acceptance discharge (both clear together on the multi-file run)." This is correct but the sequencing implication is underspecified: the multi-file run must demonstrate the REMAINING-with-advance pattern first, then the COMPLETE finish on the final trailing turn. A run that shows advance-to-second-file but never reaches COMPLETE (e.g., because it stalls on the third deliverable) discharges ADR-038's gate but not ADR-037's. The joint discharge condition should make explicit that the COMPLETE finish must follow the advance sequence in the same run, not in separate runs.

**Recommendation:** Add a parenthetical: "(The discharge run must demonstrate the full sequence in one session: REMAINING-with-advance on the first trailing turn, then COMPLETE on the final trailing turn — or equivalently, REMAINING-with-advance for each intermediate deliverable followed by COMPLETE on the final turn. Separate runs for advance and for convergence do not satisfy the joint gate.)"

#### P3-2: The FC (delegation preserved under the anchor) is stated in refutable form but the refutation signal is under-specified

**Location:** §Decision, Fitness criteria ("FC (delegation preserved under the anchor)")

**Claim:** "The anchored call 2 still delegates generation... Refutable: an inline `write` of generated content on an anchored call-2 turn."

**Observation:** The evidence from ρ.2-imp shows delegation at 9-10/10, with some no-tool-call responses. The refutation signal identifies inline `write` as the violation, but does not address the no-tool-call failure mode (the 1/10 none case in ρ.2-imp B3). A no-tool-call response on an anchored call-2 turn is a distinct violation type (the session stalls without delegating or writing inline) that the current FC wording does not capture. The FC as written is technically refutable from the Finding B inline-write shape but is silent on the premature-finish failure mode that ρ.2-imp measured at 1/10.

**Recommendation:** Extend the refutation signal: "Refutable: an inline `write` of generated content on an anchored call-2 turn, or a no-tool-call turn that terminates the action call without delegation."

---

## Section 2: Framing Audit

The framing audit reads the negative space of content selection: what alternative framings the evidence supported that the ADR did not foreground, what findings the source material contains that are absent or underweighted, and what the dominant framing reveals when inverted.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: The anchor's mechanism is interaction-design, not signal-routing**

The ADR's dominant framing is "route the computed signal forward" — the judge has already produced the remaining-work statement; the amendment stops discarding it. This is a framing about information flow (a discard bug being corrected). An alternative framing with equal support from the evidence: the advance rate improvement is an interaction-design result, not a signal-routing result. The model's next-action selection is sensitive to what text appears at the trailing position of the call-2 context. The control's 0/10 is consistent with "remaining-work content matters," but the rung-1 probe's hardcoded anchor also included an imperative ("Produce it next.") that the judge's bare statement lacks. The methods review named this as the "most consequential gap in the design" — the rung-1 hardcoded anchor was a combined naming-statement-plus-directive; the production form is a naming-statement-plus-separately-added-imperative. The B2 edge for the imperative form (10/10 vs 8/10) could be read as the interaction design (imperative placement) doing work independent of the remaining-work content.

Under this framing, the ADR's argument that "the remaining-work content is causally responsible" would be accurate for the naming-statement component but would attribute the marginal B2 gain to the wrong cause (the imperative's directive quality, not the statement's naming quality). The framing does not change the decision — the amendment ships statement + imperative in either framing — but it would change the explanation of which component does the causal work and what the tuning lever is (statement wording vs. imperative wording vs. their combination).

Belief-mapping: what would the reader need to believe for this framing to be right? They would need to believe that the calling model's next-action selection is primarily sensitive to the presence of an imperative directive at the trailing position, and that the filename content is enabling the directive to hit a specific target. The evidence is consistent with this but does not distinguish it from the dominant framing at n=10.

**Alternative framing B: The ADR-037 composition was under-specified for multi-file tasks from the start**

The dominant framing treats Finding G as a runtime discovery — the two-call mechanism was validated for single-file sessions, and multi-file non-convergence emerged from the WP-LB-K acceptance run. An alternative framing: ADR-037's FC (call-2 form preservation), by specifying that the judgment exchange is discarded byte-equal to the E4b composition, explicitly locked in the information loss that Finding G exposed. The discard was not incidental — it was a stated design property. Under this framing, ADR-038 is not a patch for an unexpected gap; it is a correction of a stated design choice whose consequences were predictable from ADR-037's own §Scope (which recorded that multi-file tasks were outside the measured scope, with the boundary "stated, not guessed across").

The ADR nods at this in §Context: "ADR-037 solved termination and, in doing so, starved progress." But the framing remains that of a discovery. The alternative framing would foreground the ADR-037 FC as a predicted consequence point — the discard was a deliberate choice, and a scope that excluded multi-file tasks meant the discard's effect on multi-file progress was deferred rather than unknown.

Belief-mapping: what would the reader need to believe for this to matter? That ADR-037's FC (call-2 form preservation) was not just a measurement-faithful property but a design choice with a predictable consequence in the multi-file regime that was in scope from the start (even if not measured). This framing would strengthen the argument for a more comprehensive scope annotation in future FCs.

**Alternative framing C: The anchor mechanism will require re-grounding as task shapes diversify**

The ADR closes with "the remaining-work content is causally isolated, so the mechanism is understood, not a lucky perturbation — it should generalize as the judge's naming reliability generalizes." This is optimistic generalization from a narrow scope (file-write deliverables, depth two, qwen3:14b). An alternative framing treats the anchor's success as specific to a narrow task class where the deliverable is a named file: the judge can name `test_string_utils.py` specifically because the task text contains that filename. For tasks where remaining work is not a named file but a functional requirement (e.g., "add error handling" or "refactor the module to support streaming"), the judge's one-sentence statement may be less specific and the anchor's advance effect may be weaker. The ADR records this as a boundary, but the optimistic generalization claim ("should generalize as the judge's naming reliability generalizes") conflates the judge's naming reliability in the measured class with naming reliability in the unmeasured class, which may not follow.

Belief-mapping: the reader would need to believe that the judge's naming reliability is primarily a function of its understanding of deliverables-as-entities rather than deliverables-as-named-files.

---

### Question 2: What truths were available but not featured?

**Underrepresented finding 1: The no-tool-call failure mode on ρ.2-imp B3**

The research log records ρ.2-imp B3 as "9/10 advance, 0 stuck, 0 other, 1 none, 9/10 delegated." The 1/10 none case on the deeper-tail base (B3, two trailing tool pairs) is not discussed in the ADR. The Consequences section reports "delegation preserved (9-10/10 across arms)" without noting that B3's 9/10 included one no-tool-call case specifically on the imperative form — the form adopted by the decision. The methods review (P3-B) recommended sub-classifying no-tool-call responses by content (finish/text/other), and the research log records this sub-classification was applied ("none-finish / none-text / none-other"). The ADR does not carry forward what the B3 none case was — whether it was a premature-finish or a stuck-text response, which have different implications for deeper-tail robustness.

This is not a fatal gap (the 1/10 none rate is within the pre-registered 2/10 ceiling), but the ADR's optimism about deeper-tail behavior would be better grounded by acknowledging that the one failure on the deeper base was a specific failure type that should be watched.

**Underrepresented finding 2: The rung-1 probe's 3/10 "other" rate in the unanchored baseline**

The rung-1 RESULTS.md records A_current as "0/10 advance, 7/10 stuck (file 1), [3/10 other]." The ADR's Consequences section characterizes A_current as "0/10 advance" without noting the 3/10 other — which represents sessions that targeted neither file 1 nor the test file (presumably writing to a third path or producing no tool call). This is relevant to understanding what the baseline failure mode actually is, and whether the anchor suppresses the "other" failure mode as well as the "stuck" failure mode. The spike's results show the anchor eliminates stuck cases, but the ADR is silent on what happens to the "other" failure mode under the anchored form.

**Underrepresented finding 3: The ρ.2 statement-only form as a viable fallback**

The methods review's incongruity discussion noted that the statement-only form (8/10 B2, 9/10 B3) passes the pre-registered threshold independently of the imperative. The ADR adopts statement + imperative as the decision but does not name statement-only as a validated fallback — if the imperative wording causes a regression in the real-OpenCode run (e.g., the imperative is too directive and suppresses tool-call formation in a different context), there is a validated simpler form that can be tried without re-running the spike. Naming the statement-only form as a validated fallback form in the Consequences or Empirical grounding sections would make the amendment's design space more legible.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing: "The judge computes the remaining-work signal; the amendment routes it forward instead of discarding it."

Inverted framing: "The judge's signal was always discarded for a reason (information-bounding, context cleanliness); the amendment is accepting a tradeoff that leaks one sentence of judgment output into the action context."

Under the inverted framing:

- The ADR-037 property that "the judgment exchange is discarded — it does not ride into call 2's context" is not just a byte-equality constraint but a principled context-bounding choice. The amendment routes one sentence of judge output forward. The ADR notes: "the judgment question/digest remain discarded; only the stripped remaining-work sentence plus the imperative carry forward." This is correct and the ADR characterizes it as Neutral (the context-bounding property "holds" because only the sentence and imperative cross). But the inverted framing would ask: what is the information-theoretic relationship between the judge's sentence and the judgment exchange it is derived from? If an observer of call 2 can partially reconstruct the judgment exchange from the routed sentence (e.g., inferring that the judge saw file 1 written and did not see the test file), then the boundary is leakier than the ADR's Neutral characterization implies.

- Claims that become weaker: "context bounded regardless of session depth" (ADR-037 §Decision Point 4's cost tiebreak rationale). The session depth does not affect the anchor's size (one sentence + one fixed string), so this specific concern does not apply — the ADR is correct that this is bounded. The inverted framing does not find a practical weakness here, but it reveals that the boundary is now "judgment sentence forward, everything else discarded" rather than "everything discarded," which is a narrower but still meaningful boundary.

- Claims that become stronger: the amendment's practical robustness argument. If the mechanism were "information flow routing," the anchor's failure mode would be the judge providing a wrong or vague sentence. The spike measured this at 0/20 wrong-naming across both bases. The inverted framing (interaction-design lens) would say the robustness comes not from routing a pre-verified signal but from positioning a specific-file reference at the trailing context position where the model's next-action selection operates — which is actually a more stable explanation for why the anchor works reliably at ceiling.

- What the document would need to address: whether the one-sentence routing creates any dependency between the judgment call's content and the action call's behavior that future wording changes to the judge prompt could disrupt. If the judge is re-tuned (per the FC-58 discipline), the remaining-work statement's form may change, and the statement + imperative composition's advance rate would need re-validation. The ADR partially addresses this with "wording revisions re-validate the affected ρ arms" in the Consequences (Negative), but the inverted framing reveals this is now a two-surface tuning constraint (judge question wording + action anchor wording) rather than one.

---

### Framing Issues

**P2-F1: The ADR's optimistic generalization from the spike scope to "should generalize" is underqualified**

**Location:** §Consequences (Positive, third bullet): "the mechanism is understood, not a lucky perturbation — it should generalize as the judge's naming reliability generalizes."

The spike measured one task class (named-file deliverables) at n=10/cell across two bases. The judge's 20/20 naming accuracy on that class is ceiling performance that may not hold for tasks where remaining work is functional (non-file-write deliverables). The ADR records the scope boundary correctly in §Consequences (Negative): "Scope is the θ class (qwen3:14b, file-write deliverables, tails to depth two)." But it then reaches past that boundary in the Positive consequences with "should generalize as the judge's naming reliability generalizes" — importing optimism from the measured class to the unmeasured class in the same section.

The optimistic generalization is not supported by evidence and is inconsistent with the scope boundary stated two bullets later. The Positive bullet should either drop the generalization or scope it: "within the measured class, the remaining-work content is causally isolated — future scope extension will require per-class re-validation of both naming reliability (ρ.1) and advance rate (ρ.2)."

**P2-F2: The ADR presents the routing-planner rejection's "heavier subsystem" characterization without evidence that a planner would materially add complexity in this context**

**Location:** §Rejected alternatives ("Per-task routing-planner")

The routing-planner alternative is rejected as "disproportionate — a heavier subsystem for a problem the judge's already-computed output solves." This is a practical engineering judgment, not an evidence-backed argument from the spike. The spike established that the signal-routing approach works; it did not measure the routing-planner alternative. The "heavier subsystem" characterization rests on an assertion about relative implementation complexity that is not grounded in the source evidence. The confabulation-surface argument ("re-opens the planner-confabulation surface Cycle 6/7 spent effort bounding") is evidence-backed (the prior loops did address planner confabulation) and is the stronger rejection argument.

The issue is not that the routing-planner is the right choice — signal-routing is clearly simpler — but that the ADR presents both a practical argument (simpler) and an evidence argument (confabulation risk) when only the confabulation argument draws on measured evidence. Presenting "heavier subsystem" as a peer argument to the confabulation evidence overstates the engineering-judgment claim's authority.

**P3-F1: The inverted-framing analysis (Question 3 above) surfaces a two-surface tuning dependency that the ADR's Negative consequences underspecify**

**Location:** §Consequences (Negative, third bullet)

The ADR notes: "like ADR-037's, [the imperative] is tunable at the FC-58 evidence bar (wording revisions re-validate the affected ρ arms)." This is correct but understates the dependency: re-tuning the judge's question wording (which generates the statement that is routed forward) also affects the action anchor, because the anchor's content is the judge's output. A practitioner who re-tunes the judge prompt per FC-58 discipline may change the statement's form and thereby change the anchored call-2's advance rate without running ρ.2 explicitly. The FC-58 discipline applies to the judge question; ρ.2 is the instrument for the action anchor. The ADR should note that judge-prompt re-validation (θ-harness) and anchor re-validation (ρ-harness) are now coupled — a judge-prompt wording change requires ρ.2 re-validation as well as θ re-validation.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R1
- P1 count this round: 1 (Section 1: P1-1 strict-dominance inconsistency)
- P2 count this round (new, non-carry-over): 5 (P2-1 composed-estimate measurement-design asymmetry; P2-2 causal-isolation confound; P2-3 framework-checklist equivocation; P2-F1 generalization overreach; P2-F2 routing-planner characterization)
- New framings or claim-scope expansions: interaction-design framing (anchor works via position/directive, not just content routing); under-specification-of-scope framing (ADR-037 FC as a predicted consequence point for multi-file tasks); two-surface tuning dependency (judge-prompt re-validation couples to ρ.2-harness)
- Recommendation: CONTINUE to next round (P1 count > 0)
