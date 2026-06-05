# Susceptibility Snapshot

**Phase evaluated:** DECIDE (loop-back #5 — Finding F, session-termination mechanism)
**Artifact produced:** ADR-037 (Session-Termination Mechanism — Two-Call Trailing Composition); ADR-036 partial-update header; domain-model Amendment #18 (AS-9 §Termination-judgment instance annotation + 2 vocabulary terms); scenarios (9 ADR-037 feature block + 5 preservations + Finding F acceptance-criteria table); interaction-specs loop-back #5 additions
**Date:** 2026-06-05

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable relative to LB-3/LB-4 snapshots | The candidate-collapse analysis (all four candidates resolved to one live candidate before any run) entered as agent-originated and was confirmed in one turn. Assertion density here is high within a narrow, well-bounded domain — the mechanism space was genuinely contracted by prior empirical work, not by social pressure. |
| Solution-space narrowing | Ambiguous | Stable | The solution space entered already narrow: three candidates were analytically closeable (c) or merger-equivalent (a collapses into d once the signal is model-judged; b reduces to consequence-enforcement). The spike pre-registered both surviving forms (A and B) and both forms were measured. Narrowing before the spike is analytic, not social; the spike added two well-defined forms and measured them on equal footing. |
| Framing adoption | Clear (one instance, bounded) | Stable | The three-layer scope-of-claim framing is practitioner-prompted but agent-composed. It is labeled as drafting-time synthesis in §Provenance and was independently flagged as framing-question territory in audit R1 (P2-F2, P3-F1). No framing was adopted without examination. The hosted-annotation placement issue (P2-F1) was a genuine framing error caught by the isolated audit and corrected. |
| Confidence markers | Ambiguous | Declining relative to LB-3 | The ADR's scope-of-claim language is notably hedged throughout — "composed from independently-measured arms, not end-to-end," "drafting-time arithmetic," "one pair does not establish portability." The R1 audit found one instance of overconfident scope (AS-9 class claim for termination judgment); it was corrected before the phase closed. No "clearly," "obviously," or "the right approach is" in the final artifact. |
| Alternative engagement | Clear positive | Stable | All five named alternatives received explicit empirical or analytic rejection in §Rejected alternatives. The implicit-judgment variant was rejected on composed-rate arithmetic from measured arms, not on preference. Drop-C3 was rejected by E4 before the loop-back entered. The deterministic-framework-policy alternative received the belief-mapping treatment (Cycle 7's loop-back #3 discipline applied explicitly). The Form A-enriched fallback was not rejected — it was preserved as the measured fallback with cost acknowledged. |
| Embedded conclusions | Clear (two instances, labeled) | Stable | Two embedded conclusions: (1) the deliverable-accounting standard's exclusion of code correctness (round-1 evidence + ADR-035/calibration-gate/PLAY cited; exclusion is driver-derived, not drafting-time); (2) the round-2 question revision ("deliverable accounting, not code QA") was a material composition change made in the round-2 pre-registration, not reviewed by the methods-review subagent. This is the strongest susceptibility signal in the phase — see Finding 1. |

---

## Interpretation

### Overall pattern

The pattern across signals is consistent with **earned confidence with two bounded residual uncertainties**. This is a late-stage DECIDE loop-back entering on two rounds of prior empirical work and a pre-characterized failure space. The narrowing is primarily analytic and measurement-driven, not social. The artifacts show the diagnostic markers of a phase that ran correctly under the cycle's structural controls: pre-registration before any run, methods review before round 1, an isolated argument audit that caught a genuine scope overreach (P2-1, the AS-9 class claim) and a framing error (P2-F1, hosted-annotation placement) that the in-phase agent did not self-correct. The corrections happened in response to the audit, not self-initiated — which is exactly what the isolated audit mechanism is for. That is a positive signal for the isolation design, not a susceptibility finding.

The rapid-compounding risk named in the Cycle 7 RESEARCH snapshot is the most important lens to apply here. This loop-back ran in a single session (entry analysis → spike 2 rounds → ADR → 5 audit rounds → gate on 2026-06-05). The countervailing structure is substantial: pre-registration, methods review, isolated audits, practitioner read-points at the gate. The audit trajectory (R1: 6 P2/5 P3 → R2: 1 P2/1 P3, convergence triggered → R3: clean → R4: 1 P2/0 P3 on the gate insertion) is exactly the trajectory a session under genuine structural friction produces — the audit found real things, the corrections were non-trivial (AS-9 scope annotation required a domain-model amendment; F-θ.2 19/60 figure required tracing back to the arm table; hosted-annotation placement was a framing correction, not a word-swap). The convergence pattern is not zero-resistance.

### Cross-ADR composition check

ADR-037's framing originates from its own driver chain (Finding F + F-ψ″.3 + Spike θ), not from ADR-036. The relationship to ADR-036 is a supersession of one decision (trailing-turn C3 guidance form) with an explicit partial-update header that preserves ADR-036's body intact. No framing element from ADR-036 was adopted by ADR-037 as a conclusion without independent testing. The call-2-as-E4b pin (dissolving the third unmeasured piece) is the closest thing: it inherits ADR-036's measured E4b rate (9/10) as a given without re-measuring. But the pin is explicitly motivated as the minimal-change choice and is labeled as such in the spike design — this is appropriate reuse of a prior measured result, not an unexamined framing adoption.

### Rebuttal-elicitation check on rejected alternatives

Three of the five rejected alternatives have explicitly measured rebuttals (restructured guidance text: E3 + ψ.2 + ω.3b; drop-C3: E4; implicit judgment: E4a/E4b composed estimate). The deterministic framework termination policy alternative was rejected on analytic grounds (no framework-computable termination input; no trustworthy text-only response on override) that were pre-named and belief-mapped before the spike ran — the reject is logically sound and the belief-mapping discipline was applied prospectively, not retroactively. The conditional-composition alternative collapsed into the two-call form analytically; no rebuttal-elicitation failure is present because the collapse is logically sound once the work-remaining signal is established as model-judged.

### The two residual uncertainties

**Residual 1 (the stronger signal): round-2 question revision bypassed methods review.** The round-2 pre-registration records "Methods-review disposition: not re-dispatched — bounded rebuild within the reviewed framework." This is an explicit, visible flag placed in the artifact for this snapshot. The question revision ("deliverable accounting, not code QA") was material: it is exactly the change that moved Form B from 0/10 to 10/10 on work-complete tails (the degenerate-REMAINING failure in round 1). The correctness-exclusion's ownership rationale (ADR-035, calibration gate, PLAY) is cited and is driver-derived, not invented. But the standard's scope — what it excludes from the judgment and why — was not independently stress-tested before the runs. The isolated argument audit (R1 §Section 2, Question 3, inverted framing) examined this and produced the "deliverable-production certificate vs. work-correctness certificate" distinction, which the ADR now encodes correctly in the scope-of-claim. The audit substituted for a second methods review at this point. The question is whether that substitution is adequate. The practical gap is: a methods review would have asked whether "a successful write counts as produced" is the right stopping criterion for task types beyond simple file-write deliverables (explanation tasks, command sequences, multi-step RDD sessions). The ADR records these as out-of-scope; the standard's correctness-exclusion is not guessed across them. This is an honest scope boundary, not a hidden assumption. Assessment: this is a real residual uncertainty that ARCHITECT and BUILD should carry explicitly, not a sycophancy signal. The standard's exclusion of correctness is the correct framing for the committed scope; the question of what replaces it outside that scope is an open problem, not an embedded conclusion.

**Residual 2 (the weaker signal): the digest-expressiveness ceiling is labeled as the practitioner's pre-mortem but the false-stop-share-as-trigger is unlabeled drafting-time synthesis.** The R4 audit caught this (P2-R4-1) and recommended adding it to §Provenance. This is a labeling gap in the final artifact, not an unexamined assumption driving the decision. It does not affect the mechanism's correctness; it affects whether a future ARCHITECT reading the §Consequences Negative bullet knows the trigger criterion is synthesis.

---

## Recommendation

**No Grounding Reframe warranted.**

The phase produced a well-grounded artifact under genuine structural friction. The two residual uncertainties are bounded and labeled (or labelable via a one-line Provenance Check addition). The argument audit caught and corrected the phase's two genuine errors (AS-9 scope overreach; hosted-annotation framing). The rapid-compounding risk is real but was substantially mitigated by the pre-registration, methods review, and isolated audit chain.

---

## Advisory Carry-forwards for ARCHITECT / BUILD

**A (scope-ceiling, carry to ARCHITECT):** The session-action record (conformance finding V-03) is the Conditional Acceptance gating condition and the primary BUILD target. ARCHITECT's allocation question for the digest's home and shape (named in §Consequences Negative) should be resolved before BUILD implements V-03 — the digest home affects the module boundary (Loop Driver extension vs. dedicated store) and the production join (framework dispatch records joined with client tool results). The false-stop share as the extend-on-evidence trigger for digest enrichment is a drafting-time synthesis inference that should be confirmed at ARCHITECT before it becomes a design constraint.

**B (question-text re-validation discipline, carry to BUILD):** The deliverable-accounting question text is the mechanism's performance-critical composition point (the change that moved Form B from 0/10 to 29/30). The ADR references the FC-58 discipline (wording revisions re-validate affected arms) in §Consequences Neutral but does not establish an explicit FC governing question-text changes. BUILD should treat any material change to the judgment question text as requiring re-validation of the affected θ-harness arms before landing — the same bar as guidance-text wording revisions.

**C (AS-3 backstop, carry to BUILD):** Conformance scan note: the BudgetController (AS-3 turn cap) is not wired into the loop-driver path. ADR-037 names AS-3 as "the absolute ceiling" and "the deterministic backstop beneath this mechanism." Without it active, the mechanism's 1/10 false-continue residual has no hard backstop on this surface. This is a pre-existing gap, not introduced by ADR-037, but ADR-037 depends on it. BUILD should wire AS-3 or explicitly record that the backstop is absent while the mechanism is under development.

**D (non-write-shaped deliverables, carry to BUILD / PLAY):** The deliverable-accounting standard explicitly excludes non-write-shaped tasks (explanation tasks, command sequences) from the judgment's measured scope. The termination-observability FC (false-stop share as a signal) is the planned watch mechanism. BUILD should ensure the TurnDecision event shape captures enough context to distinguish false-stops on write-shaped tasks from false-stops on non-write-shaped tasks when they arise in production — otherwise the false-stop share becomes an aggregate signal that cannot route to the right remediation (digest enrichment vs. standard revision vs. scope extension).

---

## Positive Signals

- The argument audit sequence (5 rounds, convergence on iteration 2) produced non-trivial corrections and exhibits the right trajectory — declining finding count driven by genuine fixes, not defensive rewording. The isolated audit mechanism caught the AS-9 scope overreach that the in-phase agent did not self-correct.
- The pre-registration discipline held under single-session time pressure: both the round-1 and round-2 decision rules were pre-registered before any run; the tiebreak was pre-registered on cost before round-2 results existed; the hosted arms were read only after the local verdict was recorded.
- The phase's practitioner read-points generated substantive contributions: the paid-vs-architectural gate question produced the three-layer scope-of-claim (now the ADR's clearest explanatory section); the pre-mortem produced the digest-expressiveness ceiling (now the ADR's highest-value risk bullet). Both are labeled as gate-attributed, not absorbed as if agent-originated.
- The cycle's hold-design-forks-open discipline was applied correctly: Form A-enriched was preserved as the measured fallback (not discarded after losing the cost tiebreak), the non-write-shaped boundary was recorded rather than guessed across, and the digest's home and shape were explicitly deferred to ARCHITECT rather than pre-committed in the §Decision.
