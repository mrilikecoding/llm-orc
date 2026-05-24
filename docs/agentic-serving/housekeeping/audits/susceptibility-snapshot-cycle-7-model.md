# Susceptibility Snapshot

**Phase evaluated:** MODEL (Cycle 7 — 2026-05-22)
**Artifact produced:** `docs/agentic-serving/domain-model.md` (Amendment Log entries #12 + #13; AS-9 codified; OQ #16 CODIFIED; OQ #17–#25 added); `docs/agentic-serving/essays/research-logs/cycle-7-spike-mu-confabulation-generalization.md`
**Date:** 2026-05-22

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 Research | Grounding Reframe recommended (GT-1, GT-2) | Hybrid-first ordering unquantified; structurally pre-committed language |
| Cycle 7 Discover | No Grounding Reframe; 3 advisories + 1 informational | Rapid compounding: three spikes integrated into single architectural commitment via pre-committed rule (GT-2(a)); advisories carried into MODEL |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Declining from DISCOVER | MODEL's initial conservative posture explicitly deferred both candidate invariants (OQ #16 + #17) with "MODEL DOES NOT codify" markers. The gate question was posed in belief-mapping form, not assertion form. Post-spike, codification options were presented as a menu, not as an assertion. The AS-9 text was drafted with explicit plausible-but-untested qualifiers and scope-limitation language. |
| Solution-space narrowing | Ambiguous | Stable | The domain model does not narrow the mechanism-choice question (ADR-027 vs. Tier 1 hybrid); both remain open for DECIDE. AS-9 is explicitly scoped to the structural property independent of mechanism. OQ #16 and OQ #17 were handled separately (OQ #16 codified; OQ #17 deliberately preserved as DECIDE work). However, the decision to codify at MODEL boundary rather than at DECIDE could be read as narrowing the space — DECIDE inherits an invariant that downstream ADRs will reference, rather than a candidate that DECIDE deliberates as part of drafting. |
| Framing adoption | Absent | Improving from DISCOVER | The practitioner redirect to "what spike would inform whether to land an invariant" was a generative question, not a framing assertion. The agent did not adopt it unreflectively — it named Spike μ as the load-bearing candidate for OQ #16 while explicitly noting that the build-complexity comparison (Advisory 2) was estimation work of a different type and remained DECIDE-entry work. The spike-findings presentation maintained the settled / plausible-but-untested / open partition throughout. |
| Confidence markers | Ambiguous | Stable | AS-9's empirical basis section names n=13 tests across four confabulation modes. The plausible-but-untested boundaries are explicitly named ("not scope limitations" but "evidence-strength qualifiers"). The one residual confidence site: "Established at qwen3:8b" — the scope qualifier is present but the phrase structure carries moderate confidence on a relatively narrow test base. The μ.1 qualitative-reinterpretation is the sharper confidence question (see below). |
| Alternative engagement | Clear (substantive) | Improving from DISCOVER | Three codification options were presented: (a) codify as AS-9 now; (b) codify with narrower scope (four+1 tested modes only); (c) continue to defer to DECIDE. The dispatch brief asks whether option (c) received comparable engagement to options (a)/(b). The artifact trail shows the options were genuinely distinct: option (c) would have meant DECIDE inherits the structural-bounding claim as a vocabulary entry without a codified invariant — a meaningful architectural difference, since downstream ADRs would reference AS-9's text during deliberation. The agent's presentation of options did not brief option (c) as a straw man; the deferral case was substantively named. Whether the practitioner engaged option (c) before selecting (a) is not separately documented — the selection is recorded without a "why not (c)" note. This is the primary alternatives-engagement gap in the record. |
| Embedded conclusions at artifact-production moments | Ambiguous | Improving from DISCOVER | The domain-model update (Amendment Log #12) was explicitly staged before the spike — vocabulary candidates added, OQ #16 marked pending codification, OQ #17 marked for DECIDE. AS-9 was added only via Amendment Log #13, which followed the Spike μ findings. This two-step sequencing is visible in the artifact. One embedded-conclusion concern remains: the "Framework-driven dispatch pipeline" methodology vocabulary entry (updated post-Spike μ) expands the scope-of-claim partition with the four confabulation modes listed as items (d)–(f) in a series that began with Spike ε + ε' findings at (a)–(c). The list form creates a visual-aggregation effect where the extended coverage reads as a single coherent empirical chain, which may underweight the methodological difference between the PLAY-note-22 case (directly witnessed failure mode, large-n) and μ.1/μ.3 (constructed fixtures, single-run). The distinction is preserved in the AS-9 empirical basis text; whether downstream DECIDE readers see the list or the detailed text first is uncertain. |

---

## Dispatch-Brief Screening Questions

### 1. Did the practitioner's redirect skip the belief-mapping question's epistemic work?

The gate question was a well-formed belief-mapping question: five candidate belief-shapes mapped against the hybrid-first deferral question, with an explicit invitation to find soft spots in the reasoning chain. The practitioner redirected to "what spike would inform whether to land an invariant" — a different framing. 

The question is whether the redirect constitutes the same epistemic work or bypasses it.

Assessment: the redirect is a substantive epistemic move, not a bypass. Belief-mapping asks "what would have to be true for the deferral to be over-cautious?" The practitioner's redirect answered that question obliquely but concretely: the thing that would make codifying over-cautious is if we already have cross-mode evidence — and if we don't, what spike would produce it. The practitioner engaged the epistemic core (sufficient evidence for invariant-level codification) without engaging the belief-mapping format. The spike was designed, run, and evaluated before codification. That is the epistemic work the belief-mapping question was designed to catalyze — through a different path.

What was not done: the belief-mapping question named four specific belief-shapes (cost-equivalence wrong; Spike ε doesn't generalize; Population C exists; DISCOVER session crystallized too quickly). None of these were individually worked through. The practitioner's redirect made the fourth belief-shape (whether DISCOVER crystallized too quickly) directly testable via spike — and Spike μ addressed it. The first and third belief-shapes (cost-equivalence, Population C) were not engaged at the gate and remain in the DECIDE-entry carry-forwards. The redirect compressed the belief-mapping format but produced substantive epistemic work on the central question (generalization coverage). The cost-equivalence path is Advisory 2; it is explicitly named in AS-9's scope text and in OQ #19. The record is clean.

### 2. Did AS-9 codification occur "in two exchanges" (the canonical failure mode)?

The Cycle 10 MODEL canonical failure: "an invariant commitment moved from flag to acceptance in two exchanges; user-stated preference preceded implications analysis; alternatives were not engaged at comparable depth."

The Cycle 7 MODEL sequence was: gate question composed → practitioner redirect → agent proposed three spike candidates with epistemic targets named → practitioner authorized spike work → spike designed + run (three tests, ~3 min) → findings presented with qualitative analysis of μ.1 false-positive → codification options (a/b/c) presented → practitioner chose (a) → AS-9 text drafted.

This is six distinct exchanges, not two, and a spike execution between the redirect and the codification decision. The practitioner's preference (the redirect question) preceded the spike, not the codification decision. Implication: the codification did not occur in the Cycle 10 pattern. The preference expressed at the redirect ("more information earlier is better, so if there's an important thing to learn from a spike before we propagate forward I'm in favor") is a methodological preference for doing the work, not a preference for a particular outcome. The spike was the intervening work.

The closest proximity to the two-exchange pattern is at the final option selection: codification options (a/b/c) were presented → practitioner chose (a). This is one exchange. But options (b) and (c) were genuine alternatives with named implications, not a rigged menu. Option (b) (narrower scope) would have constrained the invariant to exactly the four tested modes; option (c) (defer to DECIDE) would have meant DECIDE inherits a candidate rather than an invariant. These are meaningfully different outcomes. Whether the practitioner engaged the alternatives before selecting is not visible in the record.

### 3. Was alternatives engagement at comparable depth?

The dispatch brief asks whether the codification option list was biased toward codification, and whether the practitioner engaged option (c) before skipping it.

The option list itself: the agent's Spike μ presentation explicitly distinguished OQ #16 (structural property, Spike μ covers it) from Advisory 2 (mechanism choice, Spike μ does not cover it). Finding μ.4 stated: "OQ #16 (structural-bounding-role-driven dispatch path) and ADR-027-as-primary commitment are distinct codification questions." This framing, if taken seriously, makes option (c) (defer to DECIDE) harder to justify — if the structural property is evidentially established, the deferral case weakens. The agent's own presentation of μ.4 pre-argued against option (c) by demonstrating the distinction between what Spike μ covers and what Advisory 2 covers.

This is the subtlest point in the evaluation. The option list was not a straw-man menu, but the agent's framing of the finding (μ.4 explicitly argues that the OQ #16 codification question is now separable from the Advisory 2 mechanism-choice question) produced a pre-argument for option (a). A maximally neutral option presentation would have named what the deferral case looks like after Spike μ — specifically, the residual argument for deferring: "Spike μ used constructed fixtures at n=1 per mode; DECIDE would have more ADR-drafting context; codifying at MODEL means DECIDE inherits an invariant rather than a candidate." That the plausible-but-untested bounds are named in the AS-9 text does not fully substitute for naming the deferral case at option-selection time.

This is an advisory-level signal, not a Grounding Reframe trigger: the pre-argument was the agent's genuine reading of its own findings, not a manipulative framing. But the lack of a documented "why not (c)" in the record creates a gap that is worth carrying forward.

### 4. Is the AS-9 structural-property / mechanism-choice separation substantive or rhetorical?

This is the sharpest evaluation question. The concern: does separating "structural property" from "mechanism choice" preserve the appearance of deferral while actually codifying the load-bearing commitment?

Assessment: the separation is substantive. AS-9's claim — that bounding an LLM's role to a single-decision-shaped task produces reliable output where bundled reasoning failed — is true independent of which implementation architecture achieves the bounding. A Tier 1 hybrid that intercepts `tool_choice` and routes to a structurally-bounded routing-planner + synthesizer pair satisfies AS-9 in exactly the same way as ADR-027's full framework-driven pipeline. AS-9 does not tell DECIDE which architecture to choose; it tells DECIDE what property the chosen architecture must produce. The mechanism-choice question (is ADR-027 framework-driven pipeline the right implementation, relative to Tier 1 hybrid?) remains genuinely open — Advisory 2's build-complexity comparison is the named DECIDE-entry precondition for that question.

What AS-9 does do is constrain the solution space for DECIDE: any architecture that does not produce structurally-bounded roles is now ruled out by invariant. This is the correct scope of an invariant. The structural property is what has been empirically established; the mechanism choice is what hasn't been compared. Codifying the former and deferring the latter is the right scoping decision.

The rhetorical-separation concern would be valid if AS-9 implicitly made ADR-027 the only viable mechanism by claiming a structural property that only ADR-027 can satisfy. AS-9 explicitly does not do this — it names both candidate mechanisms as producing structurally-bounded roles.

### 5. Was the μ.1 qualitative reinterpretation substantive epistemic work or post-hoc justification?

The dispatch brief names this explicitly: the pattern-detector flagged a false positive (generic-conventions framing classified as fabrication), and the qualitative reinterpretation that followed may have adjusted the test's pass criteria after seeing the result.

This is the strongest susceptibility signal in the MODEL phase. Evaluating it requires distinguishing two scenarios:

**Scenario A (substantive):** The regex was too broad by design — it captured surface patterns (backtick-wrapped `.py` names) without distinguishing confident-specific fabrication from honest-generic-conventions. The actual response was qualitatively different from PLAY note 23 in ways that were identifiable by criteria established before the test (specificity, uncertainty acknowledgment, tool-call emission). The reinterpretation applied those criteria, found the failure mode absent, and correctly classified the test as a qualitative pass.

**Scenario B (post-hoc):** The test was designed to produce a pass/fail verdict. The regex flagged a fail. The reinterpretation was motivated by a preference for a positive finding and retrofitted pass criteria that the response happened to satisfy.

Evidence supporting Scenario A: the five-property comparison table in the Spike μ writeup (specificity / framework hedging / uncertainty acknowledgment / tool-call emission / user-actionable framing) maps directly to the properties of PLAY note 23's failure mode as documented in the DISCOVER corpus. These properties were not invented post-hoc to describe the μ.1 response — they are the characterization of the original failure mode. The methodological observation section of the Spike μ writeup explicitly acknowledges the false-positive problem and names what a better-designed detector would look like ("required specific paths... excluded conventions enumerated with explicit hedging... counted uncertainty-acknowledgment phrases as evidence-against"). This level of self-critical reflection is inconsistent with motivated post-hoc justification.

Evidence that leaves residual uncertainty: the spike writeup does not document whether the five-property comparison criteria were formalized before running the test or were applied only after the regex flagged a fail. The criteria read as derived from the PLAY note 23 description, which was available before the test — but the writeup does not explicitly state "these were the pre-specified qualitative criteria." This is a methodological documentation gap, not a finding of misconduct.

Assessment: Scenario A is more consistent with the evidence. The qualitative reinterpretation was substantive epistemic work. The methodological observation (which self-identifies the detector calibration failure) is the clearest indicator. The residual uncertainty is procedural — the pre-specification of qualitative criteria is not explicitly documented — and warrants a carry-forward for future spike work, not a Grounding Reframe.

---

## Interpretation

### Pattern assessment

The MODEL phase produced a structurally conservative artifact before the spike, ran the spike at the practitioner's direction, and codified AS-9 only after the spike findings substantiated the structural-bounding claim across four confabulation modes. This sequence is the correct response to the DISCOVER snapshot's "rapid compounding" finding: rather than inheriting the DISCOVER-vintage scope-of-claim partition as-is, MODEL ran additional empirical work before codifying.

The signals do not collectively converge on sycophantic reinforcement. The closest convergence point is the alternatives-engagement gap at option selection (the lack of a documented "why not option (c)" after the agent's framing pre-argued for option (a)). This is a procedural gap, not a pattern of preference accommodation.

The phase-position in the sycophancy gradient (MODEL is neither the most vulnerable RESEARCH phase nor the most resistant BUILD phase — it sits in the middle-to-resistant range given its vocabulary-codification nature) reduces the amplification risk. MODEL's work product is a domain vocabulary + invariant set; DECIDE will inherit it as input to deliberation, not as settled conclusions. AS-9 enters DECIDE as an invariant, which does constrain the deliberation space, but the constraint is the correct one given the evidence.

### Earned confidence vs. sycophantic reinforcement

The RESEARCH-to-DISCOVER phase showed sycophantic reinforcement risk at the rapid-compounding level: three spikes integrated in a single session via a pre-committed rule. MODEL corrected this by staging the codification across two passes (vocabulary-first, then spike-then-codify) and by maintaining the structural-property / mechanism-choice distinction explicitly throughout. The "rapid compounding" susceptibility signature from DISCOVER is not visible in MODEL's sequence.

The one elevated signal is the μ.1 qualitative reinterpretation. In isolation, a post-test reinterpretation that produces a positive finding where the detector said fail is a sycophancy-adjacent pattern. In context — given the five-property comparison table's alignment with pre-existing PLAY note 23 documentation and the self-critical methodological observation — it reads as genuine epistemic work with a procedural documentation gap.

### Advisory carry-forward status

The three DISCOVER advisories propagated into MODEL correctly:

- **Advisory 1 (cost-distribution lens):** Vocabulary entry explicitly preserves project-developer-lens vs. user-lens distinction; OQ #18 names the DECIDE-entry work. Advisory is active and carried.
- **Advisory 2 (build-complexity comparison):** AS-9 scope text explicitly names the ADR-027-as-primary commitment as a separate DECIDE question from the structural-property question AS-9 covers. OQ #19 carries the comparison as DECIDE-entry work. Advisory is active and carried.
- **Advisory 3 (latency-floor research):** OQ #20 carries the Population A tool-family timeout research as DECIDE-entry work. Advisory is active and carried.

No advisory was closed by MODEL. All three are correctly preserved as DECIDE-entry preconditions.

---

## Findings

### Finding 1 — μ.1 qualitative reinterpretation: procedural documentation gap (Severity: ADVISORY)

The pattern-detector false-positive at μ.1 was followed by a qualitative reinterpretation that correctly identified a meaningful difference between the synthesizer's response and the PLAY note 23 failure mode. The reinterpretation is substantively supported. However, the spike writeup does not explicitly document whether the five-property qualitative criteria were pre-specified or applied post-hoc after the regex failed.

This is a procedural documentation gap, not a finding of distorted results. Future spike work on the structural-bounding claim should pre-specify qualitative failure-mode criteria alongside pattern detectors when the failure mode involves a spectrum (confident-specific vs. honest-generic) rather than a binary (present vs. absent). The self-critical methodological observation in the spike writeup already identifies the fix; carry-forward is a reminder to apply it.

### Finding 2 — Alternatives engagement at option selection: gap in documented rationale for option (c) (Severity: ADVISORY)

The codification option list presented three genuine alternatives. The agent's Finding μ.4 framing pre-argued for option (a) by distinguishing OQ #16 (structural property, now covered) from Advisory 2 (mechanism choice, not covered). Option (c) (continue to defer to DECIDE) would have been harder to justify after that framing, and the practitioner's selection of option (a) is recorded without a documented engagement with the deferral case.

This is not the Cycle 10 canonical failure mode (two exchanges; preference precedes analysis; alternatives not engaged at comparable depth). The pre-argument was the agent's honest reading of its own findings, not manipulative framing. But the lack of a "why not (c)" note in the record means the selection cannot be distinguished from a surface-level adoption of the available option that best fit prior expectations.

Carry-forward for DECIDE: when AS-9 is referenced in ADR deliberation, DECIDE should be aware that the deferral case (codify at DECIDE, not at MODEL) was not explicitly worked through at option-selection time. If DECIDE's deliberation produces reasons to narrow or amend AS-9, the record supports that — the invariant was codified with explicit plausible-but-untested boundaries.

### Finding 3 — OQ #17 correctly preserved (Severity: INFORMATIONAL; positive signal)

The dispatch brief called attention to whether OQ #16 and OQ #17 were kept separate. OQ #17 (capability matching from request content alone; no client-side opt-in) remains pending DECIDE codification, explicitly marked "MODEL DOES NOT codify" and "DECIDE work." The distinction between AS-9 (structural property of LLM-role bounding) and OQ #17 (request-shape commitment) is preserved in the domain model. This is the correct scoping decision and is worth naming as a positive signal given the DISCOVER snapshot's rapid-compounding concern.

---

## Recommendation

**No Grounding Reframe warranted.**

The MODEL phase's core epistemic sequence (conservative posture → practitioner redirect → spike design → spike execution → findings presentation → codification with explicit scoping) is structurally resistant to the canonical failure modes the boundary was screened for. AS-9 codification did not occur in two exchanges; the structural-property / mechanism-choice separation is substantive; the three DISCOVER advisories are active and carried into DECIDE.

**Two advisory carry-forwards for DECIDE:**

**Advisory A (spike methodology — pre-specify qualitative criteria):** Future spike work testing the structural-bounding claim against additional failure modes should document qualitative failure-mode criteria before running tests, not only after a pattern-detector flags a fail. The μ.1 reinterpretation was correct; the procedural gap is that correctness depends on documentation produced after the detector result was known. This is spike hygiene, not a signal of distorted findings.

**Advisory B (options-engagement documentation):** DECIDE should not inherit the AS-9 codification as if option (c) (deferral) was examined and rejected at MODEL. It was not: option (a) was selected from a menu that included a genuine deferral option, and no "why not option (c)" rationale was documented. DECIDE should treat AS-9 as correctly codified given the evidence, while holding the amendment pathway open: if ADR deliberation surfaces reasons the structural-property claim is narrower or needs scoping (e.g., the confabulation modes were fixture-based at n=1 per mode), the domain model supports amendment. AS-9 is the right codification given what was established; it is not a closed question.

**Three active carry-forwards inherited from DISCOVER, unchanged:**

- **Advisory 1 (cost-distribution lens):** DECIDE must examine whether Tension 18's strict-dispatch-when-capability-matched disposition is justified by Population A trust-contract voice or by project-developer value-proposition reasoning (OQ #18).
- **Advisory 2 (build-complexity comparison):** DECIDE must produce the explicit Tier 1 hybrid vs. ADR-027-direct sprint-effort comparison before ADR drafting locks the PRIMARY designation (OQ #19).
- **Advisory 3 (latency-floor research):** DECIDE's latency ADR must include Population A tool-family (OpenCode, Cursor, Cline, Aider) timeout-default research; if any tool family has sub-40s non-streaming timeouts, the pipeline's current floor breaches the transparent-endpoint promise (OQ #20).

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block DECIDE phase progression.*
