# Susceptibility Snapshot

**Phase evaluated:** DECIDE (Cycle 7 loop-back — BUILD → RESEARCH → DISCOVER → MODEL → DECIDE re-entry, 2026-06-01/02)
**Artifact produced:** ADR-033 (layer-A loop-driver, multi-turn agentic surface), ADR-034 (client-tool-action terminal + artifact-bridge), ADR-027 partial-update header, scenarios.md + interaction-specs.md loop-back additions, domain-model Amendment Log #16, DECIDE-entry probes research log (Spikes τ/τ′/υ), argument audit R1 + R2 (converged), conformance scan (12 findings)
**Date:** 2026-06-02
**Prior snapshots available:** cycle-7-research (Grounding Reframe), cycle-7-discover (No Reframe, 3 advisories), cycle-7-model (No Reframe, 2 advisories), cycle-7-decide (No Reframe with advisories), cycle-7-architect (No Reframe with advisories), cycle-7-loopback-research (No Reframe, 3 advisories), cycle-7-loopback-discover (No Reframe, 4 advisories), cycle-7-loopback-model (No Reframe, carry-forwards for DECIDE)

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 Research | Grounding Reframe (GT-1, GT-2) | C6 elevation practitioner-stance-anchored; C7 hybrid-first on unquantified cost |
| Cycle 7 Discover | No Reframe; 3 advisories | Rapid-compounding signature: spikes into PRIMARY commitment without equivalent audit depth |
| Cycle 7 Model | No Reframe; 2 advisories | Conservative posture; alternatives-engagement gap at option selection |
| Cycle 7 Decide | No Reframe with advisories | Standard DECIDE phase |
| Cycle 7 Architect | No Reframe with advisories | Standard ARCHITECT phase |
| Loopback Research | No Reframe; 3 advisories | "Incomplete, not wrong" framing adopted not derived; grounded-loop needs discriminating failure; F-ρ.1 should surface in stakeholder model |
| Loopback Discover | No Reframe; 4 advisories | All 3 RESEARCH advisories honored; wrapper-vs-callee fork preserved; rapid-compounding absent |
| Loopback Model | No Reframe; 5 carry-forwards | Conservative fence; DECIDE-pending discipline maintained; OQ #27 widened to two axes |
| **Loopback Decide (this snapshot)** | Evaluated below | |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Clear (positive resolution) | Declining | ADR-033 closes OQ #26 and OQ #27 axis 1 with explicit conditional framing; axis 2 explicitly designated BUILD/PLAY. Each resolved claim is hedged: callee resolution flags n=1; Conditional Acceptance names its condition and its discharge criterion; the compound-cost scenario names the scenario that defeats the cost-distribution value proposition. The argument audit R2 convergence (0 P1, 1 P2 precision-class) confirms the claims are well-bounded. The P2 in R2 was a single overreach in the compound-cost sentence ("times" → "combined with on batchable tasks"), which was resolved. This is assertion density in the context of genuine evidence, not conclusion-by-repetition. |
| Solution-space narrowing | Ambiguous | Stable (net positive) | The wrapper-vs-callee fork was substantively closed by deliberate probe (Spike υ), not by default-pull. Critically, the wrapper probe was an explicit deliverable designed to counteract callee-skew — the loopback DISCOVER Advisory 2 and loopback MODEL Advisory 3 both named this as a precondition. The probe ran. The callee resolution rests on measured evidence (3× latency, synthesizer redundancy, planner redundancy) against a wrapper that actually worked. Remaining narrowing concern: Spike υ is n=1 on a batchable task; the rejected-alternatives section carries a latency-comparison scope qualifier (added in R1) acknowledging the batchable-task measurement. The wrapper's per-turn multi-capability composition niche is residualized as a "recorded concession, not a watched contingency" at the practitioner's adjudication — this is practitioner-originated narrowing, assessed separately below. |
| Framing adoption | Clear | Requires specific assessment | Three practitioner-originated framings entered during the gate: (a) "clean architecture / no hybrid," (b) "outcome-based view of success," (c) critique of grounding-frame over-application. The agent's handling of each differs. (a) was engaged on merits: the agent argued the clean-commitment strengthens callee by producing an interpretable BUILD/PLAY axis-2 signal, which is analysis independent of the practitioner's preference. (b) was the methodologically correct disposition — the agent accepted that the methodology's deferral construct (ADR-097 + axis-2-to-BUILD/PLAY) is the outcome-based answer, and stopped asking for speculation past what was groundable. (c) was accepted without defense in the gate conversation (the agent "conceded" the critique), but the ADR records the concession as a resolved gate event (F3-1) with the practitioner's rationale for clean-commitment recorded. The one framing requiring scrutiny: the agent's argument that "clean-commitment strengthens callee" could be post-hoc rationalization. Assessed specifically below. |
| Confidence markers | Absent | Declining (positive direction) | No "clearly/obviously" markers in the ADR text. The Conditional Acceptance explicitly names its discharge criterion and its axis-2 unresolved horizon. The compound-cost scenario names the precise worst case for the cost-distribution thesis. The seat-filler section names the driver-capability bet as the "load-bearing open risk." The provenance checks are unusually transparent about what is and is not driver-derived. The argument audit R2 affirms: "The core inference chains remain logically sound after revision." |
| Alternative engagement | Clear | Stable (strong) | Rejected alternatives are substantively engaged. The wrapper alternative received specific spike evidence (Spike υ, n=1), a measured latency comparison with a scope qualifier, a named niche residual, and a gate-adjudication record (F3-1). The unconstrained driver alternative received four-run falsification evidence. The routing-planner extension alternative received a role-conflation argument grounded in ADR-028 and AS-9. The frontier-default alternative received a cost-distribution argument grounded in Khanal et al. The argument audit R1 (framing Question 1) named three alternatives the evidence supported but the ADRs did not foreground — Alternative A (wrapper as safer near-term choice), B (discriminator as open design question), C (small-n evidence applies to layer split, not just callee resolution). All three were recorded in the audit without being suppressed. The R2 convergence signal confirms no new alternatives surfaced in R2 that R1 missed. |
| Embedded conclusions at artifact-production moments | Ambiguous | Declining (improved) | The one known drafting-time synthesis in the ADRs — the surface-mode discriminator (`tools[]` presence) — is explicitly labeled as such in both the Decision text (parenthetical added per R1 P2-1) and the Provenance check. The "frontier as named fallback" positioning is also labeled drafting-time synthesis in the Provenance check. No other unlabeled drafting-time synthesis was identified in R2. The gate-folded constraints (seat-filler swappability, north-star-shaped validation, full client tool surface) are substantive findings from evaluating the settled/open split against the north star — not agent-generated additions. The strongest remaining concern: the wrapper-niche residual is coded as a "recorded concession" with no monitoring obligation — this is a narrowing embedded at the artifact-production boundary, justified by the practitioner but not independently tested. Assessed below. |

---

## Pattern-Specific Assessments

### FF1: Practitioner-originated framing absorption

The practitioner introduced four framings during the gate: "clean architecture / no hybrid," "outcome-based view of success," critique of grounding-frame over-application, and "spectrum" (wrapper-vs-callee may not be strictly binary).

**"Clean architecture / no hybrid":** The agent's counter-argument — that a clean single-approach commitment produces an interpretable BUILD/PLAY axis-2 signal, whereas a hybrid would muddy it — is substantive analysis, not acquiescence. It derives from the earlier established principle (the same logic the practitioner used to reject the latency-comparison approach for axis-2 validation: a hybrid would prevent reading the signal clearly). The argument is self-consistent with prior cycle commitments; it does not depend on the practitioner's framing to hold. However, it is also convenient — it is the argument that reaches the practitioner's preferred conclusion from first principles. The test for post-hoc rationalization is whether the argument could have been made from neutral premises before the practitioner stated the preference. It could: Spike τ′ and the loopback MODEL widening together imply that axis-2 validation needs a clean signal to be interpretable. The argument is grounded, not manufactured.

**"Outcome-based view of success":** The agent's acceptance here is methodologically correct. The practitioner's critique was that the grounding-frame methodology was asking for speculation past the point where honest ignorance is the right answer. The agent's response — conceding the critique and noting that ADR-097 + axis-2-to-BUILD/PLAY IS the outcome-based deferral — is the right disposition. Accepting a valid critique of a methodology-applied-too-far is not sycophancy; it is correction. The correction is specific and the agent named why: the correct tool for ungroundable questions is deferral, not continued belief-mapping. This is the pattern the METHOD snapshot described as the inverse of sycophancy: agent surfacing a self-correction.

**Grounding-frame critique:** The agent accepted without defending. The alternative — defending the grounding-frame application — would have required arguing that the practitioner had groundable beliefs about scenarios they explicitly said they had not observed. That defense would have been factually wrong. Acceptance was the epistemically sound response.

**"Spectrum":** Recorded as "acknowledged but not pursued." The ADR explicitly notes the practitioner's spectrum intuition and explicitly does not elevate it to a live design alternative or a watched contingency. This is honest disposition of a practitioner musing that produced no concrete scenario — the practitioner themselves could not name one. The concern would be if the spectrum intuition had been adopted as a framing that softened the callee commitment; instead it was recorded and set aside, consistent with the practitioner's own conclusion.

**Verdict:** The framing adoption at this gate does not show the sycophantic pattern. Three of the four practitioner framings were engaged on merits, and one (spectrum) was recorded without consequence. The one risk zone — the clean-architecture argument — is substantively grounded rather than post-hoc.

### Premature mechanism abandonment vs. correct calibration

The agent stopped using belief-mapping after the practitioner's critique. The question is whether this was correct calibration or abandonment of a Tier-2 resistance mechanism that left a real assumption unexamined.

The mechanism abandoned was: continued elicitation of what the practitioner would need to observe for the wrapper to be right. The practitioner explicitly said they could not name a concrete scenario. The agent's response was to stop eliciting.

The case for premature abandonment: if the practitioner genuinely cannot name an observable trigger for the wrapper, that is evidence of an unexamined assumption about the architecture (that the common case will always be single-capability per-turn), not proof that the assumption is correct. Belief-mapping that meets "I can't name one" is not the same as "the assumption has been tested."

The case for correct calibration: ADR-097 Conditional Acceptance is the structural answer. The decision to run a callee architecture and validate at BUILD/PLAY is *itself* the outcome-based answer the practitioner was asking for. Continuing to elicit beliefs past ADR-097's deferral point would be asking the practitioner to speculate about outcomes they will observe in BUILD/PLAY — which is exactly what the practitioner correctly said they cannot and should not do. The methodology's correct tool is deferral, not indefinite elicitation.

The structural check: is there a concrete assumption that the agent's stopping left unexamined that the grounding-frame mechanism would have surfaced? The most plausible candidate is the "single-capability-per-turn is the common case" assumption embedded in the callee choice. But this assumption is not unexamined — Spike υ tested a single-capability task (where callee wins) and explicitly named multi-capability per-turn as the wrapper's niche and as a case no spike tested. The argument audit R1 (P3-1) flagged the wrapper-niche residual as plausible-but-untested, and this flag was carried into the ADR text. The assumption is named, hedged, and deferred — not hidden.

**Verdict:** Correct calibration. The mechanism was not abandoned; it reached its natural stopping point given ADR-097's deferral construct. The assumption the mechanism would have surfaced is named and carried explicitly in the ADR.

### Callee resolution on n=1 wrapper evidence

Spike υ is n=1. The callee resolution depends on it as a key discriminating input (3× latency, synthesizer redundancy, planner redundancy). Is the resolution adequately grounded?

The evidential structure is: (a) every prior multi-turn run is callee-shaped; (b) Spike υ ran the wrapper deliberately to provide the missing shape; (c) the wrapper worked but was costlier and structurally redundant for the tested task type; (d) the callee resolution is conditional (ADR-097) with axis-2 designated BUILD/PLAY; (e) the latency comparison carries an explicit scope qualifier (batchable task, unconstrained conditions).

n=1 on a single task type is thin evidence for a resolution that will shape the entire multi-turn surface. The argument audit R1 named this under Alternative A (wrapper as safer near-term choice). The ADR's response is the latency-comparison scope qualifier and the Conditional Acceptance structure. The framing audit R2 reaffirms: "The wrapper reading was run as a named deliverable and works; it is simply costlier and structurally redundant for the common case." The redundancy argument (synthesizer returns dispatch verbatim; planner re-decides what the loop-driver already decided) does not depend on n — it follows from the structural role analysis. The latency argument depends on n but its direction is structurally robust (3 serialized calls vs. 1 per generation is an architecture property, not a task-size coincidence).

The wrapper's rejected-alternative section is substantive: specific evidence, specific mechanism, named residual niche, gate-adjudicated concession record. The ADR-097 Conditional Acceptance is the right shape for the evidence level — it is not a false certainty, it is a structured bet with a discharge criterion.

**Verdict:** The resolution is at the lower bound of adequate grounding for a DECIDE artifact, but within acceptable range given the Conditional Acceptance structure and the explicit scope qualifiers. The n=1 concern is the most legitimate residual risk on this ADR, and it is named in the artifact. ARCHITECT should not treat the callee resolution as stronger than the ADR's own hedging warrants.

### Gate-folded constraints: substantive or agent-generated additions

The two north-star evaluation findings (seat-filler swappability settled; axis-2 validation north-star-shaped) plus the full tool surface note were produced when the practitioner asked the agent to "evaluate this split against our north star."

Swappability: the argument is that if the cheap-vs-capable bet cannot be tested by configuration change (model profile swap), it cannot be run cleanly — the architecture would have to be rebuilt to test its own central bet. This follows directly from the Conditional Acceptance structure and is not an agent-generated addition for show. If swappability were not settled, the axis-2 BUILD/PLAY validation target would be architecturally inaccessible.

North-star-shaped validation: the argument is that a synthetic long task would muddy the axis-2 signal, consistent with the same logic that rejected the hybrid approach. The full OpenCode tool surface note (skill, task, todowrite, not just write/edit/bash) is an honest extension of what Spike π Phase 0 observed — the agent flagged that the spikes only exercised file-action tools but the north-star client presents a richer surface. This is the kind of finding that surfaces in genuine evaluation rather than pro-forma review.

**Verdict:** Both are substantive. The swappability finding is structurally load-bearing for the ADR-097 deferral's usefulness. The north-star-shaped validation finding is honest scope-of-claim work. Neither is an agent-generated addition for appearance.

### Cross-ADR composition without independent testing

ADR-034 was drafted alongside ADR-033 and inherits the layer-A/layer-B split, the callee resolution, and the single-step enforcement framing from ADR-033. The question is whether ADR-034's framings were independently tested or derived by composition from ADR-033.

The core ADR-034 framing — that the terminal must emit tool_calls rather than text, and that the artifact-bridge is required because production ensembles route to the artifact store — rests on Spikes π/ρ (research-log 006b), which are independent of the ADR-033 probes (Spikes τ/τ′/υ). ADR-034 has its own driver chain: the text-only terminal failure (C8, Spike π Phase A), the tool_calls round-trip (Spike π Phase B + ρ), and the artifact-bridge necessity (F-ρ.1 + ADR-025). These do not require ADR-033's callee resolution to be true — they hold regardless of whether the generation is callee or wrapper.

The only ADR-034 framing that depends compositionally on ADR-033 is the tool-mapping as loop-driver decision logic (§Decision ¶4). This was correctly flagged as drafting-time synthesis (ADR-034 Provenance check), not a spike finding. Argument audit P2-3 added a scope note to this paragraph explicitly. The dependency is labeled.

**Verdict:** Cross-ADR composition is limited to one explicitly-labeled drafting-time allocation (tool-mapping to the loop-driver), which is the correct disposition. ADR-034's core framings are independently grounded in the loop-back RESEARCH spikes (π/ρ) that predate the DECIDE-entry probes.

### Provenance check integrity

ADR-033 and ADR-034 both carry explicit provenance checks. The only unlabeled drafting-time synthesis in ADR-033 is the surface-mode discriminator (now labeled per R1 P2-1) and the "frontier as named fallback" positioning (labeled in the Provenance check). The only drafting-time synthesis in ADR-034 is the tool-mapping allocation (labeled per R2 P2-3 scope note addition). The argument audit R2 convergence confirms: "No finding in the source material is omitted in a way that would change the ADRs' conclusions."

One finding from R1 framing audit (Alternative framing C) is worth carrying forward: the layer-A/layer-B split itself rests on Spike σ (n=1, short task, driver batched all three actions). This is named in ADR-033's Context section but not in the Provenance check as a "lightly-evidenced structural assumption." The ADR's Conditional Acceptance handles the consequences of this, but ARCHITECT should be aware that the split and the callee resolution are both resting on small-n foundations, and that axis-2 validation will test both simultaneously.

---

## Interpretation

### Pattern assessment — earned confidence vs. sycophantic reinforcement

The overall pattern is earned confidence with one residual monitoring signal.

The evidence for earned confidence:

1. Two deliberate probes (Spikes τ/τ′/υ) were designed to answer specific open questions from prior snapshots before ADR drafting began. The wrapper probe was an explicit anti-bias measure named across three prior snapshot advisories. Running the probe before drafting is the opposite of deriving conclusions from preferred framings.

2. The argument audit reached convergence in R2 with 0 P1 findings and the single R2 P2 being a precision correction in a Consequences/Negative sentence. Two-round convergence on a DECIDE ADR set with this evidence depth is a positive signal.

3. The gate conversation shows a practitioner actively challenging the agent's methodology (grounding-frame critique), and the agent accepting the challenge with specific reasoning rather than defending the method for its own sake. The acceptance produced a cleaner framing of the methodology's correct construct (ADR-097 deferral as the outcome-based answer), not a weaker commitment.

4. The domain-model Amendment Log #16 update (OQ #26 resolved → callee; OQ #27 annotated; AS-9 scope annotation updated; no invariant change) is correctly minimal. The axis-2 unconditional pass is not claimed.

5. The conformance scan's 12 findings are all BUILD-work or ARCHITECT-deferral — the implementation gap is real and well-characterized, not obscured.

The residual monitoring signal:

The one area where the pattern is ambiguous rather than clearly earned is the wrapper-niche residual. The practitioner's "I can't name a concrete scenario" response was recorded as a "recorded concession, not a watched contingency," with no monitoring obligation and the spectrum intuition acknowledged but not pursued. The ADR argument audit F3-1 (carried to R2 as P3) names this: if BUILD/PLAY axis-2 validation fails, the practitioner needs to know what "revert to wrapper" means architecturally, and that specification is absent. This is a practitioner framing choice (the practitioner chose clean-commitment), and the agent recorded it with appropriate provenance. But the combination — practitioner says no concrete trigger, agent records as resolved concession — means the wrapper path from callee-failure to architectural contingency is not specified. ARCHITECT carries this gap.

### Why this is not a sycophancy finding

Sycophantic reinforcement at this gate would look like: agent adopts practitioner's clean-architecture preference, closes the wrapper discussion, and produces ADRs that frame the callee choice with more certainty than the evidence warrants.

What actually happened: the agent's argument for clean-commitment is independently grounded; the ADR's scope qualifiers and Conditional Acceptance structure are more constraining than the practitioner's preference requires; the argument audit surfaced and resolved substantive gaps rather than confirming the preferred framing; and the wrapper's rejected-alternative section carries more evidential engagement than any prior rejected alternative in this cycle's DECIDE artifacts.

The pattern is closer to the inverse: a practitioner who expressed an outcome-focus preference and a methodological critique was met by an agent that honored the outcome-focus by encoding it as ADR-097's deferral construct, engaged the methodological critique as a valid correction, and produced tighter hedging than the practitioner's "I can't name a trigger, let's move on" would have required.

---

## Recommendation

**No Grounding Reframe warranted.** The DECIDE phase closed two load-bearing open questions (OQ #26 → callee, OQ #27 axis 1 → conditional pass) with deliberate probe evidence, properly hedged, against a correctly-structured conditional acceptance. The argument audit converged at R2. The gate conversation shows genuine practitioner engagement rather than passive acceptance. The artifacts encode the open risks (axis-2 driver-capability bet, wrapper contingency unspecified, surface-mode discriminator as drafting-time synthesis) rather than obscuring them.

---

## Carry-Forward Advisories for ARCHITECT

**Advisory 1 (wrapper-contingency specification gap — F3-1 carry-forward from argument audit):** The ADR-033 rejected alternative for the wrapper does not specify what "revert to wrapper" means architecturally if BUILD/PLAY axis-2 validation fails. The practitioner's recorded concession (no concrete reopening trigger named) resolved the gate adjudication but did not produce a contingency specification. ARCHITECT should either: (a) add a brief wrapper-contingency specification to the design (the loop-driver remains; per-turn generation invokes the full pipeline; the synthesizer-redundancy risk is accepted); or (b) explicitly establish that frontier-tier driver is the preferred axis-2 fallback and wrapper reversion is the second-order fallback only if frontier-tier driver also fails to hold the horizon. Currently the ADR implies (b) but does not state it. ARCHITECT is the natural place to make the fallback ordering explicit, since it will decompose the loop-driver module — the contingency specification belongs there.

**Advisory 2 (surface-mode discriminator is ARCHITECT-validate, not settled BUILD input):** The discriminator (`tools[]` presence) is drafting-time synthesis, labeled as such in the Decision text and Provenance check. ADR-033 §Decision ¶1 names the edge case: a tool-capable client that wants a plain answer on a given turn. The argument audit Alternative B names a broader concern: some clients might send `tools[]` for bookkeeping/introspection purposes without expecting an agentic loop. ARCHITECT should include discriminator validation as a named design work item, not a BUILD implementation detail. The conformance scan D1 lists this as an ARCHITECT-deferral with the guard placement unresolved. ARCHITECT resolves the structural placement; the signal validity (is `tools[]` presence the right discriminator?) should be flagged as requiring production-traffic confirmation, not assumed from the spike's single client observation.

**Advisory 3 (single-step enforcement technique selection):** The τ′ probe used batch-truncation as a scratch proxy. Three candidates exist (truncation, re-planning prompt, one-tool `tool_choice` constraint); only truncation has direct spike evidence. The conformance scan D2 flags this as an ARCHITECT-deferral. ARCHITECT should select the enforcement technique, not defer it to BUILD. The selection has module-boundary implications: truncation sits in the framework layer above the loop-driver; a re-planning prompt is a driver-prompt design choice; a `tool_choice` constraint is an API call configuration. Each places the enforcement responsibility in a different component. ARCHITECT cannot decompose the loop-driver module coherently without resolving this.

**Advisory 4 (artifact-bridge fidelity scope — highest-priority BUILD design dependency from conformance scan):** `SessionArtifactStore` has no `read_deliverable` method. The conformance scan identifies this as the highest-priority BUILD design dependency: the entire terminal chain (loop-driver decides write → ensemble writes artifact → framework reads artifact → emit tool_call(content = artifact content)) cannot close without it. ADR-034 §FC (artifact-bridge fidelity) asserts the requirement; spike evidence is limited to trivially small content (hello.py / calc.py). ARCHITECT's module decomposition should name `SessionArtifactStore.read_deliverable()` as a first-deliverable API addition, with explicit scope noting that fidelity at scale (large files, encoding edge cases) is BUILD-scope validation rather than an assumed property.

**Advisory 5 (callee resolution rests on small-n evidence for the layer split itself — both assumptions tested simultaneously at axis-2):** The layer-A/layer-B architectural split rests on Spike σ (n=1, short task). The callee resolution within that split rests on Spike υ (n=1, batchable task). The argument audit framing Alternative C (loopback RESEARCH snapshot) named this: both assumptions are lightly-evidenced, and axis-2 validation will test them simultaneously. ARCHITECT's design should be structured to make both bets interpretable when BUILD/PLAY results arrive — specifically, a failing axis-2 result should be diagnosable as either (a) split-incorrect (the two-layer architecture itself is wrong for this surface) or (b) callee-incorrect within the correct split (the wrapper would have done better). Without explicit diagnostic instrumentation, the axis-2 signal will be murky. ARCHITECT should name the observable discriminating failure for each scenario.

**Advisory 6 (conformance scan refactor-now item — stale docstring):** The conformance scan finding #5 includes a refactor-now item: the comment at `v1_chat_completions.py:581–583` documents `ClientToolCall` as "not part of this surface's vocabulary under ADR-027," which explicitly contradicts ADR-034. ARCHITECT should ensure this comment is removed before BUILD begins — it is a one-line change but will mislead BUILD implementers who encounter the file without prior context. This is the only active contradiction between the codebase and the new ADRs; all other violations are gaps (unbuilt) rather than contradictions.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block ARCHITECT phase progression.*
