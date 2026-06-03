# Susceptibility Snapshot

**Phase evaluated:** DECIDE (Cycle 7 loop-back #2 — scoped BUILD → DECIDE → ARCHITECT loop, Finding D — client-tool deliverable I/O-contract)
**Artifact produced:** ADR-035 (client-tool deliverable form contract, boundary-composed), ADR-024 partial update header
**Date:** 2026-06-03
**Prior snapshots available:** Full cycle-7 chain through cycle-7-loopback-decide (No Grounding Reframe warranted, 6 advisories carried to ARCHITECT)

---

## Prior Snapshot Trajectory (relevant to this scope)

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 loopback DECIDE | No Reframe; 6 advisories | Earned confidence; wrapper-contingency spec gap; cross-ADR composition one labeled drafting-time synthesis |
| **Loopback #2 DECIDE (this snapshot)** | Evaluated below | |

The prior snapshot at the cycle-7-loopback DECIDE gate found earned confidence with deliberate probe-before-draft discipline. The pattern to watch here is whether that discipline held under a tighter scope (one ADR, one finding, one extended session) — where the pressure to converge early is higher and the gate-exchange-derived additions require specific scrutiny.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Declining (positive) | The practitioner produced questions and probes throughout, not declarative conclusions. Steps 1–5 in the engagement trajectory are all "choose to ground further" choices; step 6 is the first convergence move, and it follows completed evidence. The ADR itself flags n=4 grounding, pipeline-narrow caveat, and Conditional Acceptance — confidence language moved toward more hedging under audit pressure (P2-1 overclaim corrected), not toward stronger assertion. |
| Solution-space narrowing | Ambiguous | Requires specific assessment (see below) | Two narrowing events occurred: (a) the where-fork was resolved toward a boundary-composed directive, which was an agent-synthesized option beyond the two original poles; (b) the granularity invariant was adopted from the χ-P6 break without testing a structured-multi-file alternative. The first is assessed under agent-proposed synthesis below. The second is the primary residual concern — it is flagged in the ADR's own provenance check and held as framing-P2-F2 at the gate. |
| Framing adoption | Ambiguous | Requires specific assessment (see below) | Two distinct absorption events with opposite risk profiles. The practitioner-originated seam framing was absorbed by the agent and extended (context-seam distinction, FC-51 mapping, scope-boundary elaboration) — practitioner-originated material flowing into agent-extended synthesis, not verbatim absorption. The agent-proposed "boundary-composed directive" synthesis was accepted by the practitioner after one breadth-grounding round. These run in opposite directions and are assessed separately. |
| Confidence markers | Absent | Declining (positive) | No "clearly/obviously" markers. The Conditional Acceptance names its discharge criteria, the provenance check labels drafting-time synthesis explicitly, and the R3 audit confirms the four gate-derived additions introduce no overclaiming. The P2-1 correction in R1 ("structural in spirit" → explicit directive-presence/form-compliance distinction) moved in the right direction. |
| Alternative engagement | Clear (mixed quality) | Stable with one gap | Three rejected alternatives are substantively engaged with named spike evidence: bridge-side shaper rejected by F-χ.1 (2/3 multi-fence ambiguity), static ensemble coupling rejected by ADR-025 + χ compliance data, schema-retry rejected by first-try compliance + ADR-024 Candidate B-strong. The harder-enforcement alternative received its deepest engagement at the gate — the practitioner's challenge produced the "neither available nor required" analysis, which the R3 audit found sound and honestly scoped. One gap: the structured-multi-file contract alternative was not probed. χ-P6 tested only the implicit multi-file case; a JSON-array structured format was not run. The ADR's provenance check names this explicitly (framing-P2-F2, held for the gate), but the alternative was not closed by evidence — it was closed by design preference. |
| Embedded conclusions at artifact-production moments | Ambiguous | Improving (one held concern) | The ADR's provenance check is unusually transparent about evidentiary levels: three gate-exchange-derived items are labeled as such, the granularity invariant's χ-P6 inference limitation is named, and the seam framing is attributed to practitioner origin at the gate. The one concern: the agent-proposed settled/open partition at commitment gating was ratified by proceeding to ARCHITECT rather than being independently restated. The practitioner answered the guarantee-justification challenge directly (a substantive contribution) but did not independently enumerate which decisions they considered settled vs. which they treated as open. The partition itself is not assessed to be wrong — it accurately tracks the evidence — but it was agent-generated and practitioner-ratified without re-derivation. |

---

## Pattern-Specific Assessments

### Agent-proposed synthesis: "boundary-composed directive" as winning option

The where-fork (bridge-side vs. ensemble-side) was resolved toward a third option that neither original pole offered: inject a destination-keyed directive at the marshalling boundary per-dispatch, keeping ensembles destination-agnostic. This synthesis was proposed in Spike χ F-χ.4, accepted by the practitioner after one breadth-grounding round (χ.2, four additional deliverable types across three dispatch types).

**Evidence-forced or framing-adopted?**

The synthesis resolves a genuine dilemma: the bridge-side pole is fragile (F-χ.1, 2/3 ambiguity with no robust rule) and the ensemble-side pole erodes ADR-025 reusability (static coupling). The synthesis emerges from the failure modes of both poles being confirmed by spike data, not from agent preference. F-χ.2 then provides direct evidence that the directive-in-input mechanism produces reliable compliance (n=4 after χ.2 breadth runs). The acceptance after one breadth-grounding round is appropriate — the practitioner chose to run χ.2 before committing, which is exactly the "ground the fork first" discipline the engagement trajectory documents.

One honest gap: the synthesis's reusability argument ("capability ensembles stay destination-agnostic") is derived from ADR-025 as a prior-ADR principle, not from spike evidence. If ADR-025's destination-agnostic library principle is wrong or over-valued, the synthesis's advantage over static ensemble coupling is weaker. This is prior-ADR composition risk, but the principle is established (not contested) and the argument audit did not flag it. **Assessment: evidence-forced within the constraint set, not agent-framing-adopted without test.**

### Practitioner-originated seam framing: absorption risk

The seam framing ("form seam as a cost specific to delegation architecture; absent in single-model flows; two-halves-of-parity") was introduced by the practitioner at the gate and folded into ADR-035 the same turn. The R3 audit verified internal consistency and found the addition sound. The agent extended the framing (context-seam distinction, FC-51 mapping to three seam types, scope-boundary elaboration citing OQ #27 and ensemble-quality-orthogonal declaration), rather than absorbing it verbatim.

**Is the extension sycophantic or substantive?**

The extension added scope-boundary precision (the semantic-coherence seam is distinct and deferred with a named accountability mechanism — FC-51/PLAY). The precision is testable against the existing architecture: FC-51 `TurnDecision` instrumentation does distinguish wrong-form from wrong-action turns, and the DISCOVER ensemble-quality-orthogonal declaration is a prior artifact. The elaboration draws on prior ADRs and named instrumentation, not on agent reasoning without grounding.

The absorption was rapid (same turn). Rapid absorption is the risk marker. But the R3 audit ran the seam framing through all four verification questions and found: claim 1 (structural) sound, claim 2 (two-halves-of-parity) consistent with ADR-034 scope, claim 3 (semantic seam excluded) consistently maintained across Decision/Conditional Acceptance/Consequences, claim 4 (mechanism vs. content quality) appropriately scoped. The R3 audit is external to the drafting session; finding the rapid absorption internally consistent after external audit is the appropriate check. **Assessment: substantive extension, not sycophantic reinforcement. Rapid absorption risk does not materialize here given the audit outcome.**

### Cross-ADR framing propagation: ADR-035 inheriting from ADR-034/033

ADR-035 inherits the seam framing, the "forms half of parity" positioning, and the FC-51 discrimination mechanism from the prior ADR corpus. The question is whether any of these framings were adopted into ADR-035 without independent testing.

- The "faithful marshalling leaves form open" position (ADR-034 fidelity FC) is independently grounded in Spike φ's separability probe against the real WP-LB-G artifact. The probe confirms the store held prose-framed content, so faithful marshalling delivers prose — the form gap is real, not inferred.
- The granularity invariant inheriting from ADR-033 F3-1 is the cleaner concern. The provenance check explicitly labels F3-1 as "recorded concession / wrapper-residual watch point, not a driver finding" and the structured-multi-file alternative as not probed. This is transparent — but the framing adoption still occurred: the granularity invariant is stated as the ADR's answer to χ-P6, where χ-P6 tested only the unparseable-implicit-convention failure mode, not a structured-format alternative. The argument audit carried this as P2-F2 at every round.
- The Conditional Acceptance shape and PLAY escalation order are labeled "drafting-time synthesis applying ADR-097's grounding filter." This is correctly labeled.

**Assessment: One non-trivial cross-ADR framing adoption without independent test — the granularity invariant as inference from F3-1 + χ-P6, where χ-P6 cannot close the structured-multi-file door it did not test. This is known and labeled in the ADR. It is not hidden; it is not correctable without running the probe. The concern is accuracy of evidentiary claims, not sycophancy.**

### Rebuttal-elicitation check: rejected harder-enforcement alternative

The harder-enforcement alternative (output_schema-as-enforcement with reject-and-retry, or submit_file structural guarantee) received its deepest engagement at the practitioner's gate challenge: "What's the drawback of harder enforcement? If we need a guarantee isn't that justified?" The agent's response produced the "neither available nor required" analysis, which the R3 audit found sound across four verified claims (form not structurally enforceable, schema-validation inherits heuristics, submit_file guarantees slot not form, "not required" conditioned on client-side affordances with explicit carve-out).

This is a positive signal for rebuttal-elicitation quality: the challenge actually deepened the analysis and produced three gate-exchange additions (decision 4's detect-and-refuse escalation form, the guarantee-analysis subsection, Conditional Acceptance delegation-precondition bullet). The practitioner's challenge was a genuine test of the rejected alternative, not a rhetorical question; the agent's response is verifiable against ADR-033/034 architecture and Spike χ evidence. The R3 audit confirms the additions hold. **Assessment: rebuttal-elicitation was adequate at the gate for the harder-enforcement alternative.**

### Agent-proposed settled/open partition ratified without re-derivation

The practitioner ratified routing to ARCHITECT "for consistency" and did not independently enumerate the settled/open partition. This is the lightest concern in the batch. The partition itself is accurately tracked (the ADR's Conditional Acceptance explicitly names what remains open: trajectory compliance, granularity invariant under real loop-driver, escalated-tier behavior, delegation-fires precondition). The agent-generated partition is not optimistic — it acknowledges the model-compliance-dependent nature of the form contract and the n=4 grounding limitation.

The risk is structural, not evidential: a practitioner-unratified partition is more likely to encode agent preferences in the settled/open boundary than a partition the practitioner independently derives. In this case, the argument audit's three rounds provide the external check the practitioner's non-re-derivation would otherwise leave absent. **Assessment: minor process gap, not a substantive error given audit coverage.**

---

## Interpretation

The overall pattern is earned confidence with two labeled concerns that are known to the ADR and open at ARCHITECT.

**Evidence for earned confidence:**

1. Five consecutive "ground further" choices before drafting (engagement trajectory steps 1–5). The diagnostic spike before D1/D2 disposition, the $0 local run to confirm D2 universality, the $0 probe to ground the where-fork, the breadth runs before ADR drafting — this sequencing matches the anti-sycophancy discipline visible in the prior loopback DECIDE snapshot.

2. Argument audit convergence across three rounds (R1: 5 issues; R2: 1 issue, Convergence-Saturation signal triggered; R3: 0 P1/P2, scoped gate-additions verified). The convergence direction is toward more hedging, not stronger claims — P2-1 weakened the structural-guarantee framing; P2-2 softened the Spike β correction claim; P2-3 corrected ADR-034 misattribution. That is the right direction.

3. The harder-enforcement alternative received its most substantive engagement at the gate, not its shallowest. This is the inverse of the declining-alternative-engagement pattern.

4. Three gate-derived additions are labeled as gate-exchange-derived in the provenance check. The ADR does not obscure the boundary between spike-grounded findings and gate-synthesis.

5. Two framing-P2s (AS-9 analogy tension, structured-multi-file alternative not tested) were held for practitioner gate decision rather than silently dropped. The practitioner saw the flags; proceeding to ARCHITECT with them open is an informed choice, not an oversight.

**The two known concerns carried to ARCHITECT:**

Concern 1 (granularity invariant — P2-F2): The granularity-invariant framing forecloses a structured-multi-file contract without testing it. χ-P6 tested an implicit multi-file convention, not a JSON-array format. The door was closed by design inference, not by evidence. This is labeled in the ADR and in every audit round. ARCHITECT does not inherit a hidden assumption — it inherits a named gap.

Concern 2 (AS-9 analogy — P2-F1): The ADR recruits AS-9 vocabulary ("structurally framework-owned") while choosing a model-compliance-dependent mechanism. The distinction between directive-presence (structural) and form-compliance (model-reliant) is made explicit in the Consequences section — but the tension with AS-9's origin (structural enforcement adopted because model trust failed under probing) is acknowledged but not fully surfaced. The R3 audit did not re-flag this as a new issue since it was a held gate item.

**Is this sycophantic reinforcement or earned confidence?**

Sycophantic reinforcement at this gate would show: the boundary-composed directive adopted from agent synthesis without testing, the seam framing absorbed verbatim and uncritically, alternative engagement declining, the guarantee challenge deflected rather than engaged. None of these hold. The synthesis is evidence-forced within the constraint set; the seam framing is substantively extended and externally audited; alternative engagement reached its highest quality at the gate; the guarantee challenge produced analysis that the R3 audit verified as sound.

The pattern is earned confidence with two labeled inheritance items (granularity-invariant probe gap, AS-9 analogy tension) that are known, named, and appropriate for ARCHITECT rather than DECIDE to resolve.

---

## Recommendation

**No Grounding Reframe warranted.**

The phase shows consistent probe-before-draft discipline, argument audit convergence toward more conservative claims, genuine gate engagement on the rejected harder-enforcement alternative, and transparent provenance labeling of drafting-time synthesis and gate-derived additions.

---

## Carry-Forward Advisories for ARCHITECT

**Advisory 1 (granularity invariant — structured-multi-file probe gap):** The granularity invariant (one dispatch → one client-tool deliverable; multi-file is across-turn loop-driver decomposition) was inferred from χ-P6's implicit-convention failure without testing a structured-format multi-file contract (e.g., JSON array of filename+content pairs). The ADR frames this as a practical anchor; ARCHITECT should confirm that the across-turn loop-driver decomposition path is architecturally designed before treating the granularity invariant as locked. If the loop-driver's multi-turn decomposition for multi-file work is not built in ARCHITECT's scope, the granularity invariant leaves multi-file deliverables architecturally unserved — not merely deferred.

**Advisory 2 (AS-9 analogy tension — carry-forward from P2-F1):** ADR-035's form contract is model-compliance-dependent. AS-9's structural enforcement was adopted in ADR-033 precisely because model trust failed under probing. ARCHITECT should not treat the form contract as structurally equivalent to single-step enforcement. The detect-and-refuse backstop (Decision 4) is the mechanism that moves toward structural enforcement if PLAY shows compliance failures. ARCHITECT's module decomposition should make Decision 4's detect-and-refuse gate a named interface point — not an optional future addition — so that the escalation path from directive-and-trust to detection gate to schema-retry has a clear architectural home.

**Advisory 3 (delegation-fires precondition — from Conditional Acceptance):** ADR-035's form contract only exercises when the loop-driver chooses to delegate via `invoke_ensemble`. The form-contract PLAY validation is gated on delegation continuing to fire. ARCHITECT's instrumentation design should include a diagnostic that distinguishes form-compliance failures from delegation-not-firing failures — otherwise PLAY results for ADR-035 will be ambiguous (is the form contract broken, or is the loop-driver bypassing delegation on the task types PLAY exercises?).

**Advisory 4 (synthesizer timeout + directive injection placement — F-χ.3 BUILD design question):** If the synthesizer (the agent the boundary directive addresses) times out and the D1 fallback activates (last successful agent's output), the fallback agent was not given the bare-output directive. Whether the directive should be injected to the synthesizer specifically or as shared context across all pipeline agents is a BUILD scenario-group decision. ARCHITECT's module decomposition should make the injection point explicit in the dispatch-input composition interface — not leave it implicit in "the boundary directive is injected" — so that F-χ.3's failure path has a defined behavior.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block ARCHITECT phase progression.*
