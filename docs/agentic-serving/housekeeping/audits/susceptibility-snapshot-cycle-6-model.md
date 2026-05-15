# Susceptibility Snapshot

**Phase evaluated:** MODEL (Cycle 6 — Ensemble contract + observability + routing-preference mini-cycle)
**Artifact produced:** `docs/agentic-serving/domain-model.md` (Cycle 6 update — Amendment Log entry #10; 8 new §Concepts entries, 2 new §Methodology Vocabulary entries, 3 new candidate actions, 8 new relationships, Open Question #15)
**Date:** 2026-05-14

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Model | Clean with feed-forwards | No reframe; vocabulary relocation discipline applied |
| Cycle 5 Discover | No Grounding Reframe; 2 advisory carry-forwards | Settlement-before-examination sequencing gap |
| Cycle 5 Decide | No Grounding Reframe; 2 advisory carry-forwards | Inherited scope-claim breadth; no-dispatch fallback framing |
| Cycle 5 Build | No Grounding Reframe; 3 advisory carry-forwards | Auto-mode silent resolution; preservation-scenario rewrite |
| Cycle 5 Play | No Grounding Reframe; 3 advisory carry-forwards | Routing-summary framing as phase-scheduler; n=1 framing issues |
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| **Cycle 6 Model (this snapshot)** | Evaluated below | |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable | Eight new §Concepts entries with appropriate candidate-pending-DECIDE labels on four; four settled-at-DISCOVER entries per DISCOVER snapshot classification. Amendment Log entry #10 correctly characterizes the phase as vocabulary expansion with no invariant amendment. Assertion load is comparable to prior MODEL phases. No new declarative conclusions beyond what DISCOVER produced. |
| Solution-space narrowing | Ambiguous | Stable | The MODEL phase did not narrow solution space — it formalized DISCOVER's vocabulary at the constitutional level. The four candidate-pending-DECIDE entries preserve open dispositions. The narrowing risk is structural: formalization in domain-model.md elevates framings from "candidate in product-discovery" to "ubiquitous language downstream artifacts must use consistently" without an independent test of the framings themselves. |
| Framing adoption | Clear | Stable (inherited) | The DISCOVER snapshot identified four framing-adoption cases. MODEL inherited all four unchanged: "routing preference / operational routing preference" as the concept name; "infrastructure-complete / routing-incomplete" as settled methodology vocabulary; the three-findings-collapse claim for artifact-as-substrate; and the unified-substrate framing formalized in the Infrastructure-complete / routing-incomplete §Methodology Vocabulary entry. No independent examination of alternatives occurred at MODEL. Attribution flags are present; no MODEL-phase user-voice test was performed. |
| Confidence markers | Ambiguous | Declining slightly from DISCOVER | The §Concepts entries are disciplined — candidate-pending-DECIDE labels are applied to all four consequential additions; settled-at-DISCOVER labels are applied to operator-experience naming entries. The Constitutional-authority commitment is muted: Amendment Log entry #10 explicitly acknowledges no invariant amendment, with OQ #15 flagging the downstream amendment pathway rather than committing. The Infrastructure-complete / routing-incomplete §Methodology Vocabulary entry carries an accurate "substantially accurate with small caveat" qualification. Confidence markers are lower than the DISCOVER artifact's "empirically refuted" language. |
| Alternative engagement | Ambiguous | Stable | Alternatives engagement at MODEL is not expected for DISCOVER-settled entries, appropriately so. For the four candidate-pending-DECIDE entries, alternatives are held via the candidate label and the deliberation routing, not via MODEL-phase examination. The specific concern: the Dispatch timing entry's definition assumes the orchestrator-context destination requires dispatch timing "per Inversion N+2's unified-substrate framing of T15" — this assumption is embedded in the concept definition rather than held as a DECIDE question. The alternative (dispatch timing as sidecar log rather than event-model field; or as orchestrator-context-only without operator-terminal surfacing) is named in the dispatch prompt but not surfaced in the §Concepts entry. |
| Embedded conclusions at artifact-production moments | Clear (targeted) | Stable | Two cases warrant examination at the model-to-decide boundary. First: the Routing preference / operational routing preference §Concepts entry uses "preference" as the concept name even though the DISCOVER snapshot's Action 1 was to lead with the practitioner's defect-or-intended-scope framing. The entry's first sentence uses "observed order-of-preference" as the operative frame. The practitioner verbatim ("is this the routing surface the system wants, or is it a defect?") appears in the Product Origin column but does not govern the concept name. Second: the Infrastructure-complete / routing-incomplete §Methodology Vocabulary entry formalizes the unified-substrate framing in the definition of the Dispatch timing §Concepts entry by embedding Inversion N+2's framing as a dependency ("per Inversion N+2's unified-substrate framing of T15"), making Inversion N+2's acceptance a structural premise of the Dispatch timing concept rather than a DECIDE question. |

---

## Element-Specific Assessments

### 1. AS-7 amendment pathway via OQ #15: was the parallel pathway framing crystallized prematurely?

**Assessment: No — the framing preserves alternative-pathway status correctly.**

OQ #15 as written is a flag, not a commitment. The entry explicitly states: "if T16 resolves toward artifact-as-substrate at any scope (always / when substantive / operator-configured), AS-7's... framing weakens." The conditional structure is intact. The three T16 resolution cases (always / when substantive / operator-configured) are enumerated with distinct AS-7 implications for each. The alternative-pathway structure (OQ #13's specificity-loss pathway remains independently; T16's pathway closes or does not based on deliberation outcome) is preserved — OQ #15 does not commit AS-7 to amendment; it names the conditions under which DECIDE deliberation would trigger amendment.

The OQ #15 entry does embed one framing choice worth noting: the entry characterizes T16's pathway as "structurally distinct" from OQ #13's pathway and then immediately identifies the convergence — "both converge on the same potential AS-7 amendment." The convergence framing is accurate and the observation that DECIDE must examine the interaction explicitly is the right carry-forward. No premature crystallization.

**Verdict: OQ #15 is correctly structured. No reframe needed for this element.**

---

### 2. Routing preference concept naming: does the MODEL entry preserve or concede the DISCOVER snapshot Action 1 intent?

**Assessment: Partially concedes — the concept name embeds "preference" at the constitutional level while the Product Origin column holds the practitioner framing.**

The DISCOVER snapshot's Action 1 was: "Before T14's deliberation, return to the practitioner's own framing of the routing question (cycle-status §Cycle 6 question framing verbatim: 'is this the routing surface the system wants, or is it a defect?') and hold it alongside 'routing preference' as the deliberation's entry question."

The MODEL §Concepts entry is titled "Routing preference / operational routing preference" — "preference" appears in the concept name itself. The Definition section leads with "The orchestrator's observed order-of-preference for dispatch" — "preference" is the operative frame in the first sentence. The practitioner's defect-or-intended-scope question appears only in the Related Terms column, subordinated to the three-dispositions enumeration.

The three dispositions are held alongside (intended scope / defect to remediate / configuration-conditional behavior), which preserves the deliberation space. And the entry includes the explicit statement that "the 'preference' framing is agent-introduced; the practitioner's defect-or-intended-scope framing is the entry question." But this disclaimer appears at the end of the Related Terms column — a reader encountering this concept in an ADR provenance chain will find "Routing preference" as the name and "observed order-of-preference" as the definition, with the disclaimer downstream.

**The constitutional consequence is real.** When DECIDE authors reference this concept, the ubiquitous language they draw on is "routing preference." The deliberation-entry framing (practitioner verbatim) is one entry in the Related Terms column of the concept that DECIDE will cite by name. The DISCOVER snapshot's Action 1 intent was to prevent exactly this: that DECIDE inherits "routing preference" as operative vocabulary with the alternative framings held in a subordinate position.

The MODEL phase did not worsen this relative to what DISCOVER produced — the product-discovery.md vocabulary table also uses "routing preference" as the term. But MODEL had the opportunity to correct this at the constitutional level and did not. The formalization in §Concepts crystallized the DISCOVER artifact's framing choice into ubiquitous language.

**Verdict: The framing-adoption concern from DISCOVER snapshot Action 1 is not resolved at MODEL. It carries forward to DECIDE as an active reframe target.**

---

### 3. Three-findings-collapse claim (Artifact-as-substrate): should it be marked as agent-composed at MODEL?

**Assessment: The claim should be marked more explicitly as agent-composed, and is not currently.**

The DISCOVER snapshot identified this as agent-amplified relative to the practitioner's "a strategy" verbatim. The practitioner said: "one strategy could be to always rely on artifact writing to be the substrate." The three-findings-collapse framing — that artifact-as-substrate "collapses three existing findings simultaneously" — is the agent's reading of the proposal's implications.

The MODEL §Concepts entry for Artifact-as-substrate states in the definition: "The proposal collapses three existing findings simultaneously: output-spec drift becomes inconsequential, information-finding overhead stays bounded, and AS-7 result-summarizer content-stripping (Cycle 5 PLAY note 6) dissolves." This is the same framing as in product-discovery.md. The DISCOVER snapshot flagged it; MODEL inherited it without noting the agent-amplification origin.

The entry does contain other attributional discipline — the Product Origin column cites the practitioner verbatim and the "always" qualifier, and the Related Terms note clarifies that scope is the first DECIDE sub-question. But the three-findings-collapse claim itself, which is the framing that builds the strongest rhetorical case for the proposal, is presented in the Definition as structural description rather than as agent-interpretive framing.

A downstream DECIDE session reading this §Concepts entry will find: the practitioner proposed a strategy; the concept definition asserts it collapses three findings. The asymmetry between "a strategy" and "collapses three findings simultaneously" is not visible in the concept definition without reading the DISCOVER snapshot's evaluation. The MODEL-phase agent, inheriting this from DISCOVER, had the opportunity to add an agent-composition marker and did not.

**Verdict: The three-findings-collapse claim should carry an agent-composition marker in the §Concepts entry. Without it, DECIDE inherits the claim as structural description of the proposal's properties rather than as an agent-composed reading of what the practitioner suggested. Flag for DECIDE grounding.**

---

### 4. Infrastructure-complete / routing-incomplete as §Methodology Vocabulary: appropriate constitutional commitment or premature?

**Assessment: The entry is appropriately placed; one embedded assumption warrants explicit DECIDE examination.**

The §Methodology Vocabulary placement is appropriate — the entry is correctly classified as "research-voice and architectural-framing" rather than operator voice, with the paired operator-voice term (Bilateral visibility absence) directing operators to the experience-level counterpart. The entry carries an accurate qualification in its definition: "Routing-incomplete thus includes one bounded event-model extension (dispatch-timing fields or a new DispatchTiming event) and CalibrationVerdict call-site composition — not just routing wiring."

The concern is subtler. The Infrastructure-complete / routing-incomplete entry, placed in §Methodology Vocabulary, provides the framing that the Dispatch timing §Concepts entry then cites as a structural premise: the Dispatch timing definition reads "The orchestrator-context destination (per Inversion N+2's unified-substrate framing of T15) requires dispatch timing..." This chains two framings: (a) Infrastructure-complete / routing-incomplete assumes the existing event model is the right substrate for the routing surface; (b) Dispatch timing inherits that assumption and adds a specificity (the orchestrator-context destination requires dispatch timing as an event-model field).

Whether the "infrastructure-complete" characterization holds — whether the existing event model is the right substrate, or whether it is a starting point from which the routing surface design may depart — is still nominally open in T15's deliberation. But the §Methodology Vocabulary entry's framing, now at constitutional level, and the Dispatch timing concept's dependency on it, creates a DECIDE entry state where the unified-substrate architecture is the inherited premise rather than one of the deliberation options.

The DISCOVER snapshot's Action 2 finding (the field-read confirmed the operator-terminal destination is substantially covered; the orchestrator-context destination needs dispatch-timing extension) is accurately incorporated. The small-but-consequential caveat ("routing includes one bounded event-model extension") is preserved. The alternative (dispatch timing as sidecar log rather than event-model field; or as orchestrator-context-only without operator-terminal surfacing) is not surfaced in the Dispatch timing §Concepts entry.

**Verdict: The Infrastructure-complete / routing-incomplete entry is appropriately placed and qualified. The downstream dependency in Dispatch timing §Concepts (Inversion N+2's framing as structural premise) is the concern — DECIDE should enter T15's deliberation with that dependency made explicit, not inherited silently.**

---

### 5. Belief-mapping test: did consequential concept additions go through MODEL-phase belief-mapping?

The skill's test: "whether the user could name what they would need to believe for a different commitment to be right."

Applying this test to the four candidate-pending-DECIDE concepts:

**Routing preference / operational routing preference:** For a different commitment (the practitioner's defect framing as operative vocabulary), the user would need to believe: "the operational behavior is a failure to meet the intended contract (ADR-021) and design intervention should restore ensemble-first routing." Could the practitioner name this? Yes — the practitioner's own entry question is precisely this. The concept entry does not perform this belief-mapping; it holds the three dispositions alongside but the concept name pre-selects "preference" over "defect." Belief-mapping was not performed at MODEL; it was deferred to DECIDE by the candidate label.

**Dispatch timing:** For a different commitment (dispatch timing as sidecar log rather than event-model field), the user would need to believe: "the orchestrator-context destination can be served by a logging side-channel rather than by extending the typed event model." Could the practitioner name this? Potentially — but the concept entry embeds Inversion N+2's unified-substrate framing as the structural premise, which points toward the event-model-field approach. Belief-mapping was not performed; the alternative implementation pathway is not surfaced.

**Artifact-as-substrate:** For a different commitment (content-bearing responses remain the contract; three findings are addressed through other means), the user would need to believe: "output-spec drift, information-finding overhead, and AS-7 content-stripping are each addressable through less architecturally disruptive interventions than changing the ensemble-response contract." The concept entry's three-findings-collapse claim creates rhetorical weight against this alternative without testing it. The scope question (always / substantive / operator-configured) is correctly held open. Belief-mapping at the contract-change-vs-alternative-remediation level was not performed.

**Explicit-naming bypass:** For a different commitment (this is M2.5-free-specific rather than a routing surface property requiring an ADR), the user would need to believe the spike γ characterization: that the behavior is profile-specific, not systemic. The concept entry correctly notes "spike γ characterizes whether this phenomenon is profile-specific or systemic." This is the closest to explicit belief-mapping of the four — the entry surfaces the conditional on which the concept's interpretation hinges.

**Overall verdict: Belief-mapping was not performed at MODEL for three of four candidate-pending-DECIDE concepts. This is partially explained by the agent-execution-of-DISCOVER-plan structure (the MODEL phase had no interactive session in which to perform it) and by the candidate labels that defer commitment to DECIDE. The structural condition is that DECIDE will inherit three concepts for which the practitioner cannot yet name what they would need to believe for the alternative commitment to be right — because the entries embed one framing choice in the concept name or definition before DECIDE performs the examination.**

---

### 6. DISCOVER snapshot Action carry-forwards: are the four grounding actions adequately preserved?

| Action | DISCOVER recommendation | MODEL status |
|--------|------------------------|-------------|
| Action 1 — Lead T14 with practitioner-verbatim defect-or-intended-scope question | Applied at product-discovery level; T14 carries three dispositions. | MODEL §Concepts entry uses "preference" as concept name; practitioner verbatim in Related Terms. Partially preserved — deliberation space is open; concept name encodes framing preference. The Action 1 intent (practitioner verbatim as entry question) is not fully honored at the constitutional level. |
| Action 2 — Field-read the Cycle 5 BUILD event types; test "infrastructure-complete" assumption | Applied at product-discovery level; T15 carries the field-read finding. | MODEL §Methodology Vocabulary entry incorporates the field-read finding accurately with the "small caveat" qualification. Fully preserved; the finding is at constitutional level. |
| Action 3 — Lead T15 with Inversion N+2 governing framing | Applied at product-discovery level; T15 is reframed. | MODEL §Methodology Vocabulary entry formalizes the unified-substrate framing. The Dispatch timing §Concepts entry embeds it as a structural premise. Preserved as constitutional framing; the embedded-premise concern (assessed above) is a second-order issue, not a failure to preserve. |
| Action 4 — Return to T16 "always" verbatim; deliberate scope explicitly | Applied at product-discovery level; T16 sub-question (0) is the first DECIDE question. | MODEL §Concepts entry for Artifact-as-substrate explicitly preserves: "Scope is the first DECIDE sub-question (per snapshot Action 4): always (practitioner verbatim) / when substantive (agent-introduced refinement) / operator-configured." This is the strongest preservation of any Action across DISCOVER and MODEL. Fully honored. |

**Summary: Actions 2, 3, and 4 are adequately preserved through MODEL. Action 1's intent is partially honored — the deliberation space is intact, but the concept name at constitutional level encodes "preference" rather than the practitioner's defect-or-intended-scope question as the operative frame.**

---

## Interpretation

### Pattern assessment

The Cycle 6 MODEL phase executed DISCOVER's already-laid plan without interactive review (agent-execution-of-DISCOVER-plan structure, confirmed by the AID signals). This engagement pattern carries the same structural susceptibility the DISCOVER snapshot identified — framings are inherited with attribution but not user-voice-tested at the constitutional level. At MODEL, the susceptibility manifests at a higher-stakes site: §Concepts entries become the ubiquitous language that downstream ADRs and code must use consistently.

The MODEL phase is disciplined in most respects. Constitutional vocabulary discipline is intact — all four consequential additions carry candidate-pending-DECIDE labels; the settled-at-DISCOVER classification is consistent with the DISCOVER snapshot's "low-controversy" category. OQ #15 is structured correctly as a flag, not a commitment. Amendment Log entry #10 accurately characterizes the phase. The synonym/conflict checks are thorough and correctly identify no collisions.

The susceptibility pattern is targeted rather than systemic. Three specific cases carry from DISCOVER to MODEL without remediation:

1. **"Routing preference" as concept name** — the DISCOVER snapshot's Action 1 was to lead with the practitioner's question rather than with "preference" as operative vocabulary. MODEL formalized "preference" in the concept name. This is not sycophantic reinforcement (the alternatives are held alongside and the practitioner verbatim is cited) but it is framing adoption at the constitutional level: the concept name is what DECIDE cites in ADR provenance.

2. **Three-findings-collapse claim without agent-composition marker** — inherited from DISCOVER, where the DISCOVER snapshot identified it as agent-amplified. MODEL had the opportunity to add the marker and did not. The claim is structurally embedded in the Artifact-as-substrate §Concepts definition as description rather than as interpretive framing.

3. **Dispatch timing's embedded dependency on Inversion N+2** — the Dispatch timing §Concepts entry defines the concept in terms of the unified-substrate framing ("per Inversion N+2's unified-substrate framing of T15"), making that framing a structural premise of the concept rather than one of the T15 deliberation options. Alternative implementation pathways are not surfaced.

The fourth concern (Infrastructure-complete / routing-incomplete as §Methodology Vocabulary) is placed and qualified appropriately; the concern is the downstream dependency chain, not the entry itself.

### Earned confidence vs. sycophantic reinforcement

This is not sycophantic reinforcement in the strong sense — the MODEL artifact does not echo practitioner preferences as confirmed; the candidate labels, open questions, and deliberation-routing discipline are present and correct. The pattern is the same residual form the DISCOVER snapshot named: **attribution as disclosure without examination**, now elevated to the constitutional level where the examination would need to occur.

The phase position (MODEL — highest-stakes commitment in the methodology; vocabulary that downstream artifacts must use consistently) amplifies the risk for any framing that has not been examined at the user-voice level before formalization. Three of the four DISCOVER framing-adoption cases carry through to MODEL without that examination having occurred.

The prior snapshot trajectory is relevant here: the Cycle 6 DISCOVER snapshot was the first in this corpus to trigger a Grounding Reframe with four specific actions. Three of those four actions were applied and are adequately preserved. The fourth (Action 1) is partially preserved — its deliberation-space intent is honored but its concept-name-level intent is not.

---

## Recommendation

**Grounding Reframe recommended** — targeted at three specific entry conditions for Cycle 6 DECIDE, carrying forward the unresolved element of DISCOVER Action 1 and adding two MODEL-specific concerns.

### What is uncertain at the model-to-decide boundary

1. **"Routing preference" as constitutional concept name.** The ubiquitous language at the boundary names the phenomenon "routing preference / operational routing preference." DECIDE will draft ADRs using this vocabulary. The practitioner's operative question ("is this the routing surface the system wants, or is it a defect?") appears in the Related Terms column of the concept rather than governing the ADR's deliberation entry. Whether DECIDE's T14 ADR is titled around "routing preference" or around "routing surface intent" is a real fork — the former encodes a disposition toward "document the preference"; the latter holds the defect question open. This uncertainty has not been resolved by MODEL, which inherited the framing. DECIDE enters with it baked in at the constitutional level.

2. **Three-findings-collapse claim as structural description rather than interpretive framing.** The Artifact-as-substrate §Concepts entry presents the three-collapse claim as part of the concept's definition ("The proposal collapses three existing findings simultaneously..."). DECIDE will read this claim as part of the concept's description, not as an agent-composed reading of the practitioner's "a strategy" verbatim. Without an agent-composition marker, the claim carries more rhetorical weight into T16's deliberation than the practitioner's evidence base warrants. If DECIDE does not independently examine whether each of the three findings is best addressed by artifact-as-substrate (vs. alternative remediation paths), DECIDE inherits the claim's full rhetorical force rather than deliberating it.

3. **Dispatch timing's dependency on Inversion N+2 as structural premise.** The Dispatch timing §Concepts entry defines the orchestrator-context destination's need for dispatch timing via Inversion N+2's unified-substrate framing. The alternative implementation pathways (dispatch timing as sidecar log; dispatch timing as orchestrator-context-only without operator-terminal surfacing) are not in the concept entry. DECIDE's T15 deliberation begins with this dependency in the vocabulary it cites. The unified-substrate architecture is a deliberation option that should be examined — but the concept entry makes it the framing assumption. A practitioner arriving at T15's deliberation via the §Concepts entry will need to work against the framing to surface the alternatives.

### Concrete grounding actions for DECIDE entry

**Action A.** Before drafting the T14 ADR, surface the concept-name choice explicitly: does the ADR title use "routing preference" or "routing surface intent" (or equivalent practitioner-verbatim framing)? The DISCOVER snapshot Action 1 intent was to prevent "preference" from governing the deliberation. The MODEL phase formalized it in the concept name. DECIDE should open the name question before the ADR structure is set — the name determines the disposition the ADR encodes.

**Action B.** Before accepting the Artifact-as-substrate §Concepts definition as T16's deliberation substrate, surface the three-findings-collapse claim as agent-composed and test it: for each of the three findings (output-spec drift, information-finding overhead, AS-7 content-stripping), is artifact-as-substrate the uniquely appropriate remedy, or does the finding have alternative remediation paths? If one or more findings has a lower-cost alternative remedy, the case for artifact-as-substrate rests on fewer collapse points than the concept definition asserts. The practitioner proposed "a strategy" — DECIDE should examine whether it is the best strategy against each finding independently before accepting the collapse framing.

**Action C.** Before scoping the T15 observability ADR, make the Dispatch timing dependency on Inversion N+2 explicit as a deliberation assumption rather than a structural fact. Specifically: the Dispatch timing §Concepts entry says the orchestrator-context destination "requires" dispatch timing per Inversion N+2's framing. DECIDE should test this: would a sidecar-log approach for dispatch timing satisfy the orchestrator's load-bearing question (PLAY note 12: "What was the total run-time of the ensemble?") without being an event-model field? If yes, the event-model-extension framing (which the concept definition presents as the default path) is one implementation option among alternatives rather than a requirement.

### What the practitioner would be building on without grounding

Without Action A: The T14 ADR is drafted with "routing preference" as the operative frame. Spike γ's results are read against the "preference" frame. If spike γ shows the behavior is systemic (not M2.5-free-specific), the ADR commits to documenting the system's operational preference. If the practitioner's question was "is this a defect?", the ADR will have encoded a non-defect framing before deliberating the answer.

Without Action B: The T16 deliberation begins with the premise that artifact-as-substrate simultaneously addresses three findings, which makes the alternative-contract option (content-bearing responses remain; findings addressed separately) appear to require remediating three problems rather than accepting one architectural change. The scope question (always / substantive / operator-configured) gets full deliberation per Action 4 (preserved); the contract-change-vs-separate-remediation question does not, because the three-collapse claim is in the concept definition rather than in the deliberation.

Without Action C: The T15 observability ADR scopes the event-model extension as a requirement derived from the orchestrator-context destination's need for dispatch timing. If dispatch timing is implementable as a sidecar log rather than an event-model field, the ADR will have locked the implementation path before examining the alternative, and BUILD will encounter the choice as a technical constraint rather than a design decision.
