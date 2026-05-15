# Susceptibility Snapshot

**Phase evaluated:** DISCOVER (Cycle 6 — Ensemble contract + observability + routing-preference mini-cycle)
**Artifact produced:** `docs/agentic-serving/product-discovery.md` (Cycle 6 update, 2026-05-14)
**Date:** 2026-05-14

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Research | Grounding Reframe triggered | Autonomous-routing gap named; three grounding actions |
| Cycle 4 Discover | Grounding Reframe triggered | Research-voice transplants; asymmetric readiness mapping |
| Cycle 4 Model | Clean with feed-forwards | No reframe; vocabulary relocation discipline applied |
| Cycle 4 Decide | Grounding Reframe recommended (1 finding) | ADR-015 evidence gap not carried into artifact |
| Cycle 4 Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE |
| Cycle 4 Build | Grounding Reframe (targeted) + 2 advisory | Pre-loaded conditional-acceptance disposition |
| Cycle 4 Play | No Grounding Reframe; 4 advisory carry-forwards | Voice-blurring at synthesis boundary; n=1 findings encoded as settled |
| Cycle 5 Discover | No Grounding Reframe; 2 advisory carry-forwards | Settlement-before-examination sequencing gap |
| Cycle 5 Decide | No Grounding Reframe; 2 advisory carry-forwards | Inherited scope-claim breadth; no-dispatch fallback reasoning at minimum threshold |
| Cycle 5 Build | No Grounding Reframe; 3 advisory carry-forwards | Auto-mode silent resolution; preservation-scenario rewrite |
| Cycle 5 Play | No Grounding Reframe; 3 advisory carry-forwards | Routing-summary framing as phase-scheduler; note 1 label overstatement; note 19 "unchanged" framing |
| **Cycle 6 Discover (this snapshot)** | Evaluated below | |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable | The artifact densely encodes findings from post-hotfix verification 2026-05-14 (findings 1-8) and PLAY notes 1-9, 20. Assertion load is comparable to Cycle 5 DISCOVER — which was also a roll-forward. The concentration point is the Skill Orchestration User's mental model section, where "empirically refuted" language appears for a 2-configuration, 1-post-hotfix-session evidence base. Elsewhere the artifact uses appropriately hedged framing ("candidate," "open question," attribution flags). |
| Solution-space narrowing | Clear (bounded) | Stable | Three-cluster structure was fixed at cycle-entry context drafting (recent commits); DISCOVER confirmed rather than extended that structure. No new clusters surfaced; the binary framings in T14, T15, T16, T17 were present at cycle-status entry and carried unchanged into product-discovery. The narrowing originated at the cycle-entry drafting phase, not during this DISCOVER update. |
| Framing adoption | Clear | Rising relative to prior cycles | Four agent-named patterns — "routing preference," "explicit-naming bypass," "infrastructure-complete / routing-incomplete," "liveness signal" — appear in product-vocabulary with attribution flags. The flags name the attribution; they do not test the framing against user voice. The Cycle 4 DISCOVER snapshot's primary finding (research-voice transplants into product-facing language) is present here in a mitigated form: the framings carry flags, but the flags function as disclosures rather than as tests. No alternative framings for "preference" vs. "defect" vs. "intended scope" appear in the vocabulary table itself — that examination is deferred to T14's deliberation note. |
| Confidence markers | Ambiguous | Stable-to-declining | Vocabulary table disposes 11 new terms with explicit attribution flags and deliberation-routing notes. The "empirically refuted" framing on the Skill Orchestration User's mental model is the highest-confidence marker in the artifact. The 5.3× variance claim and the "across both tested client configurations" generalizations are strong framings relative to their evidence base (n=2 configurations, n=1 follow-on session). Elsewhere the artifact uses candidate / open / deliberation-needed qualifications consistently. |
| Alternative engagement | Ambiguous | Stable | T14 explicitly holds the "revise mental model" vs. "system meets mental model" binary and routes the deliberation to DECIDE. T15 notes the surfaces "are not substitutes" and poses the shared-infrastructure question. T16 enumerates five sub-questions for DECIDE. T17 holds the useful-work-time vs. total-elapsed-time framing without examining the continuous-metric alternative. The alternatives *within* each binary are named; the alternatives *between* the binary frame and a spectrum framing are not examined. |
| Embedded conclusions at artifact-production moments | Ambiguous | Stable | The vocabulary disposition note (end of §Product Vocabulary) explicitly classifies all 11 new terms into deliberation-routing buckets. No term is labeled "settled" that the evidence does not support settling. The "operator-experience naming" bucket (liveness signal, tool-call-emit logging, inference-wait heartbeat, infrastructure-complete / routing-incomplete, bilateral visibility absence, information-finding overhead) is marked "low-controversy; likely settled at DISCOVER" — a conclusion embedded at artifact-production that warrants examination in this snapshot. |

---

## Element-Specific Assessments

### Pattern (a): Research-phase framings inherited into product-facing language without being tested against user voice

The dispatch brief defines the "research-phase substrate" for Cycle 6 as: Cycle 5 PLAY field notes + post-hotfix verification findings 1-8. The question is whether agent-named framings from those sources crystallized into product-discovery vocabulary without being tested against user voice.

**"Routing preference" / "operational routing preference" / "explicit-naming bypass"**

These three terms appear in the vocabulary table with attribution flags reading: "The term names a behavioral phenomenon empirically observed across multiple client configurations. Whether 'preference' is the right framing (vs. 'defect,' 'intended scope,' or 'configuration-conditional behavior') is Cycle 6 Open Question 1 / Tension 14."

The attribution flag names the alternatives correctly. But the artifact also uses the term operationally — as the active vocabulary — in the Skill Orchestration User's jobs-and-mental-models entry, in the Tension 14 framing, and in Inversions N+1 and N+3. The alternatives are held in the tension's resolution note, not tested in the product-discovery language itself. A practitioner reading the Skill Orchestration User section encounters "operational routing preference" as the working concept; they reach the alternative framings only by following the tension citation.

The risk: DECIDE inherits "routing preference" as the operative framing because it is the active vocabulary in the stakeholder-language sections. The alternative framings ("defect," "intended scope," "configuration-conditional behavior") are correctly disclosed in the tension but not given equal rhetorical weight in the user-voice sections. This is the Cycle 4 DISCOVER pattern in a mitigated form — the attribution is present, but the test against user voice is not.

**"Infrastructure-complete / routing-incomplete"**

This framing originates from the Cycle 5 PLAY susceptibility snapshot (Advisory 3) — an agent-analytical framing that reframed note 19's "unchanged" characterization. The product-discovery artifact records it as "Cycle 5 PLAY susceptibility snapshot reframing... accepted at Cycle 6 entry; replaces the prior 'observability is unchanged across cycles' framing."

The framing is accurate and load-bearing. The attribution is explicit. The question is whether it is operator voice. The Cycle 5 PLAY snapshot that introduced this framing was itself an evaluation artifact, not a practitioner-engagement record. The practitioner's voice on the observability gap is: *"the observability of the routing still seems lacking to me... This was flagged last time."* (PLAY note 19 verbatim). The practitioner's voice is absence-of-routing-telemetry, not "infrastructure-complete / routing-incomplete." The analytical distinction (infrastructure exists; routing is what is missing) is sound architecture — but it is the agent's reading of what the practitioner's observation means for the system's design state. The product-discovery vocabulary entry uses it as settled operator voice; the practitioner verbatim that grounded it was not tested against this analytical frame.

This matters because it sets up a DECIDE framing that points toward "wire the routing surface" rather than examining whether the existing event model is the right substrate for the routing surface that needs to be designed. If the infrastructure is "complete" by assumption, DECIDE scopes to routing wiring. If the infrastructure is "starting point" by examination, DECIDE may find the event model needs extending before routing wiring is meaningful.

**"Liveness signal" / "tool-call-emit logging" / "inference-wait heartbeat"**

These terms are marked "operator-experience naming (low-controversy; likely settled at DISCOVER)" in the vocabulary disposition note. The low-controversy classification is worth examining.

"Liveness signal" is the practitioner's concept (the gap was observed directly: 10+ minute console silence). The agent's contribution is the naming pattern and the two specific candidates (tool-call-emit logging; inference-wait heartbeats). Whether these two candidates are the right technical implementations — as opposed to, say, a streaming progress counter, an SSE push from the serve layer, or a client-side timeout-with-notification — is not a settled question at DISCOVER. The artifact's "two low-cost candidates" framing is directionally useful, but it narrows the implementation space at a phase that should be surfacing the space, not pre-selecting candidates. The operator's job is named correctly ("I want signal during in-flight states, not just on completion"); the two candidate patterns are agent-named and have been carried from the cycle-status drafting into the product-discovery operator jobs section without being labeled as candidates-pending-DECIDE-deliberation.

**"Artifact-as-substrate"**

This is the clearest practitioner-voice term in the new vocabulary: verbatim-attributed to the practitioner ("this also to me points to an ensemble design shape for agentic-serving... one strategy could be to always rely on artifact writing to be the substrate"). The artifact records the attribution and correctly routes to DECIDE.

The expansion work — the five sub-questions (boundary, contract, client access, cleanup, backward compat) and the three-findings-collapse claim — is agent-composed. The expansion is internally coherent and the three findings it "collapses" (output-spec drift, information-finding overhead, AS-7 summarizer content-stripping) are genuinely related. The risk is that the three-collapse framing creates a stronger case for the artifact-as-substrate proposal than the practitioner verbatim alone warrants: the practitioner observed a code output "buried in the ensemble output" and suggested writing to artifact as a strategy; the agent translated "a strategy" into "collapses three existing findings simultaneously." These are different confidence levels. The product-discovery artifact presents the three-collapse framing as the tension's substantive case, not as an agent-amplified reading of a practitioner suggestion.

**Assessment on pattern (a):** Attribution flags are present across the board — this is a material improvement over Cycle 4 DISCOVER, where transplants were unlabeled. The flags function as disclosures of provenance, but do not constitute testing against user voice. The user-voice test that DISCOVER is supposed to perform — "does this framing read in operator voice or methodology/research voice?" — is implicit in the attribution flags' categories but is not executed as a deliberate examination for any of the four framings named above. The Cycle 4 DISCOVER finding ("research-voice transplants propagating without attribution") is resolved at the attribution level; the underlying susceptibility (framings crystallizing before being tested against what stakeholders actually say and mean) is mitigated but not closed.

---

### Pattern (b): Value tensions that surfaced as spectra but collapsed into binary framings without the alternatives being examined

**T14: "Revise stakeholder mental model" vs. "system meets existing mental model"**

The tension correctly holds both resolution paths and routes the deliberation to DECIDE with an empirical test (spike γ). The binary framing is appropriate for DECIDE entry: these are genuinely alternative design commitments, not points on a spectrum. One case where the binary obscures a middle position: the tension does not examine "document the routing preference alongside the existing mental model without committing to either resolution" — a minimal-intervention third option that might be appropriate given the n=2 configuration evidence base. That said, T14 is the tension best suited for DECIDE's deliberation format; the binary framing serves a decision artifact better than it would serve a discovery artifact.

What is missing is the n-evidence qualification. The "empirically refuted" framing on the Skill Orchestration User mental model appears in the stakeholder section before T14's deliberation note. A practitioner reading the stakeholder section encounters strong language ("empirically refuted under both tested client configurations"); a practitioner reading T14's resolution paths encounters the spike γ qualification ("whether the preference is MiniMax M2.5-free-reasoning-shape-specific or systemic to the cheap-cloud-orchestrator pattern"). The two sections are coherent together; read independently, the stakeholder section's language is stronger than the evidence base (n=2 configurations, n=1 post-hotfix verification session) warrants.

**T15: "Operator-terminal surface" vs. "orchestrator-context surface"**

The tension correctly notes that the surfaces "are not substitutes" and poses the shared-infrastructure question. The binary framing here is less appropriate than in T14 because the tension's own text suggests the two surfaces may be the routing destinations of a single event-emission infrastructure — which is a spectrum position (unified source, two routable destinations) that the "operator-terminal vs. orchestrator-context" framing obscures. The Inversion N+2 form makes this explicit: "the two axes collapse into one architectural concern: bidirectional flow between the orchestrator's reasoning context and the dispatch telemetry it produces." But the tension heading (T15) presents the surfaces as alternatives rather than as destinations of a common substrate. A practitioner reading only T15 would be inclined toward "which one?" framing; a practitioner reading T15 + Inversion N+2 would be inclined toward "how do they share infrastructure?" framing. The inversion and the tension are in tension with each other on this point — the inversion is the more accurate framing of the architectural question.

This is a case where the binary framing is not the failure; the failure is that the unified-substrate framing exists in the inversion but does not inform the tension's opening question. A Grounding Reframe is warranted at DECIDE entry to ensure T15's deliberation starts from the inversion's framing (shared-infrastructure-with-two-routing-destinations) rather than the tension's binary (which surface?).

**T16: "Content-bearing" vs. "artifact-as-substrate"**

This is a genuine binary — these are mutually exclusive architectural commitments for how an ensemble response carries its deliverable. The five sub-questions frame the design surface correctly. The case that warrants attention is the "always rely on artifact writing" qualifier in the original practitioner verbatim vs. the tension's treatment.

The practitioner said "always rely on artifact writing to be the substrate." Tension 16 introduces a boundary condition: "capability ensembles whose deliverable is substantive (code, structured analyses, long-form text)." This is a reasonable refinement — the practitioner's "always" likely implied "for this class of output" rather than "for single-word answers." But the refinement narrows the scope of the commitment in a direction the practitioner did not explicitly specify, and that narrowing is agent-introduced. Whether "always" or "when substantive" is the right scope is a DECIDE question that should be deliberated, not pre-resolved at DISCOVER. The current tension framing carries the refinement as settled, which means DECIDE inherits a narrower scope than the practitioner's verbatim suggested.

**T17: "Useful-work-time" vs. "total-elapsed-time"**

This is the tension most clearly identifiable as a spectrum collapsed into a binary. The operator-experience metric question is not which one of two metrics to use; it is what the right characterization of the latency problem is. The practitioner verbatim ("it had trouble finding the right ensemble and it seemed like the time to find trivial info was long") describes an information-retrieval efficiency problem, not a choice between two metrics. "Useful-work-time vs. total-elapsed-time" is the agent's naming of what the practitioner observed — a ratio framing that may or may not be the right analytical lens. An alternative framing: the problem is the orchestrator's failure to use information it already had (it "saw it in the list" but did not use it), which is an orchestrator-reasoning-loop problem, not a metric problem. The two routing-axis sub-questions (a) and (b) embedded in T17 are the real substance; the binary framing is a meta-framing imposed on top of them that may direct DECIDE toward metric-selection deliberation when the underlying question is orchestrator-information-retention behavior.

**Assessment on pattern (b):** T14 has an appropriate binary framing with the correct caveats. T15's binary framing is in tension with Inversion N+2's more accurate unified-substrate framing — DECIDE entry needs the inversion's framing to govern, not the tension's heading. T16's binary is appropriate but carries an agent-introduced scope narrowing ("substantive" vs. "always") that reduces a practitioner verbatim without deliberation. T17's binary may be misdirecting — the framing of a ratio metric may be an analytical lens imposed on a routing-behavior problem, and DECIDE should examine whether the metric frame or the orchestrator-reasoning-loop frame is the right way to address what the practitioner observed.

---

## Interpretation

### Pattern assessment

The Cycle 6 DISCOVER artifact is the strongest-attributed product-discovery update in the corpus. Every new framing carries an attribution flag or explicit provenance note; the vocabulary disposition table routes each term to a deliberation bucket with the appropriate epistemic status. This represents a material improvement over Cycle 4 DISCOVER (where transplants were unlabeled) and over Cycle 5 DISCOVER (where settlement-before-examination occurred at the gate).

The susceptibility pattern is not the Cycle 4 form (unlabeled transplants) but a residual form: **attribution as disclosure without examination**. The flags correctly identify which framings are agent-named, but they do not substitute for the test that DISCOVER is supposed to perform — whether the framing reads in user voice, whether an operator would use this term naturally, whether the analytical lens the agent applied to an empirical observation is the right lens for what the practitioner observed. Four cases above fail this test at varying severity levels.

The engagement pattern (agent-roll-forward of prior practitioner work, with practitioner's deep engagement front-loaded at cycle-entry drafting) is the structural condition that makes this form of susceptibility likely. The practitioner authored the cycle-entry context across multiple sessions (recent commits); the DISCOVER update executed that context. The question the snapshots are designed to catch — "did the roll-forward inherit framings the practitioner accepted at entry without testing against the user voice the product-discovery is supposed to land in?" — is affirmatively answered for four specific cases:

1. "Routing preference" as operative vocabulary in stakeholder sections, while "defect" / "intended scope" alternatives are held in a tension deliberation note.
2. "Infrastructure-complete / routing-incomplete" as settled operator voice, when it is a snapshot-analytical framing of practitioner-stated absence-of-routing-telemetry.
3. T15's binary heading in tension with Inversion N+2's unified-substrate framing.
4. T16's agent-introduced scope narrowing ("substantive" vs. "always") reducing a practitioner verbatim without deliberation.

Two additional signals warrant noting without rising to Grounding Reframe severity:

- The "empirically refuted" language on the Skill Orchestration User mental model is strong relative to the evidence base (n=2 configurations, n=1 follow-on session). The claim is directionally supported; the language implies a settled empirical finding that spike γ is explicitly designed to test. A practitioner reading the stakeholder section before T14's deliberation note will encounter stronger certainty than the evidence supports.
- The 5.3× variance claim (44m 32s vs. 8m 28s) is presented in both the Tool User refinement and the cycle-status as a sharp empirical finding. n=2 under different prompts is not a variance characterization — it is two data points. The variance is real; the characterization as "5.3×" implies quantitative precision the observation does not support.

### Earned confidence vs. sycophantic reinforcement

The new tensions (T14, T15, T16, T17) are all correctly routed to DECIDE deliberation — none is labeled settled, none encodes a design commitment the evidence does not support. The vocabulary disposition table is disciplined. The post-hotfix verification findings are presented with appropriate "DISCOVER attends" framing in the cycle-status and as open questions in the product-discovery text.

The pattern above is not sycophantic reinforcement in the strong sense — the artifact does not echo back the practitioner's preferences as confirmed. It is a narrower condition: agent-analytical framings that convert empirical observations into conceptual frames and then carry those frames into product-voice sections without returning to the practitioner-verbatim level to test whether the conceptual frame matches what the practitioner actually observed and means.

### Prior advisory carry-forward status

| Advisory | Origin | Status at Cycle 6 DISCOVER |
|----------|--------|-----------------------------|
| Advisory 1 — Routing-summary framing should not schedule downstream phases | Cycle 5 PLAY snapshot | Honored. Cycle 6 opened DISCOVER + DECIDE as parallel rather than sequenced. The cycle-status treats the three clusters as linked but not gated on each other. |
| Advisory 2 — Note 1's "structurally inadequate" label vs. note body | Cycle 5 PLAY snapshot | Partially honored. Product-discovery does not inherit "structurally inadequate"; it routes the BUILD-close practice gap correctly as a scenario-addition question. DECIDE will deliberate it explicitly. |
| Advisory 3 — Note 19's "unchanged" framing | Cycle 5 PLAY snapshot | Honored. The product-discovery artifact replaces "unchanged" with "infrastructure-complete / routing-incomplete." The new framing is accurate and the reframe is attributed to the snapshot. The secondary susceptibility — that the new framing is itself agent-analytical rather than operator-voice — is the subject of Pattern (a) above. |
| BUILD Advisory 1 — Preservation-scenario amendment pattern | Cycle 5 BUILD snapshot | Not addressed in DISCOVER (appropriate scope). Active carry-forward for Cycle 6 BUILD. |
| BUILD Advisory 2 — Script-agent YAML schema constraint documentation | Cycle 5 BUILD snapshot | Not addressed in DISCOVER. Active carry-forward if BUILD touches operator docs. |
| BUILD Advisory 3 — ADR-019 §Consequences §Positive n=1 qualifier | Cycle 5 BUILD snapshot | Not addressed in DISCOVER. Not DISCOVER territory. Active carry-forward for DECIDE or BUILD. |

---

## Recommendation

**Grounding Reframe recommended** — targeted at four specific entry conditions for Cycle 6 DECIDE.

### What is uncertain

1. **"Routing preference" as operative stakeholder vocabulary.** The term is agent-named; the alternatives ("defect," "intended scope," "configuration-conditional behavior") are held in the tension deliberation note rather than given equal weight in the stakeholder-voice sections. The practitioner verbatim does not use "preference" — the practitioner observed behavior and asked "is this the routing surface the system wants, or is it a defect?" (cycle-status §Cycle 6 question framing, verbatim). DECIDE should enter with the practitioner's own question as the operative frame, not with "preference" as the settled vocabulary.

2. **"Infrastructure-complete / routing-incomplete" as settled operator voice.** This framing was introduced by the Cycle 5 PLAY susceptibility snapshot (an analytical artifact, not a practitioner-engagement record) and was accepted at entry. Whether the existing event model is the right substrate for the routing surface DECIDE will design — or whether "infrastructure-complete" is itself an assumption that DECIDE should test — is uncertain. If the event model needs extending, the DECIDE framing ("wire the routing surface") will be insufficient.

3. **T15's binary heading vs. Inversion N+2's unified-substrate framing.** The tension asks "which surface?" while the inversion correctly characterizes the architectural question as "one shared event-emission infrastructure with two routing destinations." DECIDE's T15 deliberation should start from the inversion's framing, not from the tension's either/or.

4. **T16's scope: "substantive" vs. "always."** The practitioner verbatim says "always rely on artifact writing to be the substrate." The tension introduces "outputs that are substantive (code, structured analyses, long-form text)" as the scope. This narrowing is agent-introduced. Whether the boundary should be "always," "when substantive," or "operator-configured" is a DECIDE question that the artifact has partially pre-answered.

### Concrete grounding actions for Cycle 6 DECIDE entry

**Action 1.** Before T14's deliberation, return to the practitioner's own framing of the routing question (cycle-status §Cycle 6 question framing verbatim: "is this the routing surface the system wants, or is it a defect?") and hold it alongside "routing preference" as the deliberation's entry question. Spike γ's results should be read against the practitioner's question, not against a pre-selected vocabulary.

**Action 2.** Before the observability ADR is scoped, test the "infrastructure-complete" assumption by reading the Cycle 5 BUILD event types and their current fields — not to redesign them, but to determine whether they carry what the observability surface needs or whether the "routing" problem is partly a "missing fields" problem. This is a 30-minute read-and-compare, not a spike. If the fields are sufficient, "routing-incomplete" is accurate; if the fields are insufficient, the DECIDE scoping changes.

**Action 3.** Before T15's deliberation, lead with Inversion N+2's framing: "the two axes collapse into one architectural concern." T15's open question ("should the cycle commit to one surface, both, or design them to share a common event-routing infrastructure?") should be answered after establishing whether the unified-infrastructure option is architecturally coherent, not before.

**Action 4.** Before T16's deliberation, return to the practitioner verbatim ("always rely on artifact writing to be the substrate") and deliberate the scope question explicitly: always, or when substantive? The five sub-questions (boundary, contract, client access, cleanup, backward compat) depend on this answer — scoping to "substantive" produces a narrower contract; scoping to "always" produces a simpler one with different tradeoffs. This is the first DECIDE question for T16, not a pre-answered premise.

### What the practitioner would be building on without grounding

Without Action 1: DECIDE deliberates T14 against "routing preference" as the operative vocabulary, which encodes a disposition toward "document the preference" over "remediate the defect." Spike γ's results, whatever they show, will be interpreted through that frame.

Without Action 2: The observability ADR is scoped as a "routing wiring" problem. If the event model is missing fields that the routing surface needs, the first BUILD iteration discovers the gap rather than DECIDE.

Without Action 3: T15 is deliberated as an either/or choice (operator-terminal vs. orchestrator-context). The unified-infrastructure option, which avoids the choice, is discoverable at DECIDE only if Inversion N+2 is prominently read at entry — which the tension heading actively works against.

Without Action 4: T16's five sub-questions are deliberated against "outputs that are substantive" as the established scope. The "always" option — which produces a simpler contract and may be more aligned with the practitioner's intent — is not a live deliberation option.
