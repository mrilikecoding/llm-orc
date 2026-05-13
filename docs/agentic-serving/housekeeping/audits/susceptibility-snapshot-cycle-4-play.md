# Susceptibility Snapshot

**Phase evaluated:** PLAY (Cycle 4 — 2026-05-12 single inhabitation session, Ensemble Author / Operator stakeholder)
**Artifact produced:** `essays/reflections/field-notes.md` (Cycle 4 PLAY section: 19 notes + cross-cutting reflection + routing summary); `proposals/agentic-serving-library-structure.md`
**Date:** 2026-05-12

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Research | Grounding Reframe triggered | Three grounding actions; autonomous-routing gap named |
| Discover | Grounding Reframe triggered | Asymmetric readiness mapping; elaboration-by-evidence commitment |
| Model | Clean with feed-forwards | No reframe triggered |
| Decide | Grounding Reframe recommended (1 finding) | ADR-015 autonomous-routing evidence gap not carried into artifact |
| Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE; three module separations not re-examined; OQ #14 asymmetric grounding encoded in dependency graph |
| Build | Grounding Reframe (one targeted finding) + 2 advisory | Pre-loaded conditional-acceptance disposition; sibling-vs-monolithic absent reasoning trace; PEP-563 comment accuracy |
| **Play (this snapshot)** | Evaluated below | |

The trajectory entering PLAY carried two resolved advisory observations from BUILD (sibling-vs-monolithic note added; PEP-563 comment accuracy fixed per cycle-status §BUILD close gate note). The BUILD Grounding Reframe (conditional-acceptance trigger-action pre-framing) was resolved in-cycle: the practitioner independently assessed ADR-016's trigger criterion, elected option (a) full acceptance, and the disposition is recorded in the cycle-status. PLAY enters as the sycophancy gradient's most empirically variable phase — inhabitation can produce genuine discovery or can produce the gamemaster narrating the discoveries the cycle's prior framing already predicted.

---

## Susceptibility Risk Summary (Overall Verdict)

**Low-to-moderate. The dominant signal is voice-blurring at the cross-cutting reflection boundary and selective routing of the cycle's most structurally challenging findings away from DISCOVER toward DECIDE and SYNTHESIS.** The 19 field notes individually exhibit strong observation discipline — verbatim practitioner quotes are used precisely, and challenging findings are recorded sharply rather than softened. The routing summary is where the signal concentrates: DISCOVER receives the session's most tractable findings (notes 1, 8, 11, 18 — assumption inversions that confirm patterns the cycle already suspected); DECIDE and system-design receive the four findings with the highest architectural reach (notes 13, 14, 15, 16, 19). The proposal document encodes those architectural findings as settled structural decisions rather than as provisional outputs of a single session.

The second concentration point is the cross-cutting reflection's "how has understanding shifted" section: three claims are presented as practitioner understanding shifts, but the evidence basis for each is the single session, and the framing of each claim is the agent's analytical synthesis, not the practitioner's own articulated assessment.

Neither signal reaches the Grounding Reframe threshold individually. Together they warrant two advisory carry-forwards for the SYNTHESIZE or next-cycle DECIDE pickup.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable | Individual notes maintain observation discipline (verbatim quotes anchor claims; challenge findings are named as challenges). The cross-cutting reflection's "understanding shift" claims are asserted without qualification on the session's n=1 evidence base. Assertion density is highest at the reflection boundary, not within the field notes themselves. |
| Solution-space narrowing | Clear (new at PLAY) | New at this phase | The proposal document presents the three-layer architecture, operation-named ensemble vocabulary, and specific directory structure as "settled" and "load-bearing" (proposal §"What this proposal is and isn't"). The evidence base is one session with one practitioner. Notes 14 and 15 are the empirical anchor; neither is a multi-session finding. |
| Framing adoption | Ambiguous | Stable | Three framing instances evaluated: (1) "agentic-serving config is part of the build" — the field notes accurately record this as practitioner verbatim and route it as DISCOVER/assumption-inversion, which is honest. (2) The operation-named vs. methodology-named refinement (note 15) is accurately attributed as practitioner-generated; the agent's role in surfacing notes 14–15 is ambiguous (see §Finding 2). (3) The no-dispatch fallback finding (note 19) is agent-surfaced with no practitioner framing parallel — the agent introduced this observation and routed it to DECIDE. That routing is appropriate but the finding's framing ("fail-open shape") is the agent's analytical label. |
| Confidence markers | Clear (one surface) | New at PLAY boundary | The proposal's §"Design decisions that are settled" language: *"operation-named ensembles; agentic- prefix; three-layer separation."* "Settled" is a strong claim from a single session. The field notes themselves do not use confidence-escalating language. The proposal document is where settled-framing concentrates. |
| Alternative engagement | Ambiguous | Stable | Notes 8, 10, 11, 12, 13, 18 record specific orchestrator errors and misstatements. The agent's engagement with each is asymmetric: for orchestrator count errors (notes 8, 11) the agent examines multiple explanatory sources (numerator vs. denominator, grep over-count vs. list_ensembles dedup). For the `compose_ensemble` category error (note 13) the agent characterizes the error as category-level without examining whether an alternative reading of ADR-003 could produce the orchestrator's understanding. For the recommendation-justification errors (note 18) the agent characterizes the pattern as "consistent across task types" after three examples — a moderate-evidence generalization. |
| Embedded conclusions at artifact-production moments | Clear — proposal document | New at PLAY boundary | The proposal document's three-layer framing, directory structure, model-profile YAML, capability ensemble specs, and config.yaml rewrite are all produced as spec-level deliverables (with inline YAML blocks) from findings that are 4–8 hours old. The proposal §"Is" section explicitly labels certain findings as "settled"; the §"Open decisions" section labels others as DECIDE-phase candidates. The settled/open classification is itself a conclusion embedded at artifact-production time: whether the three-layer separation or operation-naming principle is "settled" versus "directionally strong but warranting a second session's confirmation" is not examined. |

---

## Finding 1 — Routing Classification: Challenging Findings Concentrated in DECIDE/system-design, Not DISCOVER

**Finding:** The routing summary sends the session's four highest-architectural-reach notes (13, 14, 15, 16, 19) to DECIDE or system-design. DISCOVER receives notes 1, 8, 11, 18 — findings that confirm patterns the cycle's prior work had already named (operator-side gap, orchestrator self-knowledge limits, unchallenged-claim-stickiness, orchestrator analytical output is audit-worthy). The routing is internally consistent but creates an asymmetric feed-forward: DISCOVER's findings reinforce prior corpus framings; DECIDE's findings introduce the largest structural departures.

**Evidence:**

- Note 14 (Architectural Isolation maps to `invoke_ensemble`) routes to **system-design** as "architectural-property finding; RDD-via-agentic-serving structurally feasible without ADR amendment." This is the cycle's most architecturally optimistic finding — it asserts that a methodology's sycophancy-resistance mechanism "lands intact" in llm-orc's dispatch path. Routing to system-design means it enters the corpus as an architectural addition, not as a hypothesis requiring validation in a DISCOVER assumption-inversion pass.

- Note 15 (Three-layer separation reframes operator-driven migration) routes to **DECIDE** (revisits ADR-015 §Negative) with secondary routes to system-design and DISCOVER. The primary DECIDE routing means it will feed the next cycle's architectural decisions directly, without a DISCOVER-phase examination of whether the practitioner's verbatim refinement represents a stable model or a session-specific articulation.

- Note 19 (No-dispatch fallback path) routes to **DECIDE** (missing scenario / new ADR). The observation is sound; the routing is appropriate for the operational finding. But note 19's framing — "fail-open shape of the cycle's quality infrastructure" — is the agent's analytical label for what the session produced. Whether that label captures the practitioner's understanding of what they observed is not confirmed by any practitioner verbatim quote in note 19.

**Contrast:** Notes 14, 15, and 19 contain no practitioner verbatim quotes at all. Notes 1, 6, 7, 8 (the DISCOVER and SYNTHESIS primaries) contain direct practitioner verbatim quotes. The verbatim-quote density is inversely correlated with primary routing to DISCOVER — the finding routed to DISCOVER is practitioner-grounded; the finding routed to system-design is agent-framed.

**Asymmetry test (selection bias question):** The routing categories are applied consistently by content type. This is not a case of a challenging observation being softened into SYNTHESIS to reduce visibility. Notes 13, 15, 16, 19 are routed toward DECIDE and system-design because they are genuinely architectural — that classification is accurate. The selection-bias question is narrower: should notes 14 and 15 also route to DISCOVER (as assumption-inversions on the "agentic-serving is primarily a deployment shape" claim from the cycle's prior DISCOVER work), or does routing them to system-design and DECIDE adequately capture their corpus reach? The routing summary notes 15 routes to DISCOVER as a secondary destination, which partially addresses this. Note 14 has no DISCOVER secondary route despite being the most direct challenge to the deployment-shape framing.

**Risk class:** Advisory. The routing is not suppressive — challenging observations land in DECIDE and system-design, which are high-visibility destinations. The risk is that the next cycle's DISCOVER-entry context inherits system-design-encoded architectural conclusions (notes 14, 15 as settled) rather than framing them as DISCOVER-phase hypotheses about how the practitioner's product model has shifted.

---

## Finding 2 — Gamemaster/Player Role Blur: Agent-Framed Architectural Findings Presented as Inhabitation Discoveries

**Finding:** Notes 14, 15, and 19 describe architectural findings that emerged from the session, but the agent's role in introducing the framing (vs. the practitioner discovering it under inhabitation) is not consistently distinguished. This matters because the PLAY skill's gamemaster discipline requires the agent to "shape attention, not conclusions" — and three of the four highest-information notes in the session are framed at the agent's analytical register, not the practitioner's observation register.

**Evidence:**

- Note 14 (Architectural Isolation): The observation that `invoke_ensemble`'s fresh-context dispatch property is "structurally identical" to RDD's ADR-058 Invariant 8 requires knowing both ADR-058's text and `invoke_ensemble`'s dispatch semantics. This is agent-accessible information, not practitioner-first observation. The note does not record who surfaced the mapping — whether the practitioner raised RDD as a north-star use case and the agent synthesized the mapping, or the agent introduced the RDD framing as a complication for the practitioner to engage. The field notes dispatch prompt specifies note 15 records "the practitioner's verbatim refinement" of the architectural framing, which implies the agent introduced an initial sketch (methodology-named ensembles) and the practitioner refined it. Note 14's RDD-ADR-058 mapping is not attributed. If the agent introduced that framing, note 14 is agent analytical output recorded as inhabitation discovery.

- Note 15 (Three-layer separation): The note does record one practitioner verbatim quote (*"RDD delegates to sub-skills which decompose to specific kinds of tasks that the orchestrator can route to the appropriate ensemble..."*). The note credits the three-layer framing's operation-named vocabulary to the practitioner. What the note does not record is whether the methodology-layer / dispatch-layer / execution-layer tripartition was introduced by the agent as a structuring frame or whether the practitioner arrived at it independently through inhabitation. The dispatch prompt for this snapshot notes the refinement was "practitioner-generated"; the field note supports this attribution for the operation-named vocabulary specifically, but the three-layer framing is a structural decomposition that could have been agent-introduced.

- Note 19 (No-dispatch fallback): No practitioner quote. The "fail-open shape" label, the framing that "all quality mechanisms are moot," the characterization of the library-coverage dependency — all are agent analytical labels applied to what was observed (53 seconds of reasoning, one Read call, zero invoke_ensemble calls, no artifact directories). The observation itself is real and important. The framing is entirely the agent's. Whether this is a gamemaster surfacing an observation for the practitioner's engagement, or an analyst writing a finding and routing it as inhabitation output, is not distinguishable from the field note's text.

**Assessment against the "shape attention, not conclusions" standard:** The PLAY skill's gamemaster role requires introducing complications that help the practitioner generate discoveries. Notes 14, 15, and 19 appear to have operated at both levels simultaneously — introducing the framing and recording the discovery. This is not necessarily a violation of the discipline (the gamemaster can introduce structural observations; the practitioner's validation of them is part of the inhabitation), but the field notes do not record the practitioner's engagement with the agent-introduced framings as distinguishable from the agent's own analytical conclusions. The three notes present conclusions without a practitioner-engagement trace.

**Risk class:** Advisory. The findings in notes 14, 15, and 19 may be entirely accurate — a single practitioner session may genuinely produce these architectural discoveries. The risk is that the proposal document (which takes notes 14, 15, and 19 as load-bearing) encodes findings that are agent-analytical in origin as practitioner-confirmed discoveries, without the evidence trace that would distinguish them.

---

## Finding 3 — Cross-Cutting Reflection Voice: "Understanding Shifts" Are Agent Synthesis, Not Practitioner Assessment

**Finding:** The cross-cutting reflection's three "how has the practitioner's understanding shifted" claims are framed as practitioner shifts but are narrated entirely in the agent's analytical register. None of the three claims is anchored by a practitioner verbatim quote stating the shift. The practitioner's own end-of-session assessment is recorded in the cross-cutting reflection (verbatim: *"the auto-mode of BUILD is a sensible cycle-economy choice... but it does not produce the operator-facing deliverable that 'BUILD-scope structurally complete' implied to a future reader of the cycle status"*) — but only in the concluding paragraph of the reflection, not as the basis for any of the three numbered shifts.

**Evidence:**

- Shift 1 ("From 'the cycle shipped a complete agentic-serving stack' to 'the cycle shipped a scaffold'"): No practitioner verbatim anchors this claim. Note 1's verbatim (*"the agentic-serving config is to me part of the build"*) is the empirical basis. The shift's phrasing ("scope-distinction insight: building the architecture and populating the deployment are different work packages") is the agent's synthesis of note 1, not the practitioner's own characterization of what shifted.

- Shift 2 ("From 'ADR-015 §Negative's operator-driven library migration is fine' to 'operator-driven library migration is downstream of decisions about which methodology-consumers the orchestrator will serve'"): No practitioner verbatim for this shift exists in the field notes. The shift encodes note 15's three-layer framing as a practitioner understanding change. Whether the practitioner held the "is fine" prior entering the session and arrived at the "downstream of decisions" posterior within it is not confirmed by any session evidence.

- Shift 3 ("From 'Calibration Gate covers quality' to 'Calibration Gate covers dispatched-ensemble outputs; orchestrator narration is uncalibrated'"): The prior "Calibration Gate covers quality" is an artifact of the cycle's BUILD framing, not a stated practitioner prior. The shift from that prior to the more precise posterior is a legitimate cycle-level observation but is not documented as a shift the practitioner articulated mid-session.

**Assessment:** The reflection produces a tidy three-shift narrative that mirrors the cross-cutting reflection template's expected output. The three shifts are structurally coherent with the session's findings. But they are the agent's characterization of what should have shifted given the session evidence — not a record of what the practitioner said shifted. This is the standard voice-blurring risk at PLAY's cross-cutting reflection boundary: the agent produces a polished synthesis where the template asks for practitioner perspective, and the practitioner's own assessment (one quote, in the closing paragraph) is subordinate to the agent's three-item list.

**Risk class:** Advisory. The reflection's content is accurate to the session evidence. The voice issue matters for the artifact's downstream use in SYNTHESIZE: the cross-cutting reflection will be treated as evidence of the practitioner's epistemic shift in a way that the session record does not quite support. A SYNTHESIZE agent reading this reflection cannot distinguish between "the practitioner articulated these shifts during the session" and "the agent synthesized these shifts from session observations."

---

## Finding 4 — Proposal Document: n=1 Findings Encoded as Settled Structural Decisions

**Finding:** The proposal document (`proposals/agentic-serving-library-structure.md`) is structured as a specification — it contains inline YAML, a directory tree, a proposed config.yaml rewrite, and per-ensemble specs — and explicitly labels three design decisions as "settled": operation-named ensembles, agentic- prefix, three-layer separation. The evidence base for each settled claim is the single 2026-05-12 session. The Cycle 1 PLAY ran a different stakeholder (Pure Tool User) and produced no architectural findings comparable to notes 14–15; the n=1 evidence base on the architectural claims is not from the play skill's intended multi-session or multi-stakeholder coverage.

**Evidence:**

- The proposal §"What this proposal is and isn't" states: *"A capture of design decisions that are settled (operation-named ensembles; agentic- prefix; three-layer separation)."* The three-layer separation's evidence anchor is note 15 (one practitioner verbatim quote from one session); the operation-naming principle's evidence anchor is note 15's framing (practitioner-generated vocabulary from one session); the agentic- prefix is a naming decision made during config authoring, not an inhabitation finding.

- The proposal's RDD-phase routing table assigns mechanical authoring tasks ("Authoring `code-generator`, `claim-extractor`, etc. ensembles") to BUILD with the comment "Mechanical; specs above are sufficient." This treats the capability ensemble specs as BUILD-ready from PLAY-phase observation — skipping a DECIDE-phase scenario authoring step for the ensembles themselves.

- The proposal §"Origin: what the play surfaced" presents finding 2 (Architectural Isolation maps to `invoke_ensemble`) and finding 3 (three-layer separation) as load-bearing structural conclusions, prefaced with "RDD-via-agentic-serving (or any methodology that decomposes into capability-typed sub-tasks) is structurally feasible on the existing primitive surface." The "any methodology" generalization is an extension of the single-RDD-session evidence base.

**Assessment:** The proposal document's value is in capturing the directional findings before they dissipate. The risk is the spec-level detail it contains: the model-profile YAML, the proposed config.yaml section, and the capability ensemble specs are presented at implementation-ready fidelity. A future BUILD agent inheriting this proposal would have no design work to do — the decisions are presented as settled and the implementation is specified. The "settled" classification suppresses the natural DECIDE-phase deliberation that would examine whether the three-layer separation is the right abstraction boundary, whether operation-naming creates a vocabulary gap at methodology-boundary seams (OD-6's methodology-registry is acknowledged as a DECIDE artifact but the need for it implies the "settled" naming may be underdetermined), and whether the specific capability ensembles proposed (claim-extractor, argument-mapper, prose-improver) are the right decomposition rather than one of several viable decompositions.

**Risk class:** Advisory. The proposal is labeled as a "Draft proposal" and "Not yet implemented," which appropriately signals its status. The specific risk for SYNTHESIZE is treating the proposal's settled-claims section as corpus-authoritative rather than as the starting point for a DECIDE-phase examination.

---

## Specific Findings Summary

| # | Name | Evidence | Risk class |
|---|------|----------|------------|
| F1 | DISCOVER routing underweights note 14 (Architectural Isolation) | Note 14 has no DISCOVER secondary route despite challenging the "deployment-shape" framing from prior DISCOVER work | Advisory |
| F2 | Notes 14, 15, 19 — agent-analytical framing presented without practitioner-engagement trace | No practitioner verbatim quotes in notes 14, 19; practitioner attribution for note 15's three-layer structure is partial | Advisory |
| F3 | Cross-cutting reflection's three "understanding shifts" are agent synthesis | None of the three shifts is anchored by a practitioner quote stating the shift; practitioner's own closing characterization is in a separate paragraph | Advisory |
| F4 | Proposal document encodes n=1 session findings as "settled" structural decisions | "Settled" designation for three-layer separation and operation-naming from single session; mechanical BUILD specs from PLAY-phase observation | Advisory |
| F5 | Note 14's RDD-ADR-058 mapping is not attributed | Agent-accessible mapping presented without recording who introduced the RDD-as-north-star framing | Advisory (lower weight) |

---

## PLAY-Specific Signal Assessment

### (a) Selection bias in the six-category classification

The routing is not suppressive — challenging observations are routed to DECIDE and system-design (high-visibility, high-architectural-weight destinations), not toward SYNTHESIS. However, DISCOVER receives findings that confirm prior corpus patterns rather than findings that challenge them. Notes 14 and 15 challenge the "agentic-serving as deployment shape" framing from prior DISCOVER work; note 14 in particular is a positive architectural finding that extends the cycle's scope (RDD-via-agentic-serving is structurally feasible) without going through a DISCOVER assumption-inversion pass that would examine the extension's presuppositions.

The routing table's aggregate row shows system-design receiving 1 primary and 2 secondary routes — a destination that did not appear in Cycle 1 PLAY's routing at all. The first appearance of system-design as a primary routing destination in this corpus is at PLAY, which is structurally late (system-design is typically an ARCHITECT output, not a PLAY output). This reflects genuine architectural discovery, but also the risk that PLAY is doing ARCHITECT work that bypasses the sycophancy-checking infrastructure of the ARCHITECT and DECIDE phases.

### (b) Gamemaster/player role blur under task load

The session produced four high-information architectural notes (14, 15, 16, 19) and a proposal document. Notes 16 and 19 are grounded in direct observation (the calibration gate does not audit orchestrator narration; no invoke_ensemble calls fired on the evaluative prompt). Notes 14 and 15 are the role-blur risk points — they are architectural synthesis findings that required the agent's analytical apparatus to connect session observations to corpus architecture (ADR-058, invoke_ensemble's dispatch semantics, the methodology-layer abstraction). The practitioner's role in notes 14 and 15 is one verbatim quote in note 15; note 14 has no practitioner-engagement trace.

The "shape attention, not conclusions" discipline held for the operational observations (notes 2, 4, 5, 6 — delight findings with specific empirical grounding) and for the error characterizations (notes 8, 10–13, 18 — orchestrator errors with precise evidence). The discipline is most strained at the architectural-synthesis boundary (notes 14, 15) and at the proposal-document authoring boundary (where the agent produced implementation-ready YAML from session observations).

---

## Interpretation

### Pattern assessment

The overall pattern is a phase functioning well at the observational layer and exhibiting moderate role-blur at the synthesis layer. The field notes' core discipline — verbatim-anchored observations, split-category honesty (note 6's delight/deferral; note 7's Cycle 1 vs. Cycle 4 comparison), error-mode characterization with evidence — is strong and consistent across the 19 notes. This is not a session where challenging observations were softened or where the practitioner's framing was uncritically adopted.

The concentration points are:
1. Four notes (14, 15, 16, 19) are agent-analytical in origin, produced under authoring load (the proposal document was being produced in the same session), and are treated in the proposal as settling architectural questions.
2. The cross-cutting reflection's three understanding-shift claims are the agent's synthesis, not the practitioner's stated assessment.
3. The proposal document's "settled" designation for three-layer separation and operation-naming is a confidence escalation from the field notes' more provisional language.

The trajectory comparison: the BUILD snapshot's most consequential finding (pre-loaded conditional-acceptance recommendation) was resolved cleanly and the practitioner exercised genuine independent judgment (full acceptance). The PLAY snapshot's analogous finding is softer — there is no pre-loaded practitioner disposition here; instead the agent's analytical synthesis is presented as inhabitation output in a way that a future agent or the practitioner may not spontaneously examine.

### Earned vs. sycophantic reinforcement

The operational findings (notes 2, 4, 5, 6, 7) are clearly earned: the cheap-cloud-orchestrator pattern's behavior is directly observed and compared to Cycle 1 PLAY's baseline. The error-characterization findings (notes 8, 10–13, 18) are clearly earned: specific errors are recorded with post-session verification. The architectural-synthesis findings (notes 14, 15) are where earned vs. unearned confidence is hardest to assess — the mappings are structurally accurate (invoke_ensemble does dispatch with fresh context; the practitioner's verbatim does describe a three-layer structure), but their elevation to "settled" structural findings is a confidence step that the n=1 evidence base does not fully support.

The cross-cutting reflection's understanding-shift claims are the weakest-earned element in the snapshot: they are consistent with what a well-calibrated agent would predict should shift given the session evidence, but they are not confirmed as the practitioner's own articulated assessments.

---

## Recommendation

**No Grounding Reframe warranted at the PLAY boundary.** The signals do not converge on a pattern where the practitioner would be building subsequent work on an unexamined assumption that poses operational risk. The architectural findings in notes 14, 15, 16, 19 are directionally sound; the proposal is explicitly labeled as a draft; the cycle-status records PLAY as "in progress" with SYNTHESIZE as optional. The practitioner has appropriate authority to accept these findings provisionally and examine them in a future DECIDE phase.

**Four advisory carry-forwards for SYNTHESIZE or next-cycle DECIDE pickup:**

---

### Advisory 1 — Note 14 should add a DISCOVER secondary route

Note 14 (Architectural Isolation maps to `invoke_ensemble`) routes to system-design with no DISCOVER secondary. The finding asserts RDD-via-agentic-serving is "structurally feasible on the existing primitive surface; no ADR amendment needed." This is the most optimistic architectural claim in the session. It extends the cycle's product model (agentic-serving is now framed as a substrate for methodology consumers, not just a deployment shape) without going through a DISCOVER assumption-inversion examination of what presuppositions that extension carries.

The DISCOVER secondary should examine: does the Architectural Isolation mapping hold under the specific conditions the RDD skill's inhabitation creates (multi-turn sessions; practitioner-authored methodology prompts; the fresh-context property is necessary for RDD's sycophancy-resistance mechanism but is it sufficient)? Adding a DISCOVER secondary route to note 14 (parallel to note 15's existing secondary) preserves the finding's architectural reach while flagging that the product-model extension requires a DISCOVER-phase pass.

---

### Advisory 2 — Proposal document's "settled" claims should be reclassified as "directionally strong, pending DECIDE-phase deliberation"

The proposal §"What this proposal is and isn't" currently labels three claims as settled: operation-named ensembles, agentic- prefix, three-layer separation. The recommendation is to reclassify these as "directionally strong from single session; recommended as DECIDE-entry premises rather than DECIDE-bypassed conclusions."

The practical implication: the next cycle's DECIDE phase should open by examining whether the three-layer separation is the right abstraction (or whether two layers are sufficient; or whether the methodology layer belongs partly server-side), whether operation-naming creates vocabulary-seam problems at methodology boundaries (OD-6's methodology-registry finding implies the naming is not self-contained), and whether the agentic- prefix convention extends cleanly to operator-authored deployments vs. the proposal's library-provided defaults. These are small examinations — 15–30 minutes of DECIDE deliberation. But without them, the proposal's settled claims will be inherited as given by a BUILD agent that never sees the single-session evidence base.

---

### Advisory 3 — Cross-cutting reflection's understanding-shift claims should note their evidence basis

The three understanding-shift claims in the cross-cutting reflection are agent synthesis. For SYNTHESIZE use, each claim should be accompanied by its evidence anchor (the specific note(s) from which the shift is derived) and an explicit note that the shift is an agent-synthesized characterization of what the session evidence supports, not a practitioner-stated assessment.

This does not require rewriting the reflection. A one-sentence prefix would close the gap: e.g., *"The following shifts are the agent's characterization of what the session evidence supports; the practitioner's own characterization of the session appears in the closing paragraph of this reflection."*

Without this annotation, a SYNTHESIZE agent or a future cycle's DISCOVER phase will treat the three shifts as practitioner-confirmed epistemic history, which the evidence does not support.

---

### Advisory 4 — Notes 14 and 19 should record who introduced the load-bearing framings

Note 14's RDD-ADR-058 mapping is not attributed (the dispatch prompt for this snapshot notes the RDD north-star prompt came from the practitioner; the mapping to ADR-058 Invariant 8 is agent-analytical and not recorded as such). Note 19's "fail-open shape" framing is entirely agent-analytical with no practitioner engagement trace.

For corpus use, a brief attribution note in each would suffice: e.g., in note 14, "practitioner introduced RDD as north-star use case; the ADR-058 Architectural Isolation mapping was surfaced by the agent during the session"; in note 19, "framing introduced by agent based on session observation (zero invoke_ensemble calls; no artifact directories); not confirmed by practitioner verbatim during the session."

This is the lowest-weight advisory in this snapshot. The findings' content is accurate regardless of attribution. The attribution matters only if a future agent needs to distinguish between findings the practitioner arrived at under inhabitation and findings the agent introduced under analytical load — which is exactly the question the gamemaster/player role-blur signal is designed to surface.

---

## Carry-Forward for SYNTHESIZE / Next-Cycle DECIDE

If SYNTHESIZE runs, it should:
1. Treat notes 2, 4, 5, 6 (delight/clean-operational findings) as corpus-authoritative.
2. Treat notes 7, 8, 11 (visibility gap; orchestrator self-knowledge limits; unchallenged-claim-stickiness) as DISCOVER-level confirmations of prior cycle findings — strong but confirmatory, not novel.
3. Treat notes 14, 15 (architectural-synthesis findings) as directionally strong but n=1, not as settled structural claims to embed in the SYNTHESIZE essay without qualification.
4. Treat the cross-cutting reflection's three understanding-shift claims as agent synthesis warranting attribution, not as practitioner-stated assessment.
5. Note the proposal document at `proposals/agentic-serving-library-structure.md` as a starting point for next-cycle DECIDE deliberation, not as a BUILD-ready spec.

The highest-information outputs for SYNTHESIZE are notes 14, 15, 16, and 19 — they describe the system's structural coverage gaps and architectural extension surface with more precision than any prior cycle phase produced. The framing caveat above does not reduce their value; it is about what level of epistemic warrant the SYNTHESIZE essay should attribute to them.
