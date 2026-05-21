# Susceptibility Snapshot

**Phase evaluated:** RESEARCH (Cycle 7 — 2026-05-21)
**Artifact produced:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md` (post five-round argument audit)
**Date:** 2026-05-21

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 6 PLAY | Grounding Reframe recommended | Spike δ architectural claim outpaced evidence; must-delegate framing adopted without examination; framework-driven pipeline encoded as settled commitment |
| **Cycle 7 RESEARCH → DISCOVER (this snapshot)** | Evaluated below | |

The Cycle 6 PLAY snapshot recommended four grounding actions before Cycle 7 RESEARCH entry. Their outcomes are material to this evaluation:

- **Uncertainty 1 (framework-driven pipeline as "the candidate"):** Grounding action partially honored. The cycle-status's carry-forward softened three inherited claims: chain-handling is "well-grounded," routing-decision is "contested," must-delegate is "bounded to capability-matched requests." The research phase then re-opened the question empirically via Phase A (Q0 grounding), the research-methods-reviewer (who caught artifact-derived option enumerations), and Spike λ + Spike λ-paid (which validated and partially refuted the Phase A reframe, respectively). The inherited framing did not block investigation; Phase A's counter-evidence reshaped the architecture.

- **Uncertainty 2 (must-delegate framing as a value-proposition constraint):** Partially addressed in the Essay-Outline's C6 working-inference flag (Section 8 SCOPE QUALIFICATION names E6.2.1 as pending DECIDE-phase product-discovery validation). However, the gate reflection note records E6.2.1 as practitioner stance (first-person, verbatim, cycle 7 RESEARCH gate, 2026-05-21) — which is a different evidence grade than the prior cycle's agent-introduced framing. The grounding action yielded a modest evidentiary upgrade: agent framing → practitioner-stated stance. Still pending product-discovery validation.

- **Direct-completion quality contrast:** Not explicitly addressed. Section 7's C6 framing acknowledges direct completion as the empirical fallback but does not record the quality differential (note 1 / note 18 direct completions were 10-13s and arguably comparable to ensemble output). The C6 argument elevates capability-list discovery as a first-order requirement without this contrast as a named counter-consideration.

- **Methodology-level lesson (probe-1 correction):** Not addressed in the current artifact corpus; out of scope for this snapshot's focus.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable relative to Cycle 6 PLAY | C1–C4 are well-evidenced conclusions; assertion density is proportionate to empirical grounding. C6 and the C7 hybrid-first ordering carry the higher assertion density relative to their evidence base. The Abstract's "the escalation is structurally pre-committed" is a strong claim that the Argument-Graph's W7.2 + Inverted-framing acknowledgment partially qualifies. |
| Solution-space narrowing | Clear | Stable | The Phase A reframe narrowed the solution space dramatically ("Q1 reduces to contract conformance"), then Spike λ-paid reopened it. The final Essay-Outline narrows to the tiered architecture, but the narrowing is better supported than the Cycle 6 PLAY close: the Inverted-framing acknowledgment (Section 8) and three rounds of open-item carry-forwards (F3-1, F3-2, P3-3) preserve the ADR-027 alternative in visible form. |
| Framing adoption | Clear (two distinct episodes) | See note | Phase A reframe (agent-initiated): the simplification of Q1 to "tool_choice contract conformance" was adopted by the cycle before being empirically tested; Spike λ-paid refuted the reframe's strongest form. RESEARCH-gate reframe (user-initiated): the user's pushback at the gate produced the tiered architecture revision. These two episodes are structurally different in origin and should be assessed separately. |
| Confidence markers | Ambiguous | Declining relative to Cycle 6 PLAY | The Essay-Outline carries substantially more hedging than the Cycle 6 PLAY cycle-status: working-inference flags on five evidence nodes (E3.1.1, E4.2.1, E5.3.3, E6.2.1, E7.3.2), explicit SCOPE QUALIFICATION in Section 8, gate-reflection open-question list. The Abstract's "structurally pre-committed" language is the highest-confidence phrase in the document; it is partially warranted by the three-distinct-failure-modes characterization but overstates certainty relative to the hybrid-effectiveness evidence (which is zero — the hybrid has not been built). |
| Alternative engagement | Ambiguous | Improving relative to Cycle 6 PLAY | Substantially better than Cycle 6 PLAY. F2-1 is now incorporated via the Inverted-framing acknowledgment (Section 8 Tier 2). The round-3 carry-forward F2-3, round-4/5 F3-2 remain as named open items at the gate. The document names the "document and verify" leaner alternative (F3-1) as still open at P3. The belief-mapping question at the gate was composed as a challenge against C7, not as a confirmatory question. The user's response was substantive and not simply agreed with by the cycle. |
| Embedded conclusions at artifact-production moments | Clear (two sites) | Stable | **Site 1 (Phase A reframe boundary):** The Phase A synthesis crystallized "Q1 reduces to contract conformance" before the production-model probe. The Spike λ validation ran only under qwen3:14b; the reframe was adopted as "validated" at that point. The Spike λ-paid counter-finding then reversed the strongest form of the reframe, but the reframe had already shaped the Spike λ design (the three cells tested the reframe rather than testing a broader mechanism design space). **Site 2 (RESEARCH gate + user response):** The Essay-Outline revision that produced the tiered architecture (hybrid as starting commitment, ADR-027 as named escalation) was a substantial change triggered by the user's response to the belief-mapping question. The revision occurred immediately at the gate boundary. Five audit rounds verified structural soundness but did not independently examine whether the hybrid-first ordering was the most evidence-consistent choice (the Inverted-framing acknowledgment was a round-4 addition, explicitly noting this gap). |

---

## Signal Assessment: Two Reframings Separately Evaluated

### Movement 1: Phase A reframe (agent-initiated, validated empirically)

**What happened:** The Q0 empirical finding that NL-routing under tool-rich production clients is approximately zero led the agent to propose collapsing Q1 to "tool_choice contract conformance." The user authorized validation spikes. Spike λ (qwen3:14b) partially validated the reframe. Spike λ-paid (MiniMax M2.5) produced a counter-finding.

**Assessment of residual:** The reframe's strongest form was refuted. However, the Phase A reframe left a structural residue in the Essay-Outline:

1. **The latency-budget discussion (W3.3, Section 4)** was constructed around the reframe's frame: comparing routing-planner ensemble overhead vs. explicit-naming extractor overhead vs. classifier overhead, all within the server-side-interception paradigm the reframe motivated. A pre-reframe analysis might have examined latency for the ADR-027 direct pipeline rather than for hybrid interception mechanisms. The analysis is not wrong, but it inhabits the reframe's frame.

2. **C7's hybrid-first ordering.** The gate reflection note's reading note flags this directly: does the tiered architecture's "hybrid as starting commitment" still encode a Phase A residual? The Phase A reframe preferred the simpler, architecturally-preserving answer (tool_choice is already there; use it). The hybrid-first ordering also prefers the architecturally-preserving path. The gate-driven revision added ADR-027 as the escalation but did not invert the ordering. The Inverted-framing acknowledgment (Section 8) names this explicitly: "the ordering rationale (hybrid first, ADR-027 as escalation) is grounded in architectural-continuity cost, not in evidence-direction." This is an honest acknowledgment of a premise that was not tested. The Phase A residual is documented but unresolved.

3. **Spike λ-paid's three candidate diagnoses (W2.3)** were not disambiguated. The model-portability gap is established empirically, but which of the three diagnoses holds (Zen proxy stripping, MiniMax non-conformance, framework tool-list interaction) determines whether server-side interception would actually fix the gap. If the diagnosis is Zen-proxy-specific, the hybrid mechanism addresses a deployment-specific failure rather than a cross-compatibility gap. The cycle acknowledged this as an open question (gate reflection note, open question 2) but the C7 recommendation is not conditional on which diagnosis holds.

**Verdict on Movement 1 residual:** Present but documented. The Phase A reframe's residue manifests as hybrid-first ordering that rests on architectural-cost preference rather than evidence direction. The acknowledgment is explicit and proportionate. The DECIDE phase inherits this explicitly-flagged premise.

---

### Movement 2: RESEARCH-gate user-driven reframe

**What happened:** The belief-mapping question on F2-1 surfaced that the user holds the ADR-027 stance as the preferred position, with hybrid as acceptable only if it works empirically. The Essay-Outline revision restructured C7 from "hybrid architecture" to "tiered architecture with ADR-027 as structurally pre-committed escalation."

**Was the revision proportionate to the evidence?** Mostly yes, with one qualified concern.

The user's response was substantive: first-person experience with orchestrator-LLM unreliability, the project's value-proposition argument, and a structural escalation conditional. This is not a vague preference; it is grounded reasoning. The cycle's empirical evidence (Spike δ positive control, PLAY note 22 confabulation, Spike λ-paid substrate-path failure) does more strongly support the ADR-027 framing than the pre-gate hybrid framing. The revision is consistent with where the evidence points.

The qualified concern: the revision was rapid. The tiered architecture restructured C6 (elevating capability-list discovery to "first-order requirement") and C7 (naming ADR-027 as structurally pre-committed escalation) in a single gate revision, immediately following the user's response. The five audit rounds verified structural soundness, but audit rounds 1–3 were pre-revision — they verified the pre-gate hybrid framing. Rounds 4–5 verified the tiered framing. Rounds 4–5 were the rounds that added the Inverted-framing acknowledgment and the Section 8 SCOPE QUALIFICATION. The audit process did identify and correct deficiencies in the tiered framing, which is evidence that it functioned as genuine scrutiny rather than as ratification.

**Specific crystallization site: "structurally pre-committed"**

The Abstract states: "the escalation is structurally pre-committed because the orchestrator-LLM is the recurring failure surface across the cycle's empirical evidence." This phrase encodes the user's stance as a structural claim. Its warrant in the Argument-Graph (W7.2, E7.2.1–E7.2.3) is the three-distinct-failure-modes characterization, which is a round-4 addition that sharpened the language from "consistent failure surface" to "consistent failure surface across distinct failure modes." This characterization is supported by evidence (composition confabulation, Spike δ positive control, substrate-path protocol-format failure are genuinely distinct failure modes). The crystallization is earned at the pattern level.

However, "structurally pre-committed" implies a commitment that exists prior to BUILD testing. The hybrid-effectiveness measurement (W7.3, E7.3.1–E7.3.2) operationalizes the escalation trigger with criteria whose specific values are DECIDE-phase work. The phrase could be read as "the cycle has already decided to do ADR-027 regardless of BUILD results," which is not what the Evidence-Graph supports. The evidence supports "ADR-027 is structurally motivated and should be the failure path when hybrid-effectiveness criteria are not met." That is a different claim from "pre-committed." The gate reflection note's user-stated position ("if hybrid doesn't work empirically → stronger measures") is a conditional escalation, not a pre-commitment. The Abstract phrase overstates the user's stated position slightly, in the direction of the user's preference.

---

## Earned Confidence Assessment (per gate reflection note settled premises)

**Premise 1: NL-routing fraction approximately zero under tool-rich production clients (C1).**
Status: **Earned.** Evidence base is cross-cycle (Cycle 6 Spike γ cells A and B-continuation, PLAY note 18, PLAY notes 12/14 contra), n=2 model profiles, n=1 client family. Scope-of-claim caveats present in Section 2 and Section 8. This is the cycle's most empirically secure finding.

**Premise 2: tool_choice model-portability gap — framework implements it, MiniMax M2.5 via Zen does not honor it (C2).**
Status: **Earned with scope noted.** Spike λ + Spike λ-paid provide a direct A/B comparison on the same payload. The three candidate diagnoses are unresolved (which of Zen proxy stripping, MiniMax model non-conformance, or framework tool-list interaction explains the gap), but the existence of the gap is established. W2.3's epistemic status is correctly labeled. Scope-of-claim: n=1 production model profile + n=1 proxy path (Zen); other providers uncharacterized.

**Premise 3: Tiered architecture commitment (C7) — hybrid as starting commitment, ADR-027 as named escalation.**
Status: **Partially earned.** The hybrid components (C3 server-side interception, C4 framework-driven composition continuation) are individually motivated by the evidence. The ordering (hybrid-first, ADR-027 as escalation) is asserted-but-not-quantified per the Inverted-framing acknowledgment. The hybrid mechanism has zero empirical grounding as a built system — its effectiveness is entirely inferred from the evidence that motivates its individual components. The ADR-027 path has one positive control (Spike δ, framework-driven Python chaining) but that proof-of-concept runs at a lower level of integration than the full escalation path requires. The "structurally pre-committed escalation" language overstates the commitment slightly relative to what the evidence supports.

**Premise 4: Orchestrator-LLM as consistent failure surface across three distinct failure modes (C7 warrant characterization).**
Status: **Earned with precision note.** The round-4 sharpening to "consistent failure surface across distinct failure modes, not a single failure mode observed three times" is an accurate characterization of the evidence pattern: composition confabulation (PLAY note 22), Spike δ positive control (framework-driven success when orchestrator-LLM removed), and Spike λ-paid post-dispatch protocol-format failure are genuinely distinct. The characterization is not inflated. The precision note is that "failure surface" is a functional characterization derived from the evidence pattern; it is not a characterization of the orchestrator-LLM's properties in general or across deployment contexts.

**Premise 5: Direct-completion fallback as degradation surface; capability-list discovery as first-order requirement (C6).**
Status: **Premature.** This is the assessment's primary concern. The value-misalignment framing (E6.2.1) rests on practitioner stance (verbatim, gate exchange, 2026-05-21) and one PLAY-constructed stakeholder persona (E6.2.2: Skill Orchestration User's super-objective from Cycle 6 PLAY). The practitioner stance is genuine evidence of the project owner's intent, but it is not independent product-discovery validation. The "first-order requirement" elevation of capability-list discovery was driven by the user's gate-response framing about the project's value proposition. The Essay-Outline correctly flags this as a working-inference node (C6 working-inference flag in CONCLUSIONS; E6.2.1 labeled "(working inference grounded in practitioner stance; pending DECIDE-phase validation)"). However, "first-order requirement" is strong language for a claim resting on one stance + one PLAY persona. The round-4 argument-audit flagged W6.2 at P2-2 for exactly this reason. The elevation appears to have been rapid and user-preference-driven.

---

## Interpretation

### Cross-cycle comparison

The Cycle 6 PLAY snapshot diagnosed susceptibility concentrated at the PLAY-close synthesis boundary: empirically strong observation notes, but architectural conclusions outpacing evidence at the summary layer. That pattern has not recurred here in the same form. The Cycle 7 RESEARCH phase produced genuine empirical discipline: the research-methods-reviewer ran two rounds and caught artifact-derived option enumerations; the Phase A reframe was tested rather than crystallized; Spike λ-paid produced a counter-finding that refuted the reframe's strongest form; five audit rounds addressed substantive structural deficiencies rather than ratifying the initial framing.

The susceptibility pattern in Cycle 7 RESEARCH is more subtle: it concentrates at two specific moments (Phase A reframe crystallization before the production-model probe; C7 revision immediately following the user's gate response) rather than at a broad synthesis-layer overreach. The audit process partially corrected both (Phase A residual documented; Inverted-framing acknowledgment added), but neither was fully resolved.

The phase gradient (RESEARCH is the highest-risk phase for framing adoption) is partially attenuated here by the structural constraints of Essay-Outline form (ADR-092) and the multi-round audit discipline. The Abstract Synthesis bullets and Section 8 META-OBSERVATION content are the surviving sycophancy-surface sites.

### Pattern assessment: earned confidence vs. sycophantic reinforcement

**C1–C4:** Earned. The evidence base is cross-cycle, multi-configuration, empirically tested. The scope-of-claim qualifications are present and accurate. These conclusions are not inflated.

**C5:** Earned. The form-drift finding is supported by three independent observations (PLAY note 5, Spike δ form-drift-persists, PLAY note 15 as secondary corroboration). The round-4 P2-2 citation improvement strengthened rather than weakened this conclusion.

**C6:** Susceptibility present. The "first-order requirement" elevation of capability-list discovery is the Essay-Outline's weakest premise relative to its evidentiary basis. The user's gate-response stance directly shaped this elevation. The working-inference flag mitigates the risk but does not eliminate it; the DISCOVER phase will inherit this as a premise-to-be-validated rather than a settled finding.

**C7 tiered architecture:** Mixed. The ADR-027 escalation pathway is evidence-consistent. The hybrid-first ordering is asserted on architectural-cost grounds that were not quantified. The "structurally pre-committed" language in the Abstract slightly overstates the user's own conditional formulation. The Phase A residue (hybrid-first preference tracks the reframe's architecture-preserving bias) is present but explicitly acknowledged.

**Section 8 META-OBSERVATION:** The text records that the tiered framing "emerged from the cycle 7 RESEARCH gate's belief-mapping exchange (2026-05-21) on F2-1." This is honest provenance recording. It is not sycophantic attribution — it accurately describes the architectural revision's origin while preserving the audit record that verified the revision's structural soundness.

The overall pattern is consistent with a phase that managed susceptibility better than Cycle 6 PLAY but did not eliminate it. The two reframings both left traces. Movement 1's trace (hybrid-first ordering, latency analysis framed around interception mechanisms) is more structurally embedded and harder to see. Movement 2's trace (C6 elevation, "structurally pre-committed" language) is more visible and already partially flagged.

---

## Recommendation

**Grounding Reframe recommended at two targets.** The signals are sufficiently converged on the C6 elevation and the C7 ordering premise to warrant named grounding actions before DISCOVER hardens these into product-thinking commitments.

---

### Grounding Target 1: C6 capability-list discovery as "first-order requirement"

**What is uncertain:** The value-misalignment claim (E6.2.1) rests on practitioner stance + one PLAY-constructed stakeholder persona. The practitioner's stance is genuine but is not the same as product-discovery validation. The "first-order requirement" framing entered the Essay-Outline via the gate-response revision, not from the prior research work.

**What the cycle is building on without grounding:** DISCOVER's stakeholder mapping and value-tension work will receive "capability-list discovery is first-order, not documentation" as an inherited commitment. If that framing is wrong — if the Population A / Population B distinction produces a different prioritization, or if the stakeholder maps reveal that tool_choice-aware clients are a smaller population than practitioner framing suggests — DISCOVER will have been working inside a premise that product-discovery should have examined.

**Concrete grounding actions:**

1. At the DISCOVER opening, surface the C6 premise explicitly as a hypothesis to be examined: "The value-misalignment claim (direct completion is a degradation surface for llm-orc users) is a practitioner-stated stance, not yet validated by independent product-discovery. The DISCOVER phase should treat 'capability-list discovery is first-order' as a hypothesis to confirm or revise, not a settled requirement."

2. Apply the belief-mapping form to the Population A / Population B distinction: "What would have to be true for Population A (tool-call-aware clients without alternative-surface access) to be a large enough population to justify the 'first-order requirement' designation?" The answer requires characterizing the likely client population before committing DISCOVER's design work to serving it.

3. The Cycle 6 PLAY susceptibility snapshot raised the direct-completion quality contrast (notes 1/18 vs. note 7 code-generator) that the Essay-Outline did not incorporate. DISCOVER should make the quality contrast explicit when it examines the value-misalignment framing. If direct completion produces comparable results for common task shapes, capability-list discovery's "first-order" status needs a different justification (cost, observability, calibration gate access) than output quality alone.

---

### Grounding Target 2: C7 hybrid-first ordering as architectural-cost preference

**What is uncertain:** The Inverted-framing acknowledgment (Section 8) states explicitly that the hybrid-first ordering is grounded in architectural-continuity cost, not evidence direction. The architectural cost differential between hybrid and ADR-027-direct is not quantified. The Phase A residue (the reframe's architecture-preserving bias may have contributed to the hybrid-first ordering independently of the gate-driven revision) has not been examined.

**What the cycle is building on without grounding:** DECIDE will receive the hybrid as the primary architectural recommendation. If the architectural-cost differential is smaller than assumed — or if the DECIDE phase would independently arrive at ADR-027-direct given a clean prior — the hybrid becomes a BUILD investment whose primary justification is continuity with the current architecture rather than a better fit with the evidence.

**Concrete grounding actions:**

1. At the DISCOVER-DECIDE boundary (or explicitly in DECIDE's opening), surface the ordering premise as a decision rather than an inheritance: "The hybrid-first ordering rests on architectural-cost preference. Before ADRs are drafted, explicitly weigh the hybrid mechanism's implementation cost against ADR-027-direct's implementation cost. If the differential is less than one week's BUILD effort, the architectural-continuity justification is weak and ADR-027-direct should be the primary recommendation."

2. The C2 diagnosis disambiguation (gate reflection note, open question 2) should be resolved before BUILD, not during BUILD. If the model-portability gap is Zen-proxy-specific, the hybrid's server-side interception mechanism is solving a deployment-specific problem; if it is MiniMax-model-specific, the mechanism generalizes. The DECIDE-phase ADR for C3's server-side mechanism should be conditional on the diagnosis: "this mechanism is justified if the gap is model-level (not proxy-level); if proxy-level, the ADR should scope the mechanism to Zen-proxy deployments only."

3. The gate reflection note (open question 5) identified that the Essay-Outline asserts but does not quantify the architectural-cost differential. DECIDE should produce an explicit build-complexity comparison: estimated effort for hybrid Tier 1 implementation vs. ADR-027-direct implementation. If they are within the same order of magnitude, the inverted framing (ADR-027-direct as primary, hybrid as optional compatibility shim) should be evaluated as the DECIDE phase's primary recommendation rather than as the named escalation.

---

### Carry-Forward Signals for DISCOVER

The following signals from the gate audit corpus remain open and should be monitored through DISCOVER:

1. **Challenged-claim persistence not integrated into hybrid mechanism reliability profile** (argument-audit round-1 framing audit finding, not resolved across five rounds). If the hybrid mechanism surfaces dispatch results to the orchestrator-LLM, the orchestrator's post-dispatch narration may exhibit the same challenged-claim persistence documented in PLAY note 22 (four explicit corrections failed to shift the model's path assumption). DISCOVER's stakeholder map should characterize what the orchestrator-LLM's role is in Tier 1 and whether the persistence failure mode is relevant to that role.

2. **C2 scope of empirical sample** (two model profiles, one client family, one probe prompt). The architecture recommendation is conditioned on this narrow sample. DISCOVER should carry "model-portability characterization across additional orchestrator profiles" as a named DECIDE-phase spike prerequisite rather than treating C2 as fully generalized.

3. **Orchestrator LLM stakeholder role shift** under the tiered architecture is noted in the gate reflection note as a DISCOVER commitment. This is architecturally load-bearing: the gate reflection note names that in Tier 1 the orchestrator-LLM's role is constrained and in Tier 2 it is removed. DISCOVER's stakeholder map needs to surface what that shift means for the practitioner-as-operator and for clients who depend on the orchestrator's NL synthesis quality.

4. **The working-inference cluster (E3.1.1, E4.2.1, E5.3.3, E6.2.1)** is named in Section 8 as a DECIDE-phase gate. DISCOVER should not elaborate product-thinking that depends on these four nodes being validated until at least the two most consequential (E4.2.1 production-client filesystem scope disjointness; E6.2.1 value-proposition claim) have independent corroboration or explicit acceptance of residual uncertainty. The accumulation of four working-inference evidence nodes in load-bearing warrant positions remains the Essay-Outline's primary epistemic risk cluster.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block DISCOVER progression.*
