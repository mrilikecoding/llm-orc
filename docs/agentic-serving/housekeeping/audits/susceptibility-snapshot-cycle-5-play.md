# Susceptibility Snapshot

**Phase evaluated:** PLAY (Cycle 5 — 2026-05-13 two-phase session: gamemaster reconnaissance + Skill Orchestration User inhabitation via OpenCode)
**Artifact produced:** `essays/reflections/field-notes.md` (Cycle 5 PLAY section: 9 reconnaissance observations + 10 inhabitation observations + cross-cutting reflection + routing summary)
**Date:** 2026-05-13

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Research | Grounding Reframe triggered | Three grounding actions; autonomous-routing gap named |
| Cycle 4 Discover | Grounding Reframe triggered | Asymmetric readiness mapping; research-voice transplants |
| Cycle 4 Model | Clean with feed-forwards | No reframe; vocabulary relocation discipline applied |
| Cycle 4 Decide | Grounding Reframe recommended (1 finding) | ADR-015 autonomous-routing evidence gap not carried into artifact |
| Cycle 4 Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE |
| Cycle 4 Build | Grounding Reframe (one targeted) + 2 advisory | Pre-loaded conditional-acceptance disposition; resolved in-cycle |
| **Cycle 4 Play** | No Grounding Reframe; 4 advisory carry-forwards | Voice-blurring at synthesis boundary; n=1 findings encoded as settled; agent-analytical framings in notes 14, 19 without practitioner-engagement trace |
| Cycle 5 Discover | No Grounding Reframe; 2 advisory carry-forwards | Settlement-before-examination sequencing gap; four inversion questions not recorded at gate |
| Cycle 5 Decide | No Grounding Reframe; 2 advisory carry-forwards | Inherited scope-claim breadth (n=1 framework verified); no-dispatch-fallback reasoning at minimum threshold |
| Cycle 5 Build | No Grounding Reframe; 3 advisory carry-forwards | Auto-mode silent resolution of artifact-level conflicts; preservation-scenario rewrite; schema discovery embedded in artifact |
| **Cycle 5 Play (this snapshot)** | Evaluated below | |

The Cycle 4 PLAY snapshot's four advisory carry-forwards established the attribution-discipline frame this snapshot applies. Advisory 1 (note 14 DISCOVER secondary route) was honored in Cycle 5: the cycle-status records note 14's architectural mapping as a feed-forward finding that was examined under DISCOVER-phase assumption inversion. Advisory 2 (proposal's "settled" claims reclassified as directionally strong pending DECIDE) was honored: cycle-status Cycle 5 DECIDE entry records the specific inversion questions. Advisory 3 (cross-cutting reflection understanding-shifts should note evidence basis) was partially honored: the Cycle 4 PLAY field notes carry a prefatory note in the cross-cutting reflection explicitly naming the three shifts as agent synthesis; however, the Cycle 5 PLAY cross-cutting reflection does not carry the equivalent prefix, though the reflection's three shift claims are each tied more tightly to specific observations with practitioner verbatim. Advisory 4 (attribution notes for notes 14 and 19) was honored: the Cycle 4 PLAY field notes carry explicit attribution footnotes on notes 14 and 19.

The Cycle 5 BUILD snapshot's three advisory carry-forwards are assessed below under §Prior Advisory Carry-Forward Status.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable | Reconnaissance findings (1–9) are empirically dense: each is anchored by an artifact-directory diff, `execution.json` inspection, or error message verbatim. Inhabitation findings (10–19) are similarly grounded — practitioner verbatim quotes appear in notes 11, 12, and 19, and the orchestrator's narrated outputs are quoted in notes 13, 14, 15. Assertion density is lower in this cycle's field notes than in prior cycles' synthesis boundaries. The cross-cutting reflection's three shift claims are each tied to specific findings rather than presented as free-standing agent synthesis — a material improvement over the Cycle 4 PLAY pattern. The one concentration point is note 19's "Cycle 1 → Cycle 4 → Cycle 5 unchanged" framing, assessed in detail below. |
| Solution-space narrowing | Clear (inherited) | Stable — no new narrowing generated at PLAY | The narrowing that entered this PLAY was inherited from DECIDE (skill-framework-agnostic commitment; operation-named library; explicit-naming preferred dispatch contract). The field notes do not produce a new proposal document or introduce new "settled" claims. The routing summary routes BUILD-regression findings to BUILD-regression destinations and DECIDE-warranted findings to DECIDE. No new design conclusions are embedded in the PLAY artifact as settled. |
| Framing adoption | Ambiguous | Stable | The field notes adopt cycle vocabulary (AS-7 summarizer, L1+L2 dispatch surface, validation-vs-execution gap, fabrication-while-critiquing-fabrication) as analytical shorthand. The question of which framings are practitioner-generated vs. agent-introduced is addressed in the Element-Specific Assessments below. The most significant adoption concern is note 15's "fabrication while critiquing fabrication" framing, which composes two distinct observations into one analytical pattern without recording how the composition was generated. |
| Confidence markers | Ambiguous | Declining relative to prior phases | No "settled" language appears in the field notes themselves. The routing summary uses strong aggregate language ("largest BUILD-regression batch in the agentic-serving corpus's history"; "the cycle's load-bearing follow-up work is *not* in further DISCOVER-DECIDE iteration") — language that encodes a recommendation in what is nominally a classification table. This is assessed under Pattern 1 below. Note 19's "cross-cycle persistence (Cycle 1 → Cycle 4 → Cycle 5 unchanged) is itself a meta-signal" uses "unchanged" as a confidence marker for a trajectory claim whose evidential basis is thinner than the framing implies. |
| Alternative engagement | Ambiguous | Stable | The reconnaissance phase's nine probes are explicitly structured to test different cells of the dispatch surface (NL, explicit, composition, boundary). This is methodologically sounder than Cycle 4 PLAY's single-stakeholder inhabitation. However, within the inhabitation phase, alternative explanations for key observations are not deeply engaged: finding 4's `claim-extractor` runtime defect is attributed to a missing agent-level `system_prompt` without examining whether other executor paths could produce the same error; finding 6's summarizer-content-stripping is attributed to model-size limitations (qwen3:0.6b) without examining whether the problem is model-size, summarizer-prompt, or input-length; note 15's "fabrication while critiquing fabrication" is presented as a compound pattern without examining whether the fabrication and the critique arose from independent causes. |
| Embedded conclusions at artifact-production moments | Ambiguous | Declining | The Cycle 4 PLAY snapshot identified embedded conclusions in the proposal document (n=1 findings labeled as "settled"). Cycle 5 PLAY produces no equivalent proposal document. The field notes' routing summary embeds one classification conclusion that warrants examination: the decision to route notes 7 and 8 to SYNTHESIS rather than to BUILD-regression or DISCOVER. This routing is examined under Pattern 1 below. |

---

## Element-Specific Assessments

### Pattern 1: Selection bias in the six-category classification

**The question:** Are the SYNTHESIS-routed notes (3, 7, 8, 17) genuinely delight/confirmation, or would any of them be better classified differently if the framing test were applied honestly? Conversely, is note 1's BUILD-regression framing ("BUILD validation was structurally inadequate") overstating the challenge?

**Note 3 (explicit naming reliably dispatches):** The routing is accurate. The evidence is direct (four explicit-naming probes, all dispatched; artifact directories confirmed). The claim is scoped precisely: "ADR-021's explicit-naming contract works." No overgeneralization. SYNTHESIS is the correct destination; the evidence is operational, not analytical.

**Note 7 (web-searcher error-path handling is clean):** The routing to SYNTHESIS as "delight" is technically defensible — the BUILD-time smoke-test claim is verified at the live-dispatch layer. However, this routing draws on a narrow framing: the web-searcher's *error path* works. The web-searcher's *success path* was not tested (TAVILY_API_KEY unset by design). Whether a finding about error-path behavior in a component that was not exercised on its success path belongs in SYNTHESIS-delight or DISCOVER (what does the success path actually look like under Tavily?) is a classification choice. The current routing is not wrong, but it is the more optimistic of two honest routings. The alternative would be: SYNTHESIS (delight, error-path) + DISCOVER (candidate — success path exercisability is not confirmed; operator-configurable Tavily documentation has not been tested against real queries). The field notes do not record the alternative routing. **Weight: low. The finding is accurately scoped; the SYNTHESIS routing is not suppressive. The alternative routing would sharpen the success-path-gap signal rather than challenge the error-path finding.**

**Note 8 (composition pipeline respects dependencies):** The SYNTHESIS routing is accurate. The evidence is operational (three-stage explicit pipeline, first-stage failure, orchestrator halted downstream without invoking broken ensembles). The finding is also partly a positive finding about the orchestrator's multi-stage reasoning, not just the pipeline mechanics. One alternative framing worth examining: note 8 was dispatched against a pipeline that included two known-broken ensembles (`claim-extractor`, `argument-mapper`). The orchestrator's decision not to proceed past the first failure was sound; but the three-stage composition as-written would never succeed in the current library state (web-searcher → claim-extractor is already broken regardless of Tavily). Whether this should route to BUILD-regression as "composition pipeline smoke-tests pass only when broken ensembles are protected by earlier failures" is a more challenging reading. The field notes do not record this alternative. **Weight: low-to-moderate. The SYNTHESIS routing is accurate about what was tested. The alternative reading surfaces a BUILD-regression signal the current routing does not capture: the composition test passed partly because the broken ensembles were shielded by the authentication failure before they were reached.**

**Note 17 (multi-turn context preserved):** The SYNTHESIS routing is accurate. The practitioner observation is direct ("I read this file earlier. Here's the three-agent flow:" — the orchestrator reused it). The note itself offers a qualification: "Whether this improves output quality vs. single-shot is not yet tested (Query 5's stub-fabrication suggests memory of the YAML did not constrain hallucination of the LLM-backed implementation)." This qualification is honest — the finding is not unconditionally positive. The routing is SYNTHESIS with the qualification inline. No alternative routing is warranted.

**Note 1's "BUILD validation was structurally inadequate" framing:** The framing is agent-introduced. The empirical basis is specific: `validate_ensemble` and `check_ensemble_runnable` returned clean results for ensembles that fail at runtime. This is a real gap. However, the characterization "structurally inadequate" carries more architectural weight than the evidence supports. An alternative reading: the gap is a BUILD-close practice gap (ensembles were not runtime-tested before declaration), not a structural validation-layer inadequacy. The `validate_ensemble` and `check_ensemble_runnable` tools were not designed to be runtime correctness checks — they are discovery and schema checks. The gap is better framed as: "BUILD close did not include a runtime dispatch test as a verification step" rather than "the validation surface is structurally inadequate." The framing matters for downstream DECIDE work: "structural inadequacy" implies the validation tooling needs redesign; "missing BUILD-close practice" implies a scenario addition is the remediation. Note 1's routing summary text uses "BUILD validation was structurally inadequate" as a label; the note itself (the observation body) is more precise ("No scenario in `scenarios.md` requires dispatch-exercise verification before BUILD declares ensembles working"). The note body is accurate; the routing-summary label overstates. **Weight: low-to-moderate. The over-framing is in the routing summary, not the note body. A future DECIDE phase reading the routing summary may author a validation-tooling redesign scenario when a BUILD-close checklist scenario is the proportionate response.**

**Selection-bias pattern summary:** The SYNTHESIS routings are defensible on the evidence with low-weight exceptions at notes 7 and 8. The higher-weight concern is note 1's routing-summary label overstating the validation-layer diagnosis. DECIDE-routed notes (2, 6, 15, 18) do carry the cycle's challenge-the-design weight; this is not an asymmetric routing that suppresses challenging findings. SYNTHESIS carries genuinely operational findings. The framing concern is confined to the routing-summary abstraction layer, not to the observation layer itself.

---

### Pattern 2: Gamemaster/player role blur under task load

**The question:** Did the reconnaissance probes pre-load the inhabitation queries with framing the practitioner adopted rather than generated independently? Did Cycle 4 PLAY notes 16 and 19 carry into Cycle 5 as background assumptions measured against rather than tested against?

**On reconnaissance pre-loading the inhabitation queries:**

The method's two-phase design explicitly positions reconnaissance as an efficiency device: "exhaust the test surface that can be measured programmatically so the inhabited stakeholder's time attends only to what cannot be measured this way." This is methodologically sound — reconnaissance that discovers broken ensembles prevents wasted stakeholder time and allows curated query selection. However, the probe sequence had a specific framing effect that warrants examination.

Reconnaissance finding 2 established "NL framing never dispatches." Reconnaissance finding 3 established "explicit naming reliably dispatches." These two findings became the inhabitation phase's baseline orientation: "Queries 1–5 selected from gamemaster suggestions, ordered most-information-first given runtime defects established in reconnaissance." The practitioner was not discovering the explicit-naming dispatch surface during inhabitation — the gamemaster had already established it and the inhabitation queries were selected to confirm and extend it.

The "explicit naming reliably dispatches" finding (reconnaissance note 3 → SYNTHESIS note "settled-by-use") entered the inhabitation as confirmed background, not as a hypothesis the practitioner tested. Whether the practitioner would have arrived at this finding independently through OpenCode — or whether the practitioner might have observed something different with client tools available (tools absent in curl reconnaissance) — was not tested. The inhabitation phase produced no evidence that explicit naming sometimes fails under OpenCode's richer client-tool surface; this may be because it genuinely never fails, or because the queries were pre-selected to exercise the explicit-naming path rather than probe its boundaries.

**On Cycle 4 PLAY notes 16 and 19 as background assumptions:**

Cycle 4 PLAY note 16 ("Calibration Gate does not audit orchestrator's narration") and note 19 ("No-dispatch fallback path; infrastructure does not fire at all") are referenced explicitly in Cycle 5 PLAY as prior framing that the new observations extend. Note 2 of Cycle 5 reconnaissance: "This is Cycle 4 PLAY note 19 generalized." Note 6 of reconnaissance: "This is Cycle 4 PLAY note 16... in concrete operational form, but one layer deeper." Note 15 of inhabitation: "The orchestrator's natural-language narration *is* the failure surface that has no calibrator" — the same framing as Cycle 4 note 16, one session later.

The new observations are genuine extensions of the prior findings, not mere repetitions. Note 6 adds the AS-7 summarizer as a specific failure point; note 15 adds the fabrication-while-critiquing-fabrication compound pattern; note 12 adds the orchestrator's own verbatim confirmation of its execution-graph blindness. These are new empirical data, not recycled claims.

However, the framing inherited from Cycle 4 means the new observations are measured against pre-established categories (quality-infrastructure coverage gap; no-dispatch fallback path) rather than allowed to generate their own categories. The AS-7 summarizer failure (note 6) is the Cycle 5 PLAY's most operationally concrete new finding, but it is framed as a variant of the Cycle 4 quality-gap finding rather than as its own architectural category (the summarizer as a trust-inversion point where ensemble outputs are altered before reaching the orchestrator). Whether "content-stripping-and-status-inversion by the summarizer" deserves its own framing distinct from "orchestrator narration uncalibrated" is a question the Cycle 5 notes do not ask. The Cycle 4 framing is accurate as far as it goes; the question is whether it goes far enough.

**On the "explicit naming reliably dispatches" finding:**

The BUILD-close susceptibility snapshot's Advisory 1 (scope-claim breadth) noted that "explicit naming reliably dispatches" rests on reconnaissance probes under curl (no client tools). The inhabitation's OpenCode queries extended this to a client-tool-rich environment, and the finding held. This is genuine confirmation. However, the confirmation is from five queries, all of which used explicit naming per the gamemaster's recommendation. The inhabitation did not test natural-language dispatch under OpenCode (which has client tools curl doesn't), so the NL-never-dispatches finding from reconnaissance is unextended by inhabitation. Whether client-tool availability changes NL dispatch behavior remains untested.

**Role-blur summary:** The method introduces structural pre-loading that is explicit and legitimate (reconnaissance → curated inhabitation queries). The concern is narrower: the inhabitation queries were optimized to confirm and extend reconnaissance findings rather than challenge them. This means certain boundaries of the reconnaissance findings (NL dispatch under client tools; alternative failure modes for the claim-extractor defect; AS-7 summarizer as a distinct architectural category) were not tested by the inhabitation. The gamemaster-proposed queries directed practitioner attention toward the already-established explicit-naming path; the practitioner's investigations within that path were genuine, but the path selection itself was agent-directed.

---

### Pattern 3: Framing adoption in field note language

**The question:** Which analytical framings in the field notes are practitioner-generated vs. agent-introduced? Where agent-introduced framings are recorded as if settled, is this a susceptibility signal?

**Framings with clear practitioner origin:**

- "Seemed like that took a long time. Why?" (note 11) — practitioner verbatim; the framing surfaces the latency-phenomenology finding.
- "What was the total run-time of the ensemble?" (note 12) — practitioner verbatim; surfaces the execution-graph-blindness finding.
- Note 19's routing-observability concern — practitioner verbatim: *"the observability of the routing still seems lacking to me..."* This is the practitioner independently reconfirming a prior finding, not the agent recycling a frame.

**Framings that are demonstrably agent-introduced:**

- "L1+L2 dispatch surface" — appears in the method block ("exercising 9 probes across NL/explicit/composition/boundary cells of the test space, to characterize the L1+L2 dispatch surface"). This is the gamemaster's framing of what reconnaissance was measuring; not practitioner vocabulary.
- "validation-vs-execution gap" — appears in the cross-cutting reflection's "What did play reveal that the specs missed?" section as a named finding category. No practitioner verbatim uses this phrasing. The underlying observation (validate_ensemble passes; runtime fails) is empirical; "validation-vs-execution gap" is the agent's naming of it.
- "fabrication while critiquing fabrication" — note 15's heading. This composes two distinct observations: (a) the orchestrator presented stub code as ensemble output (fabrication); (b) the orchestrator simultaneously wrote "the ensemble's output habit is still too narrative" (critique). The compound pattern is agent-analytical. The two observations individually are empirically grounded (the stub code is quoted; the orchestrator's critique is quoted). Whether the compound pattern is load-bearing or rhetorical is assessed below.
- "AS-7 summarizer" as shorthand for `agentic-result-summarizer` — this is internal notation compressing an artifact name. It is not practitioner vocabulary; it is cycle-corpus vocabulary the agent introduced in earlier phases (ADR-004/AS-7). Its use in PLAY field notes is appropriate shorthand but does not constitute practitioner-settled framing.

**On "fabrication while critiquing fabrication" (note 15):**

This is the dispatch prompt's most specific framing-adoption concern. The claim is that the compound pattern "fabrication while critiquing fabrication" is agent-analytical and composes two independent observations into one.

Examining the evidence: Query 5 produced (a) stub code with `# TODO: implement...` and (b) the orchestrator's text "The ensemble's output habit is still too narrative — worth filing as a dispatch config issue." These are sequential events in one turn, not events the practitioner observed independently and the agent combined analytically. The orchestrator generated both in the same output. The compound pattern is therefore an accurate description of what happened, not a rhetorical compression of separate events. The fabrication (stub code) and the critique (ensemble output too narrative) occurred in the same orchestrator output, making the compound framing empirically warranted as a description of that output.

However, the note routes this to DECIDE as "calibration coverage gap for orchestrator natural-language narration." This routing follows the Cycle 4 PLAY framing (note 16: "orchestrator's reasoning surface remains unaudited"). The compound framing is load-bearing for the routing: if the two elements were treated as separate findings (one BUILD-regression, one DECIDE), the routing would differ. The current routing treats both as evidence for the same DECIDE-level gap. That reading is defensible — both observations point to the orchestrator's narration as an uncalibrated surface — but it elides a distinction: the stub code is a hallucination failure that is *practitioner-facing* (it would be presented to the practitioner as ensemble output); the orchestrator's critique of the ensemble is *meta-level* (it is accurate about the ensemble's behavior). Routing both to the same DECIDE framing flattens this distinction.

**Weight:** Low-to-moderate. The compound pattern is empirically warranted as a description of a single turn's output. The routing concern is that it concentrates two qualitatively different failure modes under one framing, potentially directing DECIDE toward a single architectural response (calibration coverage for orchestrator narration) when the fabrication failure might warrant a more targeted BUILD-regression response (the orchestrator should not present self-generated code as ensemble output when the ensemble returned an error).

**On "validation-vs-execution gap":**

This framing is agent-introduced and appears in the routing summary as the first of five findings "the specs missed." It is also the most consequential framing for downstream DECIDE work: it labels the BUILD-close verification surface as structurally inadequate.

The empirical basis is sound (four ensembles fail at runtime; validate_ensemble passed for all four). The framing "gap" implies the two surfaces should cover the same territory, and that the gap between them is an architectural defect rather than a scope difference. An alternative reading: `validate_ensemble` and `check_ensemble_runnable` are discovery and schema checks by design; the gap is not a validation-surface gap but a BUILD-close practice gap (no runtime dispatch test was required before declaring close). The framing "validation-vs-execution gap" is accurate as a description of what was observed; whether it correctly diagnoses the cause (tool inadequacy vs. practice gap) is a classification question the framing forecloses by naming it as a structural gap.

The Cycle 5 BUILD snapshot Advisory 2 (script-agent schema constraint) specifically addresses how BUILD-time discoveries that were not crystallized as named findings carry forward. "Validation-vs-execution gap" is a PLAY-time crystallization of a BUILD-time failure mode. Whether it crystallizes in the right direction (tool redesign vs. scenario addition) is not examined in the PLAY field notes.

---

## Interpretation

### Pattern assessment

Cycle 5 PLAY's field notes are substantially more empirically grounded than the Cycle 4 PLAY field notes at every layer that prior snapshots flagged. The reconnaissance phase produces artifact-directory-verified, error-message-anchored, execution.json-confirmed findings. The inhabitation phase produces practitioner-verbatim-anchored observations. The cross-cutting reflection ties each of its three shifts to specific noting evidence rather than presenting them as free-standing agent synthesis.

The susceptibility concentration in this PLAY is different in character from the Cycle 4 PLAY pattern. Where Cycle 4 PLAY's risk was n=1 architectural claims encoded as settled in a proposal document, Cycle 5 PLAY's risk is narrower: framing labels in the routing summary that encode architectural diagnoses (validation-vs-execution gap as structural inadequacy; note 1 label overstating the challenge) and one compound framing that may direct DECIDE toward a single architectural response when two qualitatively different failure modes are present.

Three observations compound into a pattern worth naming:

1. The routing summary uses the strongest-possible language about the BUILD-regression findings: "largest BUILD-regression batch in the agentic-serving corpus's history"; "the cycle's load-bearing follow-up work is *not* in further DISCOVER-DECIDE iteration." This language is agenda-setting, not merely classification. It is accurate that four ensembles are broken and that fixing them is the most time-sensitive follow-up. But the framing suppresses the DECIDE-routed findings (notes 2, 6, 15, 18) as lower-priority by contrast, when notes 6 (result-summarizer as trust-inversion point) and 15 (fabrication-while-critiquing-fabrication) carry architectural implications for the cycle's quality-infrastructure design that are not addressed by BUILD-regression fixes alone.

2. The note 19 "Cycle 1 → Cycle 4 → Cycle 5 unchanged" framing presents the visibility gap as a trajectory with confirmed stasis. The empirical basis is: (a) Cycle 1 PLAY notes 9–10 named bilateral visibility absence; (b) Cycle 4 PLAY note 7 reframed it concretely; (c) Cycle 5 PLAY note 19 practitioner verbatim: *"the observability of the routing still seems lacking to me... This was flagged last time."* The practitioner confirmed the gap persists and referenced a prior session. Whether the gap is "unchanged" — i.e., whether Cycle 5's new internal events (verdicts, tier-routing decisions, audit records, signal-channel aggregations) represent meaningful progress toward the gap's resolution or whether they are structurally equivalent to Cycle 4's "events fire internally but don't surface" — is a factual question the framing resolves in the stasis direction. Cycle 5 BUILD did produce new internal events. They do not yet surface to a human-visible terminal. The gap in its current form (internal events exist but don't surface) is not "unchanged" from Cycle 1 (no events fired at all) — it is narrowed at the infrastructure layer and unchanged at the operator-surface layer. The "unchanged" framing is accurate about the operator experience; it is potentially misleading about architectural progress.

3. The BUILD-close susceptibility snapshot's three advisory carry-forwards (preservation-scenario amendment pattern; script-agent YAML schema constraint; ADR-019 §Consequences §Positive n=1 qualifier) are not directly addressed in the Cycle 5 PLAY field notes. This is appropriate scope: PLAY attends to live-deployment encounters, not to DECIDE/BUILD corpus documentation gaps. The absence of these advisories from PLAY's field notes is not selection bias — it reflects the boundary between what PLAY can observe (runtime behavior) and what the advisories address (documentation and scenario completeness). The carry-forwards remain active; they should be explicitly tracked in the next cycle's DISCOVER entry, not resolved by PLAY.

### Earned confidence vs. sycophantic reinforcement

The reconnaissance findings are the most clearly earned element in the snapshot: the evidence is programmatic (curl probes), the artifact-directory diffs are verifiable, and the error messages are quoted directly. The inhabitation findings are earned at the verbatim-quote layer (practitioner self-reports are preserved) and carry moderate analytical overlay at the routing layer (compound framing of note 15; routing-summary label for note 1).

The cross-cutting reflection's three shift claims represent genuine improvement over the Cycle 4 PLAY pattern: each shift is tied to a specific finding (shift 1 → findings 4/5; shift 2 → findings 4/5/10/11; shift 3 → findings 3/6), and the reflection names its own uncertainty ("whether this improves output quality vs. single-shot is not yet tested"). The agent synthesis is more clearly demarcated than in Cycle 4 PLAY.

The confidence concentration is in the routing summary's aggregate framing, not in the observation notes themselves. This is a lower-risk concentration point than the Cycle 4 PLAY pattern (where the risk was in a proposal document encoding settled architectural claims). The routing summary is one document layer further from the artifact corpus than the proposal document was, and its "not in further DISCOVER-DECIDE iteration" framing is a scheduling recommendation, not an architectural settlement.

### Prior advisory carry-forward status

| Advisory | Origin | Status at Cycle 5 PLAY |
|----------|--------|------------------------|
| BUILD Advisory 1 — preservation-scenario amendment pattern | Cycle 5 BUILD snapshot | Not addressed in PLAY (appropriate scope). Persists as active carry-forward for Cycle 6 DISCOVER. |
| BUILD Advisory 2 — script-agent YAML schema constraint documentation | Cycle 5 BUILD snapshot | Not addressed in PLAY (appropriate scope). Persists as active carry-forward for next documentation pass. |
| BUILD Advisory 3 — ADR-019 §Consequences §Positive n=1 qualifier | Cycle 5 BUILD snapshot | Not addressed in PLAY. Persists as active carry-forward. Cycle 5 PLAY's evidence did not extend the skill-framework-agnostic claim's evidence base beyond RDD. |
| Cycle 4 PLAY Advisory 1 — note 14 DISCOVER secondary | Cycle 4 PLAY snapshot | Honored at Cycle 5 DISCOVER. Closed. |
| Cycle 4 PLAY Advisory 2 — proposal "settled" claims reclassification | Cycle 4 PLAY snapshot | Honored at Cycle 5 DECIDE. Closed. |
| Cycle 4 PLAY Advisory 3 — cross-cutting reflection attribution | Cycle 4 PLAY snapshot | Partially honored in Cycle 4 PLAY notes (prefatory attribution added); not explicitly repeated in Cycle 5 PLAY cross-cutting reflection, though shift claims are more tightly evidence-anchored. Net: substantially mitigated; not fully closed. |
| Cycle 4 PLAY Advisory 4 — notes 14/19 attribution | Cycle 4 PLAY snapshot | Honored in Cycle 4 PLAY field notes with explicit attribution footnotes. Closed for Cycle 4 PLAY. Cycle 5 PLAY note 19's "unchanged" framing is a new instance of the same pattern (framing introduced by agent at analytical register without practitioner engagement trace). See Advisory 3 below. |
| DECIDE Advisory 1 — scope-claim breadth | Cycle 5 DECIDE snapshot | Not extended by PLAY (PLAY did not produce non-RDD framework evidence). ADR-019 §Consequences §Positive still lacks inline n=1 qualifier. Persists as carry-forward; addressed under BUILD Advisory 3 above. |
| DECIDE Advisory 2 — no-dispatch-fallback empirical test | Cycle 5 DECIDE snapshot | Cycle 5 PLAY reconnaissance finding 2 provides the first live-deployment empirical data: NL framing never dispatched under five curl probes with the `minimax-m2.5-free` orchestrator. This is new evidence consistent with the "intended scope" reading at the DECIDE level. The evidence is not unambiguous: five curl probes without client tools may not represent NL dispatch under OpenCode. The inhabitation phase did not test NL dispatch under OpenCode, so the claim is still partially unexercised at the full-client-tool layer. **Assessment: partially resolved. The empirical test surface was exercised under curl; the OpenCode NL dispatch path was not tested. The advisory's "first empirical test" criterion is met under curl; the full-client-tool test remains open.** |

---

## Recommendation

**No Grounding Reframe warranted.** The signals do not converge on a pattern where the practitioner or a future agent would be building on an unexamined hidden assumption that poses operational risk. The BUILD-regression defects (four broken ensembles, summarizer content-stripping) are identified and documented with sufficient specificity to be addressed mechanically in a follow-up session. The DECIDE-routed findings (notes 2, 6, 15, 18) are architecturally consequential and correctly classified. The field notes' empirical discipline is substantially improved over the Cycle 4 PLAY baseline.

**Three advisory carry-forwards for next-cycle entry:**

---

### Advisory 1 — Routing-summary framing should not schedule downstream phases

The routing summary's aggregate conclusion — "the cycle's load-bearing follow-up work is *not* in further DISCOVER-DECIDE iteration but in BUILD-regression of the shipped artifacts" — is a scheduling recommendation embedded in what is nominally a classification artifact. This framing may be directionally correct (the four broken ensembles should be fixed before further methodology-layer iteration), but routing summaries should classify observations, not prioritize phases.

The risk is that a future session resuming from the routing summary's aggregate text would deprioritize the DECIDE-routed findings (notes 2, 6, 15, 18) on the basis of the routing summary's framing, when note 6 (result-summarizer as trust-inversion point) and note 15 (fabrication-while-critiquing-fabrication) carry architectural implications that cannot be addressed by BUILD-regression alone. The AS-7 summarizer failure mode specifically — where ensemble results are altered before reaching the orchestrator — is not a YAML defect fixable by adding a missing `system_prompt`; it is a pipeline trust-boundary question that belongs in DECIDE.

At next-cycle entry, the BUILD-regression and DECIDE-routed findings should be treated as parallel workstreams, not as sequenced phases where BUILD-regression is gating.

---

### Advisory 2 — Note 1's routing-summary label should be distinguished from the note body

The routing summary labels note 1 as "BUILD validation was structurally inadequate" while the note body is more precise: "No scenario in `scenarios.md` requires dispatch-exercise verification before BUILD declares ensembles working." These are different diagnoses with different remediation paths.

"Structurally inadequate" directs DECIDE toward validation-tooling redesign. "No scenario requires dispatch-exercise verification" directs DECIDE toward a scenario addition (a BUILD-close checklist scenario mandating runtime dispatch test before phase-close declaration). The second framing is proportionate; the first is broader than the evidence supports.

At next-cycle DECIDE entry, the scenario-addition framing should be examined explicitly rather than inheriting the routing-summary label's structural-inadequacy framing. The `validate_ensemble` and `check_ensemble_runnable` tools were not designed as runtime correctness checks and do not need redesign; the BUILD-close practice does need a mandatory runtime dispatch verification step.

---

### Advisory 3 — Note 19's "unchanged" framing requires qualification at next-cycle carry-forward

The cross-cycle trajectory framing ("Cycle 1 → Cycle 4 → Cycle 5 unchanged") is accurate about the operator-terminal experience (no colored logs, no TUI, no dispatch telemetry visible). It is potentially misleading about architectural progress: Cycle 5 BUILD did ship new internal events (verdicts, tier-routing decisions, audit records, signal-channel aggregations) that did not exist in Cycle 1 or Cycle 4. The gap is now "infrastructure-complete but routing-incomplete" — events exist on disk (`execution.json`), but they do not yet reach a human-visible surface.

At next-cycle entry, the visibility gap carry-forward should be framed as: "internal events now exist and write to `execution.json`; the gap is the absence of a human-visible surface that routes these events to the operator terminal or the orchestrator's reasoning context." This framing distinguishes what Cycle 5 BUILD closed (infrastructure) from what remains open (routing). The "unchanged" framing in note 19, if carried forward without qualification, may cause a future cycle to re-architect the internal event model rather than building the missing routing surface.

The attribution point from Cycle 4 PLAY Advisory 4 applies here: the "Cycle 1 → Cycle 4 → Cycle 5 unchanged" characterization is agent-introduced analytical framing, not a practitioner-stated assessment of the gap's trajectory. The practitioner's verbatim confirms the gap persists; the trajectory claim is the agent's reading of what "persists" means across three sessions.
