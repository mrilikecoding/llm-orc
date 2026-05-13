# Argument Audit Report — Cycle 5 DECIDE (Round 2)

**Audited documents:**
- `docs/agentic-serving/decisions/adr-019-skill-framework-agnostic-capability-library.md`
- `docs/agentic-serving/decisions/adr-020-tool-use-ensemble-shape.md`
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md`
- `docs/agentic-serving/decisions/adr-015-per-role-tier-escalation-router.md` (partial-update header revised)

**Round 1 audit read:**
- `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-5-decide.md`

**Date:** 2026-05-12

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 14 (same corpus as round 1)
- **Round 1 issues resolved or substantially resolved:** 8 of 9
- **Round 1 issues partially resolved:** 1 of 9
- **New issues introduced by revisions:** 2

---

### Round 1 Issue Status

#### Issue 1.1 (P1) — ADR-015 partial-update header describes reframing that does not match body

**Status: Resolved.**

The revised ADR-015 header now explicitly distinguishes two planes of amendment. It states that the reframing "applies to ADR-015 §Negative *as characterized in the proposal and the Cycle 4 / Cycle 5 product-discovery rendering of ADR-015 §Negative* — specifically the 'operator-driven library migration' reading" and adds: "ADR-015's body §Negative addresses one concrete instance of operator-driven migration (Topaz-metadata addition on existing ensembles); ADR-019 leaves that body bullet undisturbed." The header further specifies exactly what ADR-019 amends (the broader framing extended in the proposal and product-discovery) and what it leaves unchanged (verdict-to-tier mapping, primary-skill framing, per-skill tier defaults, rejected alternatives). A reader consulting ADR-015's body can now understand precisely what the amendment touches and what it does not. The P1 concern is closed.

---

#### Issue 2.1 (P2) — ADR-021 Topaz-skill signal path not specified

**Status: Resolved.**

ADR-021 §Per-capability dispatch contract now contains a dedicated "Topaz-skill signal path" subsection describing both dispatch shapes explicitly. The explicit-ensemble-naming path (skill framework names the ensemble directly in `invoke_ensemble` arguments) is documented as the preferred shape with a clear statement that it preserves ADR-015's pre-specified-routing commitment end-to-end. The natural-language prompt path is documented as the accepted fallback, with acknowledgment that it "reintroduces LLM-judgment routing at the capability-selection boundary" and an explanation of why the scope is narrower than what ADR-015 §(f) rejected. Both paths are now visible with their respective trade-offs.

The specific re-check (3b) commissioned for this round is addressed in new issue P3-N1 below.

---

#### Issue 2.2 (P2) — ADR-020 DuckDuckGo rejection conflates quality and on-ramp arguments

**Status: Resolved.**

ADR-020 §(a-DuckDuckGo) now explicitly separates the two arguments under labelled sub-headings. The quality argument is introduced as "sufficient on its own to reject" and is presented first. The on-ramp consideration is explicitly labelled "a secondary on-ramp consideration (acknowledged as framing extension)" and states: "This argument is an extension of note 1's literal scope (note 1 concerned missing configuration files, not search-backend quality); it is acknowledged here as additional motivation rather than as the rejection's primary basis." The section also acknowledges that "Tavily's free tier is not friction-free either (signup is required) — the rejection of DuckDuckGo rests on the quality argument." This directly addresses the round 1 concern.

---

#### Issue 2.3 (P2) — Hidden assumption that skill frameworks decompose into Topaz-aligned sub-tasks

**Status: Resolved.**

ADR-021 §Consequences §Negative now contains a dedicated bullet explicitly naming the precondition: "The orchestrator routes by Topaz skill (per ADR-015); skill frameworks composing against the orchestrator must decompose their workflows into sub-tasks that map cleanly to the eight Topaz skills." The bullet names the RDD phase-to-capability mapping problem explicitly and references the `skill-framework-capability-registry.md` artifact as the deployment-time mitigation. The constraint is no longer understated.

---

#### Issue 2.4 (P2) — ADR-021 seam-case inversion produces no falsification trigger

**Status: Resolved.**

ADR-021 §Seam-case inversion now contains a dedicated "Falsification trigger" subsection. The trigger specifies a testable observational pattern: two or more skill frameworks consuming the same Topaz-skill slot showing persistently divergent calibration verdicts on the same library ensemble across at least two consecutive Tier-Router Audit windows, where escalated-tier dispatch does not close the verdict gap. The resolution path under falsification is named (per-skill-framework capability ensembles, reopening ADR-019). The specific re-check (3a) is addressed in new issue P3-N2 below.

---

#### Issue 2.5 (P2) — ADR-019 minimum-viable set selected on RDD criterion without naming the tension

**Status: Resolved.**

ADR-019 §Working defaults now contains an explicit "Selection criterion acknowledgment" paragraph. It states: "the set's choice of five ensembles is driven by RDD's concrete BUILD-time demand (the only methodology consumer with concrete BUILD-time demand at Cycle 5). This makes the initial library shape *RDD-representative* rather than agnostically balanced across all eight Topaz slots." The paragraph distinguishes the contract-level agnostic commitment from the initial shape's single-methodology-consumer-driven selection. The un-named tension is now named.

---

#### Issue 3.1 (P3) — Inversion question 3 mislabelled as resolved at ADR-019 §Rejected alternative (c)

**Status: Resolved.**

ADR-021's inversion-question dispatch table now addresses question 3 explicitly. The table entry states the question is "not load-bearing for OD-3" and identifies ADR-019 §(c) as addressing "per-framework subdirectories (a different inversion)." The entry then names the actual wrongness conditions for the `agentic-serving/` convention (conflation of system ensembles with capability ensembles; prefix aging poorly) and explains why these are BUILD-phase authoring concerns mitigated by the `agentic-serving/README.md` commitment. The resolution is not a dismissal — the question is answered on the merits.

---

#### Issue 3.2 (P3) — ADR-020 claims calibration fires on `web-searcher` without acknowledging AUQ/HTC inapplicability

**Status: Resolved.**

ADR-020 §Consequences §Positive now contains a detailed bullet: "The Calibration Gate fires on `web-searcher`'s output via ADR-007's post-hoc result-check (structural schema verification of the returned JSON — count, fields, error-flag). Tier-Router Audit observes the dispatches; the cross-layer signal channel receives signals. ADR-014's AUQ (verbalized-confidence) and HTC (trajectory-feature) calibration components are *structurally inapplicable* to script-agent execution — there is no LLM reasoning trace inside a deterministic script — so the calibration value for `web-searcher` is on result-structure validation, not on dispatched-output confidence." This directly addresses the round 1 concern and corrects the overreach.

---

#### Issue 3.3 (P3) — Fresh-context property's "load-bearing" characterization presented as settled in §Consequences §Positive

**Status: Resolved.**

ADR-021 §Consequences §Positive now distinguishes the architectural fact from the agent-introduced framing: "The architectural fact: `invoke_ensemble`'s dispatch mechanics give each dispatched ensemble's agents `input + system_prompt` only — no orchestrator conversation history bleeds into the dispatched context. This property is real (it is how `invoke_ensemble` works); the *characterization* of it as load-bearing for sycophancy-resistance-style architectural patterns is an agent-introduced analogy from RDD's ADR-058 (Architectural Isolation) captured in Cycle 4 PLAY note 14 with an attribution flag, pending BUILD-phase or future-cycle testing in non-RDD methodology contexts." The sentence then adds: "whether the property carries the *same load-bearing significance* for non-RDD methodologies is the part that remains candidate."

---

#### Framing P1 — No DECIDE-phase framing examination of the no-dispatch fallback

**Status: Resolved.**

ADR-019 §Consequences §Neutral now contains a dedicated bullet characterizing the no-dispatch fallback as intended scope. The bullet states: "The Cycle 5 DECIDE framing resolution: the orchestrator's direct natural-language response is the appropriate behavior when no library ensemble matches the task's shape. The quality infrastructure's scope is *dispatch-conditional by design*." The bullet covers the discover gate's examination commitment, explains why evaluative and meta-tasks naturally fall on the no-dispatch path, distinguishes library-coverage expansion from no-dispatch-path quality measurement, and explicitly notes that calibration on orchestrator-own-narration is "separate infrastructure territory — not in Cycle 5 scope, not foreclosed for future cycles." The specific re-check (3c) is addressed in new issue P2-N1 below.

---

#### Framing P2 — Vocabulary terms used without acknowledging candidate status

**Status: Resolved.**

ADR-019 §Decision now opens with an explicit "Vocabulary note before the decision" paragraph. It states that "capability ensemble", "operation-named ensemble", and "three-layer architecture" "enter this ADR with the candidate status recorded in `product-discovery.md`'s vocabulary table (Cycle 5 update)" and that the ADR's "concrete library-authoring decisions and §Consequences are the DECIDE-phase test those terms' candidacy was pending." It specifies the settlement criterion: if BUILD-phase concrete authoring "confirms the terms serve operator and architecture-reader use, the product-discovery vocabulary table moves them from 'candidate under DECIDE examination' to 'settled (survived BUILD-phase authoring)' at cycle close." This directly addresses the round 1 drift-pattern concern.

---

#### Framing P2 — ADR-020 Brave/Exa/Serper adapters BUILD scope not clarified

**Status: Resolved.**

ADR-020 §Backend choice now states explicitly: "**Cycle 5 BUILD scope ships the Tavily adapter only**; Brave/Exa/Serper adapters are deferred to operator-driven extension when a deployment needs them. The adapter pattern's purpose is to keep alternative-backend authoring mechanical (one-file Python addition + environment-variable selector); the alternatives are *supported by design*, not by shipped code." The BUILD-scope boundary is unambiguous.

---

#### Framing P3 — "Three-layer architecture" term promoted to structural vocabulary without notation

**Status: Resolved.**

ADR-019 §Provenance check now contains a dedicated entry for "Three-layer architecture" that explicitly tracks its vocabulary status: "Vocabulary status: 'three-layer architecture' is marked research-voice in product-discovery (candidate for relocation to `domain-model.md` §Methodology Vocabulary). Its use as the organizing structure of this ADR (and ADR-021's inversion-question dispatch table) effectively promotes the term to settled-as-structural-vocabulary in this cycle's corpus; after BUILD-phase concrete authoring exercises the term in practice, the product-discovery vocabulary table is updated with this ADR as the settlement basis, or — if the term has not entered operator voice by BUILD close — the term is relocated to `domain-model.md` §Methodology Vocabulary as research voice." The round 1 concern is closed.

---

### P2 — Should Fix (New Issues)

**Issue P2-N1 — ADR-019 §Neutral no-dispatch fallback bullet asserts framing-examination closure but does not demonstrate the examination was conducted.**

- **Location:** ADR-019 §Consequences §Neutral, no-dispatch fallback bullet
- **Claim:** "The Cycle 5 DECIDE framing resolution: the orchestrator's direct natural-language response is the appropriate behavior when no library ensemble matches the task's shape."
- **Evidence gap:** The round 1 framing P1 issue required that DECIDE either add a §Note to ADR-019 "characterizing the no-dispatch fallback as intended scope" or "produce a brief ADR-022 documenting the framing resolution." The bullet does assert the framing resolution. What it does not do is show the reasoning chain from the discover-gate's open examination commitment to the resolution. The bullet references "Cycle 4 PLAY notes 8, 10, 11, 12, 13, 18" for the claim that "the orchestrator's reliability profile (high on derivable claims, low on integration claims; consistent across task types) is the load-bearing property for no-dispatch path quality" — but the reader cannot verify this without reading those notes. The assertion "consistent across task types per Cycle 4 PLAY notes 8, 10, 11, 12, 13, 18" is a significant empirical claim standing in for the examination. The discover gate's framing question was "coverage gap requiring infrastructure extension vs. intended scope." The bullet resolves it as intended scope, but the reasoning rests on: (a) the orchestrator's reliability profile being "consistent across task types," which is a characterization of notes 8–18's pattern; (b) calibration on orchestrator-own-narration being "separate infrastructure territory," which is asserted without showing why the discover gate's examination commitment does not require addressing it now.

  This issue is distinct from the round 1 framing P1, which was about the absence of any examination. The resolution added examination content. The remaining concern is whether the examination adequately discharges the gate's commitment, or whether it substitutes assertion for argument. The bullet is substantially better than the pre-revision state; the question is whether the quality of the reasoning meets the gate's standard. At minimum, the bullet should acknowledge that the reliability-profile claim is a characterization of the PLAY notes rather than a fresh synthesis — and note whether the notes' collective sample (6 notes across one inhabitation session) is sufficient to establish "consistent across task types" as an architectural fact rather than an observation from one inhabitation.

- **Recommendation:** Add a single qualifying sentence to the no-dispatch fallback bullet: "The consistency characterization (notes 8, 10, 11, 12, 13, 18) is from one Cycle 4 inhabitation session (n=1); the architectural conclusion that the no-dispatch path's quality is governed by the orchestrator's reliability profile rather than by missing calibration infrastructure is the cycle's adopted working position, subject to BUILD-phase and future-cycle evidence."

---

### P3 — Consider (New Issues)

**Issue P3-N1 — ADR-021 natural-language path's "narrower LLM-judgment scope" justification coherence with ADR-015's pre-specified-routing commitment.**

- **Location:** ADR-021 §Per-capability dispatch contract, Topaz-skill signal path subsection
- **Claim:** The natural-language-prompt path "reintroduces LLM-judgment routing at the *capability-selection* boundary" and "this is accepted as a *narrower* LLM-judgment scope than the alternative ADR-015 rejected in §(f)" because ADR-015 §(f) rejected output-quality classification while this path uses LLM judgment for "input-to-ensemble matching, which is structurally analogous to how `list_ensembles()` has always worked."
- **Examination:** The coherence claim requires that input-to-ensemble matching is genuinely narrower in risk than the output-quality classification ADR-015 §(f) rejected. ADR-015 §(f)'s rejection grounds were: "runtime classification reintroduces LLM-judgment into the routing path" and "the ensemble's skill metadata is what makes the orchestrator's selection meaningful at the router level; runtime inference would be redundant LLM work on data the ensemble selection already encodes." The natural-language path does not have the redundancy problem (there is no pre-encoded ensemble selection from which skill metadata can be read — the skill framework is not naming an ensemble). However, ADR-015 §(f)'s general rejection of "LLM-judgment into the routing path" applies here: the orchestrator's LLM is classifying which ensemble matches the task description, which is exactly the classification pattern the tier-escalation router was designed to avoid relying on. The "analogous to how `list_ensembles()` has always worked" claim is load-bearing but is not elaborated. It is not obvious that `list_ensembles()` consultation plus LLM-judgment matching of prompt to ensemble description is structurally analogous to `list_ensembles()` for operator browsing — the former is a routing decision; the latter is an information lookup.
- **Assessment:** The justification is coherent enough to stand, but its load-bearing claim (the analogy to existing `list_ensembles()` behavior) is asserted rather than argued. The ADR is correct that the explicit-ensemble-naming path is preferred and the natural-language path is a supported fallback — the two-tier structure is the right response to the tension. The framing "narrower LLM-judgment scope" is defensible because input-to-ensemble matching is a capability-selection question (one LLM decision per sub-task) rather than a verdict-classification question on every dispatch's output. The issue is minor: the "analogous to how `list_ensembles()` has always worked" claim should either be removed (it does not strengthen the argument and invites the above objection) or elaborated.
- **Recommendation:** Remove or replace the `list_ensembles()` analogy. The standalone justification — that input-to-ensemble matching is a pre-dispatch selection decision, not an output-quality verdict, and that skill frameworks unable to maintain library-topology knowledge accept LLM-judgment routing as a deployment trade-off — is sufficient without the analogy.

---

**Issue P3-N2 — ADR-021 falsification trigger is well-formed but the resolution-path clause introduces a claim that requires examination.**

- **Location:** ADR-021 §Seam-case inversion, Falsification trigger subsection
- **Claim:** "The resolution path under falsification would not be per-skill-framework tier overrides (those address tier-routing divergence, not output-quality divergence); it would be **per-skill-framework capability ensembles**, which is structurally the methodology-coarse library shape ADR-019 rejected."
- **Examination (re-check 3a):** The falsification trigger itself is well-formed and observable. The trigger pattern (two skill frameworks, same Topaz-skill slot, persistently divergent calibration verdicts on the same ensemble across two consecutive audit windows, escalated-tier dispatch failing to close the gap) is specific, time-bounded (two consecutive ADR-018 windows), and maps to a named measurement surface (Tier-Router-Audit drift criteria). The "filed as Cycle 5+ research territory" framing is appropriate. The trigger structure meets the ADR-018 precedent.

  The resolution-path clause, however, introduces a new claim: that the falsification scenario's appropriate response is per-skill-framework capability ensembles, which is "the methodology-coarse library shape ADR-019 rejected." The implication is that falsification of the seam-case assumption would require reopening ADR-019. This claim is structurally sound but introduces a scope question: is the only available response under falsification per-skill-framework capability ensembles, or are there intermediate options (e.g., parameterized capability ensembles that accept a skill-framework context argument, or output-formatting variants of the same ensemble differentiated by invocation parameter)?

  This is a consideration-level concern, not a logical gap. The ADR honestly acknowledges that "ADR-019's commitment would re-open under this falsification trigger" and names the two resolution options (parameterized capability ensembles or explicit acceptance that agnostic commitment was over-broad). The claim is not demonstrably wrong; it may simply be incompletely enumerated.
- **Recommendation:** Add "or (c) parameterized capability ensembles accepting a skill-framework context argument" to the resolution-path options list in the falsification trigger subsection. This does not weaken the trigger; it makes the resolution-space acknowledgment more complete.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

The three alternative framings identified in round 1 remain available from the source material. The revisions address two of them partially.

**Framing A (DuckDuckGo favors frictionlessness; Tavily favors quality).** The revised ADR-020 now explicitly acknowledges the Tavily signup cost and labels the on-ramp framing extension as secondary. The quality argument stands as sufficient. The alternative framing's strongest claim — that frictionlessness-first operators would reasonably choose DuckDuckGo — is now explicitly addressed rather than elided. The alternative framing is still available in the source material, but the ADR now shows why the quality argument prevails over it. No remaining framing issue.

**Framing B (no-dispatch fallback as primary operating mode at current library maturity).** Addressed by ADR-019 §Neutral bullet, which characterizes the fallback as intended scope and notes that "evaluative and meta-tasks naturally fall on the no-dispatch path." The alternative framing remains available in the source material (Cycle 4 PLAY note 19's "most tasks land here" observation is not quoted in ADR-019). The ADR's treatment is adequate for a DECIDE-phase framing resolution, but the source material's quantitative character of note 19 ("most tasks") is not surfaced. This is not a new P1 concern — the qualitative resolution is appropriate for DECIDE — but it is worth noting that BUILD-phase calibration evidence will provide the first empirical test of whether the 5-ensemble minimum-viable set produces a dispatch infrastructure covering more than a minority of actual task traffic.

**Framing C (agnostic commitment as provisional, pending seam-case examination).** The revised ADR-021 now explicitly frames the agnostic commitment as conditional on deployment evidence, with a falsification trigger. The alternative framing is partially operationalized within the ADR rather than left as an unexamined alternative. The revisions adequately surface Framing C's epistemic status.

---

### Question 2: What truths were available but not featured?

**Available truth 1** (skill-framework agnosticism grounded in one framework only) is now addressed in ADR-019 §Provenance check's attribution note: "The 'three-layer architecture', 'operation-named ensembles', and 'capability ensemble' framings derive from one inhabitation session (n=1) plus the Cycle 5 DISCOVER gate refinement." The note acknowledges "BUILD-phase concrete-authoring as the practical test." The truth is now featured.

**Available truth 2** (no-dispatch fallback examine-at-DECIDE commitment was a deliverable the ADRs must discharge). Addressed by ADR-019 §Neutral bullet (see Issue P2-N1 above). The delivery is present but its reasoning chain is asserted rather than argued.

**Available truth 3** (OD-6 resolution via `skill-framework-capability-registry.md` was not flagged in any ADR). Now addressed in ADR-021 §Per-capability dispatch contract and §Consequences §Negative, which reference the `skill-framework-capability-registry.md` artifact as the Cycle 5 OD-6 resolution. The truth is now featured.

---

### Question 3: What would change if the dominant framing were inverted?

The round 1 inversion analysis remains accurate in its structural claims. The revisions do not change the dominant framing; they qualify it more carefully. Specifically:

- ADR-019's rejection of methodology-coarse library is now explicitly examined through all four inversion questions, with each question dispatched to a named rejection-rationale slot. The inversion is no longer an implicit assumption; it is an examined claim.
- The falsification trigger in ADR-021 operationalizes the conditions under which the dominant framing would need to reverse — so the inverted framing now has a defined trigger path rather than being permanently foreclosed.
- The "agnostic is best-current-approximation" characterization from round 1 Framing C is now partially achieved through the vocabulary candidacy note in ADR-019 §Decision, the BUILD-phase settlement criterion, and the seam-case falsification trigger.

The dominant framing's weakest remaining exposure under inversion: the initial library shape (5 ensembles, RDD-driven) is acknowledged as RDD-representative, but the claim that the *shape principle* (operation-named, capability-fine-grained) is agnostically correct is still grounded in one methodology consumer. The seam-case falsification trigger addresses output-quality divergence; it does not address the possibility that the operation-named principle itself is RDD-vocabulary-specific (e.g., a security-review methodology might decompose differently than claim-extraction + argument-mapping, requiring different ensemble names that look operation-named from one vocabulary but methodology-coarse from another). This is a P3-level concern that would require a second methodology consumer to be visible.

---

### Framing Issues

**P2-F1 — ADR-019 §Neutral no-dispatch fallback resolution is present but the examination quality is asserted rather than argued (same as issue P2-N1 above).**

See P2-N1. The discover-gate's framing-examination commitment is discharged at a minimum threshold. Whether the quality of the reasoning — specifically the reliability-profile claim resting on six PLAY notes from one inhabitation session — meets the gate's standard is the remaining exposure.

**P3-F1 — The no-dispatch fallback bullet's claim about orchestrator reliability being "consistent across task types" is drawn from one inhabitation session without quantification.**

- **Location:** ADR-019 §Consequences §Neutral, no-dispatch fallback bullet
- **Claim:** "The orchestrator's reliability profile (high on derivable claims, low on integration claims; consistent across task types per Cycle 4 PLAY notes 8, 10, 11, 12, 13, 18) is the load-bearing property for no-dispatch path quality."
- **Available truth not featured:** Cycle 4 PLAY note 3 (field-notes, if available in source material) recorded the orchestrator routing meta-introspection to code-generation — an instance of incorrect routing, which is a no-dispatch-path failure mode. Notes from one PLAY session may not be sufficient to establish "consistency across task types" for architectural purposes.
- **Recommendation:** The claim should be qualified: "consistent within the Cycle 4 inhabitation session's task range" rather than "consistent across task types." The framing is consequential because it is the load-bearing justification for the no-dispatch path's quality not requiring calibration infrastructure.
