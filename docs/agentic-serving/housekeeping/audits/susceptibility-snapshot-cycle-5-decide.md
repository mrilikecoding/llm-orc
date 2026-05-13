# Susceptibility Snapshot

**Phase evaluated:** DECIDE (Cycle 5)
**Artifact produced:** ADR-019, ADR-020, ADR-021, ADR-015 partial-update header, `skill-framework-capability-registry.md`, scenarios.md Cycle 5 Cycle Acceptance Criteria Table, interaction-specs.md updates (Skill Orchestration User stakeholder + Cycle 5 Ensemble Author / Operator tasks)
**Date:** 2026-05-12

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Research | Grounding Reframe triggered | Three grounding actions; autonomous-routing gap named |
| Cycle 4 Discover | Grounding Reframe triggered | Asymmetric readiness mapping; research-voice transplants; agent-side sycophancy inverse |
| Cycle 4 Model | Clean with feed-forwards | No reframe; vocabulary relocation discipline applied |
| Cycle 4 Decide | Grounding Reframe recommended (1 finding) | ADR-015 autonomous-routing evidence gap not carried into artifact |
| Cycle 4 Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE |
| Cycle 4 Build | Grounding Reframe (one targeted) + 2 advisory | Pre-loaded conditional-acceptance disposition; resolved in-cycle |
| Cycle 4 Play | No Grounding Reframe; 4 advisory carry-forwards | Voice blurring at synthesis boundary; n=1 findings encoded as settled in proposal document |
| Cycle 5 Discover | No Grounding Reframe; 2 advisory carry-forwards | Settlement-before-examination sequencing gap; four inversion questions not recorded as examined at gate |
| **Cycle 5 Decide (this snapshot)** | Evaluated below | |

The DISCOVER-gate snapshot's two advisory carry-forwards were:

- **Advisory 1:** OD-3 should open by examining the seam cases of the skill-framework-agnostic commitment before treating it as settled — specifically, does Topaz-skill routing produce parity across methodology contexts?
- **Advisory 2:** The four cycle-status inversion questions for three-layer architecture should be explicitly dispatched or deferred at DECIDE entry rather than inherited silently.

Both advisories fed directly into ADR-021's structure. Advisory 1 is acknowledged in ADR-021's §Context explicitly ("snapshot Advisory 1 flagged that the agnostic commitment was settled at the gate before its seam-case inversion was examined") and discharged through the §Seam-case inversion and §Falsification trigger subsections. Advisory 2 is directly instantiated in ADR-021's inversion-question dispatch table. Whether the discharge is substantive or formal is a primary evaluation question for this snapshot.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining | Three-round audit produced 9 → 4 → 0 issues, with each revision batch traceable to specific audit findings. Practitioner gate refinement added the conjunctive falsification standard (2026-05-12), which sharpens rather than softens the commitment's epistemic status. Assertion density within the ADRs is qualified by vocabulary candidate-status notes and n=1 scoping; the density is lower than the Cycle 4 PLAY proposal's undifferentiated "settled" language. |
| Solution-space narrowing | Clear (earned convergence candidate) | Stable — narrowed at discover gate, not further narrowed within DECIDE | The skill-framework-agnostic commitment was settled at the discover gate; DECIDE inherited it as its organizing substrate. Within DECIDE, the solution space was narrowed further only at OD-2 (script-agent shape) and OD-3 (per-capability dispatch) — both via examined rejection rationales. The scope-claim breadth (RDD + Anthropic Skills + OpenAI Assistants + MCP-based frameworks) is unchanged from discover; no attempt to widen or test it was recorded within the DECIDE phase. Whether this is earned narrowing or inherited assumption is addressed in element assessment §1 below. |
| Framing adoption | Ambiguous | Declining relative to prior cycles | ADR-019's vocabulary note and provenance checks label candidate terms, attribute agent-introduced framings, and distinguish practitioner-generated from substrate-derived and drafting-time synthesis. ADR-020's DuckDuckGo revision separates quality and on-ramp arguments, explicitly acknowledging the on-ramp framing as an extension. ADR-021's fresh-context property bullet distinguishes the architectural fact from the agent-introduced analogy. The attribution discipline from prior advisory carry-forwards is substantially present. Cross-ADR framing composition is the residual risk — assessed in element §2 and §4 below. |
| Confidence markers | Ambiguous | Declining | "Settled" language is confined to practitioner-verbatim-anchored items (skill-framework-agnostic commitment, operation-named principle). Vocabulary candidate-status note in ADR-019 §Decision explicitly defers settlement to BUILD-phase authoring confirmation. The conjunctive falsification standard (element §2 below) is the one confidence-marker concentration point requiring evaluation — it is framed confidently but was practitioner-authored, not agent-introduced. |
| Alternative engagement | Ambiguous | Stable | Rejected alternatives sections have genuine depth: ADR-019 lists 5 with substantive rationales; ADR-020 lists 4 including the DuckDuckGo rebuttal-elicitation revision; ADR-021 lists 3 plus the inversion-question dispatch table. Round-1 framing audit identified rebuttal-elicitation failures on DuckDuckGo and the no-dispatch fallback; round-2 verified both were addressed substantively (not cosmetically). The natural-language routing path's "narrower LLM-judgment scope" justification was challenged through three audit rounds; the revised framing (retrieval vs. evaluative-classification distinction) stands independently. |
| Embedded conclusions at artifact-production moments | Ambiguous | Declining | The vocabulary note in ADR-019 §Decision explicitly names candidate terms and defers settlement to BUILD. This is a structural improvement over the Cycle 4 PLAY proposal. Residual instance: the no-dispatch fallback resolution (ADR-019 §Neutral bullet) asserts framing closure ("The Cycle 5 DECIDE framing resolution: the orchestrator's direct natural-language response is the appropriate behavior when no library ensemble matches") but the reasoning chain is from one inhabitation session's notes, now qualified as n=1. The round-3 audit verified the n=1 qualification is present; the qualification holds. |

---

## Element-Specific Assessments

### 1. Skill-framework-agnostic scope claim (RDD + Anthropic Skills + OpenAI Assistants + MCP-based frameworks)

The argument auditor's round-1 Available-Truth-1 flagged: "the 'skill framework agnosticism' claim is broader than any existing evidence. Only RDD has been structurally verified." The claim is n=1 extended by assertion to a family of frameworks.

The ADRs address this in two ways:

**ADR-019 §Provenance check** (§Neutral note on framing attribution): "The 'three-layer architecture', 'operation-named ensembles', and 'capability ensemble' framings derive from one inhabitation session (n=1) plus the Cycle 5 DISCOVER gate refinement. This ADR settles them as cycle-decisions on the strength of (a) the practitioner-generated gate refinement, (b) the structural fit with existing `invoke_ensemble` properties (note 14), and (c) BUILD-phase concrete-authoring as the practical test."

**ADR-019 §Decision** vocabulary note: the commitment's settlement criterion is explicitly deferred to BUILD-phase authoring, framing this as "DECIDE-phase test those terms' candidacy was pending."

What the ADRs do not do: test or qualify the scope claim's breadth beyond these acknowledgments. The scope claim (covering Anthropic Skills, OpenAI Assistants, MCP-based frameworks) remains an assertion rather than a grounded claim. No evidence from those frameworks is cited; no structural property of those frameworks is examined to confirm they decompose into Topaz-skill-tagged sub-tasks in the way the commitment requires. The basis is structural logic (if a framework decomposes into capability-typed sub-tasks, it can use Topaz as the vocabulary) extended by the practitioner's directional preference.

**Assessment:** The n=1 acknowledgment is present and correctly qualified. The scope claim's breadth is a known limitation that is labeled and deferred — the label is real (vocabulary candidate status, BUILD-phase test), not cosmetic. Whether this constitutes adequate grounding or whether the cycle closes with an unexamined scope overreach depends on the BUILD phase's ability to test the claim with even one additional framework beyond RDD. A two-framework test (RDD + one other) at BUILD close would shift this from assertion to light evidence. The current state is a labeled working assumption, not a hidden one.

**Susceptibility weight:** Low-to-moderate. The breadth claim is inherited from the discover gate's practitioner-generated refinement (earned confidence at the preference level) and labeled as candidate in the artifact. The risk is that downstream cycles treat the full scope claim as established after Cycle 5 BUILD confirms only the RDD case.

---

### 2. Conjunctive falsification standard — whether it sets a realistically unfireable bar

The practitioner's gate-conversation refinement requires both:

- **(a)** The generalized agnostic scheme does not produce good results on long-horizon tasks (assessed at task-outcome level, not sub-task calibration level)
- **(b)** Framework-encoding is empirically the only way to recover good results

**The conjunctive standard evaluation:**

The value proposition the practitioner explicitly stated is cost savings via local-free-model leverage under cheap-cloud orchestration, with long-horizon task outcomes as the measurement surface. The conjunctive standard is designed to prevent premature inversion based on sub-task verdict divergence alone. This is a legitimate design choice: the commitment's claimed value is at the outcome level, not the sub-task level; testing the commitment's falsity at the sub-task level (which would fire faster and more easily) would misrepresent what the commitment claims.

The concern about "too high a bar to ever fire" is worth examining carefully. Condition (b) — that framework-encoding is the *only* way to recover good results — is particularly demanding. In practice, if condition (a) fires (bad long-horizon outcomes), there are typically multiple remediation paths available: reconfiguring prompts, switching models, adjusting ensemble composition, adding capability ensembles, changing tier defaults. The agnostic scheme's operator can modify many parameters. Framework-encoding (ADR-019 rejected alternative (a) or ADR-021's resolution path 2) would need to be the *only* remaining path, after all other modifications are exhausted.

This is a high bar. It is not formally unreachable, but it requires a combination of: sustained long-horizon outcome failure + exhausted non-framework-encoding remediation paths + positive evidence that framework-encoding specifically is the remedy. That combination is plausible in a mature deployment with accumulated evidence but may be structurally difficult to assemble in any single cycle.

**However:** the practitioner generated this standard explicitly, and it reflects a deliberate value judgment — the practitioner's directional commitment is strong enough that they are willing to accept a high bar for reversal. The standard is not the agent's construction; it is the practitioner's own refinement of the agent's original framing. The DISCOVER-gate snapshot found that the practitioner's engagement showed "sharp decisional engagement" even under same-day cognitive load. The conjunctive standard reads as deliberate, not as compliance.

The ADR-021 falsification trigger operationalizes the sub-conjunct (a) measurement surface well (Tier-Router-Audit drift criteria as the observation surface for verdict-distribution shift, with the explicitly noted step that tier-routing-level signal alone is insufficient — task-outcome-level signal is required). Resolution path (1) — parameterized capability ensembles — is explicitly named as the lightest extension before the full falsification trigger fires, which provides an intermediate remediation step that does not require condition (b) to be true.

**Assessment:** The conjunctive standard is demanding but practitioner-generated. It is not the agent composing a bar that protects the prior commitment from challenge; it is the practitioner stating what level of evidence they would require before reversing a commitment they hold for deliberate cost-architecture reasons. The bar may effectively prevent falsification during the corpus's current lifecycle (n=1 methodology consumer; limited deployment evidence), but this is a known limitation of early-stage empirical feedback rather than a susceptibility signal. The ADR names the standard openly, including the resolution paths that do not require the full bar (per-skill-framework tier defaults for verdict-distribution divergence; parameterized capability ensembles for output-quality divergence short of the full conjunctive threshold).

**Susceptibility weight:** Low. The demanding standard reflects the practitioner's deliberate cost-architecture reasoning, not agent-side inflation of the commitment's resilience.

---

### 3. Vocabulary candidate-status acknowledgments — visible qualification vs. settled-by-use

The vocabulary note in ADR-019 §Decision is explicit: "capability ensemble", "operation-named ensemble", and "three-layer architecture" "enter this ADR with the candidate status recorded in `product-discovery.md`'s vocabulary table." The ADR then uses all three terms throughout its body.

The Cycle 5 DISCOVER snapshot's Advisory 2 concern was that terms marked candidate in the product-discovery vocabulary table would be promoted to settled-as-structural-vocabulary by their use as the organizing skeleton of the ADRs. ADR-019 §Provenance addresses this explicitly for "three-layer architecture": "its use as the organizing structure of this ADR... effectively promotes the term to settled-as-structural-vocabulary in this cycle's corpus; after BUILD-phase concrete authoring exercises the term in practice, the product-discovery vocabulary table is updated with this ADR as the settlement basis, or — if the term has not entered operator voice by BUILD close — the term is relocated to `domain-model.md` §Methodology Vocabulary as research voice."

This is a more honest treatment than any prior cycle produced at the decide boundary: the ADR explicitly names the promotion effect, describes the mechanism by which it occurs, and specifies the BUILD-phase criterion for whether the promotion stands or reverses. The qualification is visible.

**Assessment of whether the qualification holds throughout:** The terms do operate as settled-by-use throughout the ADR bodies — a reader of the §Decision or §Consequences sections encounters them without qualification inline. The qualifications are front-loaded in the §Decision opening and back-loaded in the §Provenance check. This is appropriate structure for an ADR (the decision section describes the decision using its vocabulary; the provenance section tracks that vocabulary's status), but a reader who skips the vocabulary note and the provenance section will use the terms as settled. This is inherent to the ADR format and is not a suppression of the qualification.

**Susceptibility weight:** Low. The qualification is genuinely present, correctly framed, and more prominent than comparable situations in prior cycles. The per-section density of the qualification is appropriate to the ADR format.

---

### 4. Web-searcher script-agent shape — independent justification vs. inherited ADR-019 framing

ADR-020's provenance check states the script-agent ensemble shape's driver chain is: "proposal substrate §OD-2 option (a) + ADR-019's principle that capability ensembles live in the library." The library-substrate principle is explicitly ADR-019's framing, not ADR-020's independent conclusion.

The question is whether ADR-020 contributes independent grounding for the script-agent shape or simply inherits ADR-019's library principle and applies it mechanically.

**What ADR-020 contributes independently:**

- The rejection of option (b) MCP integration on ADR-003's closed-tool-surface grounds is an independent technical argument that does not depend on ADR-019: it rests on ADR-003's prior commitment and the specific concern that MCP-as-internal-action would bypass the calibration / tier-router / audit infrastructure. This is a consequence of ADR-003 and ADR-014/015/018, not of ADR-019.
- The rejection of option (c) client-side delegation does depend on ADR-019's library-substrate principle — the argument is that client-side delegation "produces no library entry" and the methodology consumer needs "a library entry it can `invoke_ensemble` against." This argument is circular with ADR-019's commitment: it assumes the library-substrate model to reject the alternative that doesn't require it.
- The DuckDuckGo rejection is independently grounded on quality and brittleness (the quality argument is sufficient on its own per the revision; the on-ramp framing extension is labeled as secondary and acknowledged as beyond note 1's literal scope).
- The Tavily default choice is explicitly labeled "drafting-time synthesis comparing the proposal's five candidates against the on-ramp-clarity constraint" — the provenance check is honest that this is best-judgment, not research-derived.
- The AUQ/HTC inapplicability acknowledgment (post-round-1 revision) is independently correct and required engagement with the existing calibration architecture's structure.

**Assessment:** ADR-020 has mixed independence from ADR-019. The MCP rejection is strongly independent. The client-side-delegation rejection is weakly dependent on ADR-019's library-substrate principle. The backend choice is drafting-time synthesis explicitly labeled as such. The calibration-applicability treatment is independently engaged. The cross-ADR dependency is real but transparent (the provenance check names it), and the dependency is architecturally appropriate — ADR-020 operates under ADR-019's commitment and should cite it. The question of whether that commitment itself is independently grounded is addressed in §1 above; ADR-020's honest citation of ADR-019 is not itself a susceptibility signal.

**Susceptibility weight:** Low. The transparent provenance citation of ADR-019 as the driver for the library-substrate argument, combined with ADR-003-grounded independent reasoning for the MCP rejection, shows adequate independence for an ADR that is explicitly downstream of ADR-019's scope.

---

### 5. Three-round audit pattern — substantive correction vs. convergence toward audit-friendly text

**Round 1 (9 issues: 1 P1, 5 P2, 3 P3, 2 framing P1, 3 framing P2, 1 framing P3):**
- The P1 issue (ADR-015 header mismatch) required an amendment to the header's structure — substantive, not cosmetic.
- The P2 issues included the Topaz-skill signal path gap (ADR-021 — required adding a new subsection specifying both dispatch shapes with their trade-offs), the DuckDuckGo conflation (required structural argument separation), the hidden Topaz-taxonomy precondition (required a new §Negative bullet in ADR-021), the falsification trigger absence (required a new §Falsification trigger subsection in ADR-021), and the minimum-viable set selection criterion (required a new paragraph in ADR-019 §Working defaults).

These are substantive additions to the ADR bodies, not wording adjustments.

**Round 2 (4 new issues from revisions: 1 P2, 2 P3, 1 framing P2):**
The round 2 issues are themselves evidence that round 1's revisions introduced new complexity — a no-dispatch-fallback resolution bullet that asserted examination rather than conducting it, a natural-language-path justification with a load-bearing `list_ensembles()` analogy that didn't hold up, and a falsification trigger's resolution-path list that was incomplete. These issues show that the audit cycle was not simply adding qualifications to placate the auditor; the revisions introduced argumentation that was then independently pressure-tested.

**Round 3 (4 round-2 issues verified clean, 0 new):**
Round 3 is a verification audit. The fact that all four round-2 issues resolved cleanly with no new issues is consistent with either genuine convergence or with simpler revisions (epistemic qualifications and structural completions rather than architectural changes). The round-3 audit explicitly notes that the revision batch was "localized" and that no new logical gaps were introduced.

**Assessment:** The three-round audit pattern shows evidence of substantive correction rather than convergence toward audit-friendly text. The key indicator is that round-2 issues were generated by round-1 revisions — the revisions introduced arguments that were then challenged. A convergence-without-substance pattern would typically show diminishing issue density across rounds with each round's issues being weaker versions of prior concerns; instead, round 2 produced a P2 issue (the no-dispatch-fallback examination quality assertion) that required a targeted qualification with specific n=1 scoping language. That is content-level engagement, not cosmetic drift management.

The one area where the audit-driven convergence hypothesis has some purchase: the framing P1 (no-dispatch fallback examination) was resolved by assertion — the ADR states the framing resolution ("intended scope") and supports it with PLAY note citations, but the round-2 audit correctly flags that this is "assertion rather than argument." The round-2 P2-N1 recommendation was to add a qualifying sentence; the round-3 verification confirmed it was added. The final state is a qualified assertion — less exposed than the unqualified assertion, but still a reasoning closure via session-notes citation rather than via the gate's intended discursive examination. This is the audit cycle's clearest case of resolving a structural concern through framing qualification rather than substantive examination.

**Susceptibility weight:** Low overall. The no-dispatch-fallback resolution's reasoning quality is the residual exposure: the examination commitment from the discover gate was discharged at a minimum threshold (the resolution is present; the n=1 limitation is labeled), but the quality of the reasoning falls short of the gate's discursive standard. This is a scope note concern, not an architectural framing failure.

---

### 6. Practitioner gate-conversation refinement — earned confidence or pattern completion

The practitioner's refinement of the falsification standard (requiring both long-horizon outcome failure AND framework-encoding being the only recovery path) arrived in the same session that advanced the cycle from Cycle 4 BUILD close through Cycle 4 PLAY through Cycle 5 DISCOVER through Cycle 5 DECIDE. Same-day cognitive load is substantial.

**Evidence for earned confidence:**

- The refinement is architecturally precise: it distinguishes sub-task verdict divergence (tier-routing-level signal) from task-outcome-level failure (the measurement surface the commitment's value proposition actually claims). This is a non-trivial distinction that requires understanding of the three-layer architecture's separation of concerns.
- The conjunctive structure specifically targets the premature-inversion failure mode — discovering capability-level output divergence and concluding the agnostic commitment was wrong. Excluding that failure mode from the falsification trigger is deliberate, not defensive: the practitioner is naming what they would actually consider a reversal-worthy finding.
- The DISCOVER snapshot noted that the practitioner's gate responses showed "sharp decisional engagement on the methodology-consumer framing" even under same-day cognitive load. The conjunctive falsification standard is consistent with that engagement pattern.

**Evidence for pattern-completion:**

- The practitioner's prior gate conversations in this cycle have all supported the skill-framework-agnostic direction. The conjunctive standard protects the commitment from the most likely forms of early counterevidence. A practitioner who had already committed to the direction at multiple gates would naturally generate a falsification standard that excludes weaker forms of challenge.
- The DISCOVER gate settled the commitment as "not under further inversion examination this cycle" — the practitioner had already indicated they were not looking for the commitment to be revisited. The falsification standard may reflect that settled disposition rather than fresh analytical engagement with what would actually count as falsifying evidence.

**Assessment:** The balance tilts toward earned confidence, for the following reasons. The precision of the distinction (sub-task verdict divergence vs. task-outcome failure) requires familiarity with the architecture's measurement infrastructure that would not be available to a practitioner pattern-completing a commitment without engagement. The same-day cognitive load context makes it more — not less — surprising that the refinement introduced a specific technical distinction rather than a vague endorsement. The practitioner's multiple-gate engagement in this cycle has consistently been substantive rather than affirmative; the DISCOVER snapshot noted this and found no Grounding Reframe warranted. The conjunctive standard is consistent with a practitioner who understands the architecture well enough to specify what would constitute real failure.

**Susceptibility weight:** Low. The refinement is consistent with earned confidence. The risk is the one named under §2 — that the standard's demanding character means the trigger may not fire during the corpus's current early-deployment phase, even if the agnostic commitment's generalization to non-RDD frameworks proves narrower than claimed.

---

## Interpretation

### Pattern assessment

The dominant pattern is **audit-driven substantive revision with residual first-cycle scope-claim breadth.** This is the most thoroughly audited DECIDE phase in the corpus (three audit rounds; 13 total issues addressed; practitioner gate refinements at two points in the phase). The sycophancy-resistance mechanisms that were missing or weak in prior cycles — provenance checks, vocabulary candidate-status acknowledgments, framing-adoption attribution discipline — are substantially present.

The susceptibility pattern that does remain is not a convergence-toward-user-framing signal; it is a **inherited-scope-breadth signal**: the skill-framework-agnostic commitment's claimed scope (covering frameworks beyond RDD) is grounded in one framework and extended by practitioner preference and structural logic. The extension is labeled and deferred to BUILD-phase authoring as the practical test, but the test does not cover frameworks beyond RDD in Cycle 5. If subsequent cycles treat the scope claim as established after BUILD confirms only the RDD case, the labeled assumption will become a silent one.

A secondary pattern is **cross-ADR framing composition where ADR-021's per-capability dispatch contract and ADR-020's script-agent shape both depend on ADR-019's library-substrate principle.** This dependency is architecturally appropriate (ADR-020 and ADR-021 are explicitly downstream of ADR-019) and transparently labeled in their provenance checks. The dependency is not a susceptibility failure; it is the normal propagation of a cycle's organizing architectural commitment to its dependent decisions. The susceptibility risk is that ADR-019's organizing commitment is not independently re-tested by its downstream ADRs — but the audit cycle did precisely that, and the most concrete inversion (seam-case examination, OD-3 deliberation, inversion-question dispatch table) is ADR-021's primary contribution, not ADR-019's repeat.

### Earned confidence vs. sycophantic reinforcement

The three-round audit pattern, the practitioner's two gate interventions (vocabulary note introduction; conjunctive falsification standard), and the trajectory of declining susceptibility across the cycle's five prior snapshots are collectively consistent with **earned convergence** rather than sycophantic reinforcement.

The one structural residual is the no-dispatch-fallback framing resolution, which closes the discover gate's examination commitment at minimum threshold — the resolution is present, the n=1 qualification is labeled, but the reasoning quality is assertion-supported-by-notes rather than discursively argued. This is a scope note concern that falls below the Grounding Reframe threshold individually; it has been labeled and qualified to a degree that makes it visible rather than hidden.

The inherited scope-claim breadth (n=1 framework verified, broader claim labeled but not tested) is the cycle's open epistemic debt. It is the correct kind of debt to carry forward — explicitly acknowledged, with a BUILD-phase test criterion named and a vocabulary settlement process specified.

---

## Recommendation

**No Grounding Reframe warranted.** The signals do not converge on a pattern where the practitioner would be building on a hidden unexamined assumption that poses operational risk to BUILD. The cycle's primary architectural commitments are transparently labeled as working assumptions with stated test criteria; the audit cycle produced substantive corrections rather than cosmetic convergence; the practitioner's gate interventions show sharp architectural engagement.

**Two advisory carry-forwards for BUILD entry:**

---

### Advisory 1 — Scope-claim breadth: two-framework BUILD test would shift n=1 to light evidence

The skill-framework-agnostic commitment's scope claim (covering Anthropic Skills, OpenAI Assistants, MCP-based frameworks) rests on one structurally verified framework (RDD) and structural logic extended by practitioner preference. ADR-019 §Provenance labels this honestly and defers to BUILD-phase authoring as the practical test.

The BUILD test as currently scoped will confirm or challenge the commitment for RDD only — the minimum-viable capability ensemble set is selected on RDD's research-workflow demand. This is appropriate for Cycle 5 BUILD's scope. The carry-forward is not a BUILD-scope expansion; it is a recording of the epistemic gap that will persist at BUILD close.

At BUILD close or at the next cycle's RESEARCH entry: if the BUILD phase produces evidence from even one non-RDD framework decomposing against the Topaz-skill vocabulary (e.g., a security-review-as-methodology skill tag or a code-review-as-methodology decomposition using the same `claim-extractor` and `argument-mapper` ensembles), that would shift the scope claim from n=1 to light-evidence. Without it, the "any skill framework" framing in ADR-019's §Consequences §Positive should be qualified to match the actual evidence base at cycle close.

---

### Advisory 2 — No-dispatch-fallback reasoning quality: below the discover gate's discursive standard

The discover gate explicitly assigned the no-dispatch-fallback framing examination to DECIDE. ADR-019 §Neutral discharges this commitment by asserting the intended-scope resolution and supporting it with PLAY note citations (notes 8, 10, 11, 12, 13, 18), now qualified as n=1 inhabitation, single orchestrator profile.

The discharge meets a minimum threshold. The gap is that the reasoning chain — from the discover gate's open question ("coverage gap requiring infrastructure extension vs. intended scope") to the DECIDE resolution ("intended scope by design") — rests on a characterization of PLAY notes rather than on a discursive argument engaging the coverage-gap alternative framing. The coverage-gap alternative is not examined; it is bypassed by characterizing the PLAY notes as showing the orchestrator's reliability profile is adequate. Cycle 4 PLAY note 3 (orchestrator misrouted meta-introspection to code-generation) is now acknowledged inline as a routing-failure case that complicates the "consistent" claim — this is an improvement from round 1. But the acknowledgment is additive to the assertion rather than being integrated into the argument.

BUILD-phase calibration evidence on the no-dispatch path will be the first empirical test of the intended-scope resolution. If BUILD shows that dispatch rates on the 5-ensemble minimum-viable set leave a large fraction of task traffic on the no-dispatch path (consistent with Cycle 4 PLAY note 19's "most tasks land here" observation), the intended-scope framing will face its most concrete challenge — not from argument but from evidence. Recording this as an open empirical question at BUILD entry is the appropriate carry-forward.
