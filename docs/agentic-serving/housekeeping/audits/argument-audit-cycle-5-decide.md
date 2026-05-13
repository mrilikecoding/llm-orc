# Argument Audit — Cycle 5 DECIDE

**Audited documents:**
- `docs/agentic-serving/decisions/adr-019-skill-framework-agnostic-capability-library.md`
- `docs/agentic-serving/decisions/adr-020-tool-use-ensemble-shape.md`
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md`
- `docs/agentic-serving/decisions/adr-015-per-role-tier-escalation-router.md` (partial-update header)

**Source material read:**
- `docs/agentic-serving/essays/005-layer-conditional-composition.md`
- `docs/agentic-serving/product-discovery.md` (Cycle 5 update)
- `docs/agentic-serving/proposals/agentic-serving-library-structure.md`
- `docs/agentic-serving/essays/reflections/field-notes.md` (both PLAY sessions)
- `docs/agentic-serving/housekeeping/gates/cycle-5-discover-gate.md`
- `docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-5-discover.md`

**Prior ADRs read for consistency:** ADR-002, ADR-003, ADR-006, ADR-011, ADR-014, ADR-015 (body), ADR-018

**Date:** 2026-05-12

---

## Section 1: Argument Audit

### Summary

- Argument chains mapped: 14
- Issues found: 9 (1 P1, 5 P2, 3 P3)

---

### P1 — Must Fix

**Issue 1.1 — ADR-015 partial-update header describes the reframing of "operator-driven library migration" but the §Negative bullet the reframing targets is not the §Negative bullet that exists in ADR-015's body.**

- **Location:** ADR-015 partial-update header (ADR-019-induced) vs. ADR-015 §Consequences §Negative body
- **Claim:** The header states that the amendment moves working-defaults authoring into BUILD scope and reframes "operator-driven library migration" from tag-every-existing-ensemble to author-the-operation-named-capability-ensemble-set.
- **Evidence gap:** ADR-015's §Negative consequence in the body addresses Topaz-metadata migration on *existing* ensembles, which is structurally different from the capability-ensemble-set authoring question ADR-019 resolves. A reader consulting ADR-015's body to understand what the amendment changed will not find the §Negative text that matches the header's described reframing.
- **Recommendation:** Either (a) add the reframed sentence explicitly to ADR-015 §Negative under a dated amendment note, or (b) clarify in the header that the reframing applies to ADR-015's §Negative *framing as characterized in the proposal and product-discovery* rather than to a specific body bullet. Option (a) is cleaner for future readers.

---

### P2 — Should Fix

**Issue 2.1 — ADR-021's "per-capability dispatch" contract does not specify how the skill framework communicates the Topaz skill of its sub-task to the orchestrator.**

- **Location:** ADR-021 §Decision, §Per-capability dispatch contract
- **Claim:** "The skill framework (client-side) decomposes its workflow into capability-typed sub-tasks, and emits one orchestrator request per sub-task." The orchestrator then routes by Topaz skill.
- **Evidence gap:** Two dispatch surfaces are named (natural-language prompt OR explicit `invoke_ensemble` tool-call arguments). If the sub-task arrives as a natural-language prompt, the orchestrator must infer the Topaz skill from prompt content — reintroducing LLM-judgment routing, the pattern ADR-015 rejected. If the sub-task arrives as explicit tool-call arguments, the skill framework needs knowledge of the library's topology. Neither path is described.
- **Recommendation:** Add a clause specifying the Topaz-skill signal path. If carried explicitly in `invoke_ensemble` arguments, say so. If inferred from prompt content, acknowledge that this reintroduces LLM-judgment routing at the methodology-dispatch boundary and explain why it is acceptable in this context.

**Issue 2.2 — ADR-020's rejection of DuckDuckGo conflates on-ramp friction arguments with quality arguments without distinguishing their differential weight.**

- **Location:** ADR-020 §Rejected alternatives, "(a-DuckDuckGo) Default DuckDuckGo HTML scrape"
- **Claim:** "A first-encounter operator running the no-authentication-required default and getting low-quality or broken results is exactly the on-ramp gap Cycle 4 PLAY note 1 flagged."
- **Evidence gap:** Cycle 4 PLAY note 1's practitioner verbatim ("the agentic-serving config is to me part of the build") concerned absent capability ensemble library and profile file, not web-search backend specifically. Tavily's free tier requires signup — not friction-free. The DuckDuckGo rejection conflates two structurally different arguments: HTML scrape brittleness (quality) vs. note-1-style on-ramp damage (friction).
- **Recommendation:** Separate the quality argument from the on-ramp argument and acknowledge that the latter is a framing extension beyond what note 1 strictly supports. The quality argument alone is sufficient to reject DuckDuckGo as default; the on-ramp argument addition is overreach.

**Issue 2.3 — ADR-019 and ADR-021 contain a hidden assumption that skill frameworks already exist in a sufficiently decomposed form to produce capability-typed Topaz-skill sub-tasks.**

- **Location:** ADR-019 §Decision (three layers); ADR-021 §Decision, §Consequences §Negative
- **Claim:** ADR-019 states "Any skill framework that decomposes its workflow into capability-typed sub-tasks" can compose against the orchestrator.
- **Evidence gap:** Neither ADR names the precondition: skill frameworks must decompose their workflows into sub-tasks aligned to the eight Topaz skills. RDD decomposes into phases (`/rdd:research`, `/rdd:decide`, etc.) that are not inherently Topaz-skill-named. Practical integration requires either RDD's plugin internally mapping phases to Topaz skills before dispatch, or an adapter layer translating RDD's vocabulary to Topaz's. ADR-021's §Negative says "Skill-framework decomposition logic is the client's responsibility, in entirety" — understates the constraint: it's *Topaz-taxonomy-aligned* decomposition.
- **Recommendation:** ADR-021 §Consequences §Negative should add a bullet explicitly naming the constraint: skill frameworks composing against the orchestrator must decompose their workflows into Topaz-skill-tagged sub-tasks. A skill framework with its own internal vocabulary requires an adapter. This is real integration burden growing with skill-framework diversity.

**Issue 2.4 — The seam-case inversion deferred in ADR-021 produces no falsification trigger.**

- **Location:** ADR-021 §Seam-case inversion; §Consequences §Neutral
- **Claim:** "The seam-case inversion is conditional on deployment evidence, not on this ADR's design-time choice." Resolution path: per-skill-framework tier defaults extension; empirical question filed as Cycle 5+ research territory.
- **Evidence gap:** ADR-018 set a precedent for empirical deferral via falsification trigger. ADR-021 names the observation surface (Tier-Router-Audit drift criteria) and the resolution path (per-skill-framework overrides) but contains no falsification trigger: what deployment evidence would invalidate the per-capability dispatch commitment itself and require revisiting the agnostic commitment? Without this, the deferral is open-ended rather than time-bounded.
- **Recommendation:** Add a falsification trigger analogous to ADR-018's. Specify what pattern in the Tier-Router-Audit's verdict-distribution data would constitute evidence that the agnostic commitment cannot accommodate methodology-specific dispatch needs.

**Issue 2.5 — ADR-019's minimum-viable capability ensemble set is selected on the criterion "serves RDD's research workflow" while the ADR's central commitment is skill-framework-agnostic.**

- **Location:** ADR-019 §Working defaults are in Cycle 5 BUILD scope
- **Claim:** "The set is the minimum that demonstrates the principle and serves RDD's research workflow."
- **Evidence gap:** Not logically inconsistent (a skill-framework-agnostic library can have an initial set that happens to serve one framework), but creates an un-named tension: the agnostic library's initial shape is framework-specific in its selection logic. The §Provenance check honestly admits this; the §Working defaults section does not cross-reference it. A reader of §Working defaults encounters the agnostic commitment and the RDD-specific minimum set without the selection logic's acknowledgment.
- **Recommendation:** §Working defaults should add a sentence acknowledging that the minimum-viable set's selection criterion is RDD's concrete BUILD-time demand, making the initial library shape RDD-representative rather than agnostically balanced.

---

### P3 — Consider

**Issue 3.1 — Inversion question 3 ("agentic- prefix / agentic-serving/ subdirectory convention") is declared "resolved at ADR-019 §Rejected alternative (c)" but ADR-019 §Rejected alternative (c) addresses per-framework subdirectories, not the convention's wrongness conditions.**

- **Recommendation:** Either relabel as "not examined (BUILD-time authoring decision)" or add a sentence acknowledging the specific inversion (e.g., what would make the `agentic-serving/` convention wrong at BUILD-time authoring).

**Issue 3.2 — ADR-020's claim that calibration fires on `web-searcher`'s output asserts the capability "participates in the full quality stack," but ADR-014's calibration mechanism includes AUQ confidence and HTC trajectory features designed for LLM-generated outputs, not script outputs.**

- **Recommendation:** Amend §Positive bullet to note Calibration Gate fires via post-hoc result-check only — AUQ and HTC are inapplicable to deterministic script execution.

**Issue 3.3 — ADR-021's provenance note attributes the fresh-context property's load-bearing characterization as agent-introduced framing (per Cycle 4 PLAY note 14 attribution flag), but treats it as settled in §Consequences §Positive without that qualification.**

- **Recommendation:** The §Consequences §Positive bullet on fresh-context should add a qualification noting the property itself is an architectural fact of `invoke_ensemble`, but its characterization as load-bearing for sycophancy-resistance patterns is an agent-introduced analogy from RDD's ADR-058 pending BUILD-phase testing.

---

## Section 2: Framing Audit

### Q1: What alternative framings did the evidence support?

**Framing A: "DuckDuckGo-as-default favors frictionlessness; Tavily-as-default favors quality."** The source material contains genuine ambiguity here. Cycle 4 PLAY note 1's on-ramp concern is about missing configuration, not search quality. The proposal §OD-4 lists DuckDuckGo as "brittle" but names it alongside other candidates without recommendation. An alternative framing: the on-ramp-clarity constraint (tension #11) favors the fewest setup steps — DuckDuckGo needs no API key. Tavily's free-tier signup, however small, introduces a mandatory step between "install" and "working."

**Framing B: "The no-dispatch fallback (note 19) is the primary operating mode at this library maturity."** Note 19 says "most tasks land here" with the existing library state. The 5-ensemble minimum-viable set covers 5 of 8 Topaz slots; evaluative and meta-tasks may always fall through. An alternative framing would characterize the Cycle 5 commitment as deliberate narrow scope — the quality infrastructure applies to a specific class of tasks; all other tasks proceed on the no-dispatch path. Under this framing, the new ADRs would scope their quality claims to the dispatched subset.

**Framing C: "Skill-framework-agnostic is provisionally committed pending seam-case examination."** The susceptibility snapshot noted the inversion's most concrete form is deferred. The ADRs frame the commitment as architectural settlement; an alternative framing would frame it as best-current-approximation subject to seam-case qualification.

### Q2: What truths were available but not featured?

**Available truth 1:** The "skill framework agnosticism" claim is broader than any existing evidence. Only RDD has been structurally verified (Cycle 4 PLAY note 14's architectural-isolation finding). The commitment "covering RDD and any other current or emerging skill framework" is grounded in one framework and extended by assertion to others.

**Available truth 2:** The no-dispatch fallback (note 19) was explicitly assigned to DECIDE for framing examination; none of the three new ADRs examine it. The discover gate committed the examination; DECIDE did not deliver it. This is a deliverable miss, not just a framing choice.

**Available truth 3:** OD-6 (methodology-skill / capability-ensemble naming registry) was assigned to DECIDE and was not resolved within the ADRs. The skill-framework-capability-registry artifact at `skill-framework-capability-registry.md` was authored but is not flagged in any ADR as the OD-6 resolution; the cycle-status would benefit from explicit notation.

### Q3: What would change if the dominant framing were inverted?

The dominant framing across all three ADRs is: skill-framework-agnostic + three-layer + operation-named is the correct shape. Inverting: skill-framework-agnostic is a premature generalization; the right shape at this maturity is methodology-aware orchestration serving RDD well first.

Under inversion: ADR-019's rejection of methodology-coarse library weakens (at one consumer, no duplication cost; duplication only becomes cost with multiple consumers). ADR-021's "skill framework owns full decomposition" becomes a liability — RDD's skill plugin maintains Topaz-aligned decomposition logic and stays synchronized with whatever the orchestrator's library contains. Field-notes note 3 (orchestrator routed meta-introspection to code-generation) becomes more salient — the orchestrator's ability to correctly route arbitrary skill-framework sub-tasks when they arrive as natural-language prompts is unvalidated.

### Framing Issues

**P1 — No DECIDE-phase framing examination of the no-dispatch fallback.**

The discover gate explicitly assigned this examination to DECIDE; the ADRs do not contain it. Recommendation: add a §Note to ADR-019 §Consequences §Negative characterizing the no-dispatch fallback as intended scope (orchestrator direct response is appropriate behavior when no library ensemble matches; quality infrastructure scope is dispatch-conditional by design), or produce a brief ADR-022 documenting the framing resolution. The current silence leaves the discover gate's explicit commitment unresolved.

**P2 — ADR-019 and ADR-021 adopt "capability ensemble" / "operation-named ensemble" vocabulary as if settled, while product-discovery marks both as candidates under DECIDE examination.**

The ADRs use the terms throughout §Decision without qualifier. The product-discovery vocabulary table marks them as candidates. This is the drift pattern the susceptibility snapshot Advisory 2 flagged. Recommendation: ADR-019 §Decision should open with an acknowledgment that these terms are candidates under examination through this ADR's concrete library-authoring decisions — and that the ADR's §Consequences constitute the DECIDE-phase test.

**P2 — ADR-020 names Brave/Exa/Serper adapters as supported but Cycle 5 BUILD scope is not clarified.**

The proposal listed these as default-backend candidates; ADR-020's §Decision implies a multi-adapter BUILD commitment. Recommendation: clarify whether the additional adapters are in Cycle 5 BUILD scope or deferred to operator-driven extension under the same shape principle.

**P3 — The "three-layer architecture" term is used as the organizing structure of both ADRs but is marked research-voice in product-discovery.**

Using a research-voice term as the organizational skeleton effectively promotes it to settled-as-structural-vocabulary regardless of the product-discovery vocabulary table's candidate designation. Recommendation: ADR-019 §Provenance should note this and flag the term for vocabulary-table update after BUILD.
