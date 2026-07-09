# ADR-013: Session Registry Initializer-then-Resume Schema with Write-Gate Validation

**Status:** Proposed

**Date:** 2026-05-05

---

## Context

The domain concept *Session* (domain-model.md §Concepts) defines a stateful conversation between a client and the orchestrator agent, bounded by budget constraints. The Session Registry (L3) currently identifies and continues a multi-request Session (per the existing system design). Long-horizon coding agents converge on a stronger pattern: bound each session with **externalized structured state** (the methodology-voice term per domain-model.md §Methodology Vocabulary) so context survival across session boundaries is structural rather than prompt-mediated.

Cycle 4's Wave 2.B literature review (essay 005, §"Long-Horizon Reliability Infrastructure") found three converged-on artifacts in the Anthropic initializer schema: `feature_list.json` with a monotonic `passes` field (structural non-regression at the schema level, not prompt compliance); `claude-progress.txt` as an append-only narrative log; `init.sh` as a deterministic environment bootstrap. Confucius Code Agent (arXiv:2512.10398) achieves 59 percent Resolve@1 on SWE-Bench-Pro with persistent cross-session note-taking as the stated mechanism — the most prominent published evidence that this pattern is load-bearing for long-horizon coding agents. Cognition Labs explicitly states Devin does not maintain cross-session memory (mid-2025); the contrast signals that externalized state is best-practice rather than universal-deployment, but the cycle's North-Star benchmark (driving a full RDD cycle) is exactly the long-horizon shape Confucius's pattern serves.

OpenDev's Planner subagent demonstrates schema-level enforcement as a class (a)/(c) hybrid in the intervention-class taxonomy: write tools are simply absent from the planner's tool surface rather than instructed-against. This avoids the alignment tax at small model sizes (the 12-point accuracy drop from constrained decoding at 8B). The pattern transfers to the Session Registry's externalized-state surface: structural enforcement of artifact integrity rather than prompt-mediated compliance.

Mnemonic sovereignty (arXiv:2604.16548) flags a known failure mode of append-only persistence: memory-poisoning attack surface. Append-only JSONL supports auditability and session operations (resume, fork, rewind) but creates rollback limitations and a poisoning vector. **Write-gate validation is recommended in the literature but not operationalized in any reviewed system.** This is novel design work within an otherwise adoption-decision ADR (research-gate carry-forward #2) — the components asymmetry must be made visible.

The cycle's RDD-phase decomposition (essay 005, §"RDD-Cycle Decomposition") distinguishes three phase clusters by dominant decision-class. Cluster 2 (BUILD, ARCHITECT, plus portions of DEBUG and REFACTOR) exhibits continuous routing under variable inputs and is where externalized-state's value is highest. Cluster 1 (RESEARCH, DECIDE, SYNTHESIZE) and Cluster 3 (DISCOVER, PLAY, conversational MODEL) have different decision-class profiles where mandatory artifact maintenance is overhead without proportionate value.

The framing commitment from research-gate Grounding Action 2 (recorded 2026-05-05, *elaboration-by-evidence*) holds: the artifact set's responsibilities concentrate in Session Registry (L3) within ADR-002's four-layer frame, not as a cross-cutting module orthogonal to the layering.

---

## Decision

The Session Registry (L3) takes on the additive responsibility of **structured-handoff artifacts** at session boundaries. The artifact set has three adoption-derived components plus one novel-design component:

### Adoption-derived components (from Anthropic's published schema)

1. **feature-list-with-monotonic-passes** — a JSON schema cataloging features with a monotonic boolean `passes` field. Schema-level non-regression: a feature once marked `passes: true` cannot transition to `passes: false` without an explicit operator override that is audit-logged. The schema encodes structural non-regression; agent prompt compliance is not the enforcement mechanism. The schema's full structure (entry shape, identifier conventions, monotonicity rule) follows Anthropic's specification.

2. **append-only progress log** — a free-text log file capturing narrative session progress. Append-only at the filesystem level: write operations are constrained to append, not arbitrary edit. The file format follows Anthropic's `claude-progress.txt` specification (free-text with lightweight section markers).

3. **init-sh-style deterministic environment bootstrap** — an executable script that establishes the session's working environment (paths, tool availability, configuration). Deterministic in the sense that the script's output is purely a function of repository state plus operator configuration, with no LLM-mediated steps. The script is operator-authored and version-controlled with the project; the Session Registry invokes it at session-start and verifies completion before yielding to the orchestrator.

### Novel-design component (no reference implementation; per carry-forward #2 asymmetry)

4. **Write-gate validation surface** — a structural validation layer between the orchestrator's write requests and the persistent artifacts. The write-gate runs three validation classes:

   - **(i) JSON schema validation** for feature-list writes. Writes that violate the schema (including the monotonicity constraint on `passes`) are rejected; the rejection produces a typed error.
   - **(ii) Append-only constraint enforcement** for progress-log writes. Writes that attempt non-append operations (overwrite, truncate, mid-file edit) are rejected; the rejection produces a typed error.
   - **(iii) Signed-script integrity verification (tamper-detection scope)** for init.sh execution. The init.sh content is hashed at operator-authoring time and the hash is recorded in the Session Registry's configuration; execution is gated on hash match. Script modifications require explicit operator re-authoring (rotating the recorded hash). **Scope note (per argument-audit P3.2 finding, 2026-05-06):** the integrity check is *tamper-detection* — it detects modification of init.sh content between operator-authoring time and session-execution time. It does not validate that the operator-authored init.sh is itself safe; the operator's authoring step is the trust boundary. The verification's load-bearing property is that third-party modification (or accidental corruption) of the script between authoring and execution is detected; the operator's own malice or error in authoring is out of scope for this validation surface.

   Write-gate violations produce typed errors consistent with the ADR-017 typed-error pattern and do not corrupt artifact state. The validation layer is structurally enforced (the orchestrator cannot bypass it through prompt compliance) — this is the load-bearing property the write-gate provides.

### Cluster-conditional applicability

The artifact set is **required** for sessions in Cluster 2 phase contexts (long-horizon continuous routing — the BUILD/ARCHITECT/DEBUG/REFACTOR territory). The artifact set is **supported but optional** for Cluster 1 (specialist-dispatch) and Cluster 3 (conversational/exploratory) contexts. Cluster determination is a session-start decision based on the operator's session-shape declaration; default behavior is required (operators opt-out for Cluster 1 / Cluster 3 contexts rather than opt-in for Cluster 2).

**Cross-cluster sessions (argument-audit P2.2 finding, 2026-05-06).** The cycle's North-Star benchmark (driving a full RDD cycle) straddles multiple clusters within a single logical engagement — RESEARCH and DECIDE are Cluster 1, BUILD and ARCHITECT are Cluster 2, DISCOVER and PLAY are Cluster 3. A session that crosses cluster boundaries cannot be reliably pre-classified at session-start. Three dispositions for cross-cluster sessions:

- **(i) Default to required.** When session-shape declaration is ambiguous or names multiple clusters, default to required-artifact-set behavior. The cost is friction in mixed-cluster sessions where the artifact set may be overhead for sub-portions; the benefit is the structural-non-regression and narrative-continuity properties cover the load-bearing Cluster 2 sub-portions of the session.
- **(ii) Mid-session reclassification.** The Session Registry permits in-session cluster transitions through an explicit declaration mechanism. Transitioning into Cluster 2 mid-session activates the artifact set; transitioning out of Cluster 2 deactivates it (existing artifacts persist; new writes pause). The mechanism preserves operator control at the cost of a session-level state machine.
- **(iii) Always required.** The artifact set runs unconditionally regardless of cluster declaration. Cluster determination becomes documentation rather than behavior. The cost is overhead in genuinely Cluster 1 / Cluster 3 sessions; the benefit is mechanism simplicity and no false-negative misclassifications.

ADR-013 specifies disposition **(i) default to required for cross-cluster ambiguity** as the BUILD-time starting point. Disposition (ii) is available as a refinement if BUILD evidence shows reclassification is operationally useful; disposition (iii) is the fall-back if the cluster-determination surface proves unworkable in practice.

---

## Rejected alternatives

**(a) Adopt only the feature-list, omit progress-log and init.sh.** Rejected: the three-artifact set's value is compositional — feature-list provides structural non-regression; progress-log provides narrative continuity; init.sh provides deterministic environment bootstrap. Each artifact addresses a distinct failure mode (lost progress, lost context, environment drift). Adopting only one leaves two failure modes uncovered. The published evidence base (Confucius Code Agent's 59 percent SWE-Bench-Pro result) is for the composite pattern, not for any single artifact.

**(b) Adopt the three-artifact set without write-gate validation.** Rejected: Mnemonic sovereignty (arXiv:2604.16548) flags memory-poisoning attack surface as a known failure mode of append-only persistence. Without write-gate validation, the schema-level non-regression guarantee can be bypassed by direct artifact manipulation — at which point the structural enforcement reduces to prompt compliance and the pattern's class (a)/(c) intervention character collapses to class (b) prompt-suggestion. Write-gate validation is novel design work, but it is what makes the schema's structural integrity load-bearing rather than decorative.

**(c) LLM-mediated init.sh generation.** Rejected: the session-bootstrap step's value is its determinism. Inserting an LLM step into init.sh's generation reintroduces the prompt-compliance failure mode the schema-level approach was designed to replace. Operator-authored init.sh is the version of the pattern with the determinism property; LLM-generated init.sh is a different pattern that loses the load-bearing property.

**(d) Apply the artifact set to all session types unconditionally.** Rejected: Cluster 3 (conversational/exploratory) sessions do not have the structural-non-regression failure mode the artifact set addresses; the artifact set's overhead is not proportionate to its value in that context. Mandating the artifact set in those contexts is overhead without value. Cluster determination at session-start is the discrimination boundary that lets the pattern serve where it is load-bearing without imposing where it is not.

**(e) Adopt write-gate validation as a class (b) prompt-compliance instrumentation rather than structural enforcement.** Rejected: prompt compliance is the failure mode the entire schema-level approach was designed to replace (CAAF, arXiv:2604.17025: "apparent LLM reliability in safety-critical domains is often a prompt engineering artifact"). A class (b) write-gate is theatre. The cost of structural enforcement is real (novel design work), but the alternative is no enforcement.

---

## Consequences

**Positive:**
- Schema-level non-regression in feature-list addresses long-horizon judgment-decay at the structural level rather than the prompt level (CAAF mechanism)
- Append-only progress-log provides narrative continuity without requiring full-context reload across session boundaries (Confucius Code Agent's mechanism)
- Deterministic init.sh eliminates the environment-drift failure mode at session resumption
- Write-gate validation closes the memory-poisoning attack surface flagged in Mnemonic sovereignty
- Session Registry's existing responsibility (identifies and continues a multi-request Session) extends compositionally; the artifact set augments rather than replaces existing functionality
- Cluster-conditional applicability respects each phase cluster's decision-class profile rather than imposing uniform overhead

**Negative:**
- Cluster determination at session-start is an additional decision the operator (or session-start protocol) must make; default-required behavior reduces but does not eliminate the decision burden
- Write-gate validation is novel design work without literature reference implementation; the validation surface specification will require operational tuning during BUILD and may need amendment after first-deployment evidence
- Three-artifact persistence introduces filesystem state that operators must manage across sessions (filesystem hygiene is operator concern, not orchestrator concern)
- Append-only progress-log can grow unboundedly within a session series; archival policy is a deployment concern this ADR does not specify
- The signed-script integrity verification (write-gate class iii) requires operator workflow for hash rotation when init.sh changes; documentation and tooling support is a downstream concern
- Cluster 2 default-required behavior may friction-cost early adoption when operators are not yet familiar with the artifact set; opt-out is available but not the path of least resistance

**Neutral:**
- The schema choice (JSON for feature-list, free-text for progress-log, shell for init.sh) follows Anthropic's specification and is deployment-portable
- Write-gate validation interacts with ADR-017's tool-call structural validation guard — both produce typed errors but address different failure surfaces (artifact integrity vs. tool-call structure)
- The artifact set is filesystem-resident; Plexus integration (when active) ingests the artifacts as source material per AS-4 (Ingestion boundary is source material). The artifacts themselves are not written through Plexus — they are filesystem artifacts that Plexus may ingest

---

## Provenance check

- **Driver-derived content (adoption components 1–3).** The three-artifact schema (feature-list, progress-log, init.sh), the monotonic-passes constraint, the append-only-log convention, and the deterministic-bootstrap pattern are direct adoption from Anthropic's published initializer schema, surfaced via essay 005 §"ADR candidate #2" and §"Long-Horizon Reliability Infrastructure." The driver chain runs essay 005 → Anthropic engineering source. Confucius Code Agent (arXiv:2512.10398) is the empirical driver establishing the pattern's load-bearing character on long-horizon tasks.

- **Driver-derived content (cluster taxonomy).** The three-cluster phase decomposition is taken from essay 005 §"RDD-Cycle Decomposition and Mechanism Attribution." The taxonomy is essay-derived, not drafting-time synthesis.

- **Drafting-time synthesis (cluster-conditional applicability rule).** Essay 005 specifies the artifact schema but does not specify the cluster-conditional applicability rule (required for Cluster 2; supported for Cluster 1, 3). The conditional-applicability mapping is drafting-time application of the cluster taxonomy to ADR-013's scope. The choice of "Cluster 2 default-required, Cluster 1 / 3 default-optional" is drafting-time judgment about decision-class fit; the alternative ("optional everywhere; operator-driven") was rejected as not honoring the published evidence base, but the placement at "default-required for Cluster 2" specifically is drafting-time synthesis.

- **Drafting-time synthesis (write-gate validation specification — load-bearing per carry-forward #2 asymmetry).** Per essay 005's explicit caveat, write-gate validation is novel design work without any reference implementation. The four validation specifications above (JSON schema validation, append-only constraint enforcement, signed-script integrity, typed-error coupling) are drafting-time choices. Each specification is drafting-time synthesis applying converged literature mechanisms (CAAF's class (a) structural-override pattern; OpenDev's class (a)/(c) schema-level enforcement; ADR-017's typed-error pattern from codebase commit `9f86d0b`) to the write-gate's validation surface. The validation classes are not from a published reference implementation; they are drafting-time composition.

- **Drafting-time synthesis (signed-script integrity hash-rotation workflow).** The hash-rotation workflow for init.sh modifications is drafting-time addition. Essay 005 specifies the deterministic-bootstrap principle but does not specify the operator workflow for legitimate script modifications. The hash-rotation pattern is drafting-time synthesis ensuring the integrity check is workable in practice rather than purely theoretical.

- **Asymmetric treatment per research-gate carry-forward #2.** This ADR honors the asymmetry by separating Decision content into adoption-derived (items 1, 2, 3) and novel-design (item 4); the Provenance check above makes the asymmetry visible to argument-audit. Argument-audit on this ADR should concentrate on item 4 (write-gate validation) where the load-bearing question is whether the novel specification can be operationalized without a reference implementation; items 1–3 are adoption-decision discipline (verifying applicability and documenting deviations).

- **Vocabulary impact.** ADR-013 introduces three terms that will be added to the domain model on Tranche A close:
  - **Structured-handoff artifact** — proposed new term in §Concepts (operator voice)
  - **Write-gate validation** — proposed new term in §Concepts (operator voice; or §Methodology Vocabulary if drafting-time review determines it is research voice)
  - **Cluster determination at session-start** — proposed new term in §Concepts (operator voice; the cluster taxonomy itself remains methodology vocabulary)

  The "structured-handoff artifact" term sits alongside the existing methodology-voice term "externalized structured state" — the operator-voice form names what is concretely persisted (artifact); the methodology-voice form names the category (externalized state). Both belong in the domain model; the placement (Concepts vs. Methodology Vocabulary) is the open editorial question.
