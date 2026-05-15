# Argument Audit Report — Cycle 6 DECIDE (Round 2 Re-audit)

**Audited documents:**
- `docs/agentic-serving/decisions/adr-022-routing-surface-behavior.md`
- `docs/agentic-serving/decisions/adr-023-observability-event-routing.md`
- `docs/agentic-serving/decisions/adr-024-common-io-envelope.md`
- `docs/agentic-serving/decisions/adr-025-artifact-as-substrate.md`

**Prior audit reference:** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-6-decide.md`

**Source material read:**
- `docs/agentic-serving/housekeeping/cycle-status.md`
- `docs/agentic-serving/decisions/adr-019-skill-framework-agnostic-capability-library.md`
- `docs/agentic-serving/decisions/adr-007-calibration-gate-for-composed-ensembles.md`
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` (referenced)

**Genre:** ADR set (re-audit of revised versions)

**Date:** 2026-05-15

**Cycle:** 6 (agentic-serving mini-cycle; post-DECIDE revisions)

---

## Audit Metadata

**Revisions applied before this re-audit:**
- P1-A: ADR-025 — calibration-gate evaluation surface subsection added
- P2-A: ADR-022 — system prompt wording clarified re qwen3:14b over-delegation
- P2-B: ADR-024 — BUILD-sequencing dependency subsection added
- P2-C: ADR-023 — final-dispatch-before-session-close subsection added
- P2-D: ADR-025 — dial-back falsification criteria enumerated (four indicators)
- P2-E: NOT addressed — flagged as framing-audit item
- P3-A: ADR-022 — session-level vs per-profile override distinction clarified
- P3-B: ADR-023 — `CalibrationSignal` orchestrator-context destination opt-in added
- P3-C: ADR-024 / ADR-025 — ADR-004 per-invocation escape hatch reconciliation added
- P3-D: ADR-024 — BUILD-assumption note on composition substrate added

---

## Section 1: Verification of Prior Findings

### P1-A — ADR-025: Calibration gate evaluation surface under substrate-routing

**Disposition: Addressed (clean)**

The revision adds a full subsection ("Calibration-gate evaluation surface under substrate-routing") that specifies three evaluation paths: (1) summary-only default via `artifacts[0].summary` + `primary`, applicable to most capability ensembles; (2) `structured` typed payload when `output_schema:` is declared, giving the gate structural anchors; (3) artifact-content opt-in via `calibration_substrate_access: artifact` for ensembles whose quality cannot be inferred from summary alone, with `code-generator` named explicitly as the lead case. The subsection is correctly identified as "load-bearing for the Calibration Gate's substrate-routing operation" and notes that ADR-007 and ADR-014 remain current within the inline-response path while the new evaluation surface governs substrate-routing. The three-tier graduated evaluation surface fully resolves the gap — the calibration gate's critics now have a specified operating surface for every substrate-routing case, and the cost profile of each tier is honestly disclosed.

No new issues introduced by this revision.

---

### P2-A — ADR-022: Amendment effect on qwen3:14b may be directionally adverse

**Disposition: Addressed (partial — residual gap remains)**

The revision amends the system prompt wording to include "Do not pick a client-declared tool merely because the request's verb matches the client tool's verb." This explicitly targets qwen3:14b's observed failure mode (over-delegation to `write_file` for code-generation requests). The §Consequences §Negative bullet acknowledges the distinction between MiniMax under-delegation (direct completion instead of ensemble) and qwen3:14b over-delegation (client-tool delegation instead of ensemble).

The revision is a genuine improvement. However, the Consequences §Negative and the §Effectiveness paragraphs still describe the qwen3:14b risk as "uncertain expected impact" without naming the specific directional risk the prior audit flagged: that the amendment's instruction ("prefer `invoke_ensemble`") targets the MiniMax failure mode (direct-completion-as-residual), while the added clause ("do not pick client tool by verb-match") targets the qwen3:14b failure mode — but these are **two structurally distinct clauses in one amendment**. The ADR does not surface that the BUILD/PLAY characterization target should test the two clauses independently, or that qwen3:14b's compliance with the second clause is the empirically open question. The "uncertain expected impact" framing remains, and without the two-clause distinction it understates the specificity of the remaining characterization gap.

**Remaining gap (P3 severity):** The BUILD/PLAY characterization target should be stated as testing both clauses independently: does qwen3:14b under the amended prompt still delegate to client tools for capability-matched requests, and does MiniMax under the amended prompt route to `invoke_ensemble` rather than direct completion? The current ADR frames both as one "effectiveness" question. Clarifying the two-clause structure of the amendment and matching two probe conditions to the two failure modes would make the BUILD/PLAY characterization target precise rather than aggregated. This is a minor framing precision issue — the core P2-A gap (directional adverse risk) is adequately addressed by naming the two failure modes.

---

### P2-B — ADR-024: BUILD-sequencing dependency between ADR-023 and ADR-024

**Disposition: Addressed (clean)**

The revision adds §"BUILD-sequencing dependency on ADR-023" specifying that ADR-023 ships first or alongside ADR-024, enumerating the two-step sequencing (ADR-023's event types first; ADR-024's envelope construction second), and describing the graceful fallback when ADR-024 ships first without ADR-023 (diagnostics fields absent, other fields populate normally). The fallback is characterized as "the diagnostics gap is the surface-level signal that ADR-023's events have not yet shipped" — this is a pragmatic and honest characterization. The planned sequencing is stated explicitly. No new issues introduced.

---

### P2-C — ADR-023: In-turn orchestrator-context routing at session-terminal boundary

**Disposition: Addressed (clean)**

The revision adds §"Final-dispatch-before-session-close handling" specifying: in-turn routing is skipped for the final dispatch (no next turn exists to receive the observation); the end-of-session summary in `execution.json` is the persistent record regardless of whether a next turn exists; the operator-terminal destination receives all events including the final dispatch's at emission time. The asymmetry (orchestrator-context destination is a turn-boundary mechanism; the final turn has no successor) is explicitly named and the resolution is correctly characterized as structurally honest. No new issues introduced.

---

### P2-D — ADR-025: Dial-back falsification criteria

**Disposition: Addressed (clean)**

The revision adds four concrete falsification indicators:
1. Artifact-substrate latency overhead exceeds 10% of dispatch wall-clock for deliverables under 1 KB.
2. Operator PLAY friction reports for one or more capability ensembles.
3. Session-directory disk-space requiring monthly-or-more operator intervention.
4. Three or more capability ensembles declare `output_substrate: inline` opt-outs during BUILD migration.

Each indicator is measurable and fire-on-evidence rather than fire-on-discomfort. The threshold of "three or more" opt-outs distinguishing exceptions from a pattern is appropriately reasoned (one or two is an exception; three is a trend). The 1 KB size-floor threshold is derived from spike α's per-ensemble survey and cited as such. The prior audit's concern — that "perpetual deferral" was the failure mode of an undefined falsification criterion — is squarely addressed. No new issues introduced.

---

### P2-E — ADR-019 portability claim not updated

**Disposition: Not addressed — gap noted for the gate**

Per the revision brief, P2-E was not addressed and was surfaced as a framing-audit item for the practitioner. The ADR-019 §Consequences §Positive's profile-portability claim remains unqualified by spike γ's finding that routing surface behavior is model-conditional under the current system prompt. ADR-022's backward propagation updates ADR-021's NL clause but does not carry an update to ADR-019.

**Assessment of whether this creates a logical gap requiring closure before BUILD:** The gap is real but the risk level depends on what BUILD does with ADR-019's portability claim. If BUILD implements the amended system prompt (ADR-022) and treats the amendment as sufficient, the portability claim in ADR-019 becomes a stale documentation artifact. BUILD teams reading ADR-019 §Consequences §Positive to understand the orchestrator's portability guarantee will see an unqualified claim ("operators can swap orchestrator profiles within a portable routing contract") without the spike γ qualification ("routing surface behavior is model-conditional under the current prompt; the amendment mitigates but BUILD/PLAY characterizes"). This creates a documentation-level inconsistency that could mislead BUILD if ADR-019 is read as authoritative on portability.

**Recommendation for the gate:** Add ADR-019 to the backward propagation sweep before BUILD entry. The qualification is narrow — a one-sentence note on §Consequences §Positive that the profile-portability claim is at the config-layer (model profiles are interchangeable); routing-surface portability is model-conditional per spike γ and characterization is deferred to BUILD/PLAY. This is the same note ADR-022 already carries in its own §Consequences §Negative; the propagation to ADR-019 closes the documentation inconsistency.

**Severity retained at P2** — should address before BUILD, but does not block DECIDE close.

---

### P3-A — ADR-022: Session-level vs per-profile override distinction

**Disposition: Addressed (clean)**

The revision clarifies in §Consequences §Neutral and §Out of scope that the session-level override (`agentic_serving.orchestrator.system_prompt` in `config.yaml`) is an existing mechanism (per ADR-011), while per-orchestrator-profile system-prompt defaults (a new mechanism where each profile carries its own default) are follow-on territory. The distinction is now legible — operators reading the ADR can distinguish what is available now from what is flagged as future-cycle territory. No new issues introduced.

---

### P3-B — ADR-023: `CalibrationSignal` routing to orchestrator-context destination

**Disposition: Addressed (clean)**

The revision specifies that `CalibrationSignal` events are excluded from the orchestrator-context destination by default, with an opt-in config flag (`agentic_serving.observability.orchestrator_context_routes_calibration_signal: true`). The rationale (high-volume cross-layer telemetry per ADR-016; value is operator-tooling and post-hoc analysis, not in-turn orchestrator reasoning; the default `false` preserves context budget) is stated. The omission from the prior §Routing destinations discussion is now covered by the exclusion note in the Destination 1 section. No new issues introduced.

---

### P3-C — ADR-024 / ADR-025: ADR-004 per-invocation escape hatch reconciliation

**Disposition: Addressed (clean)**

ADR-025 now carries a reconciliation paragraph ("Relationship to ADR-004's existing per-invocation escape hatch") that correctly distinguishes ADR-025's `output_substrate: artifact` (a dispatch-shape commitment; where the deliverable lives) from ADR-004's escape hatch (a per-invocation skip-summarization decision within the inline-response path). The reconciliation specifies how the two compose: substrate-routed ensembles skip content summarization at the substrate layer; inline-response ensembles retain ADR-004's mandate; within the inline-response path, ADR-004's escape hatch still operates for small-output ensembles. The prior audit's concern about the two escape-hatch formulations being unreconciled is fully addressed. No new issues introduced.

---

### P3-D — ADR-024: Spike β composition-assumptions finding documented for BUILD

**Disposition: Addressed (clean)**

ADR-024 now carries §"BUILD-assumption note on composition substrate" that explicitly states spike β's headline finding (composition assumptions live in the orchestrator's reasoning surface between dispatches, not in the typed contract), explains that `output_schema:` opens structural composition predictability without displacing the orchestrator's prose-integrator role, and names the orchestrator's narrowing (to chain-selector) as a bigger architectural commitment deferred to future cycles. The risk the prior audit flagged — BUILD over-relying on `output_schema:` as composition infrastructure — is now surfaced in the ADR itself. No new issues introduced.

---

## Section 2: New Findings

### N-1 — ADR-025: Calibration gate evaluation surface introduces an unresolved question about the `structured` field's schema-validity evaluation surface

**Severity: P3 (Consider)**

The new calibration-gate evaluation subsection specifies that when `output_schema:` is declared, the gate's evaluators consume `envelope.structured` for "schema validity" and "structural anchors." But schema validation is explicitly characterized as **advisory** in ADR-024 §Decision — the synthesizer is not the drift source (spike β's reframing); schema compliance is not enforced. If schema validation is advisory at dispatch time, the calibration gate's use of `envelope.structured` for "schema validity" grounds the gate's quality verdict on a field that may not be schema-compliant. A non-compliant `structured` payload that slips through advisory validation would produce a calibration gate that either (a) accepts a structurally drift-deviant output as "Proceed" or (b) rejects a structurally deviant output as "Reflect/Abstain" — but the gate's evaluators have no reliable mechanism to distinguish schema-validity from structural correctness when the schema is advisory-only.

This is a nuance, not a blocker. The gate's evaluators are LLM agents reading the structured payload and producing a quality verdict; they can assess structural correctness without relying on schema-compliance machinery. The concern is that the subsection implies schema validation as one of the gate's evaluation surfaces when in practice the gate evaluates the payload's content, not its schema-compliance. The language "claim counts, label distributions, schema validity" overstates the gate's evaluative precision for advisory schemas.

- **Location:** ADR-025 §"Calibration-gate evaluation surface" — "`structured` (typed payload) when declared"
- **Observation:** The phrase "schema validity" is used as an evaluation anchor for the calibration gate's structured-payload consumption. Schema validation is advisory at dispatch time per ADR-024; the gate's evaluators cannot reliably distinguish advisory-schema-compliant from advisory-schema-deviant without enforcement machinery.
- **Recommendation:** Rephrase "schema validity" to "structured payload content" or "typed fields per `output_schema:`." The gate's evaluators assess whether the structured payload's content is high-quality, not whether it formally validates against a JSON Schema — the schema-advisory framing applies at dispatch time, not at evaluation time. The distinction is minor; the correction prevents a future reader from inferring that the gate runs schema-validation machinery.

---

### N-2 — ADR-023: The operator-terminal `CalibrationSignal` routing at DEBUG level and the orchestrator-context opt-in configuration flag are specified in different locations, creating a potential operator-confusion surface

**Severity: P3 (Consider)**

The `CalibrationSignal` routing behavior is now specified across two locations in ADR-023: (1) the operator-terminal destination specifies `CalibrationSignal` routes at DEBUG level (suppressed unless `--verbose` or `LOG_LEVEL=DEBUG`); (2) the exclusion from orchestrator-context destination (default off, opt-in via config flag) is specified inline in the Destination 1 section rather than in the Destination 2 section. An operator reading Destination 2 to understand what routes to orchestrator-context would not encounter the `CalibrationSignal` exclusion note — it is anchored to Destination 1's discussion. This is a minor placement issue: the exclusion note belongs in Destination 2 (or in both sections), not only in Destination 1.

- **Location:** ADR-023 §Decision §Routing destinations
- **Observation:** `CalibrationSignal` exclusion from orchestrator-context is specified in Destination 1's section; an operator consulting Destination 2 to understand its event scope would miss the exclusion.
- **Recommendation:** Add a one-sentence note in Destination 2's section: "`CalibrationSignal` events are excluded from this destination by default; see Destination 1's `CalibrationSignal` note for the opt-in mechanism."

---

## Section 3: Argument Audit (Revised ADRs)

### Per-ADR Assessment

#### ADR-022 (revised)

The decision chain is intact and the revisions do not weaken it. The system prompt amendment now carries two distinct clauses — the `prefer invoke_ensemble` clause targeting MiniMax's under-delegation and the `do not pick client tool by verb-match` clause targeting qwen3:14b's over-delegation — which are correctly grounded in the respective spike γ cell findings. The §Effectiveness section accurately represents cross-profile uncertainty. The §Consequences §Neutral distinction between session-level and per-profile override surfaces is now legible.

One minor coherence observation (carried forward as P3 from P2-A's residual): the amendment's two-clause structure is not explicitly mapped to the two-probe characterization work BUILD/PLAY should perform. This does not affect the ADR's argument chain; it is a BUILD-readiness precision gap.

#### ADR-023 (revised)

The addition of §"Final-dispatch-before-session-close handling" completes the routing semantics for the orchestrator-context destination's turn-boundary mechanism. The end-of-session summary in `execution.json` is correctly characterized as the persistent record across the boundary. The behavior is logically sound: skipping in-turn routing for the final dispatch is structurally honest; the operator-terminal destination receiving all events including the final dispatch's is consistent with that destination not depending on turn existence.

The `CalibrationSignal` opt-in placement issue (N-2) is minor and does not affect logical coherence.

#### ADR-024 (revised)

The BUILD-sequencing dependency subsection closes the BUILD-blocking dependency gap correctly. The BUILD-assumption note on composition substrate accurately represents spike β's headline finding and explicitly scopes what `output_schema:` does and does not deliver. The ADR's argument chain is stronger than before the revisions.

#### ADR-025 (revised)

The calibration-gate evaluation surface subsection is the most substantive addition and it is well-reasoned. The three-tier graduated evaluation surface (summary-only default; `structured` when declared; artifact-content opt-in) correctly maps to the cost and capability hierarchy the BUILD migration needs. The schema-validity language issue (N-1) is a minor precision gap within an otherwise sound subsection.

The dial-back falsification criteria are well-enumerated and make the always-scope commitment testable. The ADR-004 escape-hatch reconciliation closes the prior inconsistency cleanly.

### Cross-ADR Composition

No new cross-ADR composition issues emerge from the revisions. The three integration seams identified in the prior audit (ADR-022/023; ADR-023/024; ADR-024/025) remain sound. The seam between ADR-025 and ADR-007/ADR-014 (the P1-A seam) is now specified in ADR-025; ADR-007 and ADR-014 themselves carry the backward propagation sweep's supersession notes per the Step 3.7 sweep (referenced in ADR-025's backward propagation section). The composition reads as complete.

---

## Section 4: Framing Audit (Revised ADRs)

### Question 1: What alternative framings did the evidence support?

The three alternative framings from the prior audit remain relevant as framing observations; the revisions do not foreclose them or provide new counter-evidence. However, none of them represents a framing gap the revisions introduced or that materially weakens the ADRs as revised.

**Alternative framing 1 (ADR-022):** Narrow ADR-021 rather than amend the system prompt. ADR-022's rejection of this framing (§Rejected alternatives "Revise the Skill Orchestration User stakeholder mental model") remains the load-bearing case. The revision does not change this framing's availability or the rejection's adequacy.

**Alternative framing 2 (ADR-023):** Defer orchestrator-context routing to Cycle 7. The revision's final-dispatch-handling subsection does not introduce new evidence against this framing; it closes a specification gap within the chosen framing. The rejection in §Rejected alternatives remains adequate.

**Alternative framing 3 (ADR-025):** Except `web-searcher` at DECIDE rather than deferring to BUILD opt-out. The dial-back falsification criteria revision is the direct response to this framing. The criteria make the always-scope testable, which partially addresses the concern — if `web-searcher` triggers Indicator 1 (latency overhead >10% on under-1KB deliverables) or Indicator 4 (opt-out), the scope-refinement deliberation fires. The always-scope's commitment is now bounded by observable triggers rather than perpetually deferrable by discomfort.

### Question 2: What truths were available but not featured?

**Finding A (Spike β's orchestrator-as-prose-integrator open question):** The BUILD-assumption note in ADR-024 now explicitly surfaces this as context for BUILD operators. The truth is no longer absent — it is documented as the load-bearing context for `output_schema:` adoption. This prior framing issue is resolved.

**Finding B (ADR-019 portability claim not updated):** Remains un-addressed (per P2-E above). The portability qualification that spike γ made available is still absent from ADR-019. This is the one residual truth-available-but-not-featured finding that the revisions did not close.

**Finding C (`web-searcher` JSON-string-in-response convention):** ADR-024 §Decision §`output_schema:` per-ensemble declaration names `web-searcher` as an early `output_schema:` migration candidate ("their outputs have natural structure — paragraph text; list of URL+snippet records"). The implication — that `web-searcher`'s JSON-string-in-`response` convention would move to `structured` — is present by strong implication but not explicitly stated. This remains a minor framing gap but not a logical one; a BUILD implementer reading ADR-024 would understand the migration path for `web-searcher`.

### Question 3: What would change if the dominant framing were inverted?

**ADR-022's dominant framing (system prompt amendment as the operative design surface):** The inversion — narrow ADR-021's NL clause rather than amend the prompt — is fully engaged by the ADR's §Rejected alternatives. The revisions do not weaken this engagement.

**ADR-025's dominant framing (always-scope with dial-back disposition):** The inversion — substantive-deliverable scope with explicit per-ensemble criteria — is now less available as a critique because the dial-back falsification criteria make the always-scope testable. The inverted framing was most powerful when "dial back later if cumbersome" was undefined; it is less powerful when four concrete indicators specify when "later" arrives. The revisions strengthen the dominant framing's defensibility against this inversion.

### Framing Issues

**P2-E (carried forward — unaddressed):**

ADR-019 §Consequences §Positive's profile-portability claim remains unqualified by spike γ's finding. Spike γ Cell B continuation found that routing surface behavior is model-conditional under the current system prompt. ADR-022's backward propagation updates ADR-021 and ADR-004 but does not carry a qualification to ADR-019. BUILD teams reading ADR-019 to understand the portability commitment would see an unqualified claim that spike γ's data has materially weakened.

- **Location:** ADR-019 §Consequences §Positive
- **Available truth:** Spike γ Cell B's finding that qwen3:14b over-delegates while MiniMax under-delegates under identical prompts, weakening the "routing contract is portable across profiles" reading of ADR-019's portability claim.
- **Recommendation (as previously stated):** Add ADR-019 to the Step 3.7 backward propagation sweep with a one-sentence qualification on §Consequences §Positive. Severity: P2 — should address before BUILD, not a BUILD blocker.

---

## Summary

### Verification Dispositions

| Prior Finding | Severity | Disposition | Notes |
|---------------|----------|-------------|-------|
| P1-A — ADR-025 calibration gate evaluation surface | P1 | Addressed (clean) | Three-tier evaluation surface fully specified |
| P2-A — ADR-022 qwen3:14b directional risk | P2 | Addressed (partial) | Two failure modes named; two-clause test target not mapped to two probe conditions |
| P2-B — ADR-024 BUILD-sequencing dependency | P2 | Addressed (clean) | Sequencing and graceful fallback specified |
| P2-C — ADR-023 session-terminal in-turn routing | P2 | Addressed (clean) | Final-dispatch handling subsection complete |
| P2-D — ADR-025 dial-back falsification criteria | P2 | Addressed (clean) | Four concrete measurable indicators |
| P2-E — ADR-019 portability claim not updated | P2 | Not addressed | Retained for gate; documentation inconsistency for BUILD |
| P3-A — ADR-022 session vs profile override | P3 | Addressed (clean) | Distinction now legible |
| P3-B — ADR-023 CalibrationSignal orchestrator-context | P3 | Addressed (clean) | Exclusion default + opt-in config flag specified |
| P3-C — ADR-024/025 ADR-004 escape hatch | P3 | Addressed (clean) | Reconciliation paragraph complete |
| P3-D — ADR-024 spike β composition assumption | P3 | Addressed (clean) | BUILD-assumption note added |

### New Findings

| Finding | Severity | ADR | Description |
|---------|----------|-----|-------------|
| N-1 | P3 | ADR-025 | "Schema validity" language overstates gate's evaluative precision for advisory schemas |
| N-2 | P3 | ADR-023 | CalibrationSignal orchestrator-context exclusion note placed in Destination 1 section; not visible from Destination 2 |

### Issue Counts (Round 2)

| Priority | Count | Findings |
|----------|-------|---------|
| **P1** | 0 | None — the P1-A finding from Round 1 is resolved |
| **P2** | 1 | P2-E (ADR-019 not in backward propagation sweep) carried forward as unaddressed |
| **P3** | 3 | Residual from P2-A (two-clause probe precision); N-1 (schema validity language); N-2 (CalibrationSignal note placement) |

### Recommendations

**Must address before accepting ADRs:** None. The Round 1 P1 finding is resolved; the ADR set is structurally sound.

**Should address before BUILD:**
- **P2-E:** Add ADR-019 to the Step 3.7 backward propagation sweep with a portability-claim qualification noting that routing surface behavior is model-conditional per spike γ. This closes the one documentation inconsistency that BUILD teams would encounter when reading ADR-019 alongside ADR-022.

**Consider for refinement:**
- **P3 (P2-A residual):** Map the amendment's two clauses to two explicit BUILD/PLAY probe conditions — one per failure-mode direction.
- **N-1:** Replace "schema validity" with "structured payload content" in ADR-025's calibration-gate subsection to avoid implying schema-enforcement machinery that ADR-024 explicitly characterizes as advisory.
- **N-2:** Add a one-sentence cross-reference in ADR-023's Destination 2 section noting the `CalibrationSignal` exclusion specified in Destination 1.

The four ADRs as revised are well-grounded, internally consistent, and ready for the downstream DECIDE gate with only the P2-E documentation gap remaining. That gap is a backward-propagation omission, not a logical flaw in any of the four ADRs themselves.
