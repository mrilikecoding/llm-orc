# Susceptibility Snapshot

**Phase evaluated:** BUILD — Cycle 6 WP-E (ADR-022 system-prompt amendment + ADR-024 DispatchEnvelope completion + ADR-025 always-scope substrate routing + AS-7 amended; session 1 + session 2 combined; also serves as the BUILD-phase epistemic close for Cycle 6's Mode D mini-cycle)
**Artifact produced:** 8 commits (302bb5d through c4c3698); primary artifacts: `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` amendment, `EnsembleConfig` schema extensions, `SessionArtifactStore` module, Session Registry `register_close_callback` + `close_session` lifecycle hook, `_validate_invoke_arguments` refactor commit, substrate-routing branch in `OrchestratorToolDispatch` (`SubstrateRoutingConfig` + `EnsembleSubstrateReader` Protocol + five gating helpers + `_route_dispatch_to_substrate` + `_shape_calibration_evaluation_input`), 6 capability YAML migrations, serve-layer wiring (`EnsembleConfigSubstrateReader`, `EnsembleConfigOutputSchemaReader`, `get_session_artifact_store` factory, `get_orchestrator_tool_dispatch` extensions), 4 integration tests (`test_fc22_fc26_substrate_routed_envelope.py`); 2849 tests passing at WP-E close (+37 from WP-D's 2812)
**Date:** 2026-05-16

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| Cycle 6 Model | Grounding Reframe recommended (3 actions) | Framing adoption at constitutional level; 3 carry-forwards |
| Cycle 6 Decide | No Grounding Reframe; 1 pre-BUILD action (P2-E); 3 advisory carry-forwards | Earned confidence; `dispatch_id` coupled failure surface named for ARCHITECT attention |
| Cycle 6 Architect | No Grounding Reframe; 6 advisory feed-forwards; 3 closed inline at gate | Earned confidence; two-module decomposition inherited without explicit re-examination |
| Cycle 6 Build WP-B | No Grounding Reframe; 4 advisory feed-forwards | Decision 3 (propagation fixture vs. upstream fix) most significant |
| Cycle 6 Build WP-C | No Grounding Reframe; 6 advisory feed-forwards | Decision 2 (standalone `dispatch_log.json` vs. `execution.json` key) moderate |
| Cycle 6 Build WP-D | No Grounding Reframe; 5 advisory feed-forwards (2 moderate) | Commit-discipline violation (Decision 4); YAML task rewrite fork not surfaced (Decision 5); vacuous-pass accumulation pattern now two instances |

---

## Grounding Reframe Action Outcomes (WP-D Advisories entering WP-E)

**WP-D Moderate Advisory 1 — YAML task rewrite fork must be surfaced at WP-E entry.**
The WP-D snapshot recorded the fork as un-surfaced and recommended explicit practitioner choice at WP-D close or WP-E entry. Outcome: the cycle-status entry at "From Cycle 6 BUILD WP-D" explicitly names both options — (a) rewrite `default_task` to request JSON output, (b) keep schemas as documentary only — and flags it as a WP-E entry decision. Piece 7's commit message (f6fa4a3) names "WP-D snapshot moderate advisory 1 resolution per Cycle 6 WP-E entry — option (b)" explicitly. The fork was surfaced at the advisory boundary; the resolution is committed with its rationale attributed. **Status: addressed. Option (b) selected with documented rationale (spike β reframing: drift mechanism is orchestrator `input.data` override, not synthesizer non-compliance).**

**WP-D Moderate Advisory 2 — CLAUDE.md commit-discipline violation should not propagate to WP-E.**
WP-D and WP-C each bundled a structural refactor with behavioral additions. The WP-D snapshot explicitly said "WP-E should be a clean instance." Outcome: commit 54ed9e0 (`refactor: extract invoke_ensemble argument validation to free complexity budget`) landed as a separate `refactor:` prefix commit BEFORE piece 5's `feat:` commit. The commit message names the complexity-budget rationale and states "no behavior change." The behavioral piece (425edb0) follows as a distinct commit. **Status: addressed. WP-E is the clean instance the advisory requested. The two-consecutive-WP pattern does not propagate to a third.**

**WP-D Advisory 3 — PLAY-phase validation scope for `envelope.structured` requires field guide annotation.**
Outcome: not assessed as explicitly addressed by WP-E artifacts (no field guide diff was included in the examined commits). **Status: carry-forward. PLAY-phase observation agenda should include a note that `envelope.structured` yields `None` for all currently migrated ensembles by design, not defect.**

**WP-D Advisory 4 — Deferred error-envelope work should be named at WP-E entry, not left implicit.**
The cycle-status entry at "From Cycle 6 BUILD WP-D" explicitly records the deferred partial-failure scenario (item 4 in that section): "WP-E should explicitly track whether ensemble-execution work needs per-stage error surface, or whether the scenario stays vacuous through Cycle 6." WP-E did not add the partial-failure scenario; the vacuous-pass count stays at two. **Status: correctly deferred. The carry-forward is named in cycle-status; no additional WP-E instance.**

**WP-D Advisory 5 — Envelope payload location rationale should appear in codebase.**
`_route_dispatch_to_substrate`'s inline comment at line 1421-1426 of `orchestrator_tool_dispatch.py` states: "the orchestrator-LLM observes a compact `{'summary': ...}` dict in its tool-message context. The typed envelope rides along on the additive `envelope` field for downstream consumers." This names the design position (compact LLM observation vs. caller-API envelope) in the codebase. **Status: partially addressed. The comment covers the WP-E substrate-path caller; the WP-D additive-field rationale on `ToolCallSuccess.envelope`'s docstring itself was not examined. Low residual risk.**

**WP-D Advisory 6 — Vacuous-pass accumulation pattern should be tracked.**
WP-E did not add a third vacuously-passing fitness criterion. See Signal Assessment below for the FC-8 question (Decision 3 in this snapshot), which is the closest candidate for a third vacuous-pass instance. **Status: accumulator holds at two. FC-8's AS-7-amended shape warrants separate assessment.**

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Continuing decline (stable from WP-D) | 2849 tests (+37) anchor all claimed behaviors. The substrate-routing branch has 25 unit tests covering all five gating conditions; the Calibration Gate surfaces have 8 unit tests including the ADR-007 clause 2 preservation test; FC-22 and FC-26 have 4 real-component integration tests. All dispatch prompt claims can be traced to a specific test assertion. No claim is advanced without a named grounding artifact. |
| Solution-space narrowing | Absent | Stable (inherited narrowing only) | WP-E operates inside ARCHITECT/DECIDE's established envelope (always-scope substrate routing, AS-7 amended). The six autonomous decisions (see below) are BUILD-time gap resolutions, not practitioner-pressure narrowing events. The refactor-then-feat splitting decision expanded the solution space slightly (the alternative of bundling them was explicitly named and rejected by reference to WP-D advisory 2). |
| Framing adoption | Absent | Continuing decline from ARCHITECT | No practitioner framing embedded in WP-E dispatches beyond the declared auto-mode instruction ("work without stopping for clarifying questions; make the reasonable call and continue"). This instruction is an authorization, not a framing event — it does not prescribe specific design choices. |
| Confidence markers | Ambiguous (two instances) | Stable (consistent WP-C/WP-D pattern) | Decision 3 (FC-8 unchanged; FC-26 sufficient): the agent adopts the technically-correct framing ("FC-8's AST walk does not reach `_route_dispatch_to_substrate`") to close the question without examining whether FC-8's *intent* — not just its mechanism — is fully honored under AS-7 amended. Decision 4 (content-type binary default): "the rougher detection is the right shape for the first deployment" (docstring language) is a closure move that defers alternative type enumeration without naming the specific alternatives (text/plain, application/json) that have natural mappings for `text-summarizer` and `web-searcher`. Neither is a claim about unverified behavior; both use confident framing to resolve an open question rather than name it as open. |
| Alternative engagement | Absent (expected) | Stable per auto-mode declaration | Auto-mode authorization suppresses alternative surfacing to the practitioner. All six decisions have named alternatives in the dispatch prompt; none was surfaced during execution. This is expected behavior for declared auto mode (ADR-091). |
| Embedded conclusions at artifact-production moments | Clear (three instances) | Stable (consistent pattern; not rising) | Decision 3: FC-8 is left unchanged; FC-26 is presented as sufficient. The dual-structural-floor alternative (amend FC-8 to verify EITHER summarize-match OR substrate-write) is foreclosed without examination. Decision 4: `_resolve_substrate_content_type` ships as a binary (`code_generation → application/python`; else `text/markdown`), with `web-searcher`'s natural content-type (`application/json`, since its synthesizer emits JSON natively) and `text-summarizer`'s natural type (`text/plain`) unexamined. Decision 5 (calibration MVP shape): bundling deliverable content inline in checker payload rather than via ArtifactReadTool tool-call surface is acknowledged as a Cycle 7+ alternative but ships without the alternative being presented as a practitioner fork. All three are disclosed in the dispatch prompt and, for Decision 4, in the docstring; none was a fork the practitioner could choose before artifact production. |

---

## Autonomous Scoping Decision Assessments

### Decision 1 — Refactor-then-feat splitting (commit 54ed9e0 before 425edb0)

**What was decided:** When piece 5 hit the C901 complexity ceiling, the agent extracted argument validation to `_validate_invoke_arguments` as a separate `refactor:` commit before adding the substrate-routing `feat:` commit. The alternative (bundle the structural prep with the behavioral addition, as WP-C and WP-D did) was explicitly rejected by reference to WP-D moderate advisory 2 ("WP-E should be a clean instance").

**Assessment:** This is the correct response to the advisory. The refactor commit has "no behavior change" stated in its message and is mechanically verifiable (the same two validation paths, same typed errors, same dispatch position). The `feat:` commit then adds only the substrate-routing branch. The concern the dispatch prompt raises — whether this is ceremonial over-splitting rather than real structural-vs-behavioral separation — is actually backward: this is exactly what the CLAUDE.md invariant mandates and what WP-D's snapshot explicitly requested. The commit-discipline trajectory from WP-C → WP-D (violations) → WP-E (clean) is the intended pattern.

**Whether auto-mode covers this:** Yes. The scoping judgment (split vs. bundle) was explicitly pre-configured by the WP-D advisory. No fork needed to be surfaced; the advisory provided the directive.

**Severity:** No advisory. Clean resolution of a carried advisory.

---

### Decision 2 — WP-D advisory 1 resolution as option (b): schema-as-documentation, default_task preserved

**What was decided:** The three YAMLs from WP-D (`claim-extractor`, `text-summarizer`, `web-searcher`) retain their original `default_task` wording; WP-E adds three more capability ensembles (`argument-mapper`, `prose-improver`, `code-generator`) with the same documentary-schema posture. Option (a) — rewriting `default_task` to request JSON output — was rejected with reasoning from spike β (drift's mechanism is orchestrator `input.data` override; fixing synthesizer compliance targets the wrong layer).

**Assessment:** The fork was surfaced at the advisory boundary (cycle-status entry) and the resolution is named in the piece 7 commit message by advisory number and option letter. This satisfies the WP-D snapshot's requirement that "WP-E entry should explicitly choose." The residual question is whether the auto-mode resolution of option (b) displaced a practitioner-visible fork moment: the cycle-status entry reads as practitioner-facing (it is in the WP-D close summary that the practitioner reviewing WP-D would read); the piece 7 commit message names the advisory explicitly. The resolution is disclosed at both boundaries.

The substantive question — whether option (b) is the correct long-run choice — turns on spike β's reframing. That reframing is documented and internally consistent: the orchestrator's `input.data` override is the active mechanism; `default_task` is a fallback that is often overridden in production compositions. However, the consequence for PLAY-phase validation is unchanged from the WP-D advisory 3 concern: testing structured population against the six migrated ensembles will yield `envelope.structured = None` for five of them (all except `web-searcher`, which has `output_schema` declared but still produces prose unless the orchestrator's `input.data` requests JSON). The field guide annotation the WP-D advisory recommended remains unverified as written.

**Whether auto-mode covers this:** Partially. The option selection (b) is a substantive design choice, but the cycle-status entry makes the fork practitioner-visible at the WP-D close boundary. Auto mode is authorized to select when the practitioner has been given the fork at a natural review boundary (which they were).

**Severity:** Low advisory. The field guide annotation for PLAY-phase validation of `envelope.structured` remains unverified. See Advisory 1 below.

---

### Decision 3 — FC-8 unchanged; FC-26 treated as sufficient for AS-7-amended structural floor

**What was decided:** FC-8's AST test (`test_fc8_summarizer_bypass.py`) walks only `invoke_ensemble`'s method body. The substrate `ToolCallSuccess` lives in `_route_dispatch_to_substrate`, a helper method FC-8 does not walk. The agent's reasoning: FC-8's scope is technically unchanged (it finds one `ToolCallSuccess` constructor in `invoke_ensemble`'s AST — the inline path — and verifies it is inside the summarize-match block). FC-26 provides the substrate-path structural floor (substrate path → 0 harness invocations; inline path → 1). These two tests together cover the AS-7-amended dual floor.

**Assessment:** The technical claim is correct. FC-8's `_find_method` extracts `invoke_ensemble`'s method body via `ast.walk(class_def.body)`, then walks that body. `_route_dispatch_to_substrate` is a separate method; its `ToolCallSuccess` constructor at line 1427 is not in `invoke_ensemble`'s AST walk. FC-8 does not find an unguarded constructor because the substrate-path constructor is not visible to its walk. The test does not break and does not silently weaken: the substrate path is handled by FC-26.

The remaining concern from the dispatch prompt is design-intent, not mechanical: FC-8's stated purpose is "unsummarized result cannot reach the Orchestrator Runtime's context." Under AS-7 amended, the substrate path is specifically the legitimate bypass — `ToolCallSuccess` is produced without the summarizer. FC-8's AST dominance check *continues to pass correctly* because it only guards against *unintended* bypasses in `invoke_ensemble`'s own body. The substrate path is an intentional bypass that lives in a named helper. This is actually a clean separation of concerns: FC-8 guards the inline path's invariant; FC-26 guards the substrate path's invariant. No amendment to FC-8 is strictly required.

The question of whether FC-8 should be *amended to name its amended scope explicitly* (so a future engineer reading FC-8 understands it now has a sibling FC-26 that covers the substrate path) is a documentation question, not a correctness question. FC-8's module docstring says "every `ToolCallSuccess` in `invoke_ensemble` must live inside the summarize-match"; this is still true because the substrate `ToolCallSuccess` is not in `invoke_ensemble`. The AS-7-amended invariant (substrate path legitimately skips the harness) is covered by FC-26, not by an FC-8 absence. The two together constitute the dual structural floor the dispatch prompt suggests.

**Whether a third vacuous-pass accumulates here:** No. FC-8 still finds a `ToolCallSuccess` in `invoke_ensemble` (the inline path) and still verifies it is dominated by the summarize-match. The test is not vacuously passing; it is correctly scoped to the inline path. The substrate path is correctly scoped to FC-26. This is the right decomposition, not a vacuous-pass instance.

**Whether auto-mode covers this:** Yes. The FC-8 scope question is a BUILD-time structural interpretation; FC-26 provides the explicit substrate-path floor. The deferred FC-8 amendment (documenting the dual-floor relationship in FC-8's module docstring) is low-priority carry-forward.

**Severity:** Low advisory. FC-8's module docstring should note that AS-7 amended introduces a sibling floor (FC-26) covering the substrate path, so readers do not need to reconstruct the dual-floor design from the two test files independently. A two-sentence addition suffices.

---

### Decision 4 — Content-type detection binary (`code_generation → application/python`; else `text/markdown`)

**What was decided:** `_resolve_substrate_content_type` maps `topaz_skill == "code_generation"` to `application/python` and everything else to `text/markdown`. Configurable per-ensemble `content_type:` YAML declarations are deferred to Cycle 7+. The docstring explicitly names this as "best-effort" and "the right shape for the first deployment."

**Assessment:** The binary is reasonable for a Cycle 6 MVP. The concern is that `web-searcher`'s synthesizer emits JSON natively (it has `output_schema` declared; its synthesizer is designed to return structured JSON), making `text/markdown` a plausible misclassification for its artifacts. Similarly, `text-summarizer` produces plain prose, for which `text/markdown` is technically correct in MIME-type terms (Markdown is a superset of plain text) but `text/plain` is more precise. These are not behavioral defects — the artifact is written and readable regardless of content-type label — but the content-type field influences downstream tooling that reads artifacts by type.

The alternative enumeration the dispatch prompt names (`web-searcher → application/json`; `text-summarizer → text/plain`) would require an additional YAML field or a richer topaz_skill-to-content-type mapping. The docstring closes this question without naming these specific alternatives. The deferred YAML `content_type:` mechanism is the correct long-term solution; the gap is that the interim binary default is not the best-fit classification for two of the six migrated ensembles.

This is an accepted approximation, disclosed in the docstring as "best-effort." It is not a design risk that affects the substrate-routing invariant (FC-22, FC-26) or the session-lifecycle invariant (ADR-025). It is a metadata precision gap that Cycle 7+ closes via per-ensemble YAML declaration.

**Whether auto-mode covers this:** Yes. The MVP binary is a justified scope simplification. The alternative (richer mapping per topaz_skill or per YAML field) was considered and deferred with explicit Cycle 7+ attribution.

**Severity:** Low advisory. The field guide's WP-E entry should note that `web-searcher`'s substrate artifacts will be labeled `text/markdown` despite their JSON structure. If a downstream consumer (Cycle 7 artifact reader or PLAY probing) expects `application/json` for web-searcher artifacts, the label mismatch should not be misread as an artifact corruption.

---

### Decision 5 — Calibration MVP shape: content inline vs. ArtifactReadTool tool-call surface

**What was decided:** `_shape_calibration_evaluation_input` for `calibration_substrate_access: artifact` bundles the deliverable content from `raw_result.synthesis` directly in the checker payload. ADR-025's §"Calibration-gate evaluation surface" names an ArtifactReadTool tool-call surface as the intended long-term shape; the MVP inline approach is labeled Cycle 7+ territory in the docstring and commit message.

**Assessment:** The simplification is appropriate for the Cycle 6 MVP. The inline approach couples the calibration payload shape to the synthesizer's in-memory result rather than to the stored artifact; a mismatch between what was written to disk and what the calibration checker sees is theoretically possible if the content normalization differs (e.g., encoding differences). In practice the same `_extract_synthesizer_text` function feeds both the write path and the calibration payload, so the content is identical. The risk is structural (checklist architecture) rather than behavioral for the current implementation.

The ADR-007 clause 2 preservation (calibration failures swallowed by `_calibration_check_safe`) applies to the substrate path as well, correctly preserving the "calibration is advisory, not blocking" invariant. The 8 unit tests include an explicit ADR-007 clause 2 test. This is clean.

**Whether auto-mode covers this:** Yes. The simplification is disclosed and attributed as Cycle 7+. No practitioner fork was warranted at the MVP boundary.

**Severity:** No advisory beyond the existing carry-forward. The Cycle 7+ attribution is correctly placed in both the ADR-025 spec and the code docstring.

---

### Decision 6 — FC-8 and FC-26 as complementary floors without cross-reference (carry-forward from Decision 3)

This is subsumed under Decision 3. See that assessment.

---

## Interpretation

### Pattern assessment

The dominant pattern for WP-E is **clean auto-mode execution with two prior moderate advisories addressed and a strong empirical grounding floor.** The BUILD phase's empirical resistance (2849 tests; +37 new including 25 substrate-routing unit tests, 8 calibration-surface unit tests, 4 real-component integration tests; FC-22 three-surface end-to-end; FC-26 AS-7-amended dual-floor) is the strongest within Cycle 6's BUILD sequence. Each commit is mechanically traceable to one fitness criterion or ADR clause.

Two signals that were rising trajectories in WP-C and WP-D are now resolved:

- **Commit-discipline violations (WP-C Advisory 5, WP-D Advisory 2):** WP-E is the clean instance. The refactor-then-feat split is unambiguous. The pattern does not propagate.
- **YAML task rewrite fork not surfaced (WP-D Advisory 1):** The cycle-status entry named both options; the piece 7 commit message attributes the resolution by advisory number and option letter. The fork was presented at the natural review boundary even though the resolution was made in auto mode.

One signal that accumulated across WP-C and WP-D (vacuous-pass) **did not grow in WP-E.** FC-8's unchanged scope (Decision 3) is not a vacuous-pass instance; it is a correctly scoped test with a sibling (FC-26) that covers the new floor. The accumulator stays at two.

The confidence-marker pattern (two instances per WP, stable across WP-C through WP-E) continues: Decision 3 uses "FC-8 does not walk that method" as a closure move rather than opening the dual-floor documentation question; Decision 4 uses "best-effort / right shape for first deployment" to defer alternative type enumeration. Neither is a claim about unverified behavior; both are confident resolution frames applied to deferral decisions. This is the residual sycophancy-gradient signal for BUILD: not framing adoption, but closure-move language at artifact-production moments.

The most substantive carry-forward for PLAY and Cycle 7 is the **`envelope.structured = None` production state for all six migrated ensembles** (option (b) resolution confirmed by WP-E). Option (b) is coherent given spike β's reframing, but PLAY-phase probing of structured population will require bespoke test ensembles with explicit JSON-requesting task wording, not the six shipped YAMLs. Without the field guide annotation (WP-D advisory 3, still unverified), PLAY may generate false-negative findings.

### Earned confidence vs. sycophantic reinforcement

The signals are consistent with earned confidence across the build sequence. The empirical grounding is denser in WP-E than in any prior WP: the integration tests use real components (real `DispatchEventSubstrate`, real `SessionArtifactStore`, real `OrchestratorToolDispatch`, real harness), not stubs. FC-22's three-surface property is verified against actual filesystem writes. FC-26's AS-7-amended floor is verified against a recording invoker that would catch any erroneous harness call. The substrate-routing branch's five gating conditions are each covered by a distinct unit test; none passes vacuously.

The two confidence-marker instances are consistent with the WP-C/WP-D baseline — they are the same structural type (confident closure on a deferral decision) and neither encodes a design choice that blocks future correction. The FC-8 scope question (Decision 3) is the most significant of the two, because it concerns a structural test that a future BUILD engineer might read without knowing FC-26 is its substrate-path complement. But this is a documentation gap, not a design error.

The absence of a third vacuous-pass instance is a positive signal. The accumulation pattern that WP-D named as a trend does not continue into WP-E. The fitness criterion set's grounding function is not further degraded.

### BUILD-phase epistemic close assessment (Mode D)

As the BUILD-phase close for Cycle 6's Mode D mini-cycle, this snapshot assesses whether the BUILD sequence as a whole shows a pattern requiring a Grounding Reframe before PLAY entry. It does not. The trajectory across WP-B through WP-E shows:

- **Consistent test-execution grounding** at each WP boundary (from 2727 at WP-B entry to 2849 at WP-E close; +122 net).
- **Two moderate advisories surfaced and addressed** (WP-D's commit-discipline and YAML-fork advisories); no moderate advisory has now propagated unaddressed into PLAY scope.
- **Vacuous-pass accumulation stable** at two instances (WP-C and WP-D decisions); WP-E did not add a third.
- **Confidence-marker pattern stable** (two per WP, closure-move type); no escalation.
- **Framing adoption absent** across all four WPs.

The residual concerns entering PLAY are low-severity documentation gaps, not design commitments that require reframing before PLAY can produce valid field notes.

### Prior advisory carry-forward status at BUILD-phase close

| Advisory | Origin | Status at WP-E close (BUILD close) |
|----------|--------|-------------------------------------|
| `web-searcher` early-migration sequencing | DECIDE Finding 1 + ARCHITECT Advisory 1 + WP-B + WP-C + WP-D | **Resolved.** `web-searcher` migrated early in WP-E piece 7 per advisory. Indicators 1 and 4 now testable in PLAY. |
| YAML task rewrite fork (default_task) | WP-D Moderate Advisory 1 | **Resolved as option (b).** Documented in cycle-status and commit message. `envelope.structured = None` in production for all six migrated ensembles. |
| Commit-discipline violation pattern | WP-C Advisory 5 + WP-D Moderate Advisory 2 | **Resolved in WP-E.** Clean refactor-then-feat split. Pattern does not propagate. |
| Field guide annotation for `envelope.structured` PLAY validation | WP-D Advisory 3 | **Active — PLAY scope.** No field guide diff observed in WP-E commits. PLAY probing of structured population requires bespoke test ensembles. |
| Deferred error-envelope (partial-failure path) | WP-D Decision 2 | **Active — named in cycle-status.** Vacuous-pass count stays at two. No WP-E action; correctly deferred. |
| FC-8 dual-floor documentation (AS-7 amended) | WP-E Decision 3 (new) | **New — low priority.** FC-8's module docstring should note FC-26 as the substrate-path sibling. Two-sentence addition. |
| `web-searcher` content-type label (text/markdown vs. application/json) | WP-E Decision 4 (new) | **New — low priority.** Field guide WP-E entry should note the label mismatch for PLAY observers reading artifact metadata. |
| `dispatch_log.json` path convention | WP-C Advisory 1 | **Active — unaddressed by WP-E.** PLAY or Cycle 7+ territory. |
| `role: user` alternative docstring gap | WP-C Advisory 2 | **Active — low priority carry-forward.** |
| Compaction-observation interaction | WP-C Advisory 3 | **Active — PLAY observation agenda.** |
| End-to-end serve-close path (FC-24 gap) | WP-C Advisory 4 | **Active — PLAY observation agenda.** |
| `AuditDiagnostic` exclusion documentation | WP-C Advisory 6 | **Active — low priority carry-forward.** |
| Heartbeat live-deployment verification | WP-B Advisory 4 | **Active — PLAY observation agenda.** |
| Preservation-scenario amendment pattern | Cycle 5 BUILD | **Active — not triggered in WP-E (no scenario amendments).** |
| test_cli.py propagation defect (caplog) | WP-B Advisory 1 | **Dormant — WP-E did not use caplog integration tests.** |

---

## Recommendation

**No Grounding Reframe warranted** — signals are consistent with earned confidence at BUILD close. The BUILD phase's empirical grounding across the full WP-B through WP-E sequence (2849 tests, 122 net new; FC-22 three-surface end-to-end; FC-26 AS-7-amended dual floor; FC-8 correctly scoped to inline path) provides the structural resistance appropriate to this phase position. The two moderate advisories from WP-D were addressed in WP-E: the refactor-then-feat discipline was honored, and the YAML task rewrite fork was surfaced at the advisory boundary and resolved with attributed rationale. No third vacuous-pass instance accumulated. The confidence-marker pattern (two closure-move instances per WP) is stable and non-escalating.

### Advisory feed-forwards for PLAY entry

**Advisory 1 — Field guide annotation for `envelope.structured` validation required before PLAY probing (WP-D Advisory 3 carry-forward, low-moderate).**

All six migrated capability ensembles produce `envelope.structured = None` in production because their `default_task` wording requests prose, not JSON. Option (b) is the documented intentional policy. PLAY-phase probing that tests structured population against any of the six shipped ensembles will observe `None` and should not file this as a defect. The field guide's WP-E entry needs a one-paragraph note: validating `envelope.structured` population requires a bespoke test ensemble whose `default_task` explicitly instructs JSON output conforming to `output_schema:`. Without this note, PLAY observation 1 or 2 may file a false-negative finding that triggers an unnecessary debugging loop.

**Advisory 2 — FC-8 / FC-26 dual-floor relationship should be noted in FC-8's module docstring (Decision 3, low).**

FC-8's module docstring states "every `ToolCallSuccess` in `invoke_ensemble` must be dominated by the summarize-match." Under AS-7 amended, the substrate path legitimately produces a `ToolCallSuccess` without the summarize-match — but this constructor is in `_route_dispatch_to_substrate`, not in `invoke_ensemble`'s body, so FC-8 continues to pass correctly. A future BUILD engineer reading FC-8 without FC-26 context may not understand why FC-8 does not test the substrate path. A two-sentence docstring addition — noting that AS-7 amended introduces `_route_dispatch_to_substrate` as the substrate-path success arm, and that FC-26 (`test_fc22_fc26_substrate_routed_envelope.py`) is FC-8's sibling floor for the substrate path — prevents this confusion.

**Advisory 3 — `web-searcher` content-type label mismatch should be noted for PLAY observers (Decision 4, low).**

`web-searcher` artifacts are written with `content_type = text/markdown` (the binary default) despite the synthesizer emitting JSON natively. PLAY observers reading artifact metadata (checking `content_type` on stored artifacts) will find `text/markdown` for web-searcher deliverables. This is not a defect — the artifact content is valid JSON regardless of the MIME label — but it is a metadata imprecision that Cycle 7's per-ensemble `content_type:` YAML declaration resolves. The field guide should note the expected label so PLAY does not flag it as an artifact-store defect.

**Advisory 4 — PLAY observation agenda for active carry-forwards from earlier WPs.**

The following active advisories are PLAY territory and should appear on the observation agenda:
- Heartbeat live-deployment verification (WP-B Advisory 4): does the heartbeat fire during 8+ minute real-clock inference waits?
- Compaction-observation interaction (WP-C Advisory 3): does compaction mid-session produce observable artifacts at the dispatch substrate?
- End-to-end serve-close path (WP-C Advisory 4): does session close drive artifact cleanup correctly in a real serve process?
- `web-searcher` ADR-025 Indicator 1 and 4 falsification: latency overhead for small deliverables (Indicator 1) and `output_substrate: inline` opt-out count (Indicator 4) are now testable with `web-searcher`'s early migration.
