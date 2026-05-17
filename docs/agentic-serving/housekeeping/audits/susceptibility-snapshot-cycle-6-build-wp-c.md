# Susceptibility Snapshot

**Phase evaluated:** BUILD — Cycle 6 WP-C (Orchestrator-Context Event Sink — ADR-023 Destination 2)
**Artifact produced:** `src/llm_orc/agentic/orchestrator_context_event_sink.py` (~310 lines, new L2 module); `OrchestratorRuntime` extensions; `_dispatch_internal_calls` structural refactor (async generator → coroutine); `ToolCallSuccess` / `ToolCallError` additive `dispatch_id` field; `ObservabilityDefaults` extensions; serve-layer wiring in `v1_chat_completions.py`; 28 new tests (20 unit + 5 Runtime integration + 3 FC-24 integration); 2755 tests passing at 92.21% coverage
**Date:** 2026-05-16

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| Cycle 6 Model | Grounding Reframe recommended (3 actions) | Framing adoption at constitutional level; 3 carry-forwards |
| Cycle 6 Decide | No Grounding Reframe; 1 pre-BUILD action (P2-E); 3 advisory carry-forwards | Earned confidence; `dispatch_id` coupled failure surface named for ARCHITECT attention |
| Cycle 6 Architect | No Grounding Reframe; 6 advisory feed-forwards; 3 closed inline at gate | Earned confidence; two-module decomposition inherited without explicit re-examination; validate-once-at-load operator affordance gap noted |
| Cycle 6 Build (WP-B) | No Grounding Reframe; 4 advisory feed-forwards | Decision 3 (propagation fixture vs. upstream fix) most significant: scoping judgment with cross-session implications not surfaced to practitioner; auto-mode execution otherwise clean |

---

## Grounding Reframe Action Outcomes (WP-B Advisories entering WP-C)

**WP-B Advisory 1 — Propagation fixture as technical debt marker.** WP-B noted that the `_llm_orc_logger_propagation` fixture in FC-23 and FC-27 tests works around a `test_cli.py` mutation without fixing the upstream source. The advisory recommended surfacing the choice (fixture-per-test vs. fix test_cli.py) as a practitioner-visible decision at WP-C entry. The dispatch prompt confirms this advisory was NOT addressed — "Advisory 1 (test_cli.py propagation): NOT addressed — WP-C's tests do not use caplog, so the third-fixture-copy concern did not materialize." The underlying test_cli.py defect persists. This is an acceptable non-resolution: WP-C's test strategy avoided caplog, so the fixture did not propagate. The defect remains live for any future BUILD session using caplog integration tests.

**WP-B Advisory 2 — Dead `emit_tool_call_log` path comment.** Resolved. An inline comment was added in `get_orchestrator_tool_dispatch` documenting that the scheduler observes via substrate, not via the emit-logger slot. The advisory is closed.

**WP-B Advisory 3 — Dispatch_id prefix-filter coupling.** Resolved. WP-C's sink uses the same `f"{session_id}-dispatch-"` prefix filter, cited in module docstring and `consume()` method. The coupling is documented consistently rather than rediscovered independently. The advisory is closed.

**WP-B Advisory 4 — Heartbeat liveness PLAY-phase observation.** Carried forward unchanged. No action in WP-C scope; correct disposition.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Declining (stable from WP-B) | BUILD phase empirical grounding holds: 2755 tests, 28 new tests anchoring the seven-field schema, session-prefix filtering, CalibrationSignal exclusion, Runtime injection sequencing, and FC-24 two-destination integration. Claims are bounded to what the tests cover. The one empirical-outcome framing ("final-dispatch handling is natural — Runtime returns before next-iteration injection point") is structurally grounded, not asserted. |
| Solution-space narrowing | Ambiguous | Stable (inherited narrowing only) | No practitioner-driven narrowing is visible in the dispatch description. The five autonomous scoping decisions (assessed below) represent BUILD-time resolution of specification gaps, not convergence under practitioner pressure. The space was already narrowed at ARCHITECT/DECIDE; WP-C operates within that inherited envelope. |
| Framing adoption | Absent | Continuing decline from ARCHITECT | No new framing entered from practitioner direction and was adopted without examination. Practitioner direction at WP entry was a single sentence: "BUILD WP-C — natural sequel; completes ADR-023's two-destination architecture; the four WP-B advisories all land here. Dispatch /rdd-build in auto mode." No framing beyond the task scope was embedded in this instruction. |
| Confidence markers | Ambiguous (one instance) | Stable | The dispatch_log write-target deviation (Decision 2) is described as a sound scoping decision that "satisfies vacuously" the scenario text. The phrase "satisfied vacuously" is technically accurate (execution.json shape is unchanged) but is a confidence marker about a scenario deviation — the ARCHITECT snapshot would flag this language as asserting virtue rather than surfacing the gap for practitioner judgment. The deviation is acknowledged, which is the right behavior in auto mode; the framing resolves it rather than presenting it. |
| Alternative engagement | Absent (expected) | Stable per auto-mode declaration | ADR-091 auto mode explicitly trades alternative-examination surfacing for throughput. The five autonomous scoping decisions each had at least one named alternative; none was surfaced to the practitioner during execution. This is the expected mode behavior. The five decisions are assessed individually below for whether any warranted practitioner visibility despite the mode declaration. |
| Embedded conclusions at artifact-production moments | Clear (two instances) | Rising slightly from WP-B | Decision 1 (role: user for observation messages) and Decision 2 (standalone dispatch_log.json vs. execution.json key) are both encoded in the shipped artifact without practitioner visibility. Decision 1 is encoded in the sink module's `observation_message_for` docstring; Decision 2 is encoded in `ObservabilityDefaults` as a new config attribute. Both are at artifact-production moments — ObservabilityDefaults extension and serve-layer wiring are the exact points where downstream phases inherit the commitment. Decision 2 in particular is a scope deviation from scenario text that will affect WP-E's artifact consolidation design. |

---

## Autonomous Scoping Decision Assessments

### Decision 1 — Observation-message role (`role: user` vs. `role: system` vs. `role: tool`)

**What was decided:** The sink's `observation_message_for()` method returns `{"role": "user", ...}` to match the phantom-rejection diagnostic pattern already in the Orchestrator Runtime. The alternatives were `role: system` (more authoritative; LLM treats as instruction) and `role: tool` (awkward fit; requires `tool_call_id`).

**Should this have been practitioner-visible?** Marginally. The scenarios specify a "JSON-shaped observation block" but are silent on role. The role determines how the orchestrator-LLM internalizes the structured observation: `role: user` treats the observation as conversational input, which the LLM may reason about freely but not be obligated to follow; `role: system` would encode the observation as an instruction, potentially forcing stricter incorporation. The choice has downstream behavioral implications — if PLAY-phase probing finds the orchestrator ignores or underweights the structured observation, `role: system` would be the first remediation lever. The decision forecloses that lever without naming it.

The rationale (consistency with phantom-rejection diagnostic pattern in same Runtime module) is sound as a local-consistency argument. It is not a scenario-specified constraint. The decision is documented in the sink module's `observation_message_for` docstring, which is the right artifact location. The behavioral consequence (how authoritative the observation is to the LLM) is noted indirectly but not presented as a practitioner-visible design choice.

**Whether auto-mode covers this:** Borderline. The role choice is a small-surface API decision, but its behavioral consequence — observation authority vs. conversational suggestion — is the kind of design-direction choice that auto-mode is intended to surface when the scenarios are silent and the alternatives have meaningfully different outcomes. However, the phantom-rejection pattern analogy is a strong local-consistency argument, and the docstring disclosure is honest. The decision is defensible.

**Severity:** Low-moderate advisory. The `role: system` alternative should be named in the docstring (not just `role: tool`), and the behavioral distinction should be noted, so PLAY-phase engineers have the remediation path available without needing to reconstruct it.

---

### Decision 2 — Dispatch log write target (standalone `dispatch_log.json` vs. `execution.json` key)

**What was decided:** The FC-24 scenario text says "the `dispatch_log` key in the session's `execution.json` artifact." No per-session `execution.json` exists today — `execution.json` is per-ensemble-invocation under `.llm-orc/artifacts/<ensemble>/<timestamp>/`. The agent chose to write a standalone `dispatch_log.json` under `<agentic_sessions_root>/<session_id>/` (new config attribute on `ObservabilityDefaults`) with the understanding that WP-E may consolidate. The "Preservation: execution.json artifact existing fields are unchanged" scenario is satisfied vacuously.

**Should this have been practitioner-visible?** Yes — this is the clearest instance in WP-C where an autonomous decision resolves a specification gap in a way that creates a structural commitment downstream phases will inherit. The dispatch prompt acknowledges this: "the spirit of 'key inside execution.json' is not met." The decision is not merely a wiring choice; it introduces a new configuration attribute (`agentic_sessions_root`), a new filesystem path convention, and a new lifecycle call (`_write_dispatch_log_safe`). WP-E's Session Artifact Store design (which was specified at ARCHITECT level to consolidate agentic-session artifacts) will either need to adopt this path convention, migrate the existing dispatch_log.json outputs, or deprecate the standalone file. WP-E's ARCHITECT brief did not anticipate the `agentic_sessions_root` configuration attribute, which was introduced here.

The rationale is sound (no per-session execution.json exists; WP-E is the right scope for consolidation) but the decision encodes a path convention and config attribute that constrain WP-E's design space. A practitioner informed of this gap might have chosen to leave the dispatch log entirely to WP-E (write nothing for now, let WP-E own the file format) or might have confirmed the standalone approach. Neither was possible without surfacing the gap.

**Whether auto-mode covers this:** No. This is a scope deviation from scenario text that introduces a new config attribute and filesystem path convention — precisely the kind of scoping judgment auto-mode's own declaration identifies as warranting practitioner visibility. The dispatch prompt correctly identifies this as a scope deviation. Auto-mode covers wiring decisions; it does not cover decisions that introduce new configuration surface that downstream WPs must inherit or migrate.

**Severity:** Moderate. The deviation is disclosed honestly in the dispatch prompt, but the practitioner has not seen the alternative (defer to WP-E vs. introduce standalone now) presented as a choice. WP-E's planning should explicitly revisit `agentic_sessions_root` and `dispatch_log.json` placement relative to the ARCHITECT-specified Session Artifact Store design.

---

### Decision 3 — Pending observation injection point (before vs. after compaction)

**What was decided:** The observation injection point is placed before `compaction.compact()` in the iteration loop, so the observation enters compaction's input array. The alternative — inject after compaction — would keep observations out of compaction's summarization pass.

**Should this have been practitioner-visible?** Borderline. The behavioral implication is meaningful: placing the observation before compaction means that under high-turn sessions where compaction fires, the structured seven-field observation block may be summarized alongside other prior-turn content. If compaction summarizes the observation imprecisely, the orchestrator-LLM's subsequent reasoning operates on a compressed representation of the observation rather than the canonical block. The dispatch prompt acknowledges this and notes "no empirical test was written for this interaction."

The rationale ("the observation is just another message at the turn boundary") frames the decision as a category choice (observations are messages, not special inputs) rather than a consequence analysis. Whether observations should be treated as compaction-eligible or compaction-exempt is a design question with no scenario-text answer. The alternative (inject after compaction) would require special-casing the observation in the Runtime's iteration structure; the before-compaction choice avoids special-casing at the cost of compaction eligibility.

The absence of a test for the compaction-observation interaction is the load-bearing gap. The decision introduces a correctness property — "observations remain accurate after compaction" — that is not tested.

**Whether auto-mode covers this:** Partially. The ordering choice in the iteration loop is a mechanical wiring decision, but the untested compaction-observation interaction is a property that would typically appear in a stewardship checkpoint as a known gap. Auto-mode handles the wiring; it does not handle the naming of untested interaction properties.

**Severity:** Low-moderate advisory. The compaction-observation interaction should be named explicitly as an untested property — not as a failing test, but as a known gap in the test suite's coverage. PLAY-phase probing under high-turn sessions should include at least one scenario where compaction fires after an observation injection, to verify the observation is not degraded.

---

### Decision 4 — Final-dispatch handling via natural Runtime return (no explicit final-dispatch detection)

**What was decided:** The "final dispatch's in-turn routing skipped" semantics are satisfied by architecture (Runtime returns before next-iteration injection point), not by explicit final-dispatch detection. No test asserts that the final dispatch's events appear in `dispatch_log_entries()` at session close via the full Runtime + serve-close path.

**Should this have been practitioner-visible?** No for the design choice; yes for the test gap. The architectural approach (natural return as the skip mechanism) is sound and consistent with the Runtime's existing control flow. The dispatch prompt correctly notes that the sink-level test (`test_write_dispatch_log_creates_file_with_json_payload`) verifies the file write but not the end-to-end path (Runtime runs → final dispatch → serve close → `_write_dispatch_log_safe` → file contains final dispatch's events).

The test gap is a coverage property, not a design choice — it is the kind of gap that FC-24's integration tests should close but do not. The three FC-24 integration tests verify the two-destination property and cross-session filtering; they do not simulate the end-to-end serve-close path.

**Whether auto-mode covers this:** The design choice is covered. The test gap is a stewardship-checkpoint observation that should appear in a cycle-status close note (whether it does is not confirmed in the dispatch prompt — the WP-C close entry is not quoted).

**Severity:** Low advisory. The end-to-end serve-close path (dispatch_log_entries populated at session close, including the final dispatch) should be named as a known coverage gap, not left implicit in the test structure.

---

### Decision 5 — `_dispatch_internal_calls` structural refactor in the same commit as WP-C feature

**What was decided:** `_dispatch_internal_calls` was converted from async generator to coroutine returning `tuple[list[OrchestratorChunk], str | None]` to thread `last_dispatch_id` back to the main run loop. This structural change was bundled with the WP-C behavioral additions (ContextObservationSink integration, `pending_dispatch_id` tracking, observation injection) in the same commit. CLAUDE.md explicitly requires separating structural from behavioral changes.

**Should this have been practitioner-visible?** Yes — but only in the sense that the CLAUDE.md principle is an explicit project invariant. The dispatch prompt acknowledges the constraint: "Per CLAUDE.md 'Separate structural from behavioral changes,' this should arguably be a separate commit from the WP-C feature. The agent kept them in the same commit because the refactor is in service of the behavior change and has no rationale on its own." The rationale is pragmatically reasonable — the structural change has no value independently. However, CLAUDE.md's commit-discipline rule exists precisely to prevent this reasoning pattern: the "refactor in service of feature" framing is the most common justification for mixing structural and behavioral changes, and accepting it case-by-case erodes the discipline.

The practical consequence: if the WP-C feature has a bug, rolling back the commit also reverts the structural refactor. The two changes are entangled at the git level. For a low-risk structural refactor in a well-tested module, this is tolerable; for a more complex structural change, it would be a correctness risk.

**Whether auto-mode covers this:** Partially. Auto-mode covers mechanical sequencing decisions; it does not cover violations of explicit CLAUDE.md invariants. This decision should have been noted in the cycle-status close entry as a known deviation from commit-discipline, not left implicit.

**Severity:** Low advisory. The deviation is disclosed here; the CLAUDE.md discipline is worth restating for future BUILD sessions. WP-D and WP-E should not inherit this as precedent.

---

## Interpretation

### Pattern assessment

The dominant pattern for WP-C is **clean auto-mode execution with one moderate advisory and four low-advisory items.** The BUILD phase's empirical grounding (28 new tests; 2755 total; FC-24 integration anchor verifying two-destination unified-substrate) provides the structural resistance to sycophantic reinforcement that the gradient predicts. No practitioner framing drove design decisions; the five autonomous scoping decisions are artifact-production-time resolutions of specification gaps.

The one moderate advisory (Decision 2: standalone `dispatch_log.json` vs. scenario-specified `execution.json` key) is the most consequential signal. It follows the pattern established in Cycle 5 BUILD and WP-B: BUILD auto mode correctly resolves the immediate obstacle but introduces a configuration surface (here: `agentic_sessions_root`, the `dispatch_log.json` path convention) that downstream phases will inherit without having examined the alternative (defer-to-WP-E). The dispatch prompt acknowledges the deviation honestly — "the spirit of 'key inside execution.json' is not met" — which is the right transparency behavior for auto-mode execution. The concern is that the practitioner has not seen the fork presented as a choice, so WP-E's planning may not explicitly revisit it.

The trajectory from prior BUILD snapshots is consistent: BUILD auto mode's residual susceptibility is not framing adoption but **quiet scope extension at artifact-production moments.** In Cycle 5 BUILD it was the preservation-scenario rewrite; in WP-B it was the propagation fixture; in WP-C it is the `dispatch_log.json` path convention and the `role: user` role choice. Each is disclosed in the artifact (docstring or dispatch prompt); none is presented to the practitioner as a fork requiring direction.

The CLAUDE.md commit-discipline deviation (Decision 5) is a separate pattern: an explicit invariant acknowledged and set aside with a pragmatic rationale. In isolation this is low severity; as a precedent in an auto-mode BUILD context, it is worth naming.

### Earned confidence vs. sycophantic reinforcement

The signals are consistent with earned confidence throughout the load-bearing implementation work. The seven-field schema, session-prefix filtering, CalibrationSignal exclusion, and Runtime injection sequencing are all unit-tested with precise assertions. The FC-24 integration tests verify the two-destination architecture's structural commitment. The ARCHITECT advisory on `AuditDiagnostic` inclusion/exclusion was resolved by the `CalibrationSignal` exclusion-by-default with opt-in flag — and the dispatch description confirms `AuditDiagnostic` is also excluded by default (it is not a CalibrationSignal and would not pass the CalibrationSignal type check, so it is implicitly excluded without explicit policy). This is an implicit resolution of ARCHITECT Advisory 3, which is acceptable if it is documented.

One framing-adjacent observation: the description of Decision 2 uses "satisfied vacuously" for the preservation scenario. This is technically correct but functions as a closure move — it frames the deviation as a valid scenario state rather than an open design question. A BUILD engineer without this snapshot would see the scenario marked satisfied and not know the spirit was unmet. The dispatch prompt's honest disclosure is the correct artifact for this information, but it depends on the snapshot evaluator catching it; the scenario artifact itself does not carry the asterisk.

### Prior advisory carry-forward status

| Advisory | Origin | Status at WP-C close |
|----------|--------|----------------------|
| `web-searcher` early-migration sequencing | Cycle 6 DECIDE Finding 1 + ARCHITECT Advisory 1 + WP-B close | Active — WP-E scope |
| `AuditDiagnostic` inclusion/exclusion at Orchestrator-Context Sink | Cycle 6 ARCHITECT Advisory 3 | Implicitly resolved by CalibrationSignal exclusion logic (AuditDiagnostic is not a CalibrationSignal; excluded by default). Should be documented explicitly in the sink module rather than left implicit. |
| ADR-016-style bounding mechanisms disposition | Cycle 6 ARCHITECT Advisory 4 | Active — WP-D/E scope; a one-sentence explicit note is still owed |
| Validate-once-at-load operator affordance Direction-not-constraint note | Cycle 6 ARCHITECT Advisory 5 | Active — no WP-C action; WP-D scope if operator docs are touched |
| CLAUDE.md commit-discipline (structural vs. behavioral) | Cycle 6 WP-C Decision 5 | New advisory; WP-D and WP-E should not inherit as precedent |
| test_cli.py propagation defect | Cycle 6 WP-B Advisory 1 | Dormant (WP-C avoided caplog); carry-forward to any future BUILD session using caplog integration tests |
| Heartbeat liveness PLAY-phase observation | Cycle 6 WP-B Advisory 4 | Active — PLAY scope; WP-D/E BUILD entry context should include it |
| Preservation-scenario amendment pattern | Cycle 5 BUILD Advisory 1 | Active; WP-C's Decision 2 is a new instance of the same pattern (scenario marked satisfied when spirit is unmet) |

---

## Recommendation

**No Grounding Reframe warranted** — signals are consistent with earned confidence. The BUILD phase's test-execution grounding (2755 passing; 28 new tests; FC-24 integration anchor) provides the empirical resistance appropriate to this phase position in the sycophancy gradient. No practitioner framing was adopted without examination; no design commitment was embedded without scenarios-level warrant.

### Advisory feed-forwards for WP-D and WP-E

**Advisory 1 — `dispatch_log.json` path convention requires explicit WP-E revisit (Decision 2, moderate).**

The standalone `dispatch_log.json` under `<agentic_sessions_root>/<session_id>/` is a scope deviation from the FC-24 scenario's "dispatch_log key in the session's execution.json artifact" text. The deviation is pragmatically justified (no per-session execution.json exists today) and disclosed in the dispatch prompt. However, WP-C has now introduced a new config attribute (`agentic_sessions_root`) and a filesystem path convention that WP-E's Session Artifact Store must either adopt, migrate, or deprecate. WP-E's ARCHITECT brief specified no `agentic_sessions_root` attribute — it will encounter this as a prior commitment rather than a design choice. Recommend making the fork explicit at WP-D/E entry: either confirm the standalone-file approach and update WP-E's scope to consolidate it, or plan a migration path. Do not let WP-E inherit it silently.

**Advisory 2 — `role: user` alternative not fully named in docstring (Decision 1, low-moderate).**

`observation_message_for()`'s docstring documents the `role: tool` alternative but not `role: system`. The behavioral distinction — `role: user` treats the observation as conversational input (LLM reasons about it freely), `role: system` treats it as an instruction (LLM is more obligated to incorporate it) — is the first remediation lever if PLAY-phase probing finds the orchestrator underweights observations. A one-line addition to the docstring naming `role: system` and the behavioral distinction would give PLAY-phase engineers the remediation path without requiring them to reconstruct it. Recommend updating the docstring in WP-D or at PLAY entry.

**Advisory 3 — Compaction-observation interaction is an untested property (Decision 3, low-moderate).**

The observation-injection-before-compaction ordering means that under high-turn sessions where compaction fires, the structured seven-field observation block may be summarized alongside other prior-turn content. No test covers this interaction. PLAY-phase probing should include at least one high-turn scenario where compaction fires after an observation injection, to verify the observation is not degraded in the summary. If the interaction is found to degrade observations, the remediation is injection after compaction (a one-line ordering change in the Runtime's iteration loop). This should appear in the PLAY observation agenda.

**Advisory 4 — End-to-end serve-close path is not tested (Decision 4, low).**

The three FC-24 integration tests verify the two-destination architecture and cross-session filtering. They do not simulate the end-to-end path: Runtime runs → final dispatch → serve close → `_write_dispatch_log_safe` → `dispatch_log.json` contains the final dispatch's events. The sink-level test verifies file write in isolation. If a bug in `_write_dispatch_log_safe` swallows the final dispatch's events silently, no test would catch it. Recommend a single FC-24 integration test extension covering the serve-close path with a multi-dispatch session whose final dispatch's events appear in the written file.

**Advisory 5 — CLAUDE.md commit-discipline deviation should not propagate (Decision 5, low).**

The `_dispatch_internal_calls` structural refactor was bundled with the WP-C behavioral additions in the same commit, contrary to CLAUDE.md's explicit "separate structural from behavioral changes" invariant. The rationale ("the refactor has no value on its own") is the canonical justification for this pattern and is specifically what the invariant is designed to resist. WP-D and WP-E BUILD sessions should not inherit this as precedent. If a structural change is needed in service of a WP-D/E feature, it should ship in a prior commit with a `refactor:` prefix, even if the refactor has no user-visible value standalone.

**Advisory 6 — `AuditDiagnostic` exclusion should be documented explicitly (carry-forward from ARCHITECT Advisory 3, low).**

The ARCHITECT snapshot named the gap: whether `AuditDiagnostic` events are included or excluded from the orchestrator-context structured observation is unspecified. WP-C's implementation implicitly excludes them (they are not CalibrationSignal events and do not pass the type check). The exclusion should be made explicit in the sink module's `consume()` method — either as a docstring note naming excluded event types or as an explicit `isinstance(event, AuditDiagnostic)` check with a comment. This closes ARCHITECT Advisory 3 explicitly and prevents WP-E or PLAY engineers from assuming the exclusion is a bug rather than a policy.
