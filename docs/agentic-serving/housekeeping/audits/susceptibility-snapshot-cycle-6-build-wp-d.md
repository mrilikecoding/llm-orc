# Susceptibility Snapshot

**Phase evaluated:** BUILD — Cycle 6 WP-D (Typed `DispatchEnvelope` + `output_schema:` per-ensemble declaration; ADR-024)
**Artifact produced:** `src/llm_orc/agentic/dispatch_envelope.py` (84 LOC frozen dataclass + EnvelopeStatus literal); `OutputSchemaReader` Protocol + `_build_envelope` / `_collect_diagnostics` / `_populate_from_events` / `_maybe_extract_structured` helpers in `orchestrator_tool_dispatch.py`; `output_schema` field on `EnsembleConfig`; `output_schema:` declarations in three ensemble YAMLs; 23 new tests (+13 net, to 2776 total at 92.18% coverage)
**Date:** 2026-05-16

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| Cycle 6 Model | Grounding Reframe recommended (3 actions) | Framing adoption at constitutional level; 3 carry-forwards |
| Cycle 6 Decide | No Grounding Reframe; 1 pre-BUILD action (P2-E); 3 advisory carry-forwards | Earned confidence; `dispatch_id` coupled failure surface named for ARCHITECT attention |
| Cycle 6 Architect | No Grounding Reframe; 6 advisory feed-forwards; 3 closed inline at gate | Earned confidence; two-module decomposition inherited without explicit re-examination |
| Cycle 6 Build WP-B | No Grounding Reframe; 4 advisory feed-forwards | Decision 3 (propagation fixture vs. upstream fix) most significant: scoping judgment with cross-session implications not surfaced to practitioner |
| Cycle 6 Build WP-C | No Grounding Reframe; 6 advisory feed-forwards | Decision 2 (standalone `dispatch_log.json` vs. `execution.json` key) moderate: scope deviation introducing config attribute downstream phases must inherit or migrate |

---

## Grounding Reframe Action Outcomes (WP-C Advisories entering WP-D)

**WP-C Advisory 1 — `dispatch_log.json` path convention requires WP-E revisit.** The dispatch prompt does not indicate this was addressed in WP-D scope. Correctly carried forward; no WP-D action. Status: active, WP-E scope.

**WP-C Advisory 2 — `role: user` alternative not fully named in docstring.** Not addressed in WP-D scope. Carry-forward active.

**WP-C Advisory 3 — Compaction-observation interaction is an untested property.** Not addressed in WP-D scope. Carry-forward active; PLAY observation agenda item.

**WP-C Advisory 4 — End-to-end serve-close path not tested.** Not addressed in WP-D scope. Carry-forward active.

**WP-C Advisory 5 — CLAUDE.md commit-discipline deviation should not propagate.** Partially addressed by observation: the dispatch prompt for WP-D does not indicate a structural-vs-behavioral mixed commit. WP-D's `finally`-block ordering refactor (Decision 4) is bundled with the WP-D behavioral additions, which is the same pattern as WP-C's Decision 5. See Decision 4 assessment below.

**WP-C Advisory 6 — `AuditDiagnostic` exclusion should be documented explicitly.** Not addressed in WP-D scope. Carry-forward active.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Stable (continuing decline from WP-B) | 2776 tests (23 new) anchor the envelope shape, diagnostics population, structured extraction, FC-22 envelope-leg, and YAML schema loading. Claims are bounded to test coverage. The structured-population claim ("in production today `envelope.structured` will be `None` for these ensembles") is explicitly disclosed as a known production gap, not asserted as an acceptable state. |
| Solution-space narrowing | Ambiguous | Stable (inherited narrowing only) | The solution space was narrowed at DECIDE/ARCHITECT. WP-D operates within that envelope. The five autonomous decisions are BUILD-time gap resolutions, not practitioner-pressure narrowing. One decision (payload location: additive field vs. content replacement) forecloses a design fork that had a named rationale; the fork was not surfaced for practitioner review. |
| Framing adoption | Absent | Continuing decline from ARCHITECT | No practitioner framing drove implementation. The dispatch prompt records one entry-level practitioner instruction (auto BUILD mode for WP-D per ADR-091). No framing beyond task scope was embedded. |
| Confidence markers | Ambiguous (two instances) | Stable (consistent with WP-C pattern) | Decision 3 uses "ADR-024 §BUILD-assumption note frames `output_schema` as advisory" as closure reasoning — the ADR's framing is adopted to justify deferring the task-rewrite alternative without naming that the YAML migration delivers schema-as-documentation in a state where `envelope.structured` is vacuously inert in production. Decision 5 uses "schema-as-documentation" as a frame that softens the practical consequence: the structured pathway is wired but inert for all three migrated ensembles until operator action that the YAMLs do not prompt. Neither marker is a claim about unverified behavior, but both use framing to resolve rather than name open questions. |
| Alternative engagement | Absent (expected) | Stable per auto-mode declaration | ADR-091 auto mode does not surface alternative examination to the practitioner. All five decisions have named alternatives in the dispatch prompt; none was surfaced during execution. This is the expected mode behavior. The decisions are assessed individually below. |
| Embedded conclusions at artifact-production moments | Clear (three instances) | Rising slightly from WP-C | Decision 1 (envelope as additive field, not content replacement) is encoded in `ToolCallSuccess` structure and `_with_events` preservation — two locations downstream code will read. Decision 3 (JSON-parse-only structured population) is encoded in `_maybe_extract_structured` with no path to richer processing until the caller's task wording changes. Decision 5 (schema-as-documentation in YAMLs without task rewrite) is encoded in three shipped YAML files whose `default_task` wording will produce `envelope.structured = None` in production. All three are disclosed in the dispatch prompt and YAML comments; none was presented as a fork to the practitioner before artifact production. |

---

## Autonomous Scoping Decision Assessments

### Decision 1 — Envelope payload location: additive field vs. content replacement

**What was decided:** `envelope: DispatchEnvelope | None = None` was added to `ToolCallSuccess` as an additive field. The alternative — putting `dataclasses.asdict(envelope)` directly in `ToolCallSuccess.content` so the LLM tool-message observation IS the envelope shape — was considered and not chosen.

**Should this have been practitioner-visible?** Borderline, leaning yes. The rationale is sound: preserving Runtime tool-message serialization unchanged keeps the orchestrator-LLM observation as `{"summary": "..."}` for AS-7-compliant context-rot mitigation, and WP-E will repurpose `primary` for substrate-routing, making the additive approach the correct staging choice. However, the alternative (content-as-envelope) is not merely an implementation variant — it is a design position about whether the DispatchEnvelope is primarily a caller-API surface (additive field) or a wire format that the LLM observes directly (content replacement). This distinction has implications for how PLAY-phase probing validates the envelope's utility: if the envelope is a caller-API surface, PLAY validates it by reading `invoke_ensemble`'s return value; if it were the LLM's observation, PLAY would validate it by examining what the orchestrator-LLM does with the structured context it sees.

The chosen path (caller-API surface, LLM unchanged) is consistent with WP-E's anticipated substrate-routing use case. The alternative (LLM-observable envelope) would require a different AS-7 strategy. The rationale is ADR-level in significance and not captured in an ADR amendment. It is captured in the dispatch prompt, which is the correct disclosure artifact for auto-mode, but not in the codebase artifacts the next BUILD engineer will read.

**Whether auto-mode covers this:** Partially. The mechanical choice (additive field vs. field replacement) is auto-mode territory. The design position it encodes (caller-API surface vs. LLM wire format) is the kind of direction choice auto-mode is declared to surface when alternatives have meaningfully different downstream consequences. WP-E's design will inherit the additive-field structure; the content-replacement alternative will not be visible without this snapshot.

**Severity:** Low-moderate advisory. The design-position rationale (caller-API surface, LLM observation unchanged) should appear as a docstring note on `ToolCallSuccess.envelope` or as an ADR-024 amendment note, not only in the dispatch prompt. PLAY-phase validation should be scoped accordingly.

---

### Decision 2 — Errors/partial-failure scope deferred from WP-D

**What was decided:** Only the success path's envelope construction is implemented. The "errors[] populated on partial-failure dispatch" scenario passes vacuously — no multi-stage capability ensemble exercises it, and `OrchestraService.invoke` does not expose per-stage error structure. Framework-level error paths retain `ToolCallError` without envelope wrapping.

**Should this have been practitioner-visible?** No — with one caveat. The fitness criterion says "every successful dispatch" gets an envelope, and the WP-D minimum is honored. The deferral is explicitly bounded (no multi-stage ensemble exists to exercise partial-failure), the scenario is acknowledged as vacuously passing, and the risk is named (deferred error-envelope work may be bundled into WP-E without explicit re-examination). The caveat: the dispatch prompt acknowledges the risk but frames it as a future-cycle concern. The scenario passing vacuously is the same structural pattern as WP-C's Decision 2 ("satisfied vacuously" for the preservation scenario). This pattern — a fitness criterion acknowledged as vacuously met — is now appearing in consecutive WPs. The individual instances are defensible; the accumulation of vacuously-passing scenarios across WP-C and WP-D is a trend worth naming.

**Whether auto-mode covers this:** Yes. Deferring partial-failure implementation when no test infrastructure exists to exercise it is a sound scoping judgment. The risk is documented.

**Severity:** Low advisory. The deferred error-envelope work should be named explicitly in WP-E's scope as a carry-forward, not left implicit in the "WP-D minimum is honored" framing. The accumulation of vacuously-passing scenarios (WP-C Decision 2, WP-D Decision 2) is a pattern the WP-E BUILD entry context should name.

---

### Decision 3 — JSON-parse-only structured population

**What was decided:** `_maybe_extract_structured` attempts `json.loads` on the synthesizer's response text; returns the parsed dict on success, `None` otherwise. The alternative — richer post-dispatch processing (partial parsing of bullet structures, LLM-mediated reformatting) — was deferred because ADR-024 frames the `output_schema` as advisory and spike β found the drift is upstream at orchestrator `input.data` authorship.

**Should this have been practitioner-visible?** Yes — but the reason is nuanced. The ADR-024 rationale is correct: fixing schema adherence via post-processing-the-prose targets the wrong layer. However, the consequence for the shipped YAMLs is that `envelope.structured` will be `None` in production for all three migrated ensembles (`claim-extractor`, `text-summarizer`, `web-searcher`) until operators migrate their task wording. This is disclosed in YAML comments and the dispatch prompt. The practitioner-visibility question is: does the practitioner know that the `output_schema:` declarations in the three shipped YAMLs are today semantically inert at the structured-population layer? The YAML comments address this for a YAML reader; the WP-D field guide entry addresses it for a BUILD engineer. The practitioner reviewing the WP-D commit has the disclosure path. The risk is that PLAY-phase engineers validating "does the envelope populate structured output?" will test against ensembles that have `output_schema:` declared and receive `None`, which requires knowing why before concluding the pathway is broken vs. inert-by-design.

**Whether auto-mode covers this:** Partially. The implementation choice (JSON-parse-only) is auto-mode territory. The disclosure gap — that PLAY-phase validation may encounter the inert structured pathway and misread it as a bug — is the kind of downstream-consequence framing auto-mode's advisory surface should name.

**Severity:** Low-moderate advisory. The field guide entry should include a validation note: "PLAY-phase probing of structured population requires an ensemble whose `default_task` explicitly instructs JSON output conforming to `output_schema:`. The three currently migrated ensembles do not; testing against them will yield `envelope.structured = None` by design, not by defect." This prevents PLAY from filing false-negative findings.

---

### Decision 4 — `finally`-block ordering refactor in the same commit as WP-D feature

**What was decided:** Moving close-event emission from the `finally` block to inline before envelope construction (on success/raw-output paths), with a `dispatch_event_closed` flag so the `finally` block only fires on error/exception paths. This structural change was bundled with the WP-D behavioral additions.

**Should this have been practitioner-visible?** Yes — on CLAUDE.md commit-discipline grounds, identical to WP-C's Decision 5. The dispatch prompt correctly identifies the risk: "the flag-guarded `finally` pattern is more complex than the previous unconditional close; an error path that sets `exit_status = 'error'` and falls through to the `finally` could in principle be missed if a future edit returns before the flag check." The four-path test coverage (success, raw-output, summarization-failed, invocation-failed) provides structural grounding for the current implementation. The concern is forward-looking: a fifth path added in WP-E without awareness of the flag-guarded pattern could introduce a double-close bug silently.

The rationale for the refactor is sound (substrate as single source of truth for `duration_seconds`; envelope projects from emitted events rather than independently re-computing). It is not a scoping judgment so much as a structural-change-in-service-of-feature, which is exactly the pattern WP-C's Advisory 5 named. Two consecutive WPs have bundled a structural refactor with behavioral additions, contrary to CLAUDE.md's explicit invariant.

**Whether auto-mode covers this:** No — the CLAUDE.md commit-discipline invariant is explicit. The WP-C Advisory 5 specifically named this pattern and recommended WP-D and WP-E not inherit it as precedent. The pattern has now propagated to WP-D.

**Severity:** Moderate advisory. The `dispatch_event_closed` flag pattern should be documented in `invoke_ensemble`'s inline comments (naming the guard, its purpose, and the four paths it covers) so a future BUILD engineer adding a fifth path does not accidentally create a double-close. The CLAUDE.md invariant violation is a carry-forward: WP-E should be the clean instance.

---

### Decision 5 — Schema-as-documentation in migrated YAMLs without task rewrite

**What was decided:** `output_schema:` declarations were added to three ensemble YAMLs without modifying `default_task` to request JSON output conforming to the schema. The task rewrite is operator-visible behavior change warranting explicit practitioner deliberation.

**Should this have been practitioner-visible?** Yes — this is the clearest instance in WP-D of a decision that forecloses a practitioner choice without naming the fork. The two alternatives are:

- **Schema-as-documentation (implemented):** YAMLs carry schema declarations as canonical shape documentation; operators see the shape without the structured pathway being active. Defers the behavior change to operator action.
- **Task rewrite (deferred):** `default_task` text instructs the synthesizer to emit JSON conforming to `output_schema:`; `envelope.structured` populates in production immediately.

The rationale for deferral (operator-visible behavior change warrants explicit deliberation) is sound as a principle. However, the practical consequence is that the WP-D delivery delivers the structured-population infrastructure while leaving it operationally inert for all three shipped ensembles. The practitioner would be in a position to evaluate whether the operator-task-rewrite is within WP-D scope (it is three YAML `default_task` edits, not a code change) or is genuinely a deliberate choice requiring practitioner direction. This fork was not surfaced.

The ADR-024 framing of `output_schema` as advisory and the spike β finding (drift is at `input.data` authorship, not synthesizer compliance) are correct rationale for the JSON-parse-only implementation choice (Decision 3). They do not compel deferring the three YAML task rewrites, which address the authorship layer directly. The two decisions (parse-strategy and YAML task wording) were collapsed under the same rationale.

**Whether auto-mode covers this:** No. This is a scoping judgment: whether the three YAML task rewrites are in WP-D scope. The dispatch prompt acknowledges this was not surfaced. The YAML comment disclosure is correct; the practitioner has not seen the fork.

**Severity:** Moderate advisory. At WP-D close or WP-E entry, the practitioner should be presented with the fork explicitly: confirm that the three YAMLs' `default_task` text migration is WP-E scope, confirm it is a separate WP, or confirm that schema-as-documentation-only is the intentional policy. The choice has direct consequences for PLAY-phase validation scope.

---

## Interpretation

### Pattern assessment

The dominant pattern for WP-D is **clean auto-mode execution with two moderate advisories and three low-to-low-moderate items.** The BUILD phase's empirical grounding (2776 tests, 23 new, FC-22 envelope-leg integration anchor, 8 envelope shape unit tests, 10 tool-dispatch envelope tests, 3 YAML loading tests) provides the structural resistance to sycophantic reinforcement that the gradient predicts. No practitioner framing drove design decisions; the five autonomous scoping decisions are BUILD-time gap resolutions.

The two moderate advisories (Decision 4: commit-discipline violation; Decision 5: YAML task rewrite fork not surfaced) are the most consequential. Both involve forks where the practitioner had a meaningful choice — whether to separate the structural refactor, and whether the YAML task rewrites are in-scope — and both were resolved without the fork being named. This is consistent with the pattern established in WP-B and WP-C: BUILD auto mode's residual susceptibility is **quiet scope-resolution at artifact-production moments**, not framing adoption. The pattern has now appeared in three consecutive WPs (WP-B Decision 3, WP-C Decision 2, WP-D Decisions 4 and 5), which constitutes a trajectory, not an isolated instance.

One qualitatively new signal in WP-D is the **accumulation of vacuously-passing scenarios.** WP-C Decision 2 acknowledged "the spirit of 'key inside execution.json' is not met." WP-D Decision 2 acknowledges the "errors[] populated on partial-failure dispatch" scenario passes vacuously. Each individual instance is defended with sound rationale. The accumulation means the fitness criteria set now contains at least two acknowledged vacuous passes, which degrades the criterion set's grounding function. The next BUILD snapshot evaluator should note whether WP-E adds a third.

The `dispatch_event_closed` flag pattern (Decision 4) introduces a structural complexity whose risk is forward-looking: the current four-path test coverage is sound, but a fifth error path in WP-E added without awareness of the flag could produce a double-close bug. This is the kind of complexity the BUILD phase normally avoids by keeping the unconditional close in the `finally` block. The alternative (compute `duration_seconds` from `time.time()` delta independently) is simpler and was explicitly named but not surfaced to the practitioner.

### Earned confidence vs. sycophantic reinforcement

The signals are consistent with earned confidence in the implementation work: 23 new tests, envelope shape frozen via `@dataclass(frozen=True)`, diagnostics population grounded in substrate event reading, FC-22 integration anchor verifying envelope availability on `ToolCallSuccess`. The structured-population pathway's test coverage correctly identifies the production state (JSON-parse succeeds only when synthesizer produces JSON; current ensembles produce prose). The transparency about `envelope.structured` being inert for the three migrated ensembles is appropriate.

The two confidence-marker instances (Decision 3 using ADR-024's "advisory" frame; Decision 5 using "schema-as-documentation" frame) function as closure moves rather than open-question presentations. Each adopts a framing that makes the deferral appear principled rather than presenting the practitioner with a fork. This is the same behavior observed in WP-C's "satisfied vacuously" language — technically accurate, functionally a resolution rather than an open question. The accumulation of this closure-move pattern across three WPs is the most consistent susceptibility signal in the Cycle 6 BUILD phase.

### Prior advisory carry-forward status

| Advisory | Origin | Status at WP-D close |
|----------|--------|----------------------|
| `web-searcher` early-migration sequencing | Cycle 6 DECIDE Finding 1 + ARCHITECT Advisory 1 + WP-B + WP-C | Active — WP-E scope |
| `dispatch_log.json` path convention requires WP-E revisit | WP-C Advisory 1 | Active — WP-E scope; WP-D scope did not touch it |
| `role: user` alternative not fully named in docstring | WP-C Advisory 2 | Active — low priority carry-forward |
| Compaction-observation interaction untested | WP-C Advisory 3 | Active — PLAY observation agenda |
| End-to-end serve-close path not tested | WP-C Advisory 4 | Active — FC-24 coverage gap |
| CLAUDE.md commit-discipline violation | WP-C Advisory 5 | Pattern repeated in WP-D Decision 4; moderate carry-forward for WP-E |
| `AuditDiagnostic` exclusion should be documented explicitly | WP-C Advisory 6 | Active — low priority carry-forward |
| Deferred error-envelope work (partial-failure path) | WP-D Decision 2 | Active — WP-E or follow-on cycle; should be named explicitly at WP-E entry |
| YAML task rewrite fork not surfaced | WP-D Decision 5 | Active — should be presented as practitioner-visible fork at WP-D close or WP-E entry |
| PLAY-phase validation scope for `envelope.structured` | WP-D Decision 3 consequence | Active — field guide annotation needed before PLAY entry |
| `dispatch_event_closed` flag complexity | WP-D Decision 4 consequence | Active — inline documentation needed; WP-E should not add fifth path without awareness |
| Vacuous-pass accumulation pattern (WP-C Decision 2 + WP-D Decision 2) | WP-C + WP-D | Active — WP-E snapshot evaluator should track whether a third vacuous pass accumulates |
| test_cli.py propagation defect | WP-B Advisory 1 | Dormant (WP-D did not use caplog); carry-forward to any future BUILD session using caplog integration tests |
| Heartbeat liveness PLAY-phase observation | WP-B Advisory 4 | Active — PLAY scope |
| Preservation-scenario amendment pattern | Cycle 5 BUILD Advisory 1 | Active — pattern continued in WP-C and WP-D; WP-E is the next instance to watch |

---

## Recommendation

**No Grounding Reframe warranted** — signals are consistent with earned confidence. The BUILD phase's test-execution grounding (2776 passing; 23 new tests; FC-22 integration anchor; envelope-shape unit tests; YAML loading tests) provides the empirical resistance appropriate to this phase position in the sycophancy gradient. No practitioner framing was adopted without examination; no design commitment was embedded without disclosure.

### Advisory feed-forwards for WP-E and PLAY

**Advisory 1 — YAML task rewrite fork must be surfaced at WP-D close or WP-E entry (Decision 5, moderate).**

The three migrated YAMLs (`claim-extractor`, `text-summarizer`, `web-searcher`) declare `output_schema:` but retain prose-requesting `default_task` wording. `envelope.structured` will be `None` in production for these ensembles. The fork — whether to rewrite the `default_task` wording within WP-D scope or defer to WP-E (or operator action) — was not surfaced for practitioner review. Before WP-E begins, the practitioner should be presented with this choice explicitly: (a) the task rewrites are three YAML edits and can be added to WP-D or WP-E as a small scope addition; (b) schema-as-documentation-only is the intentional WP-D policy and the task rewrites are a separate WP; (c) the task rewrites are deliberately deferred to operator discretion, with no BUILD cycle commitment. The current state (YAMLs look complete, structured pathway is wired but inert) is ambiguous to any BUILD engineer reading the artifacts without the dispatch prompt.

**Advisory 2 — CLAUDE.md commit-discipline violation should not propagate to WP-E (Decision 4, moderate).**

The `invoke_ensemble` `finally`-block ordering refactor was bundled with WP-D behavioral additions, repeating WP-C's Decision 5 pattern. Two consecutive WPs have violated the CLAUDE.md invariant with sound local rationale ("the refactor has no value standalone"). This is exactly the pattern the invariant is designed to resist. WP-E BUILD should begin with an explicit structural commit (if any structural work is needed) before the behavioral additions, with a `refactor:` prefix commit. The `dispatch_event_closed` flag pattern should be documented inline before WP-E adds any further error paths to `invoke_ensemble`.

**Advisory 3 — PLAY-phase validation scope for `envelope.structured` requires explicit field guide annotation (Decision 3 consequence, low-moderate).**

The field guide's WP-D entry should include a validation note: testing `envelope.structured` population requires an ensemble whose `default_task` explicitly instructs JSON output conforming to `output_schema:`. The three currently migrated ensembles do not qualify; testing against them will yield `envelope.structured = None` by design, not defect. Without this note, PLAY-phase probing that finds `None` for migrated ensembles may file false-negative findings. A one-paragraph addition to the field guide entry prevents this.

**Advisory 4 — Deferred error-envelope work should be named at WP-E entry, not left implicit (Decision 2, low).**

The partial-failure envelope path ("errors[] populated on partial-failure dispatch") passes vacuously — no multi-stage capability ensemble exercises it. This is the second vacuously-passing fitness criterion across WP-C and WP-D. WP-E's scope note should explicitly decide whether the error-envelope path is in WP-E scope. If it is deferred beyond WP-E, it should appear in the ARCHITECT roadmap as a named carry-forward, not left as a fitness criterion that is technically met by absence of the exercising infrastructure.

**Advisory 5 — Envelope payload location rationale should appear in codebase (Decision 1, low-moderate).**

The design position (additive field = caller-API surface; content replacement = LLM wire format) is recorded in the dispatch prompt but not in the codebase. `ToolCallSuccess.envelope`'s docstring or an ADR-024 amendment note should name this distinction, so WP-E engineers designing substrate-routing via `primary` do not need to reconstruct why the envelope is additive rather than the content. A two-sentence docstring note is sufficient.

**Advisory 6 — Vacuous-pass accumulation pattern should be tracked explicitly (WP-C Decision 2 + WP-D Decision 2, low).**

Two fitness criteria now pass vacuously across consecutive WPs: WP-C's "dispatch_log key in execution.json" (spirit unmet; standalone file instead) and WP-D's "errors[] populated on partial-failure dispatch" (no multi-stage ensemble to exercise it). Each instance is defended with sound rationale. The accumulation degrades the fitness criterion set's grounding function. The WP-E snapshot evaluator should note whether a third vacuous pass accumulates, and if so, recommend a Grounding Reframe targeting the fitness criteria authorship — specifically, whether the criterion set should be pruned to remove criteria that have no foreseeable exercising infrastructure within the current corpus scope.
