# Argument Audit Report — Cycle 6 DECIDE

**Audited documents:**
- `docs/agentic-serving/decisions/adr-022-routing-surface-behavior.md`
- `docs/agentic-serving/decisions/adr-023-observability-event-routing.md`
- `docs/agentic-serving/decisions/adr-024-common-io-envelope.md`
- `docs/agentic-serving/decisions/adr-025-artifact-as-substrate.md`

**Source material read:**
- `docs/agentic-serving/housekeeping/cycle-status.md` (Spike findings subsection)
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-alpha-envelope-survey.md`
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-beta-composition-predictability.md`
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-gamma-routing-characterization.md`
- `docs/agentic-serving/domain-model.md`
- `docs/agentic-serving/product-discovery.md`
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md`
- `docs/agentic-serving/decisions/adr-004-result-summarization-mandatory.md`
- `docs/agentic-serving/decisions/adr-019-skill-framework-agnostic-capability-library.md`

**Genre:** ADR set

**Date:** 2026-05-15

**Cycle:** 6 (agentic-serving mini-cycle; post-MODEL DECIDE)

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR set (four new ADRs)
- **Argument chains mapped:** 8 (one Decision per ADR + four cross-ADR composition chains)
- **Issues found:** 8 (P1: 1, P2: 4, P3: 3)

---

### Per-ADR Breakdown

#### ADR-022 — Routing Surface Behavior

**Decision chain:** Spike γ's four cells show neither tested orchestrator routes to `invoke_ensemble` under NL framing → the system prompt lacks a `prefer invoke_ensemble when capability match exists` clause → amending the system prompt is the operative design surface → joint codification of dispositions (ii) and (iii) via amendment + acknowledged cross-profile variability.

The chain is internally coherent. Spike γ is the primary driver, and the ADR credits it specifically for the three findings that load the Decision. The Provenance check correctly distinguishes driver-derived elements from drafting-time synthesis (the joint codification; the per-profile deferral framing; the narrowing of ADR-021's NL clause wording). The Rejected alternatives section engages each option with substantive argument rather than summary dismissal.

One element requires attention (P2 below): the ADR's claim that the qwen3:14b Cell B result "strengthens disposition (iii)" in the same paragraph it claims Cell B "over-delegated to client tools" is correct, but the ADR elides a potentially consequential nuance from spike γ's Cell B continuation finding. Spike γ found that the two tested orchestrators diverged in *opposite directions* — MiniMax M2.5-free under-delegated (completed inline) while qwen3:14b over-delegated (client-tool delegation). This means the amendment's effect on qwen3:14b may not simply be "uncertain" — it may actively make qwen3:14b behavior *worse* for the routing surface, because an instruction to prefer `invoke_ensemble` adds a third competing routing instruction to a model that was already ignoring the existing prompt's precedence in the direction of over-delegating to client tools. The Consequences §Negative acknowledges uncertain effectiveness but does not explore this directional risk.

#### ADR-023 — Observability Event-Routing

**Decision chain:** DISCOVER Inversion N+2 establishes one emission substrate, two destinations → existing typed events substantially cover operator-terminal needs but lack dispatch timing → MODEL Action C surfaces three architectural alternatives → ADR-023 selects alternative (i) event-model extension, rejects (ii) sidecar and (iii) orchestrator-context-only → bounded extension: `DispatchTiming` event + `dispatch_id` correlation identifier.

The chain is sound. Inversion N+2 is a driver-derived framing (DISCOVER snapshot Action 3), not a drafting-time assumption, and the Provenance check records it as such. The rejection of alternatives (ii) and (iii) applies Inversion N+2 consistently: both alternatives create separate emission infrastructure for dispatch timing, violating the one-substrate commitment. The sidecar-log rejection is the strongest of the three: it shows that the operator-terminal destination also needs dispatch timing (Cell A-explicit's 61s code-generator dispatch had no duration line in console output), which means the sidecar approach would either produce two parallel channels or leave the operator-terminal destination without timing.

The `dispatch_id: None` transition path (P3 note below) is honest about the BUILD cost.

One structural question surfaces (P2 below): the orchestrator-context destination's "in-turn context inclusion" pattern is specified as routing events "between turns" — specifically "when the orchestrator's next turn begins, after a dispatch returns control." This is consistent with the ReAct loop shape. But the ADR does not specify what happens when the orchestrator's next turn is the *final* turn (the last reply in the session before the client disconnects). In that case, the in-turn structured observation would be assembled but the orchestrator would have no reasoning turns left to consume it. The end-of-session summary in `execution.json` covers post-hoc review, but the in-turn routing's timing assumption has an edge case the ADR does not address.

#### ADR-024 — Common I/O Envelope

**Decision chain:** Spike α found a uniform de facto envelope exists across all 8 ensembles → spike β found output-spec drift's mechanism is `input.data` override, not synthesizer deviation → spike α recommended candidate B (additive typed fields) as lowest-disruption path → ADR-024 codifies candidate B as the `DispatchEnvelope` typed contract → advisory `output_schema:` opens composition predictability without mandatory enforcement.

The chain is coherent and correctly applies the per-finding test MODEL Action B required. The rejection of candidate A (thin convention) is accurate: A addresses none of the three T16 findings. The rejection of candidate B-strong (mandatory schema enforcement) correctly applies spike β's finding that the drift is upstream of the synthesizer, making synthesizer-output enforcement catch the wrong layer.

One substantive gap is present (P2 below): the `diagnostics` field is specified to always contain `dispatch_id`, `duration_seconds`, `model_profile`, `tier`, `topaz_skill`, and `calibration_verdict`. But the ADR specifies that BUILD progressively migrates ensembles to the typed envelope, starting with `DispatchEnvelope` as the return value of `invoke_ensemble`. During the transition period — before ADR-023's `dispatch_id` field is added to all four existing event types — what values populate `diagnostics.dispatch_id` and `diagnostics.duration_seconds`? The ADR specifies that `dispatch_id` is available from ADR-023's correlation identifier, but the sequencing of the two ADRs' BUILD implementation is not specified. If ADR-024's envelope lands before ADR-023's event extension, `diagnostics.dispatch_id` and `diagnostics.duration_seconds` would be unpopulated or require fallback logic. This is a BUILD-sequencing dependency not surfaced in either ADR.

The "rejection of routing envelope through dispatch-event substrate" alternative is correctly reasoned: the envelope is a response-shape contract; the events are an observability stream; they compose via `dispatch_id` rather than merging. This is sound separation of concerns.

#### ADR-025 — Artifact-as-Substrate

**Decision chain:** Practitioner-verbatim "always" scope → spike β's per-finding test shows the three-findings-collapse claim does not hold as a single mechanism → spike β corrects to: B addresses drift, C addresses information-finding overhead and AS-7 stripping → spike γ Cell A-explicit surfaces a fourth defect (orchestrator-narration substitution, downstream of AS-7) → always-scope chosen over substantive-deliverable or operator-configured alternatives → size-floor tradeoff deliberately accepted → AS-7 amended to default-with-conditional-skip.

The chain is the most compositionally complex of the four ADRs, and it is substantially well-reasoned. The correction of the three-findings-collapse claim (from single-mechanism to two-ADR coverage) is explicit and honest. The fourth defect enumeration from spike γ Cell A-explicit is correctly sourced.

The most significant audit question concerns the AS-7 amendment (P1 below). The Amendment's framing — "default-with-conditional-skip for substrate-routed ensembles" — is principled within the context of capability ensembles. But the backward propagation sweep targets listed (ADR-007, ADR-014) carry calibration-gate logic that references result summarization as a pre-context-injection step. ADR-025 asserts that substrate-routing "dissolves" the context-rot concern at the substrate layer. This is correct for substantive-deliverable ensembles. But for the calibration gate's operation on dispatched ensembles: the Calibration Gate (ADR-007) evaluates ensemble output quality. Under substrate-routing, the calibration gate would need to evaluate the *artifact* (or its summary reference), not the inline content. ADR-025 does not address whether the calibration gate's evaluation surface changes under artifact-as-substrate, or whether the critic/evaluator agents in the ensemble themselves receive the artifact path rather than the deliverable content. If the calibration gate evaluates the `primary` summary-line (e.g., "Wrote 24-line CircularBuffer class to artifacts/...") rather than the deliverable content, the gate's quality evaluation changes significantly in character — it is evaluating metadata, not substance. This interaction is not addressed in either ADR-025 or the backward propagation sweep's consequences.

The Rejected alternatives section engages the substantive-deliverable scope with a genuine argument (absence of a quantitative threshold for "substantive"; practitioner preference for always-scope as the cleaner design rule; BUILD/PLAY evidence deferred). This is defensible. The operator-configured alternative rejection is also sound. The "defer to BUILD" rejection is the strongest of the three, correctly citing Step 3.7's propagation discipline.

---

### Cross-ADR Composition Findings

The four ADRs compose along three integration seams:

**Seam 1: ADR-022 (system prompt) + ADR-023 (event routing)**
These compose cleanly. The system-prompt amendment in ADR-022 is a prompt-layer change; ADR-023's event substrate is a dispatch-layer change. The two don't conflict. ADR-023's `DispatchTiming(phase="start")` / `(phase="end")` events capture the timing of whatever ensemble dispatch ADR-022's amendment produces, whether or not it routes via the amended clause.

**Seam 2: ADR-023 (event routing) + ADR-024 (envelope)**
These compose via `dispatch_id`. The envelope's `diagnostics.dispatch_id` correlates to ADR-023's events for a dispatch. The BUILD-sequencing dependency noted above (P2 on ADR-024) lives at this seam. Structurally the composition is correct; the dependency surfacing is a BUILD concern.

**Seam 3: ADR-024 (envelope) + ADR-025 (artifact-as-substrate)**
These compose as designed: ADR-024 provides the `artifacts[]` field structure; ADR-025 specifies when that field is populated. The `primary` field's meaning shifts between the two ADRs — ADR-024 specifies `primary` as always a string (content or reference summary), and ADR-025 specifies that for substrate-routed ensembles `primary` is a summary-line referencing the artifact. This is consistent; ADR-024's specification is explicitly anticipatory of this use ("For substrate-routed ensembles per ADR-025, this is an artifact reference summary line").

**Seam 4: ADR-025 + ADR-004 / AS-7 amendment**
This is the seam that produces the P1 finding below. The amendment narrows AS-7's scope to inline-response ensembles. ADR-004's Consequences §Negative notes that "detail lost at summarization is recoverable only through explicit retrieval" — under substrate-routing, the detail is not lost at summarization, it persists on disk in the artifact. This is a genuine improvement. But the calibration gate's operation on substrate-routed deliverables is structurally underspecified, as noted above.

---

### P1 — Must Fix

**P1-A — ADR-025: Calibration gate evaluation surface under substrate-routing is unaddressed**

- **Location:** ADR-025 §Decision §"AS-7 amendment: default-with-conditional-skip" + backward propagation §"ADR-007, ADR-014"
- **Claim:** "substrate-routed ensembles skip content summarization at the substrate layer"; backward propagation sweep targets ADR-007 and ADR-014 for supersession notes where AS-7 is referenced unconditionally.
- **Evidence gap:** ADR-007 (Calibration Gate) and ADR-014 (in-process trajectory-level calibration) both operate on dispatched ensemble outputs. Under substrate-routing, the ensemble's deliverable is on disk; the `primary` field is a summary-line. The backward propagation sweep addresses the summarization pathway but not the quality-evaluation pathway: when the Calibration Gate's critic/evaluator agents evaluate a substrate-routed ensemble's output, do they evaluate the artifact content (requiring a file-read tool or equivalent), or the `primary` summary-line, or the `structured` field if populated, or the `artifacts[0].summary`? The character of calibration-gate evaluation changes substantially depending on this. A critic evaluating "Wrote 24-line CircularBuffer class" cannot assess whether the class is correct; a critic evaluating the artifact content can. ADR-025 does not address this interaction; the backward propagation sweep targets the summarization references but not the quality-evaluation surface. The amendment to AS-7 is correct as stated; the amendment to the calibration gate's operating surface is missing.
- **Recommendation:** Add a subsection to ADR-025's backward propagation scope explicitly addressing the calibration gate's evaluation surface under substrate-routing. Two options: (a) specify that the calibration gate reads the artifact content (a new file-read capability for the critic agent), or (b) specify that the calibration gate evaluates `artifacts[0].summary` + `structured` rather than deliverable content (a scope reduction, not a capability addition). If (b), note that quality evaluation becomes metadata-evaluation rather than content-evaluation — a tradeoff honest in the ADR's §Negative. This must be resolved before BUILD, because BUILD implements the calibration gate's dispatch path and the path's behavior depends on this specification.

---

### P2 — Should Fix

**P2-A — ADR-022: Amendment's effect on qwen3:14b may be directionally adverse, not merely uncertain**

- **Location:** ADR-022 §Decision §"Effectiveness is configuration-conditional" + §Consequences §Negative
- **Claim:** "The amendment's effectiveness under qwen3:14b is uncertain — the model's reasoning shape may continue to favor client tools even with an explicit `prefer invoke_ensemble` instruction. Uncertain expected impact."
- **Evidence gap:** Spike γ Cell B (continuation) showed qwen3:14b *over-delegating* to client tools on NL framing — a different failure mode from MiniMax's under-delegation. If the amendment adds a `prefer invoke_ensemble` instruction to the system prompt, qwen3:14b may continue to delegate to client tools because qwen3:14b was already not honoring the existing prompt's tool-precedence guidance. Worse, the amendment adds a third competing routing instruction alongside the existing "client-declared tools for filesystem actions" and the implicit "direct completion as residual" — a model that was over-delegating relative to the existing prompt now has more conflicting instructions. The framing "uncertain expected impact" is accurate but understates the directional risk: the amendment is designed to pull routing toward `invoke_ensemble`, but qwen3:14b's baseline failure mode is already in the opposite direction from MiniMax's, and the amendment doesn't obviously address that failure mode. The risk is not just "may not help" but "may not help for a structurally distinct reason that the amendment doesn't target."
- **Recommendation:** Revise the §Effectiveness paragraph and §Consequences §Negative bullet on qwen3:14b to distinguish the two failure modes: MiniMax under-delegates (direct completion instead of ensemble); qwen3:14b over-delegates (client tool instead of ensemble). The amendment targets the "prefer ensemble over direct completion" gap. It does not directly target the "prefer ensemble over client-tool delegation" gap that qwen3:14b exhibits. Note this distinction explicitly; the BUILD/PLAY characterization target should test both directions of deviation, not only whether "dispatch is restored."

**P2-B — ADR-024: BUILD-sequencing dependency between ADR-023 (`dispatch_id`) and ADR-024 (`diagnostics.dispatch_id`) is unaddressed**

- **Location:** ADR-024 §Decision §"Migration path" + §Consequences §Negative
- **Claim:** "The conversion from the de facto `execution.json` shape to the typed envelope happens at the dispatch boundary." The `diagnostics` field is specified to always contain `dispatch_id` and `duration_seconds`.
- **Evidence gap:** `dispatch_id` is introduced by ADR-023 as a new correlation identifier on existing event types. `duration_seconds` comes from ADR-023's `DispatchTiming` event. ADR-024's migration path specifies that `execution.json` retains its existing shape and the envelope is constructed at the dispatch boundary — but the boundary construction of `diagnostics.dispatch_id` and `diagnostics.duration_seconds` requires ADR-023's event extension to be in place. If ADR-024's envelope ships first (BUILD may implement ADRs in any order), `diagnostics.dispatch_id` would have no value source. Neither ADR specifies a sequencing dependency or a fallback for the interim state.
- **Recommendation:** Add a migration-path note in ADR-024 specifying that `diagnostics.dispatch_id` and `diagnostics.duration_seconds` require ADR-023's event extension to be deployed first, or specify interim fallback values (`dispatch_id: None`, `duration_seconds: None`) analogous to the `dispatch_id: None` pattern ADR-023 uses for its own progressive conversion. The issue is minor in consequence but ensures BUILD does not discover an implicit dependency mid-implementation.

**P2-C — ADR-023: In-turn orchestrator-context routing has an underspecified edge case at session close**

- **Location:** ADR-023 §Decision §Routing destinations §"Destination 2 — Orchestrator-context" §"In-turn context inclusion"
- **Claim:** "When the orchestrator's next turn begins (after a dispatch returns control to the orchestrator's ReAct loop), the events emitted during the just-completed dispatch are formatted as a structured observation block prepended to the next turn's context."
- **Evidence gap:** When the dispatch is the final operation before the orchestrator produces its last turn and the session closes, the "next turn" may be the response turn returned to the client. In that case, the structured observation is assembled but the orchestrator uses it only for that final response turn — which is correct and desirable. However, the ADR does not address the case where the dispatch itself *is* the final operation (i.e., the orchestrator dispatches and then terminates the session without a subsequent reasoning turn). Under ADR-021's per-capability dispatch contract, the orchestrator returns the ensemble's output to the skill framework as the chat completion response; if the orchestrator's session ends at that point, there is no "next turn" to prepend the observation to. The end-of-session summary in `execution.json` covers the persistent record, but the in-turn routing pattern's semantics at the session-termination boundary are underspecified.
- **Recommendation:** Add a clarifying sentence specifying how the in-turn context routing behaves when the dispatch is the session's final operation: either (a) the observation is included in the final response turn if the orchestrator assembles one, or (b) the observation is omitted for the session-terminal dispatch and the end-of-session summary in `execution.json` is the authoritative record. Either (a) or (b) is acceptable; the ADR should specify which.

**P2-D — ADR-025: The "dial back later if cumbersome" disposition for the always-scope acceptance is structured as a framing hedge rather than a falsification criterion**

- **Location:** ADR-025 §Decision §"Scope: always, for capability ensembles" + §Consequences §Negative
- **Claim:** "The 'dial back later if cumbersome' disposition is structurally honest: BUILD's first deployments and the post-BUILD PLAY cycle will surface whether the always-scope produces operational friction; if so, a follow-on cycle refines the scope to *when substantive* with explicit per-ensemble criteria."
- **Evidence gap:** Spike α's size-floor finding identified specific candidate ensembles where always-scope is "structurally awkward": `web-searcher` (JSON list of URL+snippet records — already-handle-shaped), short `text-summarizer` invocations. The ADR accepts the always-scope and defers the size-floor question to BUILD/PLAY evidence. This is a legitimate decision. But the ADR does not specify what "cumbersome" means as a measurable criterion — what would BUILD/PLAY observe that would trigger the refinement? Without a falsification criterion, "dial back later if cumbersome" is a disposition that can remain perpetually deferred because "cumbersome" is never operationally defined. This is different from ADR-022's BUILD/PLAY characterization target, which specifies that spike γ cells will be re-run to check whether dispatch is restored — that is a testable outcome. The always-scope's dial-back trigger is undefined.
- **Recommendation:** Add a concrete indicator of what BUILD/PLAY evidence would constitute "cumbersome" and trigger the always-to-substantive scope refinement. Candidate criteria: operator-reported friction at the `web-searcher` boundary (e.g., sessions where the artifact path reference adds round-trip latency without value); artifact count proliferation rate that exceeds a threshold per session; or BUILD's first deployment surfacing a concrete use case where always-scope produces a session-end artifact set dominated by small-deliverable artifacts. The dial-back disposition is a sound design choice; making it actionable rather than perpetually deferrable requires at least a qualitative indicator of what would trigger it.

---

### P3 — Consider

**P3-A — ADR-022: The system prompt amendment text is specified in the Decision but its interaction with the per-orchestrator-profile override surface is mentioned only in §Out of scope**

- **Location:** ADR-022 §Decision §"Out of scope for ADR-022"
- **Observation:** The amendment's text is specified as an insertion to `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT`. ADR-011's per-session config surface allows operators to override the default system prompt via `agentic_serving.orchestrator.system_prompt`. The ADR acknowledges this in §Consequences §Neutral. A minor clarity improvement: the §Out of scope statement ("Per-orchestrator-profile system-prompt overrides — mentioned above as follow-on territory") could more precisely distinguish between operator-session-level overrides (already available; mentioned in §Neutral) and per-profile defaults (a new mechanism that would require profile-aware prompt selection). The current phrasing may conflate the two.

**P3-B — ADR-023: `CalibrationSignal` routing to the orchestrator-context destination at DEBUG level is not specified**

- **Location:** ADR-023 §Decision §Routing destinations §"Destination 1 — Operator-terminal"
- **Observation:** `CalibrationSignal` is specified to route at `DEBUG` level (suppressed unless `--verbose` or `LOG_LEVEL=DEBUG`) to the operator-terminal destination. The ADR does not specify whether `CalibrationSignal` is also routed to the orchestrator-context destination. Given that it is part of the unified event substrate, it should be routed to both destinations (or an explicit rationale for exclusion from orchestrator-context should be provided). The structured-observation format for the orchestrator-context destination is specified for `DispatchTiming` events only in the JSON example; `CalibrationSignal` is omitted.

**P3-C — ADR-024: The `diagnostics` field is always required, but the ADR's escape-hatch framing from ADR-004 (raw output small enough to pass through) is not addressed**

- **Location:** ADR-024 §Decision §"`DispatchEnvelope` dataclass" §`diagnostics`
- **Observation:** ADR-004's §Decision includes an escape hatch: "a per-ensemble or per-invocation flag may indicate that the raw output is small enough to pass through directly (e.g., a classifier returning a single label)." ADR-024 makes `diagnostics` always present (no `| None` in the type annotation). ADR-025 explicitly scopes system ensembles out of substrate-routing ("System ensembles remain inline"). But the escape hatch in ADR-004 is broader than system-ensemble-only — it applies "per-ensemble or per-invocation" which could include capability ensembles in some cases. The two escape-hatch formulations are not obviously reconciled. This is a minor inconsistency worth noting for the conformance scanner.

---

## Section 2: Framing Audit

The framing audit examines what the evidence base made available but the ADRs did not foreground.

### Question 1: What alternative framings did the evidence support?

**Alternative framing 1 (for ADR-022): Narrow ADR-021's NL clause rather than amend the system prompt**

Spike γ's disposition analysis explicitly surfaced disposition (i) — intended scope — as a live alternative: the system prompt as currently written commits to direct-completion-as-residual; the NL clause was over-broad; correcting the ADR rather than the prompt would be the architectural response. The source material (spike γ continuation data) weakens but does not eliminate this reading. Under this framing, the right response is to narrow ADR-021 to "explicit naming supported; NL framing not guaranteed" and update product-discovery's Skill Orchestration User mental model accordingly.

What would the reader need to believe: that the stakeholder mental model ("NL routes to ensemble when slot fits") is aspirational, not committed; that making the aspiration operational via a prompt amendment is the wrong design surface; and that operators choosing a specific orchestrator profile should configure it accordingly. Spike γ's data (neither tested orchestrator honors NL framing without amendment) weakens this framing — but ADR-022's own §Rejected alternatives presents a version of it under "Revise the Skill Orchestration User stakeholder mental model" and rejects it on substantive grounds (mental model carries weight independent of agent convenience; explicit naming turns the orchestrator into an "explicit-dispatch dispatcher dressed in natural-language clothing"). The rejection is adequate.

**Alternative framing 2 (for ADR-023): Defer orchestrator-context routing to Cycle 7 and ship operator-terminal only**

Spike γ Cell A-explicit demonstrated that all the data the orchestrator-context destination needs exists in `execution.json` artifacts post-dispatch. The "orchestrator structural blindness to its own execution graph" (PLAY note 12) is a real concern, but the observation is that the orchestrator's NL *narration* substituted an implementation (qwen3:14b's `s[::-1]` → orchestrator's two-pointer loop claim). Whether in-turn dispatch-event routing would have prevented this is not empirically established — the substitution occurred in Cell A-explicit *after* the AS-7 summarizer worked correctly, suggesting the orchestrator-context destination needs not just timing data but the actual deliverable content to prevent narration substitution. Artifact-as-substrate (ADR-025) is what prevents the substitution, not dispatch-event routing into the orchestrator's context.

What the reader would need to believe: that PLAY note 12's load-bearing question ("what was the total run-time?") is answerable post-hoc from `execution.json` rather than mid-session by the orchestrator; and that the session-close cost of deferred orchestrator-context is not the "keeps the orchestrator structurally blind" cost ADR-023 frames it as. The ADR engages this framing in §Rejected alternatives under "Routing all dispatch events to **only** the operator-terminal destination" and rejects it; the rejection is substantive, though the argument that deferral "is cheaper" architecturally (because the unified substrate is one build, not two) is a drafting-time judgment rather than an evidenced finding.

**Alternative framing 3 (for ADR-025): Always-scope is the right default but `web-searcher` should be explicitly excepted at ADR level, not deferred to BUILD**

Spike α's size-floor finding named `web-searcher` specifically as the ensemble where artifact-substrate is "structurally awkward" — its JSON list of URL+snippet records is already-handle-shaped and small. ADR-025 acknowledges this but defers the exception to BUILD's `output_substrate: inline` opt-out. The alternative: codify `web-searcher` (and the `agentic-calibration-checker` and `agentic-result-summarizer`) as default-inline at the ADR level, making the always-scope rule mean "all capability ensembles except those explicitly excepted in this ADR." The system-ensemble exception is already in the ADR; the capability-ensemble exception for `web-searcher` is deferred to BUILD without specifying that BUILD *should* make it.

What the reader would need to believe: that the always-scope's operational friction at `web-searcher`'s boundary is predictable from spike α's evidence and does not require BUILD observation to establish. Spike α's conclusion was that artifact-substrate on `web-searcher` is "structurally wasteful and obscures the signal behind a filesystem round-trip." The ADR's P2-D finding above (no falsification criterion for the dial-back) is this framing's strongest expression. The alternative is not more correct architecturally, but it would make the always-scope a more testable and narrower commitment.

---

### Question 2: What truths were available but not featured?

**Finding A — Spike β's open question about orchestrator-as-prose-integrator role is not addressed in any ADR**

Spike β's headline finding was that composition assumptions live in the orchestrator's reasoning surface, not in the ensemble contract. The spike ended with an open question: "For interpretation-mediated composition, is the orchestrator's prose-writing role a substrate the cycle's typed-contract work should respect, or one it should narrow over time?" This is upstream of all four ADRs — it touches ADR-022 (routing), ADR-023 (observability), ADR-024 (envelope), and ADR-025 (substrate). None of the four ADRs acknowledges this open question or addresses how the cycle's decisions interact with it.

The ADRs are technically correct in not resolving it (it is empirically open and extends beyond Cycle 6's scope). But a note in ADR-024's or ADR-025's §Consequences or §Out of scope acknowledging that the orchestrator's prose-integrator role is not addressed by the envelope or substrate contracts — and that this role is what currently makes multi-stage compositions work — would prevent BUILD from over-relying on `output_schema:` declarations as composition infrastructure when spike β showed they are drift-detection infrastructure only.

**Finding B — Spike γ's ADR-019 §Consequences §Positive portability finding is not carried forward**

Spike γ's Cell B continuation found that routing surface behavior is not model-portable under the current system prompt. The spike's §Implications for cycle artifacts beyond T14 explicitly notes: "ADR-019 (skill-framework-agnostic orchestrator). The Consequences §Positive's profile-portability claim should be qualified: profiles are Model Profile portable (config-layer mechanism works), but routing surface behavior is *not model-portable* under the current prompt." None of the four ADRs carries this forward. ADR-022 acknowledges that qwen3:14b "over-delegates" but does not connect this to ADR-019's profile-portability commitment. The backward propagation in Step 3.7 is identified as targeting ADR-004, ADR-007, ADR-014, and ADR-021 — ADR-019 is not in the sweep.

This is a gap: if ADR-019 carries a profile-portability claim in §Consequences §Positive that spike γ empirically weakened, ADR-019 should receive the same partial-update treatment ADR-021 received from ADR-022.

**Finding C — The `web-searcher` JSON-string-in-response convention is identified in spike α but not addressed in ADR-024**

Spike α's data trace found that `web-searcher` emits its output as a JSON-string (not a JSON value) within the `response` field: the structured payload is serialized to a string and lives inside `response`. The spike identified this as "an artifact of the current `response: str` contract at the dispatch boundary." ADR-024's `structured: dict | None` field would cleanly resolve this — `web-searcher`'s typed payload would move from `response` as a JSON-string to `structured` as a typed dict. But ADR-024 does not specifically name this as one of the migration benefits or describe how `web-searcher`'s JSON-string convention maps to the new envelope. The `output_schema:` declaration for `web-searcher` is mentioned in spike α but not in ADR-024's text. This is a minor gap; a note in the migration path section of ADR-024 would close it.

---

### Question 3: What would change if the dominant framing were inverted?

**ADR-022's dominant framing:** the system prompt's commitment should honor the Skill Orchestration User's NL-routes-to-ensemble mental model by amending the prompt to prefer `invoke_ensemble` when a capability match exists.

**Inverted:** the system prompt's existing precedence (direct completion as residual; client tools for filesystem; ensemble on explicit naming) is the correct routing surface for a cheap-cloud orchestrator; operators using NL framing for capability dispatch should learn to name ensembles explicitly. The Skill Orchestration User's mental model should be revised rather than the system prompt.

Under the inversion: ADR-021's NL clause narrows to "not supported under NL framing; explicit naming is the supported dispatch surface." Product-discovery's Skill Orchestration User entry is revised to reflect that the system routes by NL when the operator explicitly names an ensemble, not when the prompt is NL-framed without a name. The amendment is not authored; the profile-portability problem (spike γ Cell B) doesn't surface as an ADR concern.

What becomes weaker: the system's accessibility to skill frameworks that emit NL-shaped prompts (RDD's `rdd:*` skills, for instance, emit phase-shaped prompts without ensemble names). What becomes stronger: the architectural honesty that cheap-cloud orchestrators do not reliably self-route to ensembles on NL framing — the contract doesn't promise what the empirical evidence shows is not delivered.

ADR-022's rejection of this framing is substantive and the rejection is the load-bearing case the framing audit would validate. The inversion surfaces the strongest argument: the amendment's effectiveness is uncertain under some profiles and the "correct" behavior under the amendment is still LLM-judgment over `list_ensembles()` — a judgment the amendment instructs but doesn't guarantee. The inverted framing would produce a more conservative ADR that matches evidence more tightly at the cost of narrowing the stakeholder's expectations.

**ADR-025's dominant framing:** all capability ensembles route their deliverable through artifact-as-substrate; the size-floor tradeoff is accepted as a clean-design-rule cost.

**Inverted:** artifact-as-substrate is the right mechanism for deliverable-mediated composition shapes (code-generator → code-reviewer; prose-improver) but is structural overhead for interpretation-mediated shapes (web-searcher → claim-extractor → argument-mapper). Spike β explicitly named this distinction — the lit-review composition is interpretation-mediated; the orchestrator reads the artifact and distills for the next stage's `input.data` regardless. Under this framing, artifact-substrate for `claim-extractor` and `argument-mapper` adds disk I/O and artifact lifecycle cost without improving the orchestrator's composition predictability, because the orchestrator still mediates between stages.

What becomes more salient under the inversion: spike β's conclusion that "C reduces composition's context cost [but] does not make composition more predictable." For interpretation-mediated compositions, the always-scope adds overhead without the structural benefit that justifies it (the benefit accrues mainly for deliverable-mediated compositions). This is the most empirically grounded challenge to the always-scope decision. ADR-025's "dial back later if cumbersome" disposition implicitly acknowledges this, but without naming the deliverable-mediated vs. interpretation-mediated distinction as the relevant axis.

---

### Framing Issues

**P2-E — ADR-019 §Consequences §Positive not updated despite spike γ's portability finding**

- **Location:** ADR-019 §Consequences §Positive (not one of the four new ADRs, but within the source material)
- **Finding:** Spike γ Cell B continuation found that routing surface behavior is model-conditional — qwen3:14b over-delegates where MiniMax under-delegates. ADR-019's profile-portability claim ("operators can swap orchestrator profiles to adjust cost and capability within a portable routing contract") is materially weakened by this finding. ADR-022's backward propagation updates ADR-021's NL clause, ADR-004, and the domain model, but does not update ADR-019. This is a gap in the Step 3.7 backward propagation sweep.
- **Recommendation:** Add ADR-019 to the backward propagation sweep with a supersession note on §Consequences §Positive: the profile-portability claim is qualified — model profiles are interchangeable at the config layer, but routing surface behavior is model-conditional under the current system prompt. The amendment in ADR-022 mitigates this partially but characterization is deferred to BUILD/PLAY.

**P3-D — The orchestrator's prose-integrator role and spike β's open composition question are not carried forward to BUILD**

- **Location:** ADR-024 §Out of scope; ADR-025 §Out of scope
- **Finding:** Spike β's most consequential finding — that composition assumptions live in the orchestrator's reasoning surface, not in the typed envelope contract — is not documented as an architectural assumption or open question in the ADRs that ship the envelope and substrate. BUILD implementing `output_schema:` declarations may expect composition predictability that spike β found only the "B-strong" interpretation (typed pipeline bindings, chain-selector orchestrator) would deliver. The current ADRs ship "B-advisory" which provides drift detection, not composition predictability.
- **Recommendation:** Add a note in ADR-024 §Consequences §Positive (or §Neutral) that `output_schema:` declarations provide drift-detection infrastructure, not composition infrastructure — the orchestrator's role as prose-integrator between dispatch stages is unchanged by the envelope contract. Spike β's open question (whether the prose-integrator role should be preserved, narrowed, or replaced) is explicitly deferred as a future-cycle question.

---

## Summary

| Priority | Count | Key findings |
|----------|-------|--------------|
| **P1** | 1 | ADR-025's AS-7 amendment omits the calibration gate's evaluation surface under substrate-routing — backward propagation sweep addresses the summarization pathway but not the quality-evaluation pathway |
| **P2** | 5 | (A) ADR-022 undercharacterizes qwen3:14b amendment risk as "uncertain" rather than "directionally adverse"; (B) ADR-024/ADR-023 BUILD-sequencing dependency on `dispatch_id` and `duration_seconds` unspecified; (C) ADR-023 in-turn orchestrator-context routing at session-terminal boundary unspecified; (D) ADR-025 "dial back later if cumbersome" lacks a falsification criterion; (E) ADR-019 not in the backward propagation sweep despite spike γ's portability finding |
| **P3** | 4 | (A) ADR-022 §Out of scope conflates session-level and profile-level override surfaces; (B) ADR-023 `CalibrationSignal` routing to orchestrator-context destination unspecified; (C) ADR-024 escape-hatch reconciliation with ADR-004's per-invocation escape hatch not addressed; (D) spike β's composition-assumptions finding not documented as a BUILD assumption in ADRs 024/025 |

### Recommendations for the ADR author

The four ADRs are coherent and well-grounded in their spike evidence base. The provenance checks accurately distinguish driver-derived framings from drafting-time synthesis. The Rejected alternatives sections engage alternatives substantively rather than summarily. The evidence trail from cycle-status spike findings to ADR decisions is traceable throughout.

**Must-address before accepting ADRs:**
- P1-A: Specify the calibration gate's evaluation surface under artifact-as-substrate. This is a BUILD-blocking gap: the calibration gate's critic agents need to know whether they evaluate deliverable content (requiring artifact access) or metadata (`primary` + `structured`) under substrate-routing. Add this specification to ADR-025's backward propagation sweep and to the consequences for ADR-007 and ADR-014.

**Should-address before BUILD entry:**
- P2-A: Revise ADR-022's qwen3:14b effectiveness characterization to distinguish the two failure-mode directions and the amendment's targeting.
- P2-B: Add a BUILD-sequencing note in ADR-024 for `diagnostics.dispatch_id` and `diagnostics.duration_seconds`.
- P2-C: Clarify ADR-023's in-turn orchestrator-context routing at session-close boundary.
- P2-D: Add a concrete BUILD/PLAY indicator that would trigger the always-to-substantive-scope refinement for ADR-025.
- P2-E: Add ADR-019 to the backward propagation sweep with a portability-claim qualification.
