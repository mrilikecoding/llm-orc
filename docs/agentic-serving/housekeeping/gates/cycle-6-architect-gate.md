# Gate Reflection: Cycle 6 ARCHITECT → BUILD

**Date:** 2026-05-15
**Phase boundary:** architect → build
**Cycle:** Cycle 6 — Ensemble contract + observability + routing (mini-cycle, Mode D)

## Belief-mapping question composed for this gate

> The Cycle 6 architecture decomposed the unified event substrate's two destinations into two separate modules (Operator-Terminal Event Sink at L3; Orchestrator-Context Event Sink at L2). Inversion N+2 said "one substrate, two destinations" — but it did not specify whether "two destinations" means two modules or two consumption surfaces of one module. What would have to be true for those two destinations to fold into a single "Event Routing Surface" module with two consumption surfaces? And if those things aren't true, what's load-bearing about the separation?

## User's response

Selected "Proceed to BUILD as-is" — accepted the two-module decomposition without engaging the belief-mapping question substantively. The user has operated in autonomous-mode throughout the ARCHITECT phase (per the session directive "work without stopping for clarifying questions"), engaged deeply at the DECIDE gate (per the Cycle 6 DECIDE gate reflection, where the user engaged the pre-mortem and absorbed two refinements ADR-025 had not anticipated), and signaled at this boundary that the architectural inheritance from DECIDE was sufficient.

## Pedagogical move selected

**Teach (contingent shift from Probe).** The belief-mapping question was the initial move; the user's "proceed as-is" response is thin against that probe's depth. The contingent shift to Teach briefly surfaced the architectural argument for the separation (three factors: different layers; different consumption patterns; different dependency directions) so the rationale is documented for future readers and for BUILD's reference, rather than left implicit in the module entries. The teach moment closed the gate — no further probe.

## Commitment gating outputs

**Settled premises (the user is building on these going into BUILD):**

1. **Inversion N+2 governs Cycle 6's observability architecture** — one event-emission substrate (Dispatch Event Substrate at L1); two routing destinations (Operator-Terminal Event Sink at L3; Orchestrator-Context Event Sink at L2). The decomposition is structural; parallel-emission infrastructure for the same data is structurally prevented (FC-24).
2. **Always-scope for ADR-025 substrate-routing** — every capability ensemble in the agentic-serving library substrate-routes by default (`output_substrate: artifact`); system ensembles inline-respond; opt-out via `output_substrate: inline` per capability ensemble; the dial-back falsification criteria (5 indicators) are the empirical mechanism for re-examination.
3. **AS-7 amendment to default-with-conditional-skip is operationally honored at the architecture layer** — substrate-routed capability ensembles skip `agentic-result-summarizer` invocation; inline-response ensembles (system ensembles; opt-outs) retain mandatory summarization; the skip is enforced at Orchestrator Tool Dispatch's interposition order; ADR-004's per-invocation `raw_output` escape hatch composes without contradiction.
4. **`dispatch_id` is the single-source-of-truth correlation identifier** — generated at Orchestrator Tool Dispatch's `invoke_ensemble` entry via the substrate's `new_dispatch_id`; flows to all dispatch events for that dispatch; flows to the envelope's `diagnostics.dispatch_id`; flows to the artifact filesystem path's `<dispatch_id>` segment. FC-22 verifies cross-surface consistency. The Cycle 6 DECIDE snapshot Finding 2 advisory carry-forward is encoded structurally.
5. **The `DispatchEnvelope` is the typed return value of `invoke_ensemble`** — a shared type alongside `LlmOrcStructuralError`; fields `{status, primary, structured?, diagnostics, errors?, artifacts?}`; `output_schema:` per-ensemble declaration is advisory at dispatch time (spike β reframed output-spec drift as `input.data` override, not synthesizer compliance).
6. **The ADR-022 system-prompt amendment is the design-time intervention for routing-surface behavior** — capability-matched NL framing prefers `invoke_ensemble` over direct LLM completion AND over client-declared tools; effectiveness is per-orchestrator-profile-conditional per ADR-022 disposition (iii); cross-profile characterization is BUILD/PLAY work, not architecture work.

**Open questions (the user is holding these open going into BUILD):**

1. **ADR-022 amendment effectiveness under qwen3:14b.** Spike γ Cell B showed qwen3:14b over-delegating to client tools under NL framing; the amendment's effectiveness on this profile is uncertain. Post-BUILD PLAY re-runs the spike γ probe across Cells A and B with the amended prompt active; if qwen3:14b continues to over-delegate, per-orchestrator-profile system-prompt overrides become Cycle 7+ territory.
2. **`web-searcher` (and other small-deliverable ensembles) under always-scope substrate-routing.** The DECIDE snapshot Finding 1 advisory: BUILD should migrate `web-searcher` early (encoded in WP-E sequencing) so Indicator 1 (latency overhead for deliverables under 1 KB) and Indicator 4 (`output_substrate: inline` opt-out count) are testable before the full migration commits. Whether the always-scope holds under operational evidence is an empirical question for BUILD's first deployments + post-BUILD PLAY.
3. **`AuditDiagnostic` inclusion/exclusion at Orchestrator-Context Sink.** The exclusion-by-default policy for `CalibrationSignal` is specified; the analogous policy for `AuditDiagnostic` is unspecified. BUILD should specify when implementing the Orchestrator-Context Event Sink (per cycle-6-architect snapshot Advisory 3).
4. **Plexus-KG-as-substrate (Cycle 7+ territory).** ADR-025 §"Out of scope" surfaces this from the practitioner's DECIDE-gate pre-mortem: under active Plexus, the KG may become the durable substrate, and the filesystem-artifact path becomes the AS-8-absent path. The always-scope decision is preserved regardless; *which* substrate (filesystem vs. KG) is configuration-conditional. Not in Cycle 6 scope.
5. **`output_schema:` adoption pace and composition predictability.** Spike β reframed: output-spec drift's mechanism is the orchestrator's `input.data` override, not synthesizer compliance. ADR-024's typed `structured` field opens **structural composition predictability** when ensembles declare `output_schema:` and downstream consumers read it directly; it does **not** eliminate the orchestrator's reasoning-surface role in composition. BUILD teams implementing `output_schema:` for representative ensembles should expect drift-detection benefit, not composition-substrate replacement. Per cycle-6-architect snapshot Advisory 3 carry-forward from DECIDE.

**Specific commitments carried forward to BUILD:**

1. **BUILD-mode declaration at BUILD entry per ADR-091.** Gated recommended given the design-alternative examination character of ADR-022 routing-surface work + the cross-surface `dispatch_id` coupling. Auto-mode appropriate only if BUILD reduces to mechanical wiring after the four-module structural shape lands.

2. **`web-searcher` early-migration sequencing under WP-E.** Position `web-searcher` among the early per-ensemble migrations so Indicators 1 and 4 of the dial-back falsification criteria are observable before the migration commits. Recorded in roadmap.md WP-E.

3. **Cross-surface `dispatch_id` consistency integration test.** `test_dispatch_id_consistency_across_events_envelope_artifact_path` asserts the same `dispatch_id` value across the event stream, the envelope's `diagnostics.dispatch_id`, and the artifact path's `<dispatch_id>` segment. FC-22 is the verification anchor.

4. **`AuditDiagnostic` inclusion/exclusion at Orchestrator-Context Sink.** Resolve when implementing `consume()` and `observations_for()`; document alongside the `CalibrationSignal` exclusion policy. Default proposal (advisory): exclude by default (similar to `CalibrationSignal` — audit diagnostics are operator-tooling territory rather than in-turn orchestrator reasoning); opt-in available via `agentic_serving.observability.orchestrator_context_routes_audit_diagnostic: true`.

5. **Spike γ probe re-run at post-BUILD PLAY** across Cells A (MiniMax M2.5-free) and B (qwen3:14b via `agentic-orchestrator-offline-tools`) with the ADR-022 amended prompt active. Records per-profile dispatch behavior. If qwen3:14b continues to over-delegate, per-profile system-prompt overrides become Cycle 7+ territory.

6. **ADR-019 P2-E portability qualification verification.** Pre-BUILD action from the DECIDE snapshot: ADR-019 §Consequences §Positive should carry the spike-γ-Cell-B portability qualification. **Verified applied** at `adr-019:109` during ARCHITECT entry (cited as already-applied; no edit needed). Carry-forward to BUILD entry: confirm the qualification reads correctly under fresh review and is not subsequently removed.

7. **Three architect-phase advisory carry-forwards encoded inline before BUILD entry** (per cycle-6-architect snapshot recommendations): (a) Orchestrator-Context Event Sink separation justification added to the module entry; (b) Validate-once-at-load operator affordance Direction-not-constraint note added to Ensemble Engine Cycle 6 extension; (c) ADR-016-style bounding mechanisms disposition added explicitly to the Fitness Criteria section, closing the Cycle 4 OQ #14 carry-forward for the Cycle 6 stage set.
