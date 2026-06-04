# Conformance Scan Report

*(Recovered: the conformance-scanner returned this report as its final message
but did not write it to the output path; the orchestrator persisted it verbatim
on 2026-06-03. Findings and text are the scanner's, unedited. Same recovery as
the R1 argument audit this session — see cycle-status for the tooling note.)*

**Scanned against:** `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md` (primary); `docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md` and `docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md` (interaction context only)
**Codebase:** `src/llm_orc`
**Date:** 2026-06-03

---

## Summary

- **ADRs checked:** 1 (ADR-036 as primary; ADR-033/035 for interaction surface only)
- **Conforming:** 0 (zero ADR-036 FCs are fully satisfied by the current code)
- **Violations found:** 4

---

## Conformance Debt Table

| # | ADR / FC violated | Location | Severity | Description | Disposition |
|---|-------------------|----------|----------|-------------|-------------|
| F-1 | ADR-036 FC (directive-in-user-turn presence) | `loop_driver.py:316–326` | refactor-now | `_seat_filler_messages` prepends `_DELEGATION_GUIDANCE` as a `{"role": "system", ...}` message. The ADR's primary empirical finding is that system-slot guidance loses to a 27,925-char client system prompt (0/10 baseline; V1/V2 variants also failed). Decision 1 of ADR-036 prescribes composing the guidance into the user-turn region — on a first turn attached to the user task, on trailing turns as a standalone trailing user-role message (the C3 form). The current implementation is exactly the rejected mechanism. The docstring at line 319 names this as "The delegation system message" and line 103's module-level docstring still says "System guidance offered to the seat-filler" — the framing predates the ADR. This is the primary BUILD work item ADR-036 was written to land. | Expected structural debt; the ADR was drafted knowing this code predates it. The upcoming BUILD work package resolves this: move guidance injection from a leading `{"role": "system"}` prepend to a trailing `{"role": "user"}` standalone message on tool-result tails, and to attachment-on-first-user-turn on first turns. |
| F-2 | ADR-036 FC (delegation-rate measurability): no sink consumes `TurnDecision` for operator surfacing | `operator_terminal_event_sink.py:86–104`; gap in entire `src/` package | BUILD-work | `TurnDecision` is emitted correctly through the `DispatchEventSubstrate` by `loop_driver.py:369–389` and the dataclass's `delegated_ensemble` field (the numerator for ADR-036 Decision 3) is populated. However, no `EventSink.consume` handler in the package handles `TurnDecision` events: `OperatorTerminalEventSink.consume` dispatches on `DispatchTiming`, `TierSelection`, `CalibrationVerdictEvent`, `AuditDiagnostic`, and `CalibrationSignal` — `TurnDecision` falls through silently. `OrchestratorContextEventSink` likewise has no `TurnDecision` branch. The ADR's FC requires the rate to be "computable from emitted events alone"; events are emitted into the substrate's in-memory log, but there is no consumer that surfaces them to an operator or produces the rate. Computing the rate requires a post-hoc query of `substrate.events_for(dispatch_id)` and filtering for `TurnDecision` instances, which satisfies "no log archaeology," but the operator-visible surface (delegation_rate as a logged or queryable metric) is absent. The WP-LB-F known-unbuilt gap is confirmed. | Expected BUILD-work debt. The ADR explicitly names the instrumentation as Decision 3 and the BUILD acceptance run as the gating condition. Resolution: add a `TurnDecision` branch to `OperatorTerminalEventSink.consume` that logs the delegation event (at minimum `delegated_ensemble`, `turn_index`, `action`) and either expose the rolling rate computation there or in a dedicated rate-meter surface. |
| F-3 | ADR-036 Decision 3 FC (delegation-rate measurability): generation-shaped turn classifier absent from package | Absent from `src/llm_orc/`; present only at `scratch/spike-psi-delegation-rate/psi4a_prefilter.py` | BUILD-work | ADR-036 Decision 3 requires a deterministic generation-shaped turn classifier as the denominator for the delegation rate. The spike-validated rule (generation verb × content object × capability domain × observed-carry exclusions; reported 0/12 clear-case errors) is implemented only in `scratch/spike-psi-delegation-rate/psi4a_prefilter.py` — outside the package, in scratch space. No equivalent classifier exists anywhere in `src/llm_orc/`. Without the denominator, the `delegation_rate` metric prescribed by the ADR cannot be computed: the numerator (`TurnDecision.delegated_ensemble is not None`) is available, but the denominator (generation-shaped turns from which delegation was expected) is not. | Expected BUILD-work debt. Resolution: graduate `psi4a_prefilter.py`'s rule into the package (likely `src/llm_orc/agentic/` alongside `loop_driver.py` or a new `delegation_rate_meter.py`) and wire it into the `TurnDecision` surfacing added for F-2. |
| F-4 | ADR-036 (Rejected alternatives / `tool_choice` family closed): `single_step_enforcer.py` docstring frames `tool_choice` as a BUILD-tunable candidate | `single_step_enforcer.py:17–19` | deferred | The module docstring says: "The two untested candidates (a re-planning prompt; a one-tool `tool_choice` constraint) remain BUILD-tunable behind this module's boundary; `tool_choice` is the weakest (Spike κ: the framework does not forward `tool_choice`, and MiniMax did not honor it)." ADR-036 extends the empirical record: a third negative was measured on Ollama+qwen3:14b (Spike ψ.3 — HTTP 200, silently ignored), and the ADR closes the `tool_choice` mechanism family. The docstring's framing "remain BUILD-tunable" is now contradicted for `tool_choice` specifically. The re-planning prompt alternative remains genuinely untested and its "BUILD-tunable" characterization is still defensible. | Mild docstring drift, deferred. No behavioral consequence. Resolution: update the docstring to note `tool_choice` is empirically closed (ADR-036 ψ.3 added the third negative), leaving only the re-planning prompt as the remaining BUILD-tunable candidate. |

---

## Notes

**Pattern of non-conformance.** All four findings are pre-ADR code meeting a freshly-landed decision — none represents an independent regression. F-1 through F-3 represent the BUILD work the ADR was written to authorize. F-4 is docstring drift of low consequence.

**Expected vs. surprise split.** F-1, F-2, and F-3 are expected structural debt; no surprises in those areas. F-4 is a mild surprise (not flagged in the dispatch brief); low-severity, one-line fix.

**`TurnDecision` emission is structurally correct.** The `delegated_ensemble` field (the ADR-036 numerator) is populated and emitted through the substrate on every turn. The gap is purely at the consumer/surfacing layer (F-2) and the classifier/denominator layer (F-3). The event infrastructure does not need redesign.

**Conversation compaction system messages are not on the seat-filler path.** `conversation_compaction.py` emits `{"role": "system"}` messages, but that module is not imported by or wired into `loop_driver.py` or `client_tool_action_terminal.py`. Those system messages are part of the single-turn (ADR-027) pipeline path and do not violate ADR-036's directive-in-user-turn FC.

**ADR-033/035 interactions with ADR-036.** No new violations of ADR-033 or ADR-035 introduced by ADR-036. ADR-035's `compose_form_directive` exists and is called correctly at `loop_driver.py:352`. ADR-033's `TurnDecision.delegated_ensemble` is present and populated. The interaction surface between the three ADRs is clean.

**Severity ordering for BUILD planning.** F-1 is the highest-priority item: the BUILD acceptance gate (real-OpenCode session confirming delegation fires under V3 composition) cannot be cleared until it lands. F-2 and F-3 are co-dependent and should land together as the instrumentation work package. F-4 can be bundled with any docstring-pass and carries no urgency.
