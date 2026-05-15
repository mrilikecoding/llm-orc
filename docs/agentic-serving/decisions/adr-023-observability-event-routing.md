# ADR-023: Observability Event-Routing — Unified Emission Substrate, Two Destinations, Bounded Event-Model Extension

**Status:** Proposed

**Date:** 2026-05-15

---

## Context

Cycle 5 BUILD shipped four typed dispatch events as part of the calibration/tier-routing infrastructure: `TierSelection` (model_profile, tier, topaz_skill), `CalibrationVerdict` (Proceed/Reflect/Abstain), `AuditDiagnostic` (Tier-Router-Audit drift criteria findings per ADR-018), and `CalibrationSignal` (cross-layer calibration channel emissions per ADR-016). These events exist in code as typed dataclasses, are emitted at dispatch sites, and serialize to `execution.json` artifacts under `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/`.

Cycle 5 PLAY note 19 — sharpened by the Cycle 5 PLAY susceptibility snapshot — disclosed the observability gap as **infrastructure-complete / routing-incomplete**: the dispatch-event-emission surface is built; the dispatch-event-routing surface (events → human-visible destinations) is not. The 2026-05-14 follow-on verification finding 7 substantiated this empirically — a 28-minute serve console window captured during Cycle 5 verification showed three categories of low-signal emission (legacy YAML Pydantic warnings; coarse `tool dispatch: result kind=success` lines without ensemble identification, session ID, verdict, or tier; standard HTTP access lines) and **one category of absence**: none of the four typed events surface to console. Finding 7's liveness-signal extension (2026-05-14 follow-on) compounded the framing — a single `chunk_by_predicate` code-generator session ran 8m 28s wall-clock and emitted five total log lines, with a 10+ minute window of total console silence; every emitted line was post-hoc completion-event. The operator-side observability surface emits **completion events only** during in-flight states.

Cycle 6 DISCOVER snapshot Action 2 field-read produced two sharpening findings: the four typed events **substantially carry the operator-terminal destination's needs** (the `TierSelection` fields already cover model_profile/tier/topaz_skill operator-terminal expects; `CalibrationVerdict` carries the verdict; `AuditDiagnostic` carries drift criteria findings); the **orchestrator-context destination requires a bounded event-model extension** — dispatch duration is not a field on any existing event type, and PLAY note 12's load-bearing practitioner question (*"What was the total run-time of the ensemble?"*) cannot be answered from the existing event stream. Additionally, `CalibrationVerdict` is a bare `Literal[Proceed, Reflect, Abstain]` with no call-site context (ensemble name, dispatch identifier, timestamp) — routing it to either destination requires call-site composition.

Cycle 6 DISCOVER snapshot Action 3 surfaced **Inversion N+2**: the two observability surfaces (operator-terminal, orchestrator-context) are **routing destinations of one shared event-emission infrastructure**. The unified-substrate architectural question is answered before the "which surface?" deliberation — there is one emission infrastructure, two routing destinations. This framing reframed T15 from "design two parallel observability surfaces" to "design one event substrate; route its events to two destinations."

Cycle 6 MODEL Action C surfaced **three architectural alternatives for dispatch timing** as a DECIDE entry condition: (i) **event-model extension** — add `start_time`/`end_time` fields to existing dispatch event types, or add a new `DispatchTiming` event (the path Inversion N+2's unified-substrate framing favors); (ii) **sidecar log** — route dispatch timing to orchestrator-context only via a parallel log stream that does not participate in the typed-event surface, leaving the existing event model unchanged; (iii) **orchestrator-context-only via separate mechanism** — surface dispatch timing through the orchestrator's reasoning context without routing it through the typed-event infrastructure. MODEL Action C tests whether the sidecar-log alternative (ii) satisfies PLAY note 12's load-bearing question before scoping the event-model-extension path (i) as a requirement.

Spike γ's per-cell dispatch logs (`scratch/spike-gamma-routing/serve_cellB.log`, `serve_restored.log`, Cell A/A-explicit dispatch logs) substantiate the field-read finding empirically — Cell A-explicit's serve log captured the `code-generator` dispatch (61.44s) + `agentic-result-summarizer` dispatch (3.14s) as typed dispatch-timing data already present in `execution.json` artifacts; the operator-terminal destination received only the coarse `tool dispatch: result kind=success` lines.

---

## Decision

Codify the unified-substrate event-routing surface per Inversion N+2: **one event-emission infrastructure (existing dispatch events + bounded extension) routes to two destinations (operator-terminal, orchestrator-context).** The extension is path (i) — event-model extension via a new `DispatchTiming` event + a correlation identifier (`dispatch_id`) linking related events for one dispatch. The sidecar-log alternative (ii) is rejected because it satisfies PLAY note 12's question in isolation but violates Inversion N+2 by making dispatch-timing a separate emission infrastructure for one destination only (see Rejected alternatives).

### Event-emission substrate (single)

The substrate is the existing dispatch-event surface (`TierSelection`, `CalibrationVerdict`, `AuditDiagnostic`, `CalibrationSignal`) extended with:

1. **New event type: `DispatchTiming`** — emitted twice per dispatch:
   - `DispatchTiming(phase="start", dispatch_id, ensemble_name, model_profile, timestamp)` — emitted when the orchestrator calls `invoke_ensemble`, before the ensemble executes.
   - `DispatchTiming(phase="end", dispatch_id, ensemble_name, duration_seconds, exit_status, timestamp)` — emitted when the ensemble returns control. `exit_status` is one of `success`, `error`, `timeout`, `aborted`. `duration_seconds` is the wall-clock seconds between start and end.

2. **`dispatch_id` correlation identifier on existing events.** `TierSelection`, `CalibrationVerdict`, `AuditDiagnostic`, `CalibrationSignal` each gain a `dispatch_id: str` field. The identifier is a session-scoped monotonic counter or UUID4 — implementation-level; the typed contract is "stable identifier for one dispatch across all events emitted during that dispatch." Operators and the orchestrator-context destination can join events on `dispatch_id` to reconstruct a full dispatch picture.

3. **`CalibrationVerdict` call-site composition.** The existing `Literal[Proceed, Reflect, Abstain]` carries the verdict alone; the new `dispatch_id` field correlates back to `DispatchTiming(phase="start")` which carries the call-site context (ensemble name, model profile, timestamp). Routing requires no change to `CalibrationVerdict`'s semantic content; the correlation identifier is the call-site composition mechanism.

This is the **bounded** event-model extension: one new event type, one new field on four existing event types. Existing event semantics are unchanged; existing `execution.json` artifact contents are preserved (the extension is additive). The `dispatch_id` field is optional during the transition (Cycle 6 BUILD); a `dispatch_id: None` event corresponds to legacy non-extended emission sites that BUILD progressively converts.

### Routing destinations (two)

The two destinations route from the **same** event substrate:

#### Destination 1 — Operator-terminal (existing serve console + structured event log)

The operator-terminal destination routes events to the serve process's console output. Each event class formats to one or more human-readable lines:

- `DispatchTiming(phase="start")` → one-line: `INFO: dispatch start: ensemble=<name> profile=<model_profile> dispatch_id=<id>`
- `DispatchTiming(phase="end")` → one-line: `INFO: dispatch end: ensemble=<name> duration=<seconds> exit=<status> dispatch_id=<id>`
- `TierSelection` → one-line: `INFO: tier selection: ensemble=<name> profile=<model_profile> tier=<tier> topaz_skill=<skill> dispatch_id=<id>`
- `CalibrationVerdict` → one-line: `INFO: calibration verdict: <verdict> dispatch_id=<id>`
- `AuditDiagnostic` → one-line per drift criterion finding: `INFO: audit diagnostic: criterion=<name> finding=<value> threshold=<value> dispatch_id=<id>`
- `CalibrationSignal` → debug-level (suppressed unless `--verbose` or `LOG_LEVEL=DEBUG`): `DEBUG: calibration signal: <signal_type> value=<value> dispatch_id=<id>`

`CalibrationSignal` events are **excluded from the orchestrator-context destination by default** — they are high-volume cross-layer telemetry per ADR-016 whose value is operator-tooling and post-hoc analysis, not in-turn orchestrator reasoning. Operators can opt in to orchestrator-context routing of calibration signals via `agentic_serving.observability.orchestrator_context_routes_calibration_signal: true` if a deployment's orchestrator can use the signal productively in-turn (the default `false` preserves context budget; the opt-in is for deployments where calibration-signal routing demonstrably improves orchestrator decisions).

The format is line-oriented `key=value` for grep/awk-friendly operator consumption. Lines are emitted at `INFO` level except `CalibrationSignal` which is `DEBUG` (high-volume, low-signal at default verbosity).

#### Destination 2 — Orchestrator-context (event stream into the orchestrator's reasoning context)

The orchestrator-context destination routes events into the orchestrator's reasoning context as **structured observations** between turns. Two routing patterns are supported:

- **In-turn context inclusion.** When the orchestrator's next turn begins (after a dispatch returns control to the orchestrator's ReAct loop), the events emitted during the just-completed dispatch are formatted as a structured observation block prepended to the next turn's context. Format is JSON-shaped (consumable by the model's reasoning) rather than line-oriented:
  ```json
  {
    "dispatched": "code-generator",
    "duration_seconds": 61.44,
    "model_profile": "agentic-tier-cheap-general",
    "tier": "cheap",
    "topaz_skill": "code_generation",
    "calibration_verdict": "Proceed",
    "dispatch_id": "session-7-dispatch-3"
  }
  ```
  PLAY note 12's load-bearing practitioner question (*"What was the total run-time of the ensemble?"*) is answered directly by `duration_seconds` in this structured observation.

- **End-of-session summary.** When a session closes, the dispatch-event log for the session is summarized as a structured observation block (one entry per `dispatch_id`) and included in the session's `execution.json` artifact under a new `dispatch_log` key. The summary is consumable by post-hoc operator review and by future Plexus ingestion when Plexus is active.

The two routing patterns share the event source; they differ in **when** the events route into the orchestrator's reasoning context.

`CalibrationSignal` events are **excluded from this destination by default** (see §Destination 1 — operator-terminal — for the per-event behavior table); the opt-in flag `agentic_serving.observability.orchestrator_context_routes_calibration_signal: true` enables routing for deployments where calibration-signal context-injection demonstrably improves orchestrator decisions.

### Final-dispatch-before-session-close handling

The in-turn orchestrator-context routing pattern assumes a "next turn" exists to receive the structured observation. When a dispatch is the session's final operation before close, no next turn exists. The behavior:

- The dispatch's events route to the **end-of-session summary** in `execution.json` (the second routing pattern); this routing is unconditional and operates regardless of whether a next turn exists.
- The in-turn structured-observation context-inclusion is **skipped** for the final dispatch (no next turn to receive it). Operators reviewing the session post-hoc consume the end-of-session summary in `execution.json` (and, via ADR-025's session-dir structure, the `dispatch_log` key alongside per-dispatch artifacts).
- The operator-terminal destination receives all events including the final dispatch's events at emission time; the operator-terminal does not depend on next-turn existence.

The skipped-in-turn-routing on final dispatch is structurally honest about the asymmetry — the orchestrator-context destination is a turn-boundary mechanism, and the final turn has no successor. End-of-session summary is the persistent record across the boundary.

### Liveness-signal patterns

Two liveness-signal emission patterns ship as part of T15:

1. **Tool-call-emit logging.** When the orchestrator's response stream contains a tool-call structure (via the OpenAI-compatible streaming or non-streaming surface), the serving layer logs a one-line `INFO: tool-call emit: tool=<name> dispatch_id=<id>` **before** dispatching the tool call. This gives operators an anchor — "received tool call from cloud LLM at HH:MM:SS" — distinct from the existing post-dispatch `tool dispatch: result kind=success` line.

2. **Inference-wait heartbeat.** When a request has been open for more than `N` seconds (default `N=30`, operator-configurable via `agentic_serving.observability.heartbeat_interval_seconds`) without any tool activity (no tool-call emit, no dispatch start/end), the serving layer emits `INFO: inference wait: elapsed=<seconds> session_id=<id>` to the console. This gives operators mid-stream signal during long cloud-LLM inference waits.

Both patterns operate at the **operator-terminal destination only** — the orchestrator-context destination does not receive heartbeats (the orchestrator's reasoning surface already has natural session-level timing context). The patterns ship in Cycle 6 BUILD; default interval is operationally tunable; the patterns are added as new entries in the dispatch-timing log surface, not as new event types in the dispatch-event substrate.

### Noise-floor remediation (validate-once-at-load)

The 2026-05-14 follow-on finding 7 noted the legacy Pydantic warnings for `fan-out-test.yaml` and `plexus-graph-analysis.yaml` re-validating on every `list_ensembles()` call, emitting the same two-line warnings on each enumeration cycle. Cycle 6 BUILD changes the ensemble library validation to **validate-once-at-load**:

- At serve startup (or library reload), each ensemble YAML is validated once. Validation errors are surfaced as `WARN` lines at startup with the offending file path and the validation error.
- Subsequent `list_ensembles()` calls return the **validated subset** of the library (invalid ensembles excluded with the startup warning carrying the rationale). No re-validation occurs on enumeration.
- Operators get one validation pass at startup; the per-enumeration noise is eliminated.

The remediation interacts with ADR-019's library-loading semantics — if a YAML is added or modified at runtime (without a library reload), the validation does not re-run. Operators editing ensemble YAMLs must trigger a library reload (`SIGHUP` to the serve, or an admin endpoint, or restart) for changes to take effect. This is a small operator-affordance change; the cost is operator awareness of the reload semantic, which is consistent with ADR-011's session-boundary-config discipline.

### Out of scope for ADR-023

- **TUI dashboard / structured operator UI.** ADR-023 ships line-oriented operator-terminal output + structured orchestrator-context observations. A TUI dashboard or operator-facing UI on top of the event substrate is operator-tooling territory, not orchestrator architecture. The unified event substrate makes future UI work straightforward (consume the same event stream), but no UI ships in Cycle 6.
- **Plexus integration of dispatch-event log.** The end-of-session summary in `execution.json` is the artifact substrate; Plexus ingestion of the artifact is consistent with AS-4 (ingestion boundary is source material) and the existing artifact-ingestion shape. Active Plexus ingestion configuration is Cycle 7+ territory.
- **Per-event sampling or rate-limiting.** Default emission rates are expected to be low (a handful of events per dispatch; one heartbeat per 30 seconds). If high-volume deployments surface rate-limiting need, the operator-terminal destination's log-level filter (`CalibrationSignal` already at `DEBUG`) is the lighter mitigation; full rate-limiting is operator-tooling.
- **Per-ensemble custom event emission.** Capability ensembles may want to emit ensemble-specific events (`web-searcher` emits per-search-result events; `code-generator` emits per-syntax-pass events). The current dispatch-event substrate is **orchestrator-emitted** (events about dispatch routing and calibration), not ensemble-emitted. Per-ensemble event emission is a different question — ADR-024's envelope and ADR-025's artifact-as-substrate touch the ensemble-output surface; ADR-023 is about the dispatch surface.

---

## Rejected alternatives

### Alternative (ii) — Sidecar log: dispatch timing to orchestrator-context only via parallel log stream

A sidecar log stream emits dispatch-timing entries to a file or stream consumed only by the orchestrator-context destination; the typed-event surface is unchanged. The operator-terminal destination continues to receive only the existing post-dispatch `tool dispatch: result kind=success` lines.

**Rejected because:** the sidecar approach answers PLAY note 12's load-bearing question (*"What was the total run-time of the ensemble?"*) in isolation but violates **Inversion N+2** — it makes dispatch-timing a separate emission infrastructure for one destination only. The operator-terminal destination needs dispatch timing too: (a) for completion-event diagnostics (the operator wants to see "code-generator dispatch took 61s" in console output); (b) for liveness-signal correlation (the inference-wait heartbeat surfaces "still waiting" lines without dispatch context unless dispatch timing is in the same substrate). If dispatch-timing routes via sidecar to orchestrator-context AND via the existing console surface to operator-terminal, the implementation effectively has two parallel dispatch-timing channels — the duplication Inversion N+2 was named to prevent.

The sidecar approach also makes the call-site composition problem harder: `CalibrationVerdict`'s lack of call-site context is solved cleanly by a `dispatch_id` correlation identifier shared across all events for a dispatch; the sidecar approach requires the orchestrator-context destination to consume dispatch-timing from a separate stream and join it back to verdicts/tier-selections somehow. The event-model-extension path makes correlation a single-field addition; the sidecar makes it a cross-stream join.

This option also fails the inversion question implicit in MODEL Action C: *"What would have to be true for the sidecar-log alternative to satisfy PLAY note 12's question without requiring an event-model extension?"* The answer is: only if the operator-terminal destination does not also need dispatch timing. Spike γ's empirical evidence (Cell A-explicit's 61s `code-generator` dispatch; the operator-side absence of duration in the console line `INFO: tool dispatch: result name=invoke_ensemble kind=success`) shows the operator-terminal destination clearly does need dispatch timing. The premise of the sidecar alternative fails on inspection.

### Alternative (iii) — Orchestrator-context-only via separate mechanism (e.g., context injection at session start with rolling summary)

Dispatch timing surfaces through the orchestrator's reasoning context via a separate mechanism — context injection at session start with rolling summary, prompt-engineering, or out-of-band file-read by the orchestrator agent.

**Rejected because:** this option compounds the sidecar approach's Inversion N+2 violation — it routes dispatch timing to orchestrator-context through *yet another* mechanism (context injection / prompt-engineering / file-read) while leaving the operator-terminal destination unchanged. Three-channel dispatch-timing emission is even further from the unified-substrate Inversion N+2 commitment. It also makes the orchestrator's reasoning surface depend on a non-event mechanism (context injection or file-read) that does not compose cleanly with the existing typed-event infrastructure or with future Plexus ingestion of `execution.json` artifacts.

A subtler concern: prompt-engineering as a dispatch-timing surface couples the orchestrator's prompt budget to the dispatch-timing data volume. Sessions with many dispatches would accumulate dispatch-timing in the prompt, displacing other context. The structured-observation-between-turns pattern in destination 2 avoids this — events route into context as observations, which the model consumes and either incorporates into reasoning or summarizes via existing conversation-compaction (ADR-012).

### Routing all dispatch events to **only** the operator-terminal destination; orchestrator-context destination is not in Cycle 6 scope

A scoped alternative: ship the operator-terminal destination + liveness signals + validate-once-at-load in Cycle 6; defer orchestrator-context routing to Cycle 7+.

**Rejected because:** the orchestrator-context destination is the path that answers PLAY note 12's load-bearing question. Cycle 5 PLAY note 12 documented the orchestrator's structural blindness to its own execution graph; deferring the orchestrator-context destination keeps that blindness in place. The deferral also leaves a non-trivial future-cycle's redesign burden — the operator-terminal-only ship would implement event-routing for one destination, then Cycle 7+ would have to extend it for the second destination, when the unified substrate is the design that handles both from the start. Inversion N+2's framing — one substrate, two destinations — is *cheaper* than the scoped alternative because the substrate has to exist anyway; routing to a second destination is a structural extension, not a structural redo.

The deferral also misaligns with the Skill Orchestration User's mental model: that stakeholder asks the orchestrator dispatch-graph questions (*"What ensembles did you call? What was the total runtime?"*) — without the orchestrator-context destination, the orchestrator's answers continue to depend on the orchestrator's reasoning rather than on dispatch-event evidence the orchestrator can cite. The mental model is preserved by routing dispatch-event evidence into the orchestrator's reasoning surface.

---

## Consequences

### Positive

- **Inversion N+2 is honored architecturally.** The dispatch-event substrate is one infrastructure with two routing destinations; future destinations (TUI dashboard, Plexus ingestion, third-party monitoring) extend from the same substrate without parallel-infrastructure cost.
- **PLAY note 12's load-bearing question is answered directly.** The orchestrator-context destination's structured observations carry `duration_seconds` per dispatch; the orchestrator's reasoning surface can answer dispatch-graph questions from event evidence rather than from reasoning alone.
- **`CalibrationVerdict` call-site composition is structurally sound.** The `dispatch_id` correlation identifier shared across events for a dispatch makes joining verdicts back to call-site context a single-field operation, not a cross-stream-join.
- **The four Cycle 5 BUILD typed events are preserved.** The extension is bounded — one new event type (`DispatchTiming`), one new field on four existing event types (`dispatch_id`). No existing event semantics change; existing `execution.json` artifacts remain readable. Cycle 5 BUILD's investment in the typed-event surface compounds.
- **Liveness signals are first-class in the operator-terminal destination.** Tool-call-emit logging and inference-wait heartbeats surface in-flight state — the gap finding 7 (liveness-signal extension) named. Operators no longer experience the 10+ minute total-console-silence pattern during long inference waits.
- **Noise-floor remediation eliminates the per-enumeration validation warnings.** The 28-minute-window 8-enumeration-cycles noise pattern (finding 7) is removed; legacy YAML drift surfaces once at startup, not continuously.

### Negative

- **`dispatch_id` is a new typed field on four existing event types.** Cycle 6 BUILD's progressive conversion is needed for the events to populate the field; until conversion completes, some emission sites produce `dispatch_id: None` events. The orchestrator-context destination consuming `dispatch_id: None` events handles the case by treating those events as call-site-orphaned and not joining them to other events. The transition is observable but bounded.
- **The unified substrate routes events to two destinations with different format expectations.** The operator-terminal destination expects line-oriented `key=value`; the orchestrator-context destination expects JSON-shaped observations. The format-translation layer is implementation-level work; the typed event substrate is the single source of truth.
- **Validate-once-at-load changes operator affordance for ensemble YAML edits.** Operators editing ensemble YAMLs must trigger a library reload for changes to take effect. ADR-011's session-boundary-config discipline already establishes "changes take effect on new sessions"; the YAML-edit affordance is consistent with that but adds an operator awareness requirement.
- **The orchestrator-context destination's structured observations consume context budget.** Each dispatch contributes a small JSON observation block; high-dispatch-count sessions accumulate observations. The existing conversation-compaction (ADR-012) handles the accumulation, but operators with strict context budgets may want to disable the orchestrator-context destination (`agentic_serving.observability.orchestrator_context_routing` config flag, default `enabled`).

### Neutral

- **The bounded event-model extension keeps the dispatch-event surface narrow.** Future event types are added one at a time as need surfaces; the cycle does not pre-commit to a broader event taxonomy.
- **Liveness-signal heartbeat interval is operator-configurable.** Default 30 seconds; operators with very-low-latency or very-high-latency deployments tune `heartbeat_interval_seconds` accordingly.
- **The orchestrator-context destination's end-of-session summary in `execution.json` composes with future Plexus ingestion** — when Plexus is active (AS-8), the `dispatch_log` key in `execution.json` becomes ingestible source material per AS-4.

## Provenance check

- **Inversion N+2 framing**: Cycle 6 DISCOVER snapshot Action 3 (driver). Driver chain: DISCOVER-snapshot-derived.
- **Infrastructure-complete / routing-incomplete framing**: Cycle 5 PLAY susceptibility snapshot Advisory 3 (driver) + Cycle 6 DISCOVER snapshot Action 2 field-read (driver). Driver chain: prior-cycle-PLAY-derived + same-cycle-DISCOVER-derived.
- **Four typed events substantially carry operator-terminal needs**: Cycle 6 DISCOVER snapshot Action 2 field-read (driver). Driver chain: same-cycle-DISCOVER-derived.
- **`CalibrationVerdict` call-site composition**: Cycle 6 DISCOVER snapshot Action 2 field-read (driver). Driver chain: same-cycle-DISCOVER-derived.
- **Three architectural alternatives for dispatch timing**: Cycle 6 MODEL Action C (driver) — domain-model §Concepts entry on "Dispatch timing" enumerates (i), (ii), (iii). Driver chain: same-cycle-MODEL-derived.
- **Rejection of (ii) and (iii)**: drafting-time synthesis examining each option against Inversion N+2. The Inversion N+2 framing is driver-derived (DISCOVER snapshot); applying it to (ii) and (iii)'s structural commitments is the synthesis step.
- **`DispatchTiming` event + `dispatch_id` correlation identifier as the bounded-extension shape**: drafting-time synthesis composing the field-read finding (call-site composition needed), the dispatch-timing-fields requirement, and the Inversion N+2 commitment (one substrate). The specific shape (new event type vs. fields on existing events) is design-time judgment within the (i) event-model-extension path.
- **Liveness-signal patterns (tool-call-emit logging, inference-wait heartbeat)**: 2026-05-14 follow-on finding 7 liveness-signal extension (driver) + Cycle 6 MODEL §Concepts entry on Liveness signal (driver). Driver chain: same-cycle-follow-on-derived + same-cycle-MODEL-derived. Default interval `N=30s` is design-time judgment.
- **Noise-floor remediation (validate-once-at-load)**: 2026-05-14 follow-on finding 7 (driver). Driver chain: same-cycle-follow-on-derived. The validate-once-at-load shape is design-time judgment.
- **Two routing patterns for orchestrator-context (in-turn + end-of-session)**: drafting-time synthesis honoring the orchestrator's existing turn-boundary discipline (in-turn observations consumed during reasoning) and the existing artifact infrastructure (end-of-session summary in `execution.json`). The two patterns are not driver-derived as a pair; each pattern is derived from existing methodology mechanics.
