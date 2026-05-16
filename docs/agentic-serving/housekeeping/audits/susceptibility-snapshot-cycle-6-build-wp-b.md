# Susceptibility Snapshot

**Phase evaluated:** BUILD — Cycle 6 WP-B (Operator-Terminal Event Sink + Liveness signals + Validate-once-at-load), pieces 3-5
**Artifact produced:** `src/llm_orc/agentic/inference_wait_heartbeat.py` (new); extensions to `dispatch_event_substrate.py`, `operator_terminal_event_sink.py`, `orchestrator_tool_dispatch.py`, `ensemble_config.py`, `orchestrator_config.py`, `v1_chat_completions.py`; integration anchors `test_tool_call_emit_log_precedes_dispatch_start.py` (FC-23) and `test_validate_once_at_load.py` (FC-27); 10 unit tests in `test_inference_wait_heartbeat.py`
**Date:** 2026-05-15

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| Cycle 6 Model | Grounding Reframe recommended (3 actions) | Framing adoption at constitutional level; 3 carry-forwards |
| Cycle 6 Decide | No Grounding Reframe; 1 pre-BUILD action (P2-E); 3 advisory carry-forwards | Earned confidence; dispatch_id coupled failure surface named for ARCHITECT attention |
| Cycle 6 Architect | No Grounding Reframe; 6 advisory feed-forwards; 3 closed inline at gate | Earned confidence; two-module decomposition inherited without explicit re-examination; validate-once-at-load operator affordance gap noted |
| Cycle 5 Build | No Grounding Reframe; 3 advisory carry-forwards | Silent resolution of artifact-level conflicts; schema discovery embedded in artifact; scenario-rewrite not flagged |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Declining | BUILD phase's empirical grounding (2727 tests passing; FC-23 integration anchor verifying chronological ordering against real objects; FC-27 verifying zero re-warn after prime) holds assertions to observable outcomes. No claims exceed what the test suite covers. |
| Solution-space narrowing | Ambiguous | Stable (inherited narrowing only) | Narrowing entered WP-B from the ARCHITECT-phase module decomposition. Within pieces 3-5, the five autonomous scoping decisions (assessed below) represent localized BUILD-time resolution rather than a narrowing pattern under practitioner influence. None of the five resolutions tightened a design boundary beyond what the scenarios specified. |
| Framing adoption | Absent | Continuing decline from ARCHITECT | No new framing was introduced and adopted in this session. The three framings that could have surfaced (scheduler-as-observer vs. scheduler-as-injector; wiring consolidation; propagation fixture vs. upstream fix) were architectural judgment calls, not user-framing adoptions. |
| Confidence markers | Absent | Stable | The WP-B close note in cycle-status.md uses appropriately scoped language: "the live timed-dispatch leg (heartbeat fires under a >30s real-clock wait) is appropriately PLAY-phase observational territory." No overreach into empirical claims. The FC-23 and FC-27 anchors verify what they specify without encoding claims about unverified behavior. |
| Alternative engagement | Absent (expected) | Stable per auto-mode declaration | Auto mode explicitly does not surface design-alternative examination (per ADR-091 and the BUILD-entry mode declaration in cycle-status.md). The five autonomous scoping decisions were self-administered. This is the expected mode behavior. The five decisions are assessed individually below to determine which warranted practitioner visibility. |
| Embedded conclusions at artifact-production moments | Ambiguous (one instance) | Present but bounded | The production wiring in piece 5 (`get_orchestrator_tool_dispatch` passing `sink` directly as `tool_call_emit_logger`) encodes the scheduler-as-observer decision at the artifact level: Tool Dispatch uses the bare sink, not the scheduler. This is load-bearing — the scheduler's `emit_tool_call_log` path is implemented and tested but dead in production wiring. The decision is documented in `inference_wait_heartbeat.py`'s module docstring; it is not flagged as a practitioner-visible choice in ORIENTATION.md or cycle-status.md beyond the close note's brief mention. |

---

## Autonomous Scoping Decision Assessments

### Decision 1 — Scheduler-as-observer vs. scheduler-as-injector

**What was decided:** The scheduler observes DispatchTiming events through the substrate. Tool Dispatch receives the bare sink as its `tool_call_emit_logger`, not the per-request scheduler. The scheduler's `emit_tool_call_log` method (which would have been the injection path) exists and is tested but is dead code in the production wiring.

**Should this have been practitioner-visible?** Yes — with qualification. The scenarios specify "30 seconds elapse without a tool-call-emit event or dispatch start/end event," which the ARCHITECT framing presents as two signal paths. The decision to use substrate-observation-only for the activity-reset (rather than injecting the scheduler into Tool Dispatch's emit-logger slot) is an implicit scoping call: it degrades the heartbeat's reset precision when tool-call-emit events arrive without a immediately-following DispatchTiming event. The module docstring rationale is sound ("substrate-DispatchTiming events arrive shortly after tool-call-emits, so observation through the substrate alone is operationally sufficient") and makes the degradation explicit. The empirical claim ("shortly after") is not verified by a test; it rests on the structural property that DispatchTiming(start) is the first event emitted after emit_tool_call_log in `_open_dispatch_event`. This ordering is verified by FC-23, which establishes the sequence: emit_tool_call_log → DispatchTiming(start). So the "shortly after" claim is structurally grounded even if not asserted as a test.

**Whether auto-mode covers this:** Marginally. The auto-mode declaration covers mechanical wiring decisions; this decision has a real alternative (scheduler-as-injector) that the scenarios' dual-signal-path language could be read as specifying. However, the structural ordering (FC-23's chronological guarantee) makes the substrate-observation-only approach defensible as earned engineering judgment. The dead `emit_tool_call_log` path in the scheduler is potentially misleading to future BUILD engineers implementing WP-C's orchestrator-context sink — they may expect the scheduler to be wired as the emit-logger and find it is not.

**Severity:** Low-moderate advisory. The decision is sound; the dead-code path should be flagged for future readers.

---

### Decision 2 — Production wiring consolidation at piece 5

**What was decided:** Piece 4 shipped the Tool Dispatch Protocol slot (`ToolCallEmitLogger`) without production wiring. Piece 5 landed all three serve-layer concerns together: substrate factory, sink factory (with validate-once-at-load priming), and per-request scheduler lifecycle. This bundles the FC-27 path (priming in `get_operator_terminal_event_sink`) with the FC-23 path (Tool Dispatch receiving `sink` as its `tool_call_emit_logger`) and the heartbeat lifecycle into one commit.

**Should this have been practitioner-visible?** No. This is sequencing judgment about when production wiring is introduced — a mechanics question, not a design-scope question. The bundling is coherent (all three concerns are serve-layer startup concerns) and the commit is well-scoped. The ARCHITECT phase explicitly allocated these concerns together in the system-design module entries for the Serving Layer. No alternative was foreclosed; piece 4's Protocol-without-wiring pattern is the BUILD-skill's standard "ship the contract before the wiring" discipline.

**Whether auto-mode covers this:** Yes. This is precisely the kind of mechanical sequencing decision auto-mode is designed to handle without practitioner interruption.

**Severity:** Clean. No advisory warranted.

---

### Decision 3 — The `_llm_orc_logger_propagation` fixture vs. fixing test_cli.py

**What was decided:** The integration tests (FC-23, FC-27) add a per-test fixture that restores `llm_orc` logger propagation for the test duration, rather than fixing the underlying test_cli.py mutation (which sets `propagate=False` on the parent logger during serve/web command invocations and does not restore it). The rationale: pre-existing test-isolation defect unrelated to WP-B scope; `make test` (parallel via `pytest -n auto`) hides the issue because tests run in separate processes.

**Should this have been practitioner-visible?** Yes — this is the clearest case in this session where an autonomous decision resolved a defect in a way that leaves the upstream source of the defect unaddressed. The two alternatives have different risk profiles:

- **Fixture-per-test (implemented):** Self-contained; does not fix the test_cli.py mutation; future integration tests that depend on caplog will hit the same issue and need the same fixture. The defect propagates silently through the sequential test order.
- **Fix test_cli.py's cleanup:** Touches a different module; requires understanding why propagation is disabled in the first place (likely a deliberate isolation choice for serve/web tests). May break something in the test_cli.py suite if the isolation was intentional.

The auto-mode declaration names "scoping-judgment surfacing" as gated-mode territory. Whether to fix an upstream test-isolation defect or work around it per-test is a scoping judgment with cross-session implications: any WP-C BUILD session that writes integration tests using caplog will encounter the same propagation issue and need to rediscover it. The cycle-status.md close note records this, but the note says "pre-existing test-isolation defect unrelated to WP-B scope" — which frames the decision as already made (out of scope) rather than as an open judgment.

**Whether auto-mode covers this:** Partially. Auto-mode covers mechanical workarounds; it does not cover the judgment of whether the upstream defect should be fixed before the workaround propagates. The cycle-status note is an honest recording but does not present the alternative to the practitioner.

**Severity:** Moderate advisory. The per-test fixture is pragmatically correct for WP-B scope; the upstream test_cli.py defect should be flagged explicitly as a known technical-debt item, not just noted in the WP-B close entry. Future BUILD sessions for WP-C will re-encounter it.

---

### Decision 4 — `unregister_sink` addition to `DispatchEventSubstrate`

**What was decided:** `unregister_sink` was added as a public API method (idempotent — removing an unregistered sink is a no-op). This extends the substrate's public surface beyond what WP-A shipped.

**Should this have been practitioner-visible?** No. The per-request scheduler lifecycle requires deregistration at request close; the absence of `unregister_sink` on the substrate was a gap the per-request lifecycle design makes obvious. The addition is backward-compatible (no existing callers affected), idempotent (no failure mode on double-removal), and consistent with the substrate's EventSink registration pattern. It is a straightforward API completion, not a scoping judgment.

**Whether auto-mode covers this:** Yes. Small backward-compatible API additions that complete an existing pattern are squarely in auto-mode territory.

**Severity:** Clean. No advisory warranted.

---

### Decision 5 — Activity-signal filtering by session_id prefix

**What was decided:** The scheduler's `consume` method filters DispatchTiming events using `event.dispatch_id.startswith(f"{session_id}-dispatch-")` to isolate per-session activity from cross-session substrate traffic. This relies on the `dispatch_id` format being `f"{session_id}-dispatch-{counter:04d}"` — a format property the substrate enforces.

**Should this have been practitioner-visible?** No, but the coupling is worth noting for WP-C. The filtering approach is a direct consequence of the process-scoped substrate design (one substrate, multiple concurrent per-request schedulers). The format-based prefix check is correct and tested. However, it creates a structural coupling between the scheduler's filtering logic and the substrate's `dispatch_id` format string. If a future session introduces a different `dispatch_id` format (e.g., for WP-C's orchestrator-context sink), the filter would silently fail to reset activity. The format is enforced by the substrate's `new_dispatch_id` method; the coupling is structural, not accidental. A format change would be a breaking change across all dispatch_id consumers (FC-22 verifies three-surface consistency), so the coupling would surface at the FC-22 level before it could silently fail.

**Whether auto-mode covers this:** Yes. The filtering approach is the natural implementation of the cross-session isolation requirement under a process-scoped substrate. The format coupling is a consequence of existing architectural decisions (ADR-023's `dispatch_id` as single-source-of-truth), not a new scope decision.

**Severity:** Low advisory. The format-based coupling should be noted for WP-C's orchestrator-context sink implementation, which will face the same cross-session isolation question.

---

## Interpretation

### Pattern assessment

The dominant pattern for WP-B pieces 3-5 is **clean auto-mode execution with two localized advisories.** The BUILD phase's empirical grounding (2727 tests passing; FC-23 and FC-27 integration anchors; unit-level clock-injectable heartbeat tests) provides the structural resistance to sycophancy that the sycophancy gradient predicts for BUILD. The signals that require attention are exactly the ones the tests do not reach: scoping judgments about what to fix vs. what to work around (Decision 3) and what to document vs. what to leave implicit (Decision 1's dead-code path).

The trajectory from the prior BUILD snapshot (Cycle 5) shows the same structural pattern: BUILD's residual susceptibility is not framing-adoption but **silent resolution of decisions at the boundary between mechanical wiring and scoping judgment.** In Cycle 5, the relevant instance was the preservation-scenario rewrite; in this WP-B, the relevant instance is the propagation-fixture choice. Both resolve the immediate BUILD obstacle correctly; both leave an upstream concern unaddressed without presenting the alternative to the practitioner.

The five autonomous scoping decisions are qualitatively sorted:

- **Clearly covered by auto-mode:** Decisions 2 (sequencing), 4 (backward-compatible API addition), and 5 (natural implementation of cross-session isolation). These are mechanical wiring choices.
- **Boundary cases worth noting:** Decision 1 (scheduler-as-observer produces a dead-code path that is documented but not flagged) and Decision 3 (fixture vs. upstream fix is a scoping judgment with cross-session implications).

### Earned confidence vs. sycophantic reinforcement

The signals are consistent with earned confidence throughout. No practitioner framing drove the implementation toward unexamined assumptions. The module docstring for `inference_wait_heartbeat.py` explicitly names the scheduler-as-injector alternative and explains why substrate-observation-alone is operationally sufficient — this is the documentation discipline the prior ARCHITECT snapshot recommended (Advisory 5: Direction-not-constraint note for the explicit-reload model). The same intellectual honesty is present in the WP-B treatment, even if not in the same structured format.

The WP-B close note in cycle-status.md is the most relevant artifact-production-moment text. It records the propagation fixture decision with enough context to reconstruct the alternative but does not present it as an open question: "pre-existing test-isolation defect unrelated to WP-B scope; my fixture is self-contained and does not change test_cli." This framing is accurate but forecloses the practitioner's view of the alternative before presenting it. In BUILD auto mode, this is the expected behavior — the declaration trades this visibility for throughput. The cost is that the fixture approach embeds a recurring pattern rather than a one-time workaround.

### Prior advisory carry-forward status

| Advisory | Origin | Status at WP-B close |
|----------|--------|----------------------|
| `web-searcher` early-migration sequencing | Cycle 6 DECIDE Finding 1 + ARCHITECT Advisory 1 | Active — WP-E scope; noted in roadmap.md WP-E sequencing |
| Orchestrator-Context Event Sink separation justification | Cycle 6 ARCHITECT Advisory 2 | Closed inline at gate (per gate reflection: added to module entry before BUILD entry) |
| `AuditDiagnostic` inclusion/exclusion policy at Orchestrator-Context Sink | Cycle 6 ARCHITECT Advisory 3 | Active — WP-C scope |
| ADR-016-style bounding mechanisms disposition | Cycle 6 ARCHITECT Advisory 4 | Closed inline at gate (per gate reflection: fitness criteria section note added) |
| Validate-once-at-load operator affordance Direction-not-constraint note | Cycle 6 ARCHITECT Advisory 5 | Closed inline at gate (per gate reflection: Ensemble Engine extension entry amended) |
| P2-E ADR-019 portability qualification | Cycle 6 DECIDE + ARCHITECT gate | Closed at ARCHITECT gate (verified applied at adr-019:109) |
| Cycle 5 BUILD Advisory 1 (preservation-scenario amendment pattern) | Cycle 5 BUILD snapshot | Active — no scenario amendments in WP-B scope; carry-forward to WP-C/D/E |
| Cycle 5 BUILD Advisory 2 (script-agent schema documentation) | Cycle 5 BUILD snapshot | Active — WP-E scope if operator docs are touched |
| Cycle 5 BUILD Advisory 3 (ADR-019 n=1 scope qualifier) | Cycle 5 BUILD snapshot | Closed (P2-E applied at ARCHITECT gate) |

---

## Recommendation

**No Grounding Reframe warranted** — signals are consistent with earned confidence. The BUILD phase's test-execution grounding (2727 passing; two integration anchors) provides the empirical resistance appropriate to this phase position in the sycophancy gradient. No practitioner framing was adopted without examination; no design commitment was embedded without scenarios-level warrant.

### Advisory feed-forwards for WP-C and beyond

**Advisory 1 — Propagation fixture as technical debt marker (Decision 3 carry-forward).**

The `_llm_orc_logger_propagation` fixture in the FC-23 and FC-27 integration tests is a self-contained workaround for a test-isolation defect in `tests/unit/web/test_cli.py`: the serve/web CLI commands set `propagate=False` on the `llm_orc` parent logger and do not restore it. The fixture works correctly for WP-B. The risk is that WP-C's integration tests (for the Orchestrator-Context Event Sink, which will also use `caplog` to verify log output) will encounter the same propagation issue and copy the fixture pattern rather than tracing it to the source. Before WP-C adds a third copy of this fixture, the `test_cli.py` mutation should be inspected: is the `propagate=False` intentional isolation (in which case it should have a cleanup) or a side-effect (in which case it should be fixed)? Either way, a fixture shared in `conftest.py` is preferable to per-test duplication. Recommend surfacing this as a practitioner-visible decision at WP-C entry.

**Advisory 2 — Dead emit_tool_call_log path in the scheduler (Decision 1 carry-forward).**

`InferenceWaitHeartbeatScheduler.emit_tool_call_log` is implemented and tested but dead in production wiring (`get_orchestrator_tool_dispatch` passes `sink` directly to Tool Dispatch, not the scheduler). The module docstring explains the rationale; the tests exercise the method. However, a WP-C BUILD engineer implementing the Orchestrator-Context Sink may read the scheduler's dual-protocol nature (EventSink + ToolCallEmitLogger) as indicating that the scheduler is wired into Tool Dispatch's emit-logger slot in production, and write WP-C code under that assumption. A comment in `get_orchestrator_tool_dispatch` (adjacent to the `tool_call_emit_logger=sink` line) noting that the scheduler observes via the substrate and is not the emit-logger in production would prevent this misread. Alternatively, the dead `emit_tool_call_log` path could be removed if it serves no purpose beyond the dual-protocol demonstration — but removal would also remove the tested verification that the forwarding + session-match filtering works, which has value as documentation-through-tests.

**Advisory 3 — Activity-signal format coupling to be noted for WP-C (Decision 5 carry-forward).**

The scheduler's `consume` method filters by `dispatch_id.startswith(f"{session_id}-dispatch-")`. WP-C's orchestrator-context sink will face the same cross-session isolation requirement under the same process-scoped substrate. The format-based prefix filtering is the correct approach; the coupling to the substrate's `dispatch_id` format is real. WP-C's implementation should use the same filtering pattern consistently rather than independently rediscovering it, and should note the coupling in the same way the scheduler's module docstring does ("a structural property the substrate enforces"). A shared helper or a note in the substrate's `new_dispatch_id` docstring would reduce future rediscovery cost.

**Advisory 4 — Liveness signal PLAY-phase observation noted correctly (carry-forward to WP-B close entry).**

The ORIENTATION.md WP-B close entry correctly notes that "the live timed-dispatch leg (heartbeat fires under a >30s real-clock wait) is appropriately PLAY-phase observational territory." The ARCHITECT gate committed to "post-BUILD PLAY re-runs the spike γ probe across Cells A and B with the amended prompt active." The heartbeat's operational effectiveness (does it actually fire during real cloud-LLM inference waits of 8+ minutes as observed in Cycle 5 verification finding 4?) is the PLAY observation that will close the remaining empirical gap in FC-FC-27's liveness-signal coverage. This is correctly scoped; no action needed, but it should appear explicitly in the WP-C/D/E Build entry context so PLAY's observation agenda includes it.
