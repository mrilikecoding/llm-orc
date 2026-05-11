# Conformance Scan Report — Cycle 4 DECIDE-Phase ADRs

**Scanned against:**
- ADR-012 (`adr-012-conversation-compaction-five-layer-pipeline.md`)
- ADR-013 (`adr-013-session-registry-initializer-then-resume-schema.md`)
- ADR-014 (`adr-014-calibration-gate-trajectory-level-extension.md`)
- ADR-015 (`adr-015-per-role-tier-escalation-router.md`)
- ADR-016 (`adr-016-upward-l0-l1-read-only-signal-channel.md`)
- ADR-017 (`adr-017-tool-call-structural-validation-guard.md`)
- ADR-002 (`adr-002-four-layer-architecture-plexus-optional.md`) — layering rule interaction
- ADR-007 (`adr-007-calibration-gate-for-composed-ensembles.md`) — ADR-014 substrate
- ADR-011 (`adr-011-orchestrator-llm-is-a-model-profile.md`) — ADR-015 instantiation

**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc/`

**Date:** 2026-05-06

---

## Summary

- **ADRs checked:** 9 (6 new Cycle 4 ADRs + 3 prior ADRs as interaction surfaces)
- **Conformance surfaces examined:** 8 (per scan brief)
- **Violations found (structural cleanup territory):** 0
- **Implementation gaps (new-feature scenario territory):** 10
- **Partial conformance (ADR partially implemented):** 3
- **Implementation present, ADR-conformant:** 5

---

## Conformance Debt Table

| # | ADR | Violation / Gap | Type | Location | Commit Reference | Classification |
|---|-----|-----------------|------|----------|-----------------|----------------|
| 1 | ADR-012 | Five-layer conversation compaction pipeline — no implementation | missing | `src/llm_orc/agentic/` (no compaction module) | — | Implementation absent — full implementation gap |
| 2 | ADR-012 | Layer 0: persist-large-tool-results (>50,000 chars to disk with 2 KB preview) — not implemented | missing | `src/llm_orc/agentic/orchestrator_tool_dispatch.py` invoke path | — | Implementation gap |
| 3 | ADR-012 | Layer 3: session-notes nine-section template — no structure exists | missing | `src/llm_orc/agentic/session_registry.py` (no notes field) | — | Implementation gap |
| 4 | ADR-012 | Layer 4: LLM semantic summary with circuit-breaker (3-failure limit, per-session state reset) — not implemented | missing | `src/llm_orc/agentic/` (no compaction module) | — | Implementation gap |
| 5 | ADR-012 | `LlmOrcStructuralError` base class (ADR-017 §Shared typed-error base class) — Layer 4 failure typed error requires it | missing | `src/llm_orc/models/base.py:99` — existing `ToolCallingNotSupportedError` is `NotImplementedError` subclass, no shared base | — | Implementation gap (prerequisite of ADR-017 shared base) |
| 6 | ADR-013 | Session Registry structured-handoff artifacts — no implementation | missing | `src/llm_orc/agentic/session_registry.py` (tracks only `SessionState` with `turn_count`, `token_spend`) | — | Implementation absent — full implementation gap |
| 7 | ADR-013 | Write-gate validation surface (JSON schema validation, append-only constraint, signed-script integrity) — not implemented | missing | `src/llm_orc/agentic/session_registry.py` | — | Implementation gap |
| 8 | ADR-013 | Cluster determination at session-start (Cluster 2 default-required, Cluster 1/3 optional) — not implemented | missing | `src/llm_orc/agentic/session_start.py`, `src/llm_orc/agentic/session_registry.py` | — | Implementation gap |
| 9 | ADR-014 | In-process trajectory-level calibration (AUQ verbalized-confidence + HTC trajectory features) — not implemented | missing | `src/llm_orc/agentic/calibration_gate.py` | — | Implementation absent — full implementation gap |
| 10 | ADR-014 | Calibration verdict trichotomy (Proceed / Reflect / Abstain) — not implemented; existing gate returns `QualitySignal` | missing | `src/llm_orc/agentic/calibration_gate.py:53` (`QualitySignal = Literal["positive", "negative", "absent"]`) | — | Implementation gap |
| 11 | ADR-014 | Time-decay windowing on trajectory features — not implemented | missing | `src/llm_orc/agentic/calibration_gate.py` | — | Implementation gap |
| 12 | ADR-015 | Per-role tier-escalation router in Tool Dispatch (L2 interposition) — not implemented | missing | `src/llm_orc/agentic/orchestrator_tool_dispatch.py` (no tier-selection logic; direct invocation to `self._operations.invoke`) | — | Implementation absent — full implementation gap |
| 13 | ADR-015 | Topaz skill metadata field on ensemble YAML — not present on any ensemble | missing | `.llm-orc/ensembles/` (no `topaz_skill` or equivalent field in any YAML) | — | Implementation gap; requires one-time migration |
| 14 | ADR-015 | Per-skill tier defaults configuration (8 skills × 2 tiers = 16 Model Profile slots) — not present | missing | `.llm-orc/config.yaml` (no tier-defaults section) | — | Implementation gap |
| 15 | ADR-016 | L0→L1 read-only signal channel — not implemented | missing | `src/llm_orc/agentic/calibration_gate.py`, `src/llm_orc/core/execution/` (no cross-layer signal emission) | — | Implementation absent — full implementation gap (ADR is conditional-acceptance status) |
| 16 | ADR-016 | Bounding mechanisms (a)–(e) — none implemented | missing | `src/llm_orc/agentic/calibration_gate.py` | — | Implementation gap (ADR conditional-acceptance; first-deployment evidence pending) |
| 17 | ADR-016 | FC-2 (static import check) and FC-3 (cycle detection) do not recognize calibration-channel exception | missing | `tests/` (FC-2 and FC-3 have **no test files** — see Surface 7 below) | — | Implementation gap: update blocked on FC-2/FC-3 themselves not existing as automated checks |
| 18 | ADR-017 | Structural validation guard (phantom_tool_call pattern detection + cross-check) — not implemented | missing | `src/llm_orc/agentic/orchestrator_tool_dispatch.py` (no pattern scan in `dispatch()` or `invoke_ensemble()`) | — | Implementation absent — full implementation gap |
| 19 | ADR-017 | Shared `LlmOrcStructuralError` base class with `error_kind`, `dispatch_context`, `recovery_action_required`, `operator_diagnostic` fields — not implemented | missing | `src/llm_orc/models/base.py:99` — `ToolCallingNotSupportedError(NotImplementedError)` is the only typed error; no shared structural base | `9f86d0b` | Implementation gap; existing typed error would be the first `error_kind` entry |
| 20 | ADR-016 (mech. a) | Fresh-context isolation for audit-subagent dispatches claimed as "direct architectural precedent" — no codebase infrastructure for audit-subagent dispatch pattern | wrong-structure | `src/llm_orc/agentic/` — the claim in ADR-016 §mechanism (a) that "the methodology itself uses fresh-context isolation for audit-subagent dispatches" and "the pattern is established in the codebase (audit-subagent dispatch infrastructure)" is **overstated**; the codebase has no audit-subagent dispatch infrastructure; the pattern exists in the RDD methodology ensembles but not in llm-orc's own orchestrator code | — | Implementation absent — the "direct architectural precedent" claim in ADR-016 requires qualification |

---

## Per-Surface Findings

### Surface 1 — ADR-016 mechanism (a): Architectural Isolation precedent

**ADR claim:** Mechanism (a) (fresh-context isolation in the L1 calibration consumer) has "direct architectural precedent" because "the methodology itself uses fresh-context isolation for audit-subagent dispatches" and "the pattern is established in the codebase (audit-subagent dispatch infrastructure)."

**Codebase finding:** No audit-subagent dispatch infrastructure exists in `src/llm_orc/`. A comprehensive grep for `audit`, `subagent`, `sub.agent`, `fresh.*context`, `context.*fresh`, `fresh.*evaluat`, `dispatch.*audit` across all source files returns zero results in `src/`. The methodology's citation-auditor, argument-auditor, and susceptibility-snapshot-evaluator patterns are realized through the ensemble YAML library (`.llm-orc/ensembles/`) and the RDD plugin, not as a codebase-internal pattern that the orchestrator's L1 code instantiates.

**Verdict:** The claim of "direct architectural precedent" is overstated with respect to the codebase. The isolation pattern is real in the methodology's ensemble-based audit dispatches but does not exist as a coded infrastructure pattern within `src/llm_orc/`. For BUILD purposes, the mechanism is implementable from the ensemble dispatch pattern — the architectural isolation property is achieved through each audit ensemble invocation running in its own independent execution context. The ADR should be read as claiming methodology precedent (valid) rather than codebase-infrastructure precedent (overstated).

**Classification:** wrong-structure — the ADR's precedent claim is more accurate as "methodology-level pattern, expressible through ensemble dispatch" rather than "codebase infrastructure."

---

### Surface 2 — ADR-017 typed-error pattern at commit `9f86d0b`

**ADR claim:** Commit `9f86d0b feat: raise typed error when provider rejects tool calling per-model` provides "direct codebase precedent" for the typed-error pattern. ADR-017 specifies a shared `LlmOrcStructuralError` base class with common fields: `error_kind`, `dispatch_context`, `recovery_action_required`, `operator_diagnostic`.

**Codebase finding:** Commit `9f86d0b` exists and modifies `src/llm_orc/models/openai_compat.py` to raise `ToolCallingNotSupportedError` (defined at `src/llm_orc/models/base.py:99`) when a provider returns HTTP 400 with "does not support tools". The error class is:

```python
class ToolCallingNotSupportedError(NotImplementedError):
    """Raised when a model that does not support tool calling is asked to."""
```

**Gap assessment:** The codebase precedent establishes the typed-error *pattern* (raise a named typed exception rather than a generic `RuntimeError`) but does not implement the *shared base class* ADR-017 specifies. `ToolCallingNotSupportedError` has no `error_kind`, `dispatch_context`, `recovery_action_required`, or `operator_diagnostic` fields — it is a plain exception with a message string. The ADR's shared base class specification (`LlmOrcStructuralError` or equivalent) is an implementation gap. The existing error would become the first concrete subclass once the base class is introduced.

**Classification:** partial conformance — the typed-error discipline (pattern) matches the ADR's intent; the shared base class with common fields is an implementation gap.

---

### Surface 3 — ADR-014 composition with ADR-007 (Calibration Gate substrate)

**ADR-007 substrate:** ADR-007 is fully implemented. `CalibrationGate` at `src/llm_orc/agentic/calibration_gate.py` provides post-hoc result-checking for composed ensembles. The gate tracks per-session per-ensemble calibration records, runs an `EnsembleBackedChecker` on in-calibration ensembles, and produces `QualitySignal` (`positive | negative | absent`). Tool Dispatch interposes `mark_composed` on `compose_ensemble` success and `check_and_record` on `invoke_ensemble` before summarization. FC-12 is verified via `tests/integration/test_tool_dispatch_calibration_boundary.py`.

**ADR-014 layer:** ADR-014 extends ADR-007 with in-process trajectory-level calibration: AUQ verbalized-confidence (System 1 soft propagation + System 2 binary gate), HTC trajectory features (token-level entropy, attention-weight distributions, decision-confidence trajectories), and a verdict trichotomy (Proceed / Reflect / Abstain). None of these extensions exist in the codebase. `CalibrationGate._compute_status()` operates solely on accumulated `QualitySignal` values from the post-hoc checker; there is no trajectory-feature extraction, no AUQ confidence signal, no verdict trichotomy.

**Classification:** Partial conformance on ADR-007 (fully implemented); implementation absent — full implementation gap for ADR-014's extensions.

---

### Surface 4 — ADR-015 and ADR-003 closed five-tool surface / Tool Dispatch interposition

**Current Tool Dispatch implementation:** `OrchestratorToolDispatch` in `src/llm_orc/agentic/orchestrator_tool_dispatch.py` routes the closed five-tool surface (`invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`). The `invoke_ensemble` method at line 307 delegates directly to `self._operations.invoke(...)` after argument validation. The existing interpositions on the `invoke_ensemble` path are, in order: (1) Autonomy Policy gate, (2) Calibration Gate `check_and_record` (post-result), (3) Result Summarizer Harness. No tier-selection or escalation router exists between the tool call and `self._operations.invoke(...)`.

**ADR-015 gap:** The per-role tier-escalation router that ADR-015 specifies would interpose between the `invoke_ensemble` call and `self._operations.invoke(...)`: reading the dispatched ensemble's Topaz skill metadata, consuming a calibration verdict from ADR-014, and selecting a cheap-tier or escalated-tier Model Profile. None of this exists. The Topaz skill metadata field does not appear in any ensemble YAML in `.llm-orc/ensembles/`. The per-skill tier defaults configuration surface does not exist in `.llm-orc/config.yaml`.

**Classification:** Implementation absent — full implementation gap for ADR-015. The existing Tool Dispatch structure is the correct host for the router (it already interposes on the `invoke_ensemble` path) — no structural cleanup needed before ADR-015 BUILD work begins.

---

### Surface 5 — ADR-013 Session Registry (current state vs. extended responsibility)

**Current Session Registry implementation:** `SessionRegistry` at `src/llm_orc/agentic/session_registry.py` identifies and continues a multi-request Session. The implementation resolves identity from the OpenAI `user` field, first-user-message prefix hash, or cold-start UUID, and maintains an in-memory `dict[SessionIdentity, SessionState]`. `SessionState` tracks `turn_count` and `token_spend`. No structured-handoff artifacts exist.

**ADR-013 gap:** ADR-013 adds three adoption-derived components (feature-list-with-monotonic-passes, append-only progress log, init-sh-style deterministic environment bootstrap) and one novel-design component (write-gate validation). The existing `SessionState` dataclass has no artifact fields, no cluster-determination logic, and no write-gate surface. The init.sh hash-based integrity verification does not exist.

The current Session Registry's responsibility (identifies and continues a multi-request Session) is exactly the substrate ADR-013 extends; the extension is additive, not contradictory. No structural cleanup is required before ADR-013 BUILD work begins.

**Classification:** Implementation absent — full implementation gap for the ADR-013 extensions. The existing SessionRegistry structure is ADR-conformant for its current scope (pre-ADR-013).

---

### Surface 6 — ADR-012 Conversation Compaction (current state)

**Current implementation:** No Conversation Compaction module exists in `src/llm_orc/agentic/`. The Orchestrator Runtime at `src/llm_orc/agentic/orchestrator_runtime.py` mentions "Conversation Compaction" in its module docstring (line 21: "Aware of Routing Decisions and Conversation Compaction") but has no implementation — the concept appears in the system-design's module description for the Runtime but has not been built. The system-design module table at `system-design.md:78` describes the Runtime as "Aware of Routing Decisions and Conversation Compaction; unaware of summarization, Plexus, Autonomy, or Calibration."

**ADR-012 gap:** The five-layer pipeline (persist-large-tool-results, cache-edit, idle-expiry, session-notes template, LLM-summary with circuit-breaker) does not exist. No context threshold logic, no tool-result size check, no session-notes structure, no circuit-breaker state.

**Classification:** Implementation absent — full implementation gap. The `orchestrator_runtime.py` module docstring's mention of Compaction awareness is forward-looking documentation rather than implemented behavior.

---

### Surface 7 — Layering rule discipline (FC-2 static import check, FC-3 cycle detection)

**Current state:** FC-2 (dependency edges point from higher layer to same-or-lower layer only; verified by static inspection of module imports against L0-L3 assignment) and FC-3 (no cycles in the dependency graph; verified by static cycle detection) are specified in `system-design.agents.md` lines 360-361 as fitness criteria and referenced in `system-design.md:56` as structural guarantees.

**Test coverage:** No test files for FC-2 or FC-3 exist in the test suite. The test files that exist for other fitness criteria follow a naming pattern (`test_fc4_runtime_import_surface.py`, `test_fc8_summarizer_bypass.py`, `test_fc9_session_start_contract.py`, `test_fc11_autonomy_gate.py`). FC-2 and FC-3 have no corresponding test files. Cycle-1 archive notes (line 71 of `cycle-1-agentic-serving.md`) state "FC-1/FC-2/FC-3 pass" at WP-B close, but the pass is by assertion and inspection rather than automated test.

**ADR-016 interaction:** ADR-016's Consequences section states: "ADR-002's layering rule now has an exception that future ADRs and FCs must understand — the static import check (FC-2) and cycle detection (FC-3) need updating to recognize the calibration-channel exception." This update cannot be made to tests that do not yet exist as automated artifacts. The ADR-016 consequence is blocked on FC-2 and FC-3 first being implemented as automated checks before they can be amended.

**Classification (FC-2 and FC-3 themselves):** Implementation absent — FC-2 and FC-3 are specified fitness criteria but not automated tests. This is a pre-existing gap, not introduced by Cycle 4.

**Classification (ADR-016's FC-2/FC-3 update requirement):** The ADR-016 required update to FC-2/FC-3 cannot be executed until FC-2 and FC-3 exist as automated checks. Scenario writing for ADR-016 BUILD work should include: (a) implement FC-2 as an automated static import check across the `src/llm_orc/agentic/` module graph; (b) implement FC-3 as an automated dependency-graph cycle detection; (c) then amend both to recognize the calibration-channel exception as a permitted upward signal path.

---

### Surface 8 — ADR-011 Orchestrator Model Profile (ADR-015 interaction)

**Current implementation:** `OrchestratorConfig` at `src/llm_orc/agentic/orchestrator_config.py` holds `model_profile: str`, resolved by `OrchestratorConfigResolver` from `config.yaml`. The profile is a session-boundary value — it is fixed at session-start and does not change within a session (ADR-011 conformant). No tier-escalation or per-dispatch profile selection exists at the orchestrator level. The orchestrator's Model Profile is a single value per session, not a tier-aware surface.

**ADR-015 compatibility:** ADR-015's tier-escalation router operates on the *dispatched ensemble's* Model Profile, not the orchestrator's. The orchestrator's session-boundary-scoped Model Profile (ADR-011) is correctly untouched by ADR-015's design. FC-13 (changing the orchestrator Model Profile requires touching only Orchestrator Configuration and Session start-logic — not Runtime, not Tool Dispatch) continues to hold under ADR-015's proposed interposition because the router selects a tier for the dispatched ensemble, not for the orchestrator itself.

**Classification:** Implementation present, ADR-conformant. The orchestrator's Model Profile surface correctly honors ADR-011's session-boundary discipline. ADR-015's tier router would be additive to Tool Dispatch (not the orchestrator config), preserving ADR-011 compliance.

---

## Conformance Status by ADR

| ADR | Status | Conformance | Notes |
|-----|--------|-------------|-------|
| ADR-007 (substrate for ADR-014) | Accepted | **Conformant** | `CalibrationGate` fully implemented; FC-12 satisfied |
| ADR-011 (substrate for ADR-015) | Accepted | **Conformant** | Session-boundary Model Profile correctly implemented; FC-13 holds |
| ADR-002 (layering rule) | Accepted | **Partial** | Layering rule stated and architecturally observed; FC-2 and FC-3 are unimplemented as automated checks |
| ADR-012 (Conversation Compaction) | Proposed | **Full implementation gap** | No compaction module; concept mentioned in Runtime docstring only |
| ADR-013 (Session Registry artifacts) | Proposed | **Full implementation gap** | SessionRegistry exists for current scope; ADR-013 extensions absent |
| ADR-014 (Calibration Gate trajectory extension) | Proposed | **Full implementation gap** | ADR-007 substrate in place; trajectory layer, verdict trichotomy, time-decay windowing absent |
| ADR-015 (Tier-escalation router) | Proposed | **Full implementation gap** | Tool Dispatch host exists; router logic, Topaz metadata, tier-defaults config absent |
| ADR-016 (L0→L1 signal channel) | Proposed (conditional acceptance) | **Full implementation gap** | No cross-layer signal channel; bounding mechanisms absent; FC-2/FC-3 updates blocked on FC-2/FC-3 not existing as automated tests |
| ADR-017 (Tool-call structural validation guard) | Proposed | **Full implementation gap** | No phantom-tool-call pattern detection; `LlmOrcStructuralError` shared base class absent; typed-error discipline present from `9f86d0b` but fields not yet structured |

---

## Notes

### All Cycle 4 ADRs are implementation gaps, not structural violations

None of the six new ADRs (ADR-012 through ADR-017) describe existing code that contradicts the architecture. The codebase at Cycle 4 DECIDE close is in a correct pre-implementation state for all six ADRs. There are no `refactor:` commits required before BUILD begins — no existing code that exists and shouldn't, no wrong-structure cases where the right behavior is in the wrong path.

The single partial exception is the ADR-016 mechanism (a) precedent claim, which is wrong-structure in the documentation rather than the code: the ADR overstates "direct codebase precedent" when the precedent is methodology-level (ensemble-based audit dispatch in the RDD plugin) rather than codebase-infrastructure. This does not require a code change; it warrants a clarifying note in the ADR or scenario writing that builds the mechanism from ensemble dispatch primitives rather than from an assumed codebase infrastructure.

### Scenario writing guidance from this scan

Based on the implementation gap table, Cycle 4 scenarios fall into two tiers:

**Tier 1 — structural prerequisites that unlock other work:**
- FC-2 automated static import check (unlocks ADR-016's FC-2 amendment)
- FC-3 automated cycle detection (unlocks ADR-016's FC-3 amendment)
- `LlmOrcStructuralError` base class (unlocks ADR-017, ADR-012 Layer 4 error, ADR-013 write-gate errors, ADR-014 Abstain error, ADR-016 mechanism (e) malformed-signal error — all five reference the shared base)

**Tier 2 — independent module BUILD work:**
- ADR-017 phantom-tool-call guard (narrowest scope; most direct codebase precedent; suitable early BUILD)
- ADR-013 Session Registry artifact set (additive to existing `SessionRegistry`; cluster-determination surface is the novel-design risk area)
- ADR-012 Conversation Compaction five-layer pipeline (L1 elaboration in the Orchestrator Runtime; circuit-breaker state is novel complexity)
- ADR-014 Calibration Gate trajectory-level extension (ADR-007 substrate in place; trajectory feature extraction is the novel-without-codebase-precedent work)
- ADR-015 Tier-escalation router (requires Topaz skill metadata migration on all ensembles + per-skill tier-defaults config surface + ADR-014's calibration verdict as data input)
- ADR-016 L0→L1 signal channel (depends on ADR-014's trajectory features being extractable at L0; bounding mechanisms (b) and (d) are novel design work flagged as conditional-acceptance pending first-deployment evidence)

### FC-2 and FC-3 gap — pre-existing, not Cycle 4-introduced

The absence of FC-2 and FC-3 as automated tests predates Cycle 4. Cycle-1 archive notes that "FC-1/FC-2/FC-3 pass" at WP-B close, but this was by manual inspection at the time. As the codebase grows with new modules from ADRs 012–017, an automated FC-2 check becomes load-bearing: each new module in `src/llm_orc/agentic/` must not import from modules in a higher layer (L2/L3) without explicit permission. FC-4's static test (`test_fc4_runtime_import_surface.py`) demonstrates the pattern that FC-2 should follow — an AST-based import walker checking the allowlist/denylist per module. FC-3 should walk the full dependency graph and assert no cycles.

### ADR-016 conditional-acceptance scope

ADR-016's conditional-acceptance status (synthetic-data and structural-transfer validation completed; first-deployment evidence pending) is appropriate given the codebase's pre-implementation state. No BUILD work needs to proceed on ADR-016 before the conditional-acceptance moves to full acceptance — the ADR explicitly sequences its own validation pathway. The scan does not find any code that needs to be removed or restructured before ADR-016 BUILD begins; the status is gap-waiting-on-evidence, not violation-requiring-cleanup.

### Typed-error base class — cross-ADR coupling surface

ADR-017 specifies the shared `LlmOrcStructuralError` base class as a cross-ADR coordination mechanism because five of the six new ADRs produce typed errors: ADR-012 (Layer 4 circuit-breaker failure), ADR-013 (write-gate validation rejections), ADR-014 (Abstain verdict propagation), ADR-016 (malformed-signal rejection), and ADR-017 (phantom_tool_call rejection). The existing `ToolCallingNotSupportedError` at `src/llm_orc/models/base.py:99` is the natural `error_kind: "tool_call_rejected_per_model"` entry in the eventual base class. BUILD-time finalization of the base class (naming, field types, `recovery_action_required` value set) should happen early in Cycle 5 BUILD, before any of the five ADRs' typed errors are implemented individually — otherwise each ADR's error will need refactoring to the shared base after the fact.
