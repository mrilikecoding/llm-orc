# Conformance Scan Report — Cycle 6 DECIDE-Phase ADRs

**Scanned against:**
- ADR-022 (`adr-022-routing-surface-behavior.md`) — system-prompt amendment for `invoke_ensemble` under NL framing
- ADR-023 (`adr-023-observability-event-routing.md`) — unified event substrate; `DispatchTiming` + `dispatch_id`; two destinations; liveness signals; validate-once-at-load
- ADR-024 (`adr-024-common-io-envelope.md`) — typed `DispatchEnvelope` contract; `output_schema:` per-ensemble declaration
- ADR-025 (`adr-025-artifact-as-substrate.md`) — always-scope artifact-as-substrate; session-dir at `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/`; AS-7 amendment

**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc/`

**Branch / commit:** `agentic-serving` @ `7854fce`

**Date:** 2026-05-15

---

## Summary

- **ADRs checked:** 4 (ADR-022 through ADR-025)
- **Existing conformance items:** 10
- **BUILD-scope gaps (ADRs defer to Cycle 6 BUILD):** 21
- **Structural debt violations (code contradicts an accepted ADR):** 3
- **Adjacent observations (non-violation; coverage gaps and annotation drift):** 4

---

## Section 1 — Existing Conformance

Code that already implements the new ADRs' commitments, or where the ADR codifies already-existing behavior.

| # | ADR | What already conforms | Location |
|---|-----|-----------------------|----------|
| C1 | ADR-022 | `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` constant is defined and wired as the default that `OrchestratorConfigResolver.resolve()` returns under `orchestrator.get("system_prompt", DEFAULT_...)`. The amendment is an insertion into this constant. | `orchestrator_config.py:77–126`, `orchestrator_config.py:298–299` |
| C2 | ADR-022 | The operator-override path (`agentic_serving.orchestrator.system_prompt` in `config.yaml`) is already implemented via `orchestrator.get("system_prompt", DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT)`. Operators can override the amended prompt per ADR-022 §Out of scope. | `orchestrator_config.py:298–299` |
| C3 | ADR-023 | Four typed dispatch event dataclasses (`TierSelection`, `CalibrationVerdict`, `CalibrationSignal`, `AuditDiagnostic`) exist in code as typed objects. The infrastructure-complete/routing-incomplete characterization in ADR-023 §Context is accurate. | `tier_router.py:151–175`, `calibration_gate.py:168`, `calibration_signal_channel.py:160–210`, `tier_router_audit.py:200–228` |
| C4 | ADR-023 | The existing `_log_dispatch_result` function already emits an INFO-level line per tool dispatch. The line `"tool dispatch: result name=%s kind=success"` is the incumbent surface that ADR-023's per-event INFO lines replace and augment. | `orchestrator_tool_dispatch.py:789–806` |
| C5 | ADR-023 | The `list_ensembles` orchestrator tool dispatches through `OrchestraService.read_ensembles()` → `ResourceHandler.read_ensembles()`. The validation-on-every-call behavior is confirmed: `ResourceHandler.read_ensembles()` calls `EnsembleLoader.load_from_file()` for each YAML on every enumeration. The validate-once-at-load commitment has a clear, localized remediation target. | `resource_handler.py:63–96`, `ensemble_config.py:239–268` |
| C6 | ADR-024 | The de facto `{ensemble, status, input, results, metadata, synthesis}` shape is produced by `ArtifactManager.save_execution_results()` writing `execution.json`. The migration path (envelope wraps de facto shape at dispatch boundary; `execution.json` retains `metadata`) has a clean separation surface. | `artifact_manager.py:54–88` |
| C7 | ADR-024 | The `invoke_ensemble` return path through `ResultSummarizerHarness` already produces a `ToolCallSuccess(content={"summary": summary})` shape. This is the current incumbent the `DispatchEnvelope` replaces as the return value. The `raw_output` escape hatch (ADR-004) is present in code and composes with the substrate-routing conditional skip (ADR-025 / AS-7 amendment). | `orchestrator_tool_dispatch.py:490–508` |
| C8 | ADR-025 | The existing artifact path root (`.llm-orc/artifacts/`) is constructed by `ArtifactManager` using `self.base_dir / ".llm-orc" / "artifacts"`. The new session-dir path (`.llm-orc/agentic-sessions/`) is a parallel root at the same filesystem level — no deep restructuring is required, only a new construction site. | `artifact_manager.py:54–67` |
| C9 | ADR-025 | The `session_id` string threads through `OrchestratorRuntime`, `OrchestratorToolDispatch.dispatch()`, and `invoke_ensemble()` already. Session identity is established at request time via `SessionRegistry.resolve_identity()`. The session_id is available at the dispatch boundary where artifact path construction must occur. | `orchestrator_tool_dispatch.py:327`, `orchestrator_runtime.py:255` (session_id threading confirmed by grep) |
| C10 | ADR-025 | All six capability ensembles (`code-generator`, `web-searcher`, `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`) are present under `.llm-orc/ensembles/agentic-serving/`. The substrate migration target is complete and enumerable. | `.llm-orc/ensembles/agentic-serving/*.yaml` |

---

## Section 2 — BUILD Scope

Code that must be authored or modified to implement the new ADRs' commitments. This is Cycle 6 BUILD's work substrate.

### ADR-022 — System-prompt amendment

| # | Gap | Location | Resolution |
|---|-----|----------|------------|
| B1 | The new routing-preference paragraph is absent from `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT`. The prompt currently teaches the five-tool surface and one-kind-per-turn discipline but has no clause directing the orchestrator to prefer `invoke_ensemble` over direct completion or client tools when a capability match exists. | `orchestrator_config.py:77–126` | Insert the paragraph from ADR-022 §Decision between the "Do not pick a client-declared tool for questions about llm-orc's own state" paragraph (~line 104–109) and the "When you need a client-declared tool" paragraph (~line 111–115). The text is verbatim in the ADR. |
| B2 | Two system-prompt tests in `test_orchestrator_config.py` pin on the current prompt content. The `test_default_orchestrator_system_prompt_biases_toward_internal_tools` test checks for `"first"` near `list_ensembles` and `"do not"` or `"never"` somewhere in the prompt. The `test_default_orchestrator_system_prompt_teaches_retry_convention` test checks for presence of tool names and `"needs_client_tool"`. Neither test pins wording that would break under the amendment, but adding the new paragraph should be accompanied by a test that verifies the amendment's key clause (`prefer invoke_ensemble` near capability-match framing). | `tests/unit/agentic/test_orchestrator_config.py:564–638` | Add a third system-prompt test asserting `"prefer"` or `"invoke_ensemble"` appears in a context adjacent to capability-match language. Follow the existing `surrounding` pattern used in `test_default_orchestrator_system_prompt_biases_toward_internal_tools`. |

### ADR-023 — Event substrate and routing

| # | Gap | Location | Resolution |
|---|-----|----------|------------|
| B3 | `DispatchTiming` event dataclass does not exist. There is no `phase`, `dispatch_id`, `ensemble_name`, `model_profile`, `timestamp`, `duration_seconds`, or `exit_status` carrying type in the codebase. | `src/llm_orc/agentic/` (missing) | Author `DispatchTiming` as a frozen dataclass in a new or existing module (e.g., `dispatch_events.py` or extend `tier_router.py`). Two emission shapes per ADR-023: `phase="start"` and `phase="end"`. |
| B4 | `dispatch_id: str` field is absent from `TierSelection`, `CalibrationVerdict` (`Literal["proceed", "reflect", "abstain"]` — not a dataclass, just a type alias), `AuditDiagnostic`, and `CalibrationSignal`. The correlation identifier ADR-023 requires does not exist in any of the four event types. Note: `CalibrationVerdict` is a bare `Literal` type alias, not a dataclass — adding `dispatch_id` to it requires promoting it to a dataclass or creating a wrapper. | `tier_router.py:152–175`, `calibration_gate.py:168`, `tier_router_audit.py:200–228`, `calibration_signal_channel.py:160–210` | Add `dispatch_id: str | None` (optional, `None` for legacy call sites during transition) to `TierSelection`, `AuditDiagnostic`, and `CalibrationSignal`. For `CalibrationVerdict`, either promote from a `Literal` type alias to a dataclass or create a `VerdictWithContext` wrapper dataclass per ADR-023's call-site-composition requirement. |
| B5 | No per-event INFO emission to operator terminal. The existing `_log_dispatch_result` function emits a single coarse line per dispatch (`"tool dispatch: result name=%s kind=success"`). The per-event lines ADR-023 specifies (`"INFO: dispatch start: ..."`, `"INFO: tier selection: ..."`, `"INFO: calibration verdict: ..."`, etc.) are absent. | `orchestrator_tool_dispatch.py:789–806` | Replace or augment `_log_dispatch_result` with per-event emission. Each event type maps to one line format per ADR-023 §Destination 1. |
| B6 | No orchestrator-context destination routing. Dispatch events are not structured into observations for the orchestrator's next-turn reasoning context. No JSON-shaped observation block is injected between turns. | `orchestrator_tool_dispatch.py`, `orchestrator_runtime.py` | Implement the in-turn context-inclusion pattern per ADR-023 §Destination 2. At each turn boundary, format the just-completed dispatch's events as a structured observation block and prepend to the next-turn context. |
| B7 | No `tool-call-emit` liveness logging. The serving layer does not emit `INFO: tool-call emit: tool=<name> dispatch_id=<id>` before dispatching a tool call. | `v1_chat_completions.py`, `orchestrator_runtime.py` | Add the pre-dispatch log line at the point where an orchestrator tool call is identified and before the dispatch is initiated. The emit is operator-terminal only. |
| B8 | No inference-wait heartbeat. No mechanism emits `INFO: inference wait: elapsed=<seconds> session_id=<id>` when a request has been open for more than N seconds without tool activity. | `v1_chat_completions.py`, `orchestrator_runtime.py` (missing) | Implement the heartbeat per ADR-023 §"Liveness-signal patterns". Default `N=30s`. The heartbeat is operator-configurable via `agentic_serving.observability.heartbeat_interval_seconds`; add the config key to `OrchestratorConfig` resolution. |
| B9 | Ensemble validation fires on every `list_ensembles()` call (per-enumeration). `ResourceHandler.read_ensembles()` calls `EnsembleLoader.load_from_file()` for each YAML file on every call, re-running full Pydantic validation and dependency graph checks. The legacy warning for `fan-out-test.yaml` and `plexus-graph-analysis.yaml` (which lack some fields present in fully-specified ensembles) emits on each enumeration cycle. | `resource_handler.py:63–96`, `ensemble_config.py:239–268` | Implement validate-once-at-load: cache the validated ensemble set at startup (or library-reload) using the existing `ConfigurationManager` caching infrastructure. Subsequent `list_ensembles()` calls return the cached validated subset. Invalid ensembles surface a `WARN` at startup with path and error; they are excluded from subsequent listings without re-warning. |
| B10 | `dispatch_log` key in `execution.json` does not exist. The end-of-session summary structure ADR-023 §"Destination 2 — Orchestrator-context" requires is absent from current artifacts. | `artifact_manager.py` (missing) | Add `dispatch_log` key to `execution.json` shape at session close. Each entry is one per `dispatch_id` with the structured observation fields. |

### ADR-024 — `DispatchEnvelope` contract

| # | Gap | Location | Resolution |
|---|-----|----------|------------|
| B11 | `DispatchEnvelope` dataclass does not exist. There is no typed `status`, `primary`, `structured`, `diagnostics`, `errors`, or `artifacts` contract in the codebase. `invoke_ensemble` returns a `ToolCallSuccess(content={"summary": summary})` dict, not a typed envelope. | `src/llm_orc/agentic/` (missing) | Author the `DispatchEnvelope` frozen dataclass per ADR-024 §Decision. Fields: `status: Literal[...]`, `primary: str`, `structured: dict | None`, `diagnostics: dict`, `errors: list[dict] | None`, `artifacts: list[dict] | None`. |
| B12 | `invoke_ensemble` return value is not the typed envelope. The current return is `ToolCallSuccess(content={"summary": summary})` from `ResultSummarizerHarness`. The migration path (construct `DispatchEnvelope` at the dispatch boundary from `execution.json` + dispatch events) is not implemented. | `orchestrator_tool_dispatch.py:490–508` | Migrate `invoke_ensemble`'s return path to construct and return a `DispatchEnvelope`. The `diagnostics` field requires ADR-023's `dispatch_id` + `DispatchTiming` events to be available first (BUILD sequencing: ADR-023 ships before or alongside ADR-024). |
| B13 | `output_schema:` is absent from all six capability ensemble YAMLs. None of the six ensembles declares a typed output schema. | `.llm-orc/ensembles/agentic-serving/*.yaml` | Progressively add `output_schema:` to ensembles whose `default_task` specifies a structured format. ADR-024 identifies `text-summarizer` and `web-searcher` as early migration candidates; `claim-extractor` had output-spec drift and benefits from schema declaration. |
| B14 | `diagnostics` field naming. The de facto artifact shape uses the key `metadata`; ADR-024 renames the field to `diagnostics` at the envelope layer. The rename must not change the `execution.json` artifact format (migration deferred to Cycle 7+). | `artifact_manager.py:72–83`, `orchestrator_tool_dispatch.py:490–508` | Construct `envelope.diagnostics` from `execution.json["metadata"]` at the dispatch boundary (mapping rename). Confirm no code reads `envelope["metadata"]` to avoid silent breakage; the `execution.json` key itself remains `metadata` per ADR-024 §Migration path. |

### ADR-025 — Artifact-as-substrate

| # | Gap | Location | Resolution |
|---|-----|----------|------------|
| B15 | Session-dir path (`.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/`) does not exist in code. No code constructs or writes to this path. The `session_id` value exists at dispatch time but is not used for artifact path construction. | `artifact_manager.py:54–67` (wrong path), `orchestrator_tool_dispatch.py:449–509` (no path construction) | Implement session-dir artifact path construction. `<session_id>` format: `<iso-8601-datetime>-<short-uuid>` (set at session start by `SessionRegistry`). `<dispatch_id>` format: per-session monotonic counter (`dispatch-001`, etc.). The `dispatch_id` must be established at the same time as the ADR-023 correlation identifier (they are the same value). |
| B16 | `<session_id>` generation does not produce the ISO-8601 + short-UUID format ADR-025 specifies. `SessionRegistry.resolve_identity()` produces an identity value, but its format is not the `<iso-8601-datetime>-<short-uuid>` shape the ADR prescribes. | `session_registry.py` (identity format not inspected; format mismatch expected) | Verify current identity format; if it does not match ADR-025's format (`<iso-8601-datetime>-<short-uuid>` for natural chronological sort), add a dedicated `session_id` field distinct from the registry identity value. |
| B17 | All six capability ensembles lack `output_substrate: artifact` declaration (the default for capability ensembles in Cycle 6 BUILD). | `.llm-orc/ensembles/agentic-serving/*.yaml` | Add `output_substrate: artifact` to each capability ensemble YAML. The `EnsembleConfig` loader needs to parse this field; add to `EnsembleConfig` dataclass or YAML schema. |
| B18 | `output_substrate:` is not a recognized field in `EnsembleConfig`. The loader will ignore or reject unknown fields. | `src/llm_orc/core/config/ensemble_config.py` | Add `output_substrate: Literal["artifact", "inline"] = "inline"` to `EnsembleConfig` (default `"inline"` preserves existing behavior; Cycle 6 BUILD sets `"artifact"` for capability ensembles via the YAML or a config-layer default). |
| B19 | Calibration Gate evaluation surface under substrate-routing is not implemented. The existing gate reads the ensemble output directly from the response context; under substrate-routing, only the artifact reference + summary are in context. The `calibration_substrate_access: artifact` per-ensemble declaration is not implemented. | `calibration_gate.py:455–490`, `orchestrator_tool_dispatch.py:486–489` | Implement the three-tier evaluation surface per ADR-025 §"Calibration-gate evaluation surface": (1) summary-only default; (2) `structured` typed payload when `output_schema:` is declared; (3) file-read opt-in for ensembles declaring `calibration_substrate_access: artifact`. |
| B20 | AS-7 conditional-skip is not in code. `invoke_ensemble` always routes through `ResultSummarizerHarness` for non-`raw_output` results (ADR-004 escape hatch is the only skip). The substrate-routing conditional skip from AS-7 as amended by ADR-025 is not implemented. | `orchestrator_tool_dispatch.py:490–508` | After `DispatchEnvelope` is constructed (B12), add the AS-7 skip condition: if `ensemble.output_substrate == "artifact"`, skip `ResultSummarizerHarness` invocation; the `primary` field and `artifacts[]` in the envelope already carry summary-shaped content. |
| B21 | Session artifact cleanup on session close is not implemented. No code removes session-dir trees when a session closes. | `session_registry.py`, `v1_chat_completions.py` | Implement session-close cleanup for `retention: session` artifacts per ADR-025 §Cleanup. The serve process must track active sessions and on session close remove `.llm-orc/agentic-sessions/<session_id>/`. |

---

## Section 3 — Structural Debt

Code that violates an accepted ADR (prior or new). These require either `refactor:` reconformance commits or a superseding ADR. This section is the load-bearing input to scenario writing.

### DEBT-1: `_log_dispatch_result` emits a single coarse line, violating ADR-023's operator-terminal destination contract (accepted ADR)

**ADR:** ADR-023 (accepted 2026-05-15) §Destination 1 — Operator-terminal.

**What exists:** `orchestrator_tool_dispatch.py:789–806` defines `_log_dispatch_result`, which emits exactly one INFO line per dispatch: `"tool dispatch: result name=%s kind=success"` on success, or the error line on failure. This is the line the Cycle 5 PLAY verification (2026-05-14 finding 7) captured — it is the *only* dispatch-time observable in a 28-minute serve window, and it carries neither ensemble identification beyond the tool name `invoke_ensemble`, nor session ID, verdict, tier, nor duration.

**Why it's structural debt, not just BUILD-scope:** ADR-023 is now accepted; its operator-terminal destination format is a committed contract. The existing coarse line actively contradicts the committed format. The debt is not merely "a new feature not yet built" — it is the presence of the wrong output format at the operative log site. The existing line must be replaced, not supplemented, because keeping it alongside the per-event lines would double-emit on every dispatch.

**Location:** `src/llm_orc/agentic/orchestrator_tool_dispatch.py:789–806`

**Resolution:** Replace `_log_dispatch_result` with per-event emission per ADR-023 §Destination 1 format. The `DispatchTiming(phase="end")` event carries `duration_seconds` and `exit_status`; the INFO line becomes `"INFO: dispatch end: ensemble=<name> duration=<seconds> exit=<status> dispatch_id=<id>"`. The function is private; call sites are `dispatch()` at line 377 and `_route()` (indirectly). The replacement is a structural change — existing tests asserting on the `"tool dispatch: result name=%s kind=success"` log format will need updating.

---

### DEBT-2: Per-enumeration ensemble re-validation contradicts ADR-023's validate-once-at-load commitment (accepted ADR)

**ADR:** ADR-023 (accepted 2026-05-15) §"Noise-floor remediation (validate-once-at-load)".

**What exists:** `ResourceHandler.read_ensembles()` calls `EnsembleLoader.load_from_file()` for every `*.yaml` in every ensemble directory on each call. `load_from_file()` runs full `EnsembleConfig` Pydantic validation, dependency-graph checks, and cross-ensemble cycle detection on every invocation. The `fan-out-test.yaml` and `plexus-graph-analysis.yaml` files — which lack fields present in fully-specified capability ensembles — emit Pydantic-level warnings or partial-parse results on each enumeration. These warnings appear in the 28-minute serve-console window finding 7 documented.

**Location:** `src/llm_orc/services/handlers/resource_handler.py:63–96`, `src/llm_orc/core/config/ensemble_config.py:239–268`

**Resolution:** Implement startup-time validation with cached result. At serve startup or library reload, validate each YAML once; cache the `list[EnsembleConfig]` result. `read_ensembles()` returns from cache on subsequent calls. Invalid ensembles emit one `WARN` at startup with file path and error text; they are excluded from the cached valid set. Note: `ConfigurationManager` already has caching infrastructure (`Results are cached and invalidated when source files change`, `config_manager.py:477`) — the validate-once pattern should integrate with the existing cache invalidation mechanism rather than introducing a parallel caching layer.

---

### DEBT-3: Unconditional AS-7 summarization path contradicts the AS-7 amendment from ADR-025 (accepted ADR)

**ADR:** ADR-025 (accepted 2026-05-15) §"AS-7 amendment: default-with-conditional-skip"; ADR-004 partial update.

**What exists:** `invoke_ensemble` in `orchestrator_tool_dispatch.py:490–491` unconditionally calls `self._harness.summarize(result, raw_output=raw_output)` after every successful dispatch. The only skip path is `raw_output=True` (ADR-004's per-ensemble escape hatch for small outputs). There is no conditional skip for substrate-routed ensembles. Since ADR-025 has been accepted, the unconditional summarizer invocation contradicts the amended AS-7 invariant for the capability-ensemble substrate-routing case.

**Severity note:** This debt has zero operational impact today because the substrate-routing infrastructure (B15–B21) does not yet exist — no ensemble is actually routing its deliverable through an artifact path. The debt becomes load-bearing once BUILD implements substrate-routing. Flagged here as structural debt because the ADR is accepted and the code's unconditional path will need to be made conditional before BUILD can close.

**Location:** `src/llm_orc/agentic/orchestrator_tool_dispatch.py:490–491`

**Resolution:** After `DispatchEnvelope` construction lands (B12), add the skip condition: if the dispatched ensemble's `output_substrate` is `"artifact"`, return the envelope directly without calling `self._harness.summarize()`. The `raw_output` escape hatch (ADR-004) remains operative for inline-response ensembles. The two skip paths compose: `raw_output=True` (operator escape hatch for small outputs) and `output_substrate="artifact"` (substrate-routing skip) are independent conditions — the summarizer is bypassed if either is true.

---

## Section 4 — Adjacent Observations

Non-violation observations that intersect the ADRs and warrant BUILD-phase attention.

**OBS-1: Two system-prompt tests are fragile under the ADR-022 amendment.** `test_default_orchestrator_system_prompt_biases_toward_internal_tools` checks for `"first"` near `list_ensembles` (line 617–628) and `"do not"` / `"never"` somewhere in the prompt. The amendment adds `"Do not pick a client-declared tool merely because..."` which strengthens the `"do not"` presence test — but the proximity test for `"first"` near `list_ensembles` passes only if the insertion point does not displace `list_ensembles` from the tested vicinity. The test is robust under the amendment's specified insertion point (between lines ~104 and ~111, with `list_ensembles` defined at lines ~88–91). A BUILD scenario should verify the amended prompt passes both existing tests before the new conformance test (B2) is added.

**OBS-2: `CalibrationVerdict` is a bare `Literal` type alias, not a dataclass.** ADR-023 requires `dispatch_id: str` on `CalibrationVerdict`. Since `CalibrationVerdict = Literal["proceed", "reflect", "abstain"]` is a type alias (not a dataclass), adding a field to it requires a structural change: either promote `CalibrationVerdict` to a dataclass carrying `verdict: Literal[...]` + `dispatch_id: str | None`, or create a new `VerdictWithContext` dataclass and update all emission sites. The type-alias form is pervasive — `TierRouter.select_tier()` returns a `TierSelection`, and `CalibrationGate.verdict_for()` returns a `CalibrationVerdict`; both are consumed downstream including in the auditor and in tests. The structural change scope for this field addition is broader than the other three event types. A BUILD scenario should plan the `CalibrationVerdict` promotion as a distinct structural-change commit (separate from the `dispatch_id` field additions to the three dataclasses).

**OBS-3: `fan-out-test.yaml` and `plexus-graph-analysis.yaml` are library artifacts, not agentic-serving ensembles.** Both live at the top-level `.llm-orc/ensembles/` directory (not under `agentic-serving/`). The Pydantic re-validation warnings ADR-023 identifies come from the per-enumeration validation of these files. Under validate-once-at-load (DEBT-2 remediation), these files will surface their validation issues at startup and be excluded from the cached valid set. Operators will need to be aware that these ensembles are excluded; the startup `WARN` will surface why. This is expected behavior under the ADR — noted here so BUILD does not treat the startup warnings as a new regression.

**OBS-4: The `agentic-orchestrator-offline-tools.yaml` profile (new Cycle 6 working-tree addition) is not yet tested for system-prompt delivery.** The profile uses `provider: openai-compatible/ollama` with tool calling enabled. ADR-022's amendment is designed to affect behavior under `qwen3:14b` via this profile. No unit test verifies that the `agentic-orchestrator-offline-tools` profile satisfies `OrchestratorConfigResolver.resolve_validated()` correctly (the profile must be present in the Model Profile library for session start to succeed). This is a BUILD verification task, not a structural violation.

---

## Notes

### Load-bearing section: Section 3 (Structural Debt)

The three structural debt items have different urgency profiles:

- **DEBT-1** (`_log_dispatch_result` coarse logging) is the highest-urgency BUILD-phase structural change. The coarse log line must be replaced as part of implementing ADR-023's operator-terminal destination. Tests asserting on the current log format (`"tool dispatch: result name=invoke_ensemble kind=success"` in spike γ's serve logs) will break under the replacement — this is intended, as those tests are asserting on the wrong format. The BUILD scenario should be authored as `refactor:` commit (log-format change only) before the `feat:` commit adding per-event emission.

- **DEBT-2** (per-enumeration re-validation) is the highest-urgency operational fix. The legacy Pydantic warnings from `fan-out-test.yaml` and `plexus-graph-analysis.yaml` are emitting on every `list_ensembles()` call, which fires on every `invoke_ensemble` dispatched from the orchestrator. In a session with multiple dispatches, the noise floor is significant. The `ConfigurationManager` already has file-change-based cache invalidation; the validate-once fix integrates with it. This is a `refactor:` commit with no behavior change for operators whose ensembles are valid.

- **DEBT-3** (unconditional AS-7 summarization) has no current operational impact but must be addressed before BUILD closes substrate-routing. The fix is a conditional branch addition to `invoke_ensemble`; it is a `feat:` commit (new conditional behavior) rather than `refactor:` (since the condition evaluates a new field).

### ADR-023/024/025 BUILD sequencing is load-bearing

ADR-024's `DispatchEnvelope.diagnostics` depends on ADR-023's `dispatch_id` and `DispatchTiming` duration data. ADR-025's substrate routing depends on ADR-024's `artifacts[]` field being present in the envelope. The BUILD sequencing ADR-024 specifies — (1) ADR-023 event extension first, (2) ADR-024 envelope construction second, (3) ADR-025 substrate routing and AS-7 skip third — is enforced by data dependencies, not just convention. BUILD scenarios should be sequenced accordingly.

### No prior-ADR violations detected

Scanning the codebase against ADRs 001–021 found no new violations beyond those tracked in prior conformance scans. The Cycle 5 DECIDE scan's COMPATIBLE/NOTE items (agentic-coding-helper promotion, development/code-review placement) are resolved — `code-generator.yaml` and all six capability ensembles exist under `agentic-serving/`. The prior BUILD-scope items from the Cycle 5 scan are fully implemented.
