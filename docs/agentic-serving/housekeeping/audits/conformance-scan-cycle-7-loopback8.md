# Conformance Scan Report — Cycle 7 Loop-back #8

**Scanned against:** `docs/agentic-serving/decisions/adr-041-destination-validity-gate.md`
**Context ADRs (not scanned for debt):** ADR-034, ADR-035, ADR-040
**Codebase:** `src/llm_orc/agentic/`
**Date:** 2026-06-11

## Summary

- **ADRs checked:** 1 (ADR-041)
- **Conforming items:** 4 (the spike exercised the correct mechanism; `destination_path` is already threaded end-to-end; the terminal's `FormRefusedError` degradation path is production-ready; the `artifact_store` injection is wired at the factory)
- **Expected-temporary items (BUILD-WORK):** 10
- **Violations (permanent contradiction of an accepted ADR):** 2

The implementation correctly demonstrates the mechanism under the env-gate. All ten BUILD-WORK items are spike scaffolding that must be promoted to permanent behavior. The two violations are test gaps: the production path can silently drift from the ADR contract without failing CI.

## Conformance Debt Table

| ID | ADR | Violation / Gap | Type | Location | Classification | Resolution |
|----|-----|-----------------|------|----------|----------------|------------|
| BW-01 | ADR-041 §Status | `LLMORC_SPIKE_PI_GATE=parse` guard makes the parse-check FormGate a no-op in production and CI | missing | `artifact_bridge.py:91` | BUILD-WORK (expected-temporary) | Remove the env-gate check; `_spike_pi_form_check` fires unconditionally (or is absorbed into the installed gate — see BW-04) |
| BW-02 | ADR-041 §Status | `LLMORC_SPIKE_PI_GATE=parse` guard makes the `_spike_pi_invalid` probe return `False` unconditionally outside the spike harness | missing | `loop_driver.py:131` | BUILD-WORK (expected-temporary) | Remove the env-gate early-return from `_spike_pi_invalid` as part of the de-gate pass |
| BW-03 | ADR-041 §Status | `LLMORC_SPIKE_PI_GATE=parse` guard makes `_spike_pi_recover` return the original envelope unchanged in production | missing | `loop_driver.py:838` | BUILD-WORK (expected-temporary) | Remove the env-gate early-return from `_spike_pi_recover`; recovery becomes unconditional |
| BW-04 | ADR-041 §Decision 1/2 | `_spike_pi_form_check` is called as an inline side-effect at `marshal:160` rather than installed as `self._form_gate` via the FC-57 seam. The seam remains the passthrough at all construction sites, even under the spike env-gate. | wrong-structure | `artifact_bridge.py:160` (inline call), `artifact_bridge.py:132` (passthrough still installed) | BUILD-WORK (expected-temporary) | Convert `_spike_pi_form_check` into a `FormGate`-compatible callable and pass it as `form_gate=` at the `ArtifactBridge` construction site in `v1_chat_completions.py:499`; remove the inline call. This is the zero-Terminal-edits property ADR-035 §Decision 4 / FC-57 prescribes. |
| BW-05 | ADR-041 §Decision 2 | `FormGate` typedef is `Callable[[str\|bytes, str\|None], str\|bytes]` with the second argument named `destination_tool`. The parse-check needs `destination_path` to derive the file extension. The spike bypasses the type by calling `_spike_pi_form_check(content, destination_path)` directly, out of band. | wrong-structure | `artifact_bridge.py:59` (typedef), `artifact_bridge.py:90` (`_spike_pi_form_check` signature) | BUILD-WORK (expected-temporary) | Extend `FormGate` to three arguments: `Callable[[str\|bytes, str\|None, str\|None], str\|bytes]` with signature `(content, destination_tool, destination_path)`. Update `marshal:161` to pass both. Update `_passthrough_form_gate` and the refusing-gate test fixture accordingly. See interface reconciliation note below. |
| BW-06 | ADR-041 §Decision 3 | `LoopDriver.__init__` parameter `artifact_store` and field `self._spike_pi_store` carry `# SPIKE π — revert` comments marking them temporary | wrong-structure | `loop_driver.py:431`, `loop_driver.py:441` | BUILD-WORK (expected-temporary) | Remove spike comments; rename `self._spike_pi_store` to a permanent name (e.g. `self._artifact_store`); the parameter itself is permanent once de-gated |
| BW-07 | ADR-041 §Decision 3 | `get_loop_driver` factory passes `artifact_store=get_session_artifact_store()` annotated with a `# SPIKE π (loop-back #8)` comment marking it as temporary | wrong-structure | `v1_chat_completions.py:480–482` | BUILD-WORK (expected-temporary) | Remove the spike comment block; the wiring is permanent once de-gated |
| BW-08 | ADR-041 §Decision 4 | `_SPIKE_PI_MAX_REDISPATCH = 2` is spike-named. ADR-041 explicitly calls the redispatch cap a BUILD tuning parameter. | wrong-structure | `loop_driver.py:121` | BUILD-WORK (expected-temporary) | Rename to `_FORM_REDISPATCH_CAP` (or equivalent permanent name); remove the spike-scoped comment block at `loop_driver.py:114–122` |
| BW-09 | ADR-041 §Decision 3 | `_spike_pi_recover` method carries the `SPIKE π — env-gated … REVERT at spike close` docstring and comment. The behavior is permanent; only the name and env-gate guard are temporary. | wrong-structure | `loop_driver.py:817–866` | BUILD-WORK (expected-temporary) | Rename to `_recover_invalid_form` (or equivalent); remove env-gate guard (BW-03) and spike-comment header; the call site at `loop_driver.py:592` also carries `# SPIKE π — revert at close` and must be cleaned up |
| BW-10 | ADR-041 §Decision 3 | `ArtifactNotFoundError` and `ArtifactReference` imports carry `# SPIKE π — revert at close` comment at the import line | wrong-structure | `loop_driver.py:86–88` | BUILD-WORK (expected-temporary) | Remove the spike comment; these imports become permanent once recovery is promoted |
| V-01 | ADR-041 §Decision 2 | `test_apply_work_marshals_with_the_decided_tool` asserts `bridge.destinations == ["write"]` but never asserts `bridge.destination_paths`. The `_RecordingBridge` records `destination_paths` but no test checks the field — `destination_path` threading can silently break without a CI failure. | missing | `tests/unit/agentic/test_client_tool_action_terminal.py:227` | VIOLATION | Add `assert bridge.destination_paths == ["sort.py"]` to the existing test to confirm the path is actually threaded to `marshal` |
| V-02 | ADR-041 §Decision 1 | No test exercises the parse-check gate logic directly, even via `form_gate` injection. The env-gated path is inert in CI; `_spike_pi_form_check` is untested against the cases it must catch (invalid Python, invalid JSON, valid Python, non-.py/.json extension). | missing | `tests/unit/agentic/test_artifact_bridge.py` — no parse-gate test class | VIOLATION | Add a `TestParseCheckFormGate` class: (a) invalid `.py` content raises `FormRefusedError`; (b) invalid `.json` content raises `FormRefusedError`; (c) valid `.py` content passes; (d) non-`.py`/`.json` extension passes uninspected. Install via `form_gate=` seam, not env-var. |

## Conforming Items — What BUILD Inherits Without Modification

**C-01 — `destination_path` is already threaded end-to-end.**
`marshal`'s signature has carried `destination_path: str | None = None` since the spike was wired. The Terminal at `client_tool_action_terminal.py:127` passes `destination_path=outcome.file_path`. The `_RecordingBridge` in tests captures it (`test_client_tool_action_terminal.py:183–197`). No plumbing work is needed; only the test assertion (V-01) is missing.

**C-02 — Terminal's `FormRefusedError` degradation path is production-ready.**
`client_tool_action_terminal.py:129` already catches `FormRefusedError` in the same `except` clause as `ArtifactNotFoundError` and degrades to a dispatch-failure `stop`. Installing the parse-check gate via `form_gate=` requires zero Terminal changes — the FC-57 zero-Terminal-edits criterion is satisfied structurally.

**C-03 — `artifact_store` injection into `LoopDriver` is wired at the factory.**
`v1_chat_completions.py:482` passes `artifact_store=get_session_artifact_store()` to `LoopDriver`. Once the spike comments are removed (BW-06, BW-07), this wiring is permanent without further structural change.

**C-04 — `_spike_pi_resolve_content` covers both inline and substrate deliverables.**
`loop_driver.py:147–175` resolves inline `envelope.primary` and substrate-routed content from the store. The substrate path is exercised when the store is present (which it is at `v1_chat_completions.py:482`). The docstring's "Inline deliverables only" scoping in `_spike_pi_recover:835` is aspirationally stale rather than a code error — the implementation handles both. BUILD should remove the scoping caveat when renaming the method (BW-09).

## Notes

### Interface Reconciliation: FormGate Signature vs. `destination_path` (BW-04, BW-05)

The current `FormGate` typedef takes `(content, destination_tool)`. The parse-check needs `destination_path` for file-extension derivation. The spike avoided the mismatch by calling `_spike_pi_form_check(content, destination_path)` as a parallel side-call at `marshal:160`, leaving `self._form_gate(content, destination_tool)` unchanged at `marshal:161` — two separate calls rather than one installed gate.

Three reconciliation paths:

1. **Expand to three arguments** — `Callable[[str | bytes, str | None, str | None], str | bytes]` with `(content, destination_tool, destination_path)`. Both values are available; a gate uses whichever it needs. ADR-041's "the full destination path alongside the FC-57 `destination_tool`" is consistent with this reading. Updates required: the typedef at `artifact_bridge.py:59`, `_passthrough_form_gate` at line 70, the `marshal` call at line 161, and the refusing-gate fixture in `test_client_tool_action_terminal.py:407`. No new interfaces.

2. **Replace the second argument** — swap `destination_tool` to `destination_path`. Discards the named semantic of the seam; no production gate currently keys on `destination_tool` (the passthrough ignores both), but the name has documentation value (FC-57). Discouraged.

3. **Closure at construction** — build the gate as a closure that captures destination context. Requires the bridge to be constructed or the gate swapped per-turn, which contradicts the single-constructor-with-injected-gate design. Discouraged.

Option 1 is the path of least surprise and is consistent with ADR-041's language. The BUILD design amendment should commit this explicitly.

### ADR-040 Interaction on the Recovered Path (incidental)

`record_action` at `loop_driver.py:595` fires once after `_spike_pi_recover` returns the final envelope, so the ADR-040 `produced` set sees exactly one write record per destination regardless of how many re-dispatch attempts occurred. This is correct for ADR-040 semantics. However, each re-dispatch attempt inside `_spike_pi_recover` calls `_delegate_generation`, which writes an intermediate artifact to the `SessionArtifactStore` with no corresponding action record. Intermediate artifacts accumulate in the store without action records and are never surfaced to the client. This is a store hygiene gap, not an ADR-040 violation, but worth a BUILD watch item — a cleanup pass on the intermediate artifacts after the final envelope is chosen would prevent unbounded accumulation in long sessions.

### `_SPIKE_PI_MAX_REDISPATCH` as a Constant vs. Config

ADR-041 §Decision 4 names the cap a BUILD tuning parameter. The value `2` is appropriate for the validated spike behavior (n=5×2 across Arm A/E). BUILD should rename the constant (BW-08) but need not promote it to config unless operational evidence suggests the cap requires per-deployment tuning. A module-level constant with a docstring citing the spike evidence is sufficient.

### Scope of BW-04 Relative to the Smoke Finding

The smoke finding was that `FormRefusedError` raised at the Terminal degrades to a `stop` that ends the OpenCode client loop. `_spike_pi_recover` was introduced to self-heal before the Terminal sees the refusal. The inline side-call at `marshal:160` fires before `self._form_gate:161`, so the Terminal's gate remains the final arbiter whether or not recovery succeeds. Installing via `form_gate=` (BW-04) doesn't change this ordering — it consolidates the two calls into one seam while preserving the semantics. The key point for BUILD: the `stop`-ends-loop behavior at `client_tool_action_terminal.py:129` is correct and must not be changed.
