# ADR-017: Tool-Call Structural Validation Guard

**Status:** Proposed

**Date:** 2026-05-05

---

## Context

Cycle 4's Wave 3.A behavioral spike (essay 005 §"The Behavioral Spike") surfaced a phantom tool-call confabulation pattern at the cheap-cloud tier. Trial 3's exact text from a MiniMax M25 Free dispatch via OpenCode Zen: *"The tool call has been made and the result is displayed above as a `role:tool` observation."* Zero tool-call structures were emitted. The orchestrator wrote prose claiming a tool call had occurred without producing the structural artifact.

The class of failure is documented in the cycle's prior lit-review work (Wave 1.A): *"models like Ollama 7B/14B often hallucinate tool calls as plain text after the first interaction"* is a practitioner-documented finding for small local models. The novel element in Cycle 4's spike is observation of the same failure mode at the cheap-cloud tier on a substantially more capable model than the 7B/14B local-model class. The failure pattern is qualitatively distinct from the failure modes the cycle's prior literature scan covered for cloud tiers (judgment-decay, attention drift, instruction recency bias).

**Important scope condition:** Trial 3 used an adversarial diagnostic prompt (*"Output nothing else. Just the tool call. No prose, no explanation, no narration"*) rather than conditions representative of operational agentic sessions. The 1-of-3 confabulation rate is confounded by the prompt design and **should not be used to calibrate the urgency of the runtime-side guard**. The spike evidence is suggestive that the failure mode can occur at cheap-cloud, not that it occurs at high frequency in operational conditions.

The intervention class for this failure mode is class (a) deterministic-override in the CAAF taxonomy: validate the structural correspondence between orchestrator claims and tool-call structures, reject mismatches at the structural level rather than relying on the orchestrator to self-correct. Class (a) override is qualitatively different from prompt-mediated correction (CAAF arXiv:2604.17025: *"apparent LLM reliability in safety-critical domains is often a prompt engineering artifact"*).

The codebase precedent is the recent typed-error path established by commit `9f86d0b feat: raise typed error when provider rejects tool calling per-model`. That commit codified the pattern of producing a typed error when a provider's tool-calling capability is structurally absent; ADR-017 extends the same typed-error pattern to the structural-mismatch case (orchestrator claims tool call, structure absent).

The framing commitment from research-gate Grounding Action 2 (recorded 2026-05-05, *elaboration-by-evidence*) holds: structural validation responsibilities concentrate within Tool Dispatch (L2) rather than warranting a dedicated validation module. The placement preserves ADR-002's four-layer frame; the elaboration is within-layer.

---

## Decision

The Tool Dispatch (L2) interposition logic adds a **structural validation guard** between the orchestrator's response and the dispatch path. The guard runs on every orchestrator response and applies the following structural correspondence check:

### Detection

The guard scans the orchestrator's response text for **tool-call claim patterns** — prose that asserts a tool call has occurred or is occurring. The pattern set includes:

- *"the tool returned ..."* / *"the tool call returned ..."*
- *"I called X and the result was ..."*
- *"the result of X is displayed above"* / *"as the observation above shows"*
- *"after running X ..."* / *"after invoking X ..."*
- Pattern variants in the operator's deployment locale (the pattern set is operator-extensible at deployment configuration)

The pattern set is conservative — only patterns that explicitly assert tool-call occurrence are flagged. Patterns that describe future intent (*"I will call X"*, *"I am going to invoke X"*) are not flagged; those are pre-call narration that may or may not be followed by an actual tool-call structure.

### Validation

For each tool-call claim pattern detected, the guard cross-checks whether a corresponding tool-call structure was emitted in the same response:

- **Match:** the response contains both the prose claim and a structurally-valid tool-call structure naming the same tool. The dispatch proceeds.
- **Mismatch:** the response contains the prose claim but no corresponding tool-call structure. The guard rejects the response.

### Rejection

A mismatch produces a typed `phantom_tool_call` error consistent with the existing typed-error path established by commit `9f86d0b`. The error includes:

- The detected prose claim (the substring that asserted the tool call)
- The list of tool-call structures actually emitted (zero or more — may be partial mismatch where some calls were emitted but the prose claim references a different one)
- The dispatch context (which orchestrator turn, which session)

The orchestrator receives the typed error and must take a different action — re-emit the response with actual tool-call structures, reformulate the dispatch, or abstain. The typed error does not produce silent retry; the orchestrator's reasoning surface receives the structural feedback.

### Shared typed-error base class (cross-ADR coordination, argument-audit P3.5 finding 2026-05-06)

The new ADRs in Cycle 4 reference the typed-error pattern from commit `9f86d0b` at multiple validation surfaces — ADR-012 (Layer 4 compaction failures), ADR-013 (write-gate validation rejections), ADR-014 (Abstain verdict propagation), ADR-016 (mechanism (e) malformed-signal rejection), and ADR-017 (phantom_tool_call rejection). For BUILD coherence, ADR-017 specifies that all new typed errors derive from a shared `LlmOrcStructuralError` base class (or equivalent name to be finalized in BUILD) parallel to commit `9f86d0b`'s pattern, with the following common fields:

- `error_kind` — one of {`tool_call_rejected_per_model` (existing), `phantom_tool_call`, `compaction_layer_4_failure`, `write_gate_rejection`, `calibration_abstain`, `malformed_signal`, ...}
- `dispatch_context` — orchestrator turn identifier, session identifier
- `recovery_action_required` — one of: `reformulate` (orchestrator must rephrase or restructure the dispatch), `escalate` (orchestrator retries with a different tier per ADR-015), `abstain` (orchestrator drops the dispatch entirely without retry), `operator_intervention_required` (the error cannot be recovered orchestrator-side; operator must act — e.g., Layer 4 circuit-breaker has tripped per ADR-012, or severe-drift fail-safe mode has activated per ADR-016 mechanism (d)). The fourth value distinguishes orchestrator-recoverable errors from operator-action-required errors; per round-2 argument-audit P3 finding 2026-05-06, mapping operator-intervention errors to `abstain` would be semantically imprecise. Final value set is BUILD-time finalization territory.
- `operator_diagnostic` — operator-readable description of the structural condition that triggered the error

The shared base class enables BUILD-time error-handling cohesion across the new ADRs without requiring each ADR to re-specify the typed-error infrastructure. Naming and field finalization is BUILD work; ADR-017 is the natural place to specify the shared pattern because its scope is the most narrowly typed-error-centered.

### Minimal default pattern set with operator-extension surface

The guard's default pattern set is **minimal rather than calibrated** (per argument-audit P2.6 finding 2026-05-06). The cycle's spike evidence (Wave 3.A Trial 3) was confounded by adversarial prompt design and cannot calibrate the pattern set against operational confabulation frequency. The minimal default reflects this evidentiary state honestly: the patterns documented above are derived from the spike's observed text plus general agent-narration patterns, not from a dataset of confabulation frequencies under operational conditions.

The cost-asymmetry between false-positives (operator-visible session disruption) and false-negatives (orchestrator-side fabrication) is real but not the primary justification for the minimal default. The primary justification is **the spike evidence does not support a richer default pattern set**. A richer default would presuppose calibration data the cycle does not have.

The pattern set is **operator-extensible at deployment configuration** so operators can add patterns observed in their deployment without altering the core guard logic. Operators are expected to extend the pattern set as deployment evidence accumulates — this is the operational discovery path, not a fallback. The default pattern set is the BUILD starting point; deployment-specific patterns are the operational refinement surface.

---

## Rejected alternatives

**(a) Prompt-mediated correction (instruct the orchestrator to not confabulate).** Rejected per CAAF's prompt-engineering-artifact finding. Prompt instruction is class (b); it produces apparent reliability that is prompt-mediated rather than structural. The intervention class for confabulation is class (a) deterministic override at the dispatch layer — the orchestrator cannot violate the structural check by ignoring an instruction.

**(b) Silent retry on detected confabulation (re-prompt the orchestrator without typed error).** Rejected: silent retry hides the failure mode from the orchestrator's reasoning surface. The orchestrator does not learn from the rejection; downstream reasoning may continue to incorporate the phantom call's claimed effects. Typed error is the structural feedback mechanism that lets the orchestrator's reasoning incorporate the rejection.

**(c) Block all responses containing tool-call-claim patterns (over-detection by default).** Rejected: the conservative false-positive discipline is load-bearing. Over-detection produces operator-visible session disruption; the cycle's evidence base does not justify aggressive detection. Conservative detection plus operator-extensibility is the calibration that matches the evidence shape.

**(d) Defer the guard until multi-iteration scale evidence (treat the spike's single-trial confabulation as below evidentiary threshold for in-cycle structural validation).** Rejected: the failure mode's class (Wave 1.A's small-local-model documentation; codebase precedent at commit `9f86d0b`) is established independently of the cycle's spike evidence. The guard codifies a documented class of structural mismatch that has codebase precedent for typed-error treatment; the spike's contribution is the cheap-cloud observation, which extends the failure mode's known surface but does not change the intervention class. The spike's prompt-design caveat applies to *frequency calibration* (how often to expect confabulation under operational conditions), not to *whether the structural guard is justified*.

**(e) Validate via prose-content semantic check (LLM judges whether the prose claim is consistent with the tool-call structures).** Rejected: this reintroduces LLM-judgment into the validation path, which has documented reliability problems (CAAF). Structural validation against the tool-call structure presence is class (a) override; semantic validation by LLM is class (b) prompt-judgment.

**(f) Detect intent patterns (*"I will call X"*) in addition to assertion patterns.** Rejected per the conservative false-positive discipline. Intent patterns are pre-call narration that may legitimately be followed by an actual tool-call structure; flagging them produces operator-visible session disruption when the orchestrator was simply narrating before acting. The detection scope is restricted to assertion patterns where the prose claims tool-call occurrence.

---

## Consequences

**Positive:**
- Closes a documented failure-mode surface (phantom tool-call confabulation) at the structural level, consistent with the existing typed-error path from commit `9f86d0b`
- Class (a) deterministic override is the literature-supported intervention class for this failure mode (CAAF arXiv:2604.17025; intervention-class taxonomy from essay 005 Wave 1.A)
- Conservative false-positive discipline matches the spike's evidence shape — the guard runs but does not over-trigger; pattern-set extensibility lets deployment evidence inform calibration without altering core guard logic
- Typed error provides structural feedback to the orchestrator's reasoning surface; the orchestrator can incorporate the rejection rather than receiving silent retry
- Composes cleanly with ADR-013's write-gate validation and ADR-014's calibration verdict — all three use the typed-error pattern from commit `9f86d0b`, producing a coherent error-surfacing model across the new ADRs

**Negative:**
- Pattern-based detection has inherent false-positive risk; conservative discipline reduces but does not eliminate it
- Pattern-set extensibility introduces an operator configuration surface that may need maintenance as new patterns are observed
- The guard adds latency overhead per response (regex/pattern scan plus tool-call structure cross-check); the overhead is bounded but non-zero
- The guard does not address all confabulation cases — patterns the operator has not configured will pass through; the discipline is "structural validation where pattern-detection succeeds" not "structural validation universally"
- BUILD-time work to specify and tune the default pattern set is novel-design effort without a published reference pattern set

**Neutral:**
- The guard operates in Tool Dispatch (L2 interposition), invisible to the orchestrator's reasoning surface except through the typed error on rejection
- The codebase precedent at commit `9f86d0b` is the structurally-similar pattern; ADR-017 extends the same typed-error pattern to a different failure-mode surface
- The guard's value scales with the failure-mode's frequency in deployment; if confabulation is rare under operational conditions (the spike's prompt-design caveat suggests it may be), the guard runs but rarely triggers — that is the correct operational character
- Pattern-set localization (the operator-extensibility for deployment locale) is a deployment concern, not an architecture concern

---

## Provenance check

- **Driver-derived content (failure mode, intervention class, codebase precedent).** The phantom tool-call confabulation failure mode is documented in essay 005 §"The Behavioral Spike" (Wave 3.A Trial 3 observation) and Wave 1.A (small-local-model practitioner-documented finding). The class (a) intervention classification is from the cycle's intervention-class taxonomy, surfaced via essay 005 Wave 1.A. The typed-error pattern is the existing codebase precedent at commit `9f86d0b`.

- **Driver-derived content (scope condition on spike evidence).** The caveat that Trial 3's adversarial prompt confounds frequency calibration is essay-derived (essay 005 §"The Behavioral Spike" explicitly flags this). ADR-017 carries the caveat into the conservative false-positive discipline.

- **Drafting-time synthesis (specific pattern set).** Essay 005 specifies the structural-validation guard but does not specify the detection pattern set. The pattern set in this ADR's Decision section is drafting-time synthesis. The patterns are derived from the spike's observed text (*"the result is displayed above"*) plus general agent-narration patterns; the operator-extensibility is drafting-time addition acknowledging that the default pattern set will not cover all deployment cases.

- **Drafting-time synthesis (conservative false-positive discipline).** Essay 005 does not specify a false-positive vs. false-negative tradeoff. The conservative discipline (under-detection preferred over over-detection) is drafting-time judgment applying the spike's prompt-design caveat. The alternative (aggressive detection) was rejected for the documented reasons.

- **Drafting-time synthesis (typed error name `phantom_tool_call` and error fields).** The error name and field set are drafting-time synthesis. Commit `9f86d0b`'s precedent establishes the typed-error pattern but uses a different error name appropriate to that case (per-model tool-calling rejection). ADR-017's error name is parallel-by-construction.

- **Drafting-time synthesis (operator-extensibility configuration surface).** The pattern-set operator-extensibility model is drafting-time addition. Essay 005 does not specify deployment configuration for the guard; the extensibility is judgment-based addition responsive to the conservative-discipline-but-deployment-evidence-informed tension.

- **Vocabulary impact.** ADR-017 introduces two terms candidate for domain-model addition at Tranche-C close:
  - **Tool-call structural validation** — proposed new term in §Concepts (operator voice; the guard's category)
  - **Phantom tool-call** — proposed new term in §Methodology Vocabulary (research voice; the failure-mode name) — alternatively, may live in §Concepts if the operator works with the term operationally during deployment debugging

- **Asymmetric DECIDE budget per research-gate carry-forward #4.** ADR-017 is in the "novel architectural territory" group but is the most bounded of the four (the failure mode is documented, the intervention class is settled, the codebase precedent is direct). Argument-audit on this ADR should concentrate on (i) the conservative false-positive discipline (whether the pattern-set scoping correctly trades operator-disruption against fabrication-pass-through), (ii) the operator-extensibility configuration surface (whether the BUILD-time tuning burden is justified), and (iii) the composition with ADR-013 and ADR-014 (whether the typed-error pattern coheres across the three ADRs). The class (a) intervention classification and codebase-precedent extension are adoption-decision discipline.
