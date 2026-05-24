# ADR-029: Response-Synthesizer Ensemble Specification

**Status:** Proposed

**Date:** 2026-05-22

---

## Context

ADR-027 establishes the framework-driven dispatch pipeline (plan → dispatch → synthesize) as the primary direction for the agentic-serving chat-completions surface. The Synthesize stage replaces the orchestrator-LLM's post-dispatch synthesis (where the orchestrator-LLM composed the chat-completion response from dispatch results — empirically the locus of the Cycle 6 PLAY note 22 composition confabulation and the Spike λ-paid F-paid-4 substrate-path file-read failure mode).

Cycle 7 Spike ε (2026-05-21) and Spike ε' (2026-05-22) empirically established the response-synthesizer ensemble's behavior at qwen3:8b across the relevant test surfaces:

- **Spike ε (3 tests):** PLAY-note-22 historical confabulation case + Spike δ positive control + simple single-capability lookup. Results: 0 fabrications on the hardest test (synthesizer correctly reported claim-extractor was Planned-but-not-run rather than confabulating); 0 fabrications on the positive control; 1 Rule 4 rounding violation on simple lookup (`402,300` vs source `402,329`). Most consequential finding: the orchestrator-LLM's PLAY-note-22 confabulation pattern does NOT survive the architectural move to a structurally-bounded synthesizer role at the same cheap tier (qwen3:8b).
- **Spike ε' (7 probes):** direct-completion path (4 request shapes), numerical-density fidelity (2 shapes), multi-turn continuity (2 shapes). Results: 5 clean OK + 2 with characterized findings. Four findings: ε'.1 Rule 5 framing systematically omitted across direct-completion responses; ε'.2 rounding drift mode-specific (Mode 1 precise-figure rounding; Mode 2 large-number millions rendering); ε'.3 multi-turn continuity works under synthesizer-only architecture when prior turns are in input; ε'.4 scope-of-claim partition tightens substantially.
- **Spike μ (3 tests at MODEL):** confabulation-mode generalization across path hallucination, substrate-path-as-deliverable, coherent factual errors uncalibrated. Results: μ.1 transforms (confident-specific fabrication → honest-generic-conventions with explicit hedging); μ.2 structurally avoided (text-only synthesizer surface); μ.3 bounded by Rules 1 + 5.

The empirical baseline is captured in the spike ensemble `.llm-orc/ensembles/spike-cycle7-epsilon-response-synthesizer.yaml`. This ADR specifies the response-synthesizer ensemble's contract for production: input shape, the strict-fidelity rule set, model-profile constraints, dispatch contract, and the open DECIDE-phase design questions (Rule 5 scope per OQ #23; Rule 6 framework-convention enumeration; rounding-drift mitigation playbook per OQ #24).

---

## Decision

**Adopt the response-synthesizer ensemble as the Synthesize-stage mechanism** in the framework-driven dispatch pipeline (per ADR-027). The ensemble is a system ensemble under the `agentic-` prefix convention (per ADR-019), located at `.llm-orc/ensembles/agentic-serving/agentic-response-synthesizer.yaml` or equivalent.

### Input contract

The response-synthesizer ensemble receives one structured input per chat-completions request:

```
ORIGINAL REQUEST:
  messages: <messages[] array from the chat-completions request — full conversation history>
  model: <model field from the request>

PLAN:
  action: "dispatch" | "direct"
  ensemble: "<ensemble-name>" | null
  rationale: "<one-sentence rationale from the routing-planner per ADR-028>"

DISPATCH RESULTS:
  <when action=dispatch and dispatch succeeded: structured representation of the dispatched ensemble's envelope per ADR-024 — primary content, artifact reference summary, diagnostics>
  <when action=dispatch and dispatch failed: structured error from the dispatch — failure type, message, recovery-path advice>
  <when action=direct: empty / null>
```

The synthesizer's input is structured (not a `finish_reason: tool_calls` requiring further LLM-issued tool calls). The synthesizer does not have access to the dispatched ensemble's substrate paths directly — substrate routing per ADR-025 produces summary-shaped content in the envelope's `primary` and `artifacts[0]` fields; the synthesizer reads those, not the underlying substrate files.

Multi-turn continuity is preserved by including the full `messages[]` array in the input. Native `messages[]` handling at the framework layer (serialize prior turns into the synthesizer's input shape) is mechanical ARCHITECT-phase work per Spike ε' Finding ε'.3.

**AS-10 compliance:** the synthesizer's input is derived entirely from the chat-completions request content (the ORIGINAL REQUEST + the routing-planner's PLAN + DISPATCH RESULTS computed from that request); the synthesizer does not consume client-side opt-in signals beyond what AS-10 (per ADR-026) names as constitutionally permitted (OpenAI-protocol-native fields). The synthesizer's output (`message.content` + `finish_reason: stop`) is OpenAI-protocol-native; the synthesizer does not introduce out-of-band signals into its output that would bypass the transparent-endpoint promise.

### Output contract

The response-synthesizer produces the user-facing chat-completion response: `message.content` (a string conforming to OpenAI chat-completion semantics) and `finish_reason: stop`. The synthesizer does not emit tool calls; the chat-completion response carries content alone.

### Strict-fidelity rule set

The synthesizer operates under a strict-fidelity rule set the spike work validated empirically. **Five rules from Spike ε + ε' empirical baseline, plus Rule 6 codified from Spike μ.1:**

- **Rule 1 — Use only DISPATCH RESULTS content for substantive claims.** Substantive factual claims in the response (numbers, names, statements about the dispatched work) must be sourced from DISPATCH RESULTS. The synthesizer does not invent content; it does not draw on its own training data to substantiate claims when DISPATCH RESULTS is the authoritative source.
- **Rule 2 — Do not fabricate results for Planned-but-not-run ensembles.** When the PLAN names an ensemble but the dispatch failed or did not run (e.g., schema-non-conformance, infrastructure error), the synthesizer reports the planned-but-not-run state honestly rather than fabricating output the missing ensemble would have produced. (This is the Cycle 6 PLAY note 22 failure mode the synthesizer-only architecture dissolves; Spike ε ε.1 empirical evidence.)
- **Rule 3 — Do not invent operational metadata.** Timing, dispatch counts, model identifiers, and other operational metadata are not invented. The synthesizer reports operational details only when DISPATCH RESULTS carries them.
- **Rule 4 — Cite figures verbatim (no rounding, no restatement-drift).** Numerical figures from DISPATCH RESULTS appear in the response in their source form. Rounding-drift is a documented failure mode (per Spike ε T3 + Spike ε' Finding ε'.2 — Mode 1 precise-figure rounding, Mode 2 large-number millions rendering). Rule 4 is the *target* the synthesizer is prompted toward; the mitigation playbook (system-prompt sharpening + tier escalation + runtime fidelity check) is downstream-phase work per OQ #24.
- **Rule 5 — Honest direct-completion framing when DISPATCH RESULTS is empty.** When `action: direct` (no capability match) or dispatch failed irrecoverably, the response declares the direct-completion mode explicitly ("this answer was generated directly without dispatching a specialist ensemble" or framework-determined equivalent). Rule 5 delivers the configuration-honesty sub-promise (per OQ #18 split; per ADR-032). The framing requirement scope (always present vs. conditional vs. over-specified) is OQ #23 — see "Rule 5 framing requirement scope" below.

**Rule 6 — Framework-convention enumeration in direct-completion mode (codified from Spike μ.1):** When the request asks about file paths, framework conventions, or implementation specifics under `action: "direct"`, the synthesizer enumerates generic conventions with explicit hedging + uncertainty acknowledgment + clarification request rather than fabricating confident-specific paths. Spike μ.1 observed the synthesizer naturally exhibits this pattern under Rule 1 (no invention) + Rule 5 (honest framing); Rule 6 codifies the pattern to preserve it across future cheap-tier model substitutions where the natural-emergence may not transfer.

The rule set is operationalized via the synthesizer ensemble's system prompt; the spike ensemble's prompt is the BUILD-phase starting point. The rules are a contract the synthesizer is prompted to satisfy; the empirical floor (n=13 tests + 4 confabulation modes at qwen3:8b; 0 fabrications across the audit depth) is the cycle's evidence the prompted contract holds.

### Rule 5 framing requirement scope (OQ #23)

Spike ε' Finding ε'.1 observed the synthesizer systematically omits Rule 5 framing across 4 direct-completion responses. The omission is empirically consistent — the synthesizer's natural response shape is outcome-focused without the meta-framing layer. Two interpretations:

- **Rule 5 is load-bearing.** Operators need the transparency signal; document-degradation policy depends on it; Population A's configuration-honesty corroboration (per OQ #18 — Cline #10551 + OpenCode #20859) names the absence of the signal as the degradation signal. Under this interpretation, the synthesizer's prompt is sharpened to mandate Rule 5 framing more strongly; runtime validation checks the response for the framing marker.
- **Rule 5 is over-specified.** The response is correctly outcome-focused; the meta-framing adds noise that degrades user experience without delivering Population A-actionable signal. Population A's degradation signal (per OQ #18) is configuration *dishonesty* — the endpoint silently substituting models or context windows — not the absence of meta-framing per se. Under this interpretation, Rule 5 is relaxed; honest response labeling delivers via response *headers* or *metadata* fields (not inside `message.content`), separate from the synthesizer's content surface.

**This ADR adopts the load-bearing interpretation as the BUILD default, with the over-specified path available as a falsification trigger.** Rule 5 framing is mandatory in the synthesizer's system prompt; runtime validation checks the response for a framing marker; absence of the marker triggers a Calibration Gate Reflect verdict (per ADR-014). If production traffic surfaces evidence that Rule 5 framing degrades user experience without delivering Population A-actionable signal (e.g., Population A operators report the framing as noise), the falsification path is to migrate honest response labeling to response headers / metadata (per ADR-032's honest-response-labeling mechanism — the headers mechanism is the natural escape hatch).

The BUILD default + falsification trigger pattern preserves the configuration-honesty sub-promise (per OQ #18) — Population A receives a signal somewhere (either in content or in headers); the locus is the design choice.

### Rounding-drift mitigation playbook (OQ #24)

Spike ε T3 and Spike ε' Finding ε'.2 characterized two distinct rounding-drift modes: Mode 1 precise-figure rounding (402,042 → "~402,000"); Mode 2 large-number millions rendering (84,358,845 → "84.4 million"). Rule 4 reduces but does not eliminate drift. The mitigation playbook combines three mechanisms:

- **System-prompt sharpening.** Strengthen Rule 4 in the synthesizer's prompt with explicit examples and verbatim-quotation patterns. The spike ensemble's prompt is the starting baseline; BUILD-phase iteration refines.
- **Tier escalation policy for direct-completion (when dispatch fires but synthesizer drifts).** Calibration Gate Reflect on rounding-drift triggers Tier-Escalation Router (per ADR-015) to escalate the synthesizer to a higher-tier model for the retry. The existing infrastructure handles the escalation; the calibration criterion is the new piece.
- **Runtime fidelity check.** A lightweight post-synthesis pass extracts numerical figures from the response, compares against DISPATCH RESULTS, and rejects the response if drift exceeds threshold. Reject triggers a synthesizer retry with the fidelity violation in the input as additional context. The threshold (exact match vs. tolerance for last-digit rounding vs. semantic equivalence) is BUILD-phase design.

The playbook is BUILD-phase design work; ADR-029 names the three-mechanism shape and the mitigation hierarchy (system-prompt → tier escalation → runtime fidelity check, in increasing operational complexity). Production traffic determines which mechanisms are sufficient; the runtime fidelity check is the load-bearing fallback if system-prompt + tier-escalation does not bound drift to acceptable rates.

### Model profile and tier

The response-synthesizer ensemble is **cheap-tier by default** — qwen3:8b via local Ollama is the empirical baseline (Spike ε + ε' + μ; n=13 tests; 0 fabrications across 4 confabulation modes).

Operators may override the synthesizer's model profile via the ensemble's YAML (per ADR-011's session-boundary config discipline). The override surface remains available; the framework's empirical-grounding scope is the default profile. Tier escalation per ADR-015 + ADR-018 operates within the synthesizer ensemble — calibration verdicts on the synthesizer's output drive escalation per the rounding-drift mitigation playbook above.

### Calibration Gate integration

The Calibration Gate (per ADR-007, ADR-014) operates within the synthesizer ensemble. Three Reflect-trigger criteria apply (in addition to the existing post-hoc and trajectory-level criteria):

- **Rule 5 framing absence** — when `action: direct` and Rule 5 framing marker is absent from `message.content`, the gate reflects.
- **Rule 4 rounding-drift** — when runtime fidelity check (above) reports drift exceeding threshold, the gate reflects.
- **Rule 1 fabrication signal** — when post-hoc audit (e.g., dispatch-results-vs-content cross-check) detects substantive content not sourced from DISPATCH RESULTS, the gate reflects.

The gate's Reflect verdict triggers tier-escalation per ADR-015; recurring Reflect triggers feed the Tier-Router Audit drift criteria per ADR-018.

---

## Rejected alternatives

### Orchestrator-LLM-as-synthesizer (chat-completion response composed by the orchestrator-LLM in a ReAct turn)

The chat-completions surface continues to route through `OrchestratorRuntime`; the orchestrator-LLM composes the chat-completion response after dispatch via its normal ReAct loop (call `invoke_ensemble`; receive result; compose response).

**Rejected because:** this is the architecture ADR-027 supersedes on the chat-completions surface. Cycle 6 PLAY note 22 documented composition confabulation under this architecture (8 cache-hit web-searcher dispatches + 0 claim-extractor dispatches + fabricated final response); Spike λ-paid F-paid-4 documented substrate-path file-read failure under forced-dispatch paths (orchestrator-LLM tries to chain through `read_file` of dispatch substrate paths; production clients cannot execute file reads against the server's filesystem). AS-9 codification names the structural-bounding property; the orchestrator-LLM-as-synthesizer is the bundled-reasoning surface AS-9 names as failure-prone.

The structurally-bounded synthesizer (this ADR's mechanism) was empirically tested across 13 tests at qwen3:8b spanning 4 confabulation modes with 0 fabrications. The architecture trade is empirically grounded.

### Deterministic template synthesis (the framework composes the chat-completion response without an LLM)

The framework reads DISPATCH RESULTS and produces the chat-completion response from a deterministic template (e.g., "Here is the result: <primary content>; for details see <artifact reference>"). No LLM in the Synthesize stage; latency is sub-millisecond.

**Rejected because:** template synthesis has an information-loss floor — it cannot adapt the response to the request's framing, multi-turn context, or capability-specific output shapes. Population A's chat-completions clients expect natural-language responses framed appropriately for the user's request; a template-produced response fails on requests where the dispatch result requires summarization, narrative framing, or context-aware integration.

Template synthesis also has a multi-turn continuity problem — the synthesizer's role under multi-turn requests (where prior turns provide context relevant to interpreting the current dispatch result) is reasoning-shaped, not template-fillable. Spike ε' C1/C2 empirically established multi-turn continuity works under the LLM-based synthesizer when prior turns are in input; template synthesis cannot handle this.

The deterministic-template path remains relevant as a fallback when the LLM-based synthesizer itself fails irrecoverably (LLM unavailable; persistent schema-non-conformance). ADR-029 does not foreclose a template-based fallback at that layer; the rejected alternative is template-as-primary-mechanism.

### Mandate Rule 5 framing only in headers / metadata (not in `message.content`)

Rule 5's configuration-honesty signal is delivered exclusively via response headers or metadata fields (e.g., a `served-by` header); the synthesizer's `message.content` is content-only without meta-framing.

**Rejected because:** Population A clients vary in how they surface response headers and metadata to the user. Aider's session UI surfaces some metadata; OpenCode's per-provider tool framework may or may not. The content-layer signal is universally visible; the header/metadata-layer signal depends on client implementation. The falsification trigger above leaves the path open if production evidence indicates the content-layer framing degrades user experience — but the BUILD default keeps the signal in the universally-visible layer.

The header/metadata mechanism remains active under ADR-032 (honest response labeling); Rule 5 framing in content is the synthesizer's *content-layer* commitment, ADR-032's headers are the *response-layer* commitment. Both can coexist; ADR-029 does not require choosing between them.

### Don't codify Rule 6 (framework-convention enumeration); leave as undocumented emergent behavior

Spike μ.1 surfaced the framework-convention enumeration pattern but it could be left as an unprompted emergent behavior of the synthesizer's natural response shape under Rules 1 + 5.

**Rejected because:** Rule 6 codifies the pattern observed at qwen3:8b; the pattern's generalization to other cheap-tier models is plausible-but-untested. Codifying the rule in the synthesizer's prompt strengthens the pattern across model substitutions and makes the convention explicit for future-cycle evaluation. The cost of codification is small (a few lines in the synthesizer's prompt); the benefit is reduced model-substitution risk.

---

## Consequences

### Positive

- **Synthesize-stage reliability is empirically grounded.** Spike ε + ε' + μ established 0 fabrications across 13 tests + 4 confabulation modes at qwen3:8b. The synthesizer's behavior on the post-dispatch synthesis task is characterized.
- **The synthesizer satisfies AS-9 structurally.** The role is single-decision-shaped (produce response from given structured input); the LLM does not chain through tool calls, file reads, or multi-step reasoning. AS-9's structural-bounding property is preserved by the synthesizer's contract.
- **The C4 failure mode (substrate-path file-read attempts) is structurally prevented.** The synthesizer's input is structured (envelope summary content); the synthesizer cannot issue file-read tool calls — its output surface is text content alone. The orchestrator-LLM's emergent chain-through-file-read pattern (Spike λ-paid F-paid-4) is removed.
- **Configuration honesty is delivered structurally** (per OQ #18 split). Rule 5's mandatory framing on direct-completion responses delivers Population A-actionable transparency signal; the synthesizer cannot silently disguise direct-completion as ensemble-dispatched.
- **Multi-turn continuity works** under the synthesizer-only architecture when prior turns are in input (Spike ε' C1/C2 validated). The ARCHITECT-phase work to serialize prior turns into the synthesizer's input is mechanical, not design-open.
- **Rule 6 codifies the framework-convention enumeration pattern** observed in Spike μ.1; the pattern is preserved across future model substitutions via prompt-level codification rather than emergent behavior.
- **Calibration Gate integration uses existing infrastructure.** The three Reflect-trigger criteria (Rule 5 framing absence, Rule 4 rounding-drift, Rule 1 fabrication signal) extend the existing ADR-014 mechanism; the Tier-Router Audit per ADR-018 absorbs the new criteria into the existing drift detection.

### Negative

- **Cheap-tier reliability beyond qwen3:8b is plausible-but-untested.** Operators substituting models bear the characterization burden; the framework's empirical floor is qwen3:8b-grounded across 13 tests + 4 confabulation modes.
- **Rule 4 rounding-drift is not eliminated by prompting alone** (Spike ε T3 + Spike ε' Finding ε'.2). The mitigation playbook (system-prompt → tier escalation → runtime fidelity check) is BUILD-phase design; the runtime fidelity check is operational complexity if drift bounds are tight.
- **Rule 5 framing under the load-bearing interpretation may degrade user experience** if Population A surfaces the framing as noise. The falsification trigger (migrate to headers/metadata) is named; production traffic may exercise the trigger.
- **The synthesizer is invoked on every chat-completions request** that reaches the Synthesize stage. Synthesizer outages or persistent calibration Reflect verdicts propagate to chat-completions response failure. The operational profile is comparable to the routing-planner ensemble (both invoked per-request); the deployment has two new always-invoked ensembles instead of one.
- **The synthesizer's input contract is non-standard relative to existing capability ensembles.** Existing ensembles take user-shaped prompts; the synthesizer takes a structured `(REQUEST + PLAN + DISPATCH RESULTS)` input. The non-standard input contract is the price of structural bounding — bundling the role with a standard prompt-shape would re-introduce the orchestrator-LLM-style ambiguity.

### Neutral

- **The response-synthesizer ensemble is a system ensemble** under the `agentic-` prefix. Operators do not author it; it ships as part of the agentic-serving framework. Operator-facing complexity stays in the capability list and orchestration policy, not in the synthesizer itself.
- **Spike ε / ε' / μ's empirical baseline is retained as the BUILD starting point.** The scratch ensemble YAML migrates to production; tests (including the regression suite from spike work) are preserved.
- **The synthesizer's `topaz_skill` is `summarization`** per ADR-015's 8-skill taxonomy. The synthesizer's role is structurally a summarization task (produce response from structured input); calibration verdicts feed the tier-router as a summarization ensemble.
- **The Rule 5 BUILD default + falsification trigger** is an explicit two-state design — the cycle commits to the load-bearing interpretation now while preserving the path to migrate the signal if production evidence warrants. This is structurally the same pattern ADR-022 used (commitment + cross-profile-deferred-to-BUILD/PLAY characterization), with one important contextual difference: ADR-022's amendment was a system-prompt intervention on the orchestrator-LLM's reasoning shape under tool-rich-client suppression — an architecture the cycle later rejected as structurally insufficient for the routing-decision surface. Rule 5 operates in the response-synthesizer's structurally-bounded context where the tool-rich-client-suppression failure mode does not apply (the synthesizer's input is structured; no tools are declared in the synthesizer's context). The pattern (commitment with named falsification trigger; production evidence as the validation surface) transfers; the failure mode ADR-022 hit does not transfer because the synthesizer's operating context differs structurally.

## Provenance check

- **Empirical baseline at qwen3:8b across 13 tests + 4 confabulation modes**: Spike ε + Spike ε' + Spike μ research logs (driver). Driver chain: same-cycle empirical spikes.
- **Strict-fidelity rule set (Rules 1-5)**: Spike ε ensemble's system prompt (driver, tested rules) + Spike ε' validation of direct-completion path under the rules (driver). Driver chain: same-cycle spike-derived.
- **Rule 6 candidate (framework-convention enumeration)**: Spike μ.1 research log (driver). Driver chain: same-cycle MODEL-boundary spike.
- **Rule 5 framing scope as load-bearing-default-with-falsification-trigger**: Spike ε' Finding ε'.1 (driver, observed omission) + OQ #18 Population A configuration-honesty corroboration (driver, Cline #10551 + OpenCode #20859) + drafting-time synthesis (the two-state design with named trigger conditions). Driver chain: same-cycle spike + same-cycle research + drafting-time analytical engagement.
- **Rounding-drift mitigation playbook (system-prompt → tier escalation → runtime fidelity check hierarchy)**: Spike ε T3 + Spike ε' Finding ε'.2 (drivers, characterized two drift modes) + ADR-014 + ADR-015 + ADR-018 (drivers, existing calibration infrastructure) + drafting-time synthesis (mechanism shape). Driver chain: same-cycle spikes + prior-ADRs + drafting-time engagement.
- **Multi-turn continuity via prior-turns-in-input**: Spike ε' Finding ε'.3 (driver). Driver chain: same-cycle spike.
- **C4 failure mode (substrate-path file-read attempts) prevented by structured input**: Essay-Outline 006 §C4 (driver) + ADR-027 (driver, framework-driven pipeline framing). Driver chain: same-cycle essay + same-cycle ADR.
- **AS-9 structural-bounding property the synthesizer satisfies**: domain-model §AS-9 (driver). Driver chain: MODEL-phase codification.
- **Rejected alternative — orchestrator-LLM-as-synthesizer**: Cycle 6 PLAY note 22 (driver, composition confabulation) + Spike λ-paid F-paid-4 (driver, substrate-path failure) + AS-9 (driver, structural-bounding property). Driver chain: prior-cycle PLAY + same-cycle spike + same-cycle invariant.
- **Rejected alternative — deterministic template synthesis**: Spike ε' C1/C2 (driver, multi-turn continuity requires LLM-based reasoning) + drafting-time synthesis (information-loss floor argument). Driver chain: same-cycle spike + drafting-time engagement.
- **Rejected alternative — Rule 5 in headers/metadata only**: drafting-time synthesis examining Population A client-implementation variance against the universally-visible-content-layer signal. Driver chain: drafting-time analytical engagement; the falsification path leaves the alternative available.
- **Rejected alternative — don't codify Rule 6**: drafting-time synthesis weighing codification cost vs. model-substitution risk under the structural-property generalization claim. Driver chain: drafting-time engagement.
- **Calibration Gate integration with three new Reflect-trigger criteria**: drafting-time synthesis extending ADR-014's existing mechanism. Driver chain: prior-ADR-derived + drafting-time engagement.
