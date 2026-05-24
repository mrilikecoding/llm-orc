# ADR-030: `tool_choice` Disposition for the Agentic-Serving Chat-Completions Surface

**Status:** Proposed

**Date:** 2026-05-22

---

## Context

Cycle 7 Spike κ (2026-05-21) established the framework's current `tool_choice` handling via source-code inspection: D0 — zero handling. The `_ChatCompletionsRequest` Pydantic model at `src/llm_orc/web/api/v1_chat_completions.py:401-408` declares fields `model`, `messages`, `stream`, `tools`, `user` only and has no `tool_choice` field; Pydantic's default `extra="ignore"` silently discards the parameter at the request-parsing boundary. No code path in `src/llm_orc/` reads or forwards `tool_choice`. Clients sending the parameter receive HTTP 200 responses with no signal that the parameter was ignored.

The current behavior is a **configuration-honesty footgun**: tool-call-aware clients (Population A, per ADR-026 + the domain-model concept) sending `tool_choice` reasonably assume the parameter is honored according to the OpenAI contract. The empirical reality (silent strip) violates AS-10 (capability matching from request content alone — per ADR-026) only in the sense that the violation surfaces as a Population A degradation signal of the kind Cline #10551 (context-window divergence) and OpenCode #20859 (subagent model substitution) document — Population A's degradation signal is configuration dishonesty, and silently stripping a parameter the client sent is the canonical case.

Spike λ-paid (Cycle 7) separately tested whether the cross-compatibility-relevant production model (paid MiniMax M2.5 via OpenCode Zen) would honor `tool_choice={"name":"invoke_ensemble"}` if it reached the model. Result: HTTP 200, direct LLM completion, `tool_calls` absent — the model does not honor the parameter under tool-rich conditions. Spike κ's D0 finding made Spike λ-paid's failure-locus question moot (the parameter never reached the model), but the latent finding stands: even with framework-level implementation, the cross-compatibility-relevant production model does not enforce explicit-naming via `tool_choice` semantics.

Cycle 7 product-discovery Tension 19 names three operationally distinct dispositions:

- **(i) Implement `tool_choice` handling** — add the field to the Pydantic model; route requests carrying explicit `tool_choice={"name":"<ensemble>"}` through a deterministic dispatch path bypassing the routing-planner ensemble.
- **(ii) Explicitly reject** — return a 4xx error (or HTTP 200 with structured advisory) when `tool_choice` is present, signaling honestly that the parameter is not supported.
- **(iii) Reframe out of scope** — `tool_choice` is documented as unsupported; the framework continues to silently strip (status quo).

Tranche 2 framing audit F2 surfaced an operator-foreclosure concern: the Essay-Outline §C7's "PRIMARY direction" framing (ADR-027 as PRIMARY; hybrid as conditional alternative) forecloses operators with `tool_choice`-aware client populations who want explicit-naming intent honored deterministically. The F2 recommendation: surface the hybrid as an **orthogonal mechanism** for a self-selected Population A sub-segment, not as a "conditional alternative" subordinate to ADR-027.

This ADR resolves Tension 19 in light of AS-10 (configuration honesty implications of the status quo), Spike κ + Spike λ-paid findings, and the F2 framing carry-forward.

---

## Decision

**Adopt disposition (i) — implement `tool_choice` handling — as the cycle's commitment, with implementation deferred to a follow-on cycle.** Cycle 7 ships a bridge mechanism (disposition (ii) variant) that addresses the configuration-honesty footgun without blocking ADR-027 BUILD on full interception work.

### Architectural commitment: `tool_choice` is an orthogonal mechanism for `tool_choice`-aware Population A clients

When implemented, `tool_choice` handling is **orthogonal to the ADR-027 framework-driven dispatch pipeline**. The interception layer at the request boundary handles requests carrying `tool_choice={"name":"<ensemble>"}` or `tool_choice={"type":"function","function":{"name":"<ensemble>"}}` directly — bypassing the routing-planner ensemble — and dispatches the named ensemble via the existing `OrchestratorToolDispatch` machinery (per ADR-021), with the dispatch result fed into the response-synthesizer (per ADR-029) for response composition.

The orthogonal positioning reflects the operator-foreclosure concern Tranche 2 framing audit F2 surfaced: `tool_choice`-aware client populations are a real Population A sub-segment (skill frameworks that include explicit ensemble naming in their `tool_choice` payloads; future tool-call-aware client populations) whose explicit-naming intent should be honored deterministically when expressed. ADR-027 + ADR-030's full implementation honor that intent.

`tool_choice={"auto"}` (the OpenAI default) is treated as equivalent to absent `tool_choice` — the routing-planner ensemble runs normally; the planner's decision is the routing decision. `tool_choice={"required"}` (no specific function named, but a function call required) routes through the routing-planner with a constraint that the planner must produce `action: "dispatch"` (or, if no capability match, the framework returns an error per the OpenAI contract — "required" cannot be satisfied if no tool is appropriate).

### Cycle 7 bridge mechanism — disposition (ii) variant (provisional)

For Cycle 7 BUILD (where the focus is ADR-027 + ADR-028 + ADR-029 implementation), the framework adopts the bridge mechanism: **return HTTP 200 with a `tool_choice`-specific structured advisory when `tool_choice` is present, naming the parameter as unsupported in the current build and pointing operators toward the disposition (i) follow-on**.

**Provisional mechanism marker:** the bridge mechanism is explicitly a **provisional commitment**, not a stable architectural state. It exists to address the AS-10 configuration-honesty footgun (silent-strip status quo) without blocking ADR-027 BUILD on full interception work. The provisional state is named so that downstream artifacts (cycle-status, ORIENTATION.md, system-design.md, conformance audits) can identify the bridge advisory as a transitional construct rather than a long-lived architectural surface. The bridge mechanism is replaced when disposition (i) implementation lands per the follow-on trigger below.

**Follow-on trigger (enforcement structure for the disposition (i) commitment):** the disposition (i) full implementation is evaluated for delivery at the **next DECIDE gate after ADR-027 reaches production deployment**.

**The disposition (i) commitment itself is not evidence-conditional.** The silent-strip configuration-dishonesty footgun is rejected regardless of `tool_choice`-honoring client volume — AS-10 + the Population A configuration-honesty trust contract make the silent strip unacceptable at any client volume. The bridge advisory addresses the footgun in the interim; disposition (i) is the durable replacement; the cycle's commitment is to ship disposition (i) before the bridge advisory becomes the long-term architectural surface.

**The sprint-scoping priority of the disposition (i) work IS evidence-conditional.** If ADR-027's production deployment surfaces evidence that `tool_choice`-honoring clients are an active Population A sub-segment (operators with client-side skill frameworks emitting `tool_choice` shapes; new client populations adopting the parameter), disposition (i) is highest-priority sprint work in the follow-on cycle. If evidence suggests `tool_choice`-honoring clients are vanishingly rare in the deployment's traffic, disposition (i) may be **deprioritized within the follow-on cycle** (other sprint work takes precedence) but is **not abandoned** — the bridge advisory persists until disposition (i) ships, and the next DECIDE gate after that re-evaluates the sprint-scoping priority. Downstream cycles either deliver disposition (i) or explicitly revise the commitment itself via an ADR amendment that re-grounds the configuration-honesty argument; "deprioritize without delivery" without amendment is not a valid path.

#### Bridge advisory specification (self-contained per Cycle 7)

The bridge advisory is specified here as the canonical implementation note for Cycle 7 BUILD; ADR-032's honest response labeling mechanism carries the signal at the response layer but does not specify the bridge's content:

- **Signal trigger:** the framework's request-parsing layer detects a `tool_choice` field in the chat-completions request body. The detection happens *before* the request flows through ADR-027's pipeline (the Pydantic `_ChatCompletionsRequest` model is extended to accept and observe `tool_choice` without honoring it for routing).
- **Signal content:** the framework emits a typed metadata signal on the response. The signal's canonical key is `tool_choice_handling`; the canonical value for the bridge state is `"deferred"`. A human-readable message accompanies the signal (operator/client-developer-readable): "`tool_choice` parameter received but not used for routing in this build; routing handled by the framework-driven dispatch pipeline (ADR-027)."
- **Signal delivery layers** (all three layers carry the signal; redundancy intentional per ADR-032):
  - **Response header:** `X-LLM-Orc-Tool-Choice-Handling: deferred` (or the framework's chosen `served-by` header family from ADR-032's BUILD-phase design; the field name is conventional, not load-bearing).
  - **Response body metadata:** the response includes a `metadata.tool_choice_handling: "deferred"` field (or equivalent within the response shape ADR-032's body-metadata mechanism establishes).
  - **Response body content (Rule 5-adjacent):** for direct-completion responses (`action: "direct"` in ADR-027's pipeline), the response-synthesizer's Rule 5 framing is extended to acknowledge the `tool_choice` parameter ("this answer was generated directly without dispatching a specialist ensemble; `tool_choice` was received but not honored for routing in this build"). For dispatch responses, content-layer acknowledgment is not required (the configuration-honesty signal is delivered at headers/metadata layers).
- **Out of scope for the bridge:** `tool_choice={"auto"}` (the OpenAI default value) is treated as equivalent to absent `tool_choice` and does not trigger the advisory (the request flows through ADR-027's pipeline normally without signal). The advisory fires only when `tool_choice` is explicitly set to a non-default value.

The bridge mechanism is intermediate. It does not block on disposition (i) implementation; it does prevent the silent-strip configuration-dishonesty footgun. Cycle 7 BUILD ships the bridge per the specification above; follow-on cycle implements disposition (i) per the follow-on trigger above; ADR-030 is amended (not superseded) when disposition (i) lands.

Disposition (iii) (silent strip status quo) is explicitly rejected. The status quo creates a Population A degradation signal of the kind OQ #18 validation named as load-bearing (Cline #10551; OpenCode #20859); shipping ADR-027 while leaving the configuration-dishonesty footgun in place would defeat one of the configuration-honesty sub-promise's structural commitments.

### Disposition (i) implementation scope (follow-on cycle)

The follow-on cycle's `tool_choice` interception work breaks into four items (per OQ #19 build-complexity comparison):

1. **`tool_choice` field at the request boundary** (1-2 days) — add the field to `_ChatCompletionsRequest`; typed parsing for the three OpenAI shapes (`"auto"`, `"required"`, `{"type":"function","function":{"name":"X"}}`); unit tests for parsing + integration test for request acceptance.
2. **Ensemble-name resolver** (1 day) — when `tool_choice={"name":"<ensemble>"}` or the function-call shape, resolve against the framework's loaded-ensemble registry; reject unknown names with structured error; emit typed observability event for the resolution decision.
3. **Server-side dispatch path** (2-3 days) — new code path in `v1_chat_completions.py` that, on `tool_choice` interception, calls `OrchestratorToolDispatch.dispatch()` directly (bypasses the routing-planner ensemble); the dispatch result feeds the response-synthesizer per ADR-029 normal contract.
4. **Migration / regression tests** (1 day) — non-intercepted requests continue through ADR-027's routing-planner path; intercepted requests fire the new path; the bridge advisory is removed (replaced by the interception's normal behavior).

Total: ~5-7 days follow-on. The work is layered on ADR-027 (does not require restructuring ADR-027 or its sub-mechanisms); the routing-planner and response-synthesizer ensembles' contracts are unchanged.

### Out of scope for ADR-030

- **`tool_choice={"type":"function","function":{"name":"X"}}` honoring across model populations.** Spike λ-paid established that paid MiniMax M2.5 does not honor the parameter at the model layer; whether other production models honor it is empirically open. The disposition (i) implementation honors the parameter at the **framework** layer (the framework intercepts and dispatches the named ensemble); model-layer honoring is separate and orthogonal to ADR-030.
- **Custom non-OpenAI-standard `tool_choice` extensions.** Some custom clients may invent their own variants of `tool_choice` semantics. The framework honors the OpenAI-protocol-native shapes only; non-standard variants pass through to disposition (ii)'s bridge advisory or to the follow-on cycle's structured-error response.
- **Tool-call-aware behavior in the routing-planner ensemble.** Whether the routing-planner ensemble can consume `tools[]` content from the request (the OpenAI-protocol-native `tools` array) to inform its routing decision is a routing-planner enhancement; ADR-030 governs `tool_choice` disposition, not tool-list consumption. The routing-planner ensemble per ADR-028 receives the request content including `tools[]`; how it uses that content is the planner's design choice (current behavior: the planner ignores `tools[]` in its routing decision).

---

## Rejected alternatives

### Disposition (ii) full ("explicitly reject" — return 4xx error when `tool_choice` is present)

Return an HTTP 400 (or 422) when `tool_choice` is present in the request, with a structured error declaring the parameter unsupported.

**Rejected because:** the 4xx response is too strong relative to the bridge mechanism's behavior. Population A clients sending `tool_choice` typically also send the standard OpenAI chat-completions payload — the request is valid for processing without `tool_choice` semantics; rejecting it forces clients to detect and remove the parameter before retrying. The bridge mechanism (HTTP 200 with structured advisory) achieves the configuration-honesty goal without breaking workflows that would succeed under ADR-027's pipeline regardless.

The 4xx response also creates a Population A degradation signal in the opposite direction — clients that legitimately send `tool_choice` as a hint (not as a hard constraint) lose access to the chat-completions surface entirely until they update their client configurations. Configuration honesty is preserved by the bridge advisory; the harsher 4xx response over-corrects.

### Disposition (iii) ("reframe out of scope" — silent strip status quo)

Continue current behavior; document `tool_choice` as unsupported in the framework's API documentation; Pydantic continues to silently discard the parameter.

**Rejected because:** AS-10 + the Population A configuration-honesty trust contract make the silent strip unacceptable. Population A's degradation signal (per OQ #18 — Cline #10551 + OpenCode #20859) names the silent-divergence-between-configured-and-actual-behavior pattern as the canonical config-dishonesty bug. Silent-strip is precisely that pattern.

Documentation does not mitigate the footgun. Population A clients typically configure llm-orc via base-URL only; their developers do not read llm-orc-specific documentation as part of normal operation. The framework's own protocol-contract behavior must be honest at the wire-level; documentation-only mitigation fails the Population A trust contract that AS-10 codifies.

### Disposition (i) with Cycle 7 immediate implementation

Implement `tool_choice` handling fully in Cycle 7 BUILD; ship ADR-027 + ADR-030 together.

**Rejected because:** the build-complexity comparison (OQ #19) showed the Tier 1 hybrid's `tool_choice` interception work items 1-3 totaling ~4-6 days; Cycle 7 BUILD's primary load is ADR-027 + ADR-028 + ADR-029 + supporting infrastructure (~16 person-days median per OQ #19). Folding the full disposition (i) implementation into Cycle 7 grows BUILD scope by ~30% and concentrates risk on a single deployment.

The deferral preserves the disposition (i) commitment without growing Cycle 7's BUILD risk. The bridge mechanism prevents the configuration-honesty footgun in the interim; the follow-on cycle delivers the full mechanism after ADR-027's empirical evidence informs the implementation (e.g., production traffic showing how often Population A clients send `tool_choice` may inform priority and design refinements).

### Implement `tool_choice` honoring as a routing-planner-internal mechanism (rather than at the request boundary)

Instead of intercepting `tool_choice` at the chat-completions handler, pass it through to the routing-planner ensemble (per ADR-028 input contract extension); the planner reasons about whether to honor `tool_choice` semantics.

**Rejected because:** this re-introduces LLM judgment into a request-boundary deterministic-routing surface. `tool_choice={"name":"<ensemble>"}` is OpenAI-protocol-native explicit-naming intent; honoring it does not require reasoning. Routing it through the planner adds latency (the planner is the cheap-tier qwen3:8b ensemble at p50=10s) and removes the determinism the OpenAI contract conveys.

The routing-planner-internal alternative also conflates two distinct mechanisms — the planner's natural-language inference (its role under AS-9 + ADR-028) with the deterministic dispatch path for explicit-naming intent. Keeping these mechanisms separate (planner for NL inference; interception for explicit `tool_choice`) preserves the bonus-path layering ADR-028 names.

---

## Consequences

### Positive

- **AS-10's configuration-honesty implication is preserved structurally.** The silent-strip footgun is removed in Cycle 7 BUILD via the bridge advisory; the follow-on cycle's full disposition (i) implementation completes the honest handling.
- **Operator-foreclosure concern (F2) is addressed.** `tool_choice`-aware Population A sub-segments — skill frameworks that include explicit ensemble naming in their `tool_choice` payloads; future tool-call-aware client populations — have a path to deterministic honoring. The hybrid is positioned as orthogonal to ADR-027 (mechanism for a self-selected sub-segment), not as a "conditional alternative" subordinate to the routing-planner.
- **The disposition (i) implementation is structurally layered on ADR-027.** The interception bypasses the routing-planner; the dispatch executes via `OrchestratorToolDispatch` (per ADR-021); the result feeds the response-synthesizer (per ADR-029). No changes to ADR-027's three-stage pipeline are required to implement the disposition.
- **Cycle 7 BUILD risk is contained.** The bridge mechanism is a small addition (advisory mechanism per ADR-032); Cycle 7's primary load remains ADR-027 + ADR-028 + ADR-029 + supporting infrastructure. The follow-on cycle delivers the full mechanism with ADR-027 empirical evidence informing the design.
- **The OpenAI-protocol-native contract is honored at the framework layer.** When disposition (i) is implemented, framework-level honoring of `tool_choice={"name":"<ensemble>"}` is deterministic — the framework dispatches the named ensemble regardless of which model serves the synthesizer. Model-layer honoring (whether the synthesizer model would have called the function natively) is decoupled.

### Negative

- **The follow-on cycle commitment is real.** ADR-030 names disposition (i) as the cycle's commitment; the follow-on cycle must deliver the implementation. If the follow-on cycle is delayed or descoped, the bridge advisory persists longer than designed — Population A clients live with the structured-advisory state rather than the deterministic honoring.
- **The bridge advisory itself adds operational complexity.** The advisory mechanism (per ADR-032) is a new response surface; Population A clients may or may not surface it usefully. The advisory is the best Cycle 7 BUILD can offer; full configuration-honesty arrives with disposition (i).
- **Model-layer non-honoring (per Spike λ-paid) remains a latent gap.** Even with disposition (i)'s framework-layer interception, the framework honors `tool_choice` semantically (dispatches the named ensemble) — the *model* that serves the response is the synthesizer, not the ensemble whose name was in `tool_choice`. Clients expecting the OpenAI-protocol-native contract where the synthesizer model itself produces the tool call may find the divergence subtle. This is the structural cost of the framework-driven pipeline: the ensemble-dispatch path is named and honored, but the synthesizer model identity in the response is the response-synthesizer ensemble's model, not the client-named function-providing model.
- **`tool_choice={"required"}` may surface a Population A friction point.** When required is set but no capability match exists, the framework must either error (per OpenAI contract — "required cannot be satisfied if no tool is appropriate") or honor the spirit of the request by dispatching a generic completion ensemble. The follow-on cycle's design choice; ADR-030 names the constraint without resolving it.

### Neutral

- **The bridge mechanism is replaceable by the full implementation.** The bridge advisory mechanism does not impose architectural constraints on the follow-on cycle; removing the bridge and shipping the full interception is a clean replacement.
- **`tool_choice` honoring as a feature is orthogonal to the routing-planner.** The routing-planner ensemble does not need to know about `tool_choice` — the interception is at the request boundary; the planner's input contract (per ADR-028) does not consume `tool_choice` semantics. The bonus-path layering (per ADR-028) is preserved.
- **Custom `tool_choice` variants pass through to the bridge advisory.** Non-OpenAI-standard `tool_choice` shapes (custom clients inventing their own semantics) are handled as if standard `tool_choice` were present — they receive the bridge advisory. The framework does not attempt to interpret variant semantics.

## Provenance check

- **Spike κ D0 finding (framework strips `tool_choice` at input)**: Spike κ research log (driver). Driver chain: same-cycle empirical spike.
- **Spike λ-paid model-layer non-honoring**: Spike λ-paid research log (driver). Driver chain: same-cycle paid spike.
- **Tension 19 three dispositions (implement / explicitly reject / reframe out of scope)**: product-discovery-cycle-7-update Tension 19 (driver) + domain-model §Concepts "`tool_choice` strip-at-input" (driver). Driver chain: same-cycle product-discovery + same-cycle domain-model entry.
- **AS-10 configuration-honesty implication of the status quo**: ADR-026 (driver). Driver chain: same-cycle ADR.
- **OQ #18 Population A configuration-dishonesty as the canonical degradation signal (Cline #10551; OpenCode #20859)**: Tranche 1 research note `cycle-7-oq-18-cost-distribution-validation.md` Q3 finding (driver). Driver chain: same-cycle DECIDE-entry research.
- **Tranche 2 framing audit F2 carry-forward (orthogonal mechanism positioning)**: cycle-status §"Framing-audit findings carried forward to DECIDE/ADR-drafting/gate consideration" F2 entry (driver). Driver chain: same-cycle Tranche 2 audit.
- **OQ #19 build-complexity scoping for disposition (i) work items (1-2 + 1 + 2-3 + 1 = ~5-7 days)**: Tranche 1 research note `cycle-7-oq-19-build-complexity-comparison.md` Tier 1 hybrid work items 1-3 + 8 (driver). Driver chain: same-cycle DECIDE-entry research; the work-item decomposition is identical between Tier-1-as-primary and disposition-(i)-as-orthogonal-extension because the same code paths are implemented.
- **Bridge mechanism — disposition (ii) variant with structured advisory in response metadata**: drafting-time synthesis bridging the configuration-honesty AS-10 implication to ADR-032's honest response labeling mechanism. Driver chain: same-cycle ADR-032 + drafting-time analytical engagement.
- **Rejected alternative — disposition (ii) full (4xx error)**: drafting-time synthesis weighing the 4xx response's break-on-existing-workflows against the bridge mechanism's process-with-advisory shape. Driver chain: drafting-time engagement.
- **Rejected alternative — disposition (iii) silent strip**: ADR-026 AS-10 + OQ #18 (drivers, configuration-honesty requirement). Driver chain: same-cycle ADR + same-cycle research.
- **Rejected alternative — Cycle 7 immediate implementation**: OQ #19 build-complexity comparison (driver, ~16 person-days median ADR-027-direct cost + ~5-7 days additional for disposition (i)). Driver chain: same-cycle research + drafting-time scope-judgment.
- **Rejected alternative — routing-planner-internal `tool_choice` honoring**: ADR-028 routing-planner contract (driver) + drafting-time synthesis (LLM-reasoning vs. deterministic-routing trade for an OpenAI-protocol-native deterministic intent). Driver chain: same-cycle ADR + drafting-time engagement.
- **Out-of-scope item — model-layer honoring**: Spike λ-paid (driver, paid MiniMax M2.5 does not honor `tool_choice` at the model layer). Driver chain: same-cycle paid spike; the framework-layer / model-layer decoupling is drafting-time positioning.
- **Out-of-scope item — routing-planner consumption of `tools[]`**: ADR-028 routing-planner input contract (driver). Driver chain: same-cycle ADR; the carve-out preserves the planner's existing contract while leaving the enhancement available.
