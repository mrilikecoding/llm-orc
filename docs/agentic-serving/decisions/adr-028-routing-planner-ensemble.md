# ADR-028: Routing-Planner Ensemble Specification

**Status:** Proposed

**Date:** 2026-05-22

---

## Context

ADR-027 establishes the framework-driven dispatch pipeline (plan → dispatch → synthesize) as the primary direction for the agentic-serving chat-completions surface, with the orchestrator-LLM removed from the routing-decision and post-dispatch-synthesis surfaces. The Plan stage requires a structurally-bounded role (per AS-9) operating on chat-completions request content alone (per AS-10, ADR-026) that produces a deterministic dispatch plan the framework executes without further LLM reasoning.

Cycle 7 Spike ζ (2026-05-21) empirically established the routing-planner ensemble's mechanism viability at qwen3:8b across a 20-prompt battery spanning six prompt shapes (explicit-naming, NL clear-match single capability, NL ambiguous-match, NL multi-capability composition, NL no-match, adversarial). Results: 100% JSON conformance + 100% schema validity + 90% strict capability-match (100% with defensible judgment in ambiguous cases). Latency p50=10s, p90=13s, mean=10s via local Ollama. The spike ensemble (`spike-cycle7-epsilon-response-synthesizer.yaml` and its planner sibling; reused for Spike μ confabulation generalization) is the empirical baseline this ADR promotes to production specification.

This ADR specifies the routing-planner ensemble's contract: input shape, output schema, model-profile constraints, dispatch contract, and the bonus-path layering for explicit-naming and `tool_choice` (where ADR-030 implements `tool_choice` handling).

---

## Decision

**Adopt the routing-planner ensemble as the primary Plan-stage mechanism** in the framework-driven dispatch pipeline (per ADR-027). The ensemble is a system ensemble (per ADR-025 categorization — operator does not author it; it ships as part of the agentic-serving framework) under the `agentic-` prefix convention (per ADR-019), located at `.llm-orc/ensembles/agentic-serving/agentic-routing-planner.yaml` or equivalent.

### Input contract

The routing-planner ensemble receives one input per chat-completions request:

```
ORIGINAL REQUEST:
  messages: <messages[] array from the chat-completions request>
  model: <model field from the request>
  tools: <tools[] array, if present>

CAPABILITY LIST:
  <list of capability ensembles available in the deployment's library, each with name + description + topaz_skill tag, sourced from the framework's loaded-ensemble registry>
```

No client-side opt-in signal is permitted in the input per AS-10. The capability list is the framework's authoritative view of available capability ensembles (sourced from the loaded-ensemble registry per ADR-019); the planner does not query the registry or perform retrieval beyond the supplied list.

### Output contract

The routing-planner ensemble produces structured JSON output conforming to the dispatch-plan schema:

```json
{
  "action": "dispatch" | "direct",
  "ensemble": "<ensemble-name>" | null,
  "rationale": "<one-sentence explanation of the routing decision>",
  "input": "<input string for the dispatched ensemble; required when action=dispatch>"
}
```

- `action: "dispatch"` — the planner identified a capability match; the framework dispatches the named `ensemble` with the planner-prepared `input`.
- `action: "direct"` — no capability match exists for the request; the framework falls through to the response-synthesizer's direct-completion path (per ADR-029 + ADR-032).
- `ensemble` is `null` when `action: "direct"`; otherwise it names an ensemble in the supplied capability list.
- `rationale` is a one-sentence operator-readable explanation. The framework logs the rationale in dispatch events (per ADR-023 observability event routing); the rationale does not feed into downstream stages.
- `input` is the string passed to the dispatched ensemble (when `action: "dispatch"`). The planner constructs it from the ORIGINAL REQUEST's relevant user message content; the framework does not transform the planner's `input` before dispatching.

**Multi-step composition** (per OQ #21) is open as a downstream-phase design question. The initial BUILD adopts **single-step planner + framework chain-heuristic** as the default: the planner produces a single dispatch step; multi-step composition (`web-searcher → claim-extractor` chains, etc.) emerges from the framework's chain-heuristic detecting that one ensemble's output is suitable input for another (per Spike δ's `resolve_input(step.input, results)` pattern). Multi-step planner output (sequence of dispatch steps in a single plan) is a candidate extension; production traffic diversity may warrant it (BUILD/PLAY signal).

### Model profile and tier

The routing-planner ensemble is **cheap-tier by default**. The empirical baseline is qwen3:8b via local Ollama (Spike ζ — 100% JSON conformance + 90% strict capability-match across 20-prompt battery; p50=10s latency).

Operators may override the routing-planner's model profile via the ensemble's YAML (per ADR-011's session-boundary config discipline) for deployment shapes where the default does not suit. The override surface remains available; the framework does not lock the routing-planner to qwen3:8b. Operator override risks are operator-borne; the framework's empirical-grounding scope is the default profile.

The Calibration Gate (per ADR-007, ADR-014) operates within the routing-planner ensemble; calibration verdicts on the planner's output feed the tier-escalation router (per ADR-015) — if the planner ensemble persistently calibrates Reflect on production traffic, escalation to a higher tier is the existing infrastructure's response.

### Latency posture

The routing-planner ensemble's p50=10s / p90=13s latency at qwen3:8b exceeds the R2-1 research-methods-reviewer latency bound (1.0s absolute / 20% relative) — every chat-completions request pays approximately one bare-LLM-completion latency for the routing decision. Per DISCOVER 2026-05-21 practitioner framing, latency is a **tuning concern not a structural blocker**; the cycle's primary commitment is correctness of the routing decision, with latency tuning as downstream-phase work.

Tuning axes named in DISCOVER (each independent; pursued separately based on operational evidence):

- **Classifier pre-filter** — a lightweight per-request classifier that short-circuits the routing-planner when explicit-naming or no-capability-match is statically determinable. Reduces planner invocations on the populations of requests the classifier can handle.
- **Cached planner decisions** — memoize plans for identical or near-identical requests. Reduces planner invocations on the population of repeated request shapes.
- **Smaller faster planner model** — alternative cheap-tier models with lower latency (e.g., qwen3:4b, phi-3.5-mini) if their reliability profile is comparable. Operator-configurable per the model-profile override mechanism.
- **Streaming synthesizer response** — parallel synthesis with downstream client display, masking some routing latency (relevant for streaming-default clients per OQ #20 — OpenCode and Aider).

The latency ADR (per ADR-031) integrates these tuning axes with the OQ #20 Population A timeout findings. ADR-028 does not foreclose any tuning axis; it specifies the routing-planner's contract independent of which tuning axes are pursued.

### Bonus-path layering for explicit-naming and `tool_choice`

The routing-planner is the **primary** mechanism per ADR-027. Two bonus paths layer onto the routing-planner for operator-deployment shapes that warrant them:

- **Explicit-naming extractor at the request boundary** — a regex- or parser-based extractor that detects explicit ensemble naming patterns in the user message (e.g., `"dispatch the code-generator capability"`, `"invoke web-searcher with..."`). When the extractor matches, the framework can short-circuit the routing-planner and dispatch directly. The extractor is a Population-mediated bonus path — Population A clients that pass-through explicit naming (e.g., a skill framework that includes the ensemble name in the prompt) benefit; Population A clients that send pure NL framing are routed by the planner.
- **`tool_choice` interception at the request boundary** — when ADR-030 implements `tool_choice` handling, requests carrying `tool_choice={"name":"<ensemble>"}` short-circuit the routing-planner. The hybrid extension is meaningful only under the "implement `tool_choice` handling" disposition (per ADR-030).

Both bonus paths are optional extensions. The routing-planner ensemble's contract is unchanged by the presence or absence of either; bonus paths intercept *before* the planner runs, not after.

### Ensemble structure

The routing-planner ensemble's internal structure follows the existing ensemble authoring patterns (per ADR-019, ADR-020 — tool-use ensemble shape, operation-named convention). The ensemble's `default_task` specifies the JSON output schema; the synthesizer agent within the ensemble produces the output. The ensemble's YAML names its `topaz_skill` as `tool_use` (per ADR-015's Topaz 8-skill taxonomy) — the routing decision is structurally a tool-use task (choose which tool to invoke from a capability list).

The empirical baseline ensemble (`spike-cycle7-epsilon-response-synthesizer.yaml`-sibling planner) is retained as the BUILD-phase starting point. BUILD work hardens the spike ensemble — promotes it from scratch artifact to production ensemble; adds the 20-prompt battery as a regression test suite; integrates with the framework's loaded-ensemble registry; adds dispatch event emission for plan-emitted lifecycle events (per ADR-023).

**The `topaz_skill: "tool_use"` classification is pragmatic, not load-bearing.** The routing decision is structurally a tool-use task (choose which tool to invoke from a capability list), but the planner's empirical failure mode at qwen3:8b is schema-non-conformance under complex inputs rather than tool-selection error per se. If calibration evidence under production traffic suggests a different Topaz skill better captures the planner's failure mode (e.g., `summarization` for produce-structured-output-from-given-context roles; `instruction_following` for adherence-to-output-schema failures), the classification can be revised at ARCHITECT or BUILD without structural consequence — the tier-router and audit infrastructure operate against whatever skill the ensemble declares.

---

## Rejected alternatives

### Rule-based classifier (deterministic capability matcher without LLM)

A deterministic classifier (regex patterns, keyword matching, structural rules) maps request content to capability identifiers. No LLM in the Plan stage; latency is sub-millisecond.

**Rejected because:** a deterministic classifier has a fundamental model-quality-floor brittleness — it works well on the populations of requests its rules cover and fails opaquely on populations they don't. The 20-prompt battery Spike ζ tested includes ambiguous-match and adversarial prompt shapes that rule-based classifiers handle poorly; the routing-planner ensemble's 90% strict capability-match + 100% defensible-judgment-match across the battery establishes a capability range deterministic classifiers cannot match without a brittle accumulation of patterns.

The classifier's rules also have to be maintained per capability; adding a new capability ensemble requires updating the classifier's rules. The routing-planner ensemble naturally accommodates new capabilities via the supplied capability list — the planner's input contract adapts at deployment time. The maintenance asymmetry favors the planner.

The classifier remains viable as the **classifier pre-filter** tuning axis above (deterministic short-circuit for clearly-determinable cases, with the planner as fallback). The rejected alternative is the classifier-as-primary mechanism, not the classifier-as-pre-filter.

### Embedding-similarity router (capability matched by vector similarity to ensemble descriptions)

The framework embeds the request content and capability descriptions; the highest-similarity ensemble is dispatched.

**Rejected because:** embedding-similarity routing has a capability-coverage gap — it routes by surface similarity (does this request look like this ensemble's description?) rather than capability semantics (does this ensemble's role fit this request?). Population A's request shapes vary substantially from the capability descriptions operators author; an embedding-similarity router's accuracy depends on the embedding model's training corpus including conventions both client populations and operators use. The routing-planner ensemble reasons about the match in natural language — capable of bridging vocabulary gaps the embedding model would miss.

Embedding-similarity also has a no-capability-match floor problem — there is always some most-similar ensemble. The routing-planner can return `action: "direct"`; an embedding-similarity router needs a threshold mechanism the operator tunes, which surfaces the same calibration problem in a different form. The planner ensemble's output schema includes the `direct` action natively.

### LLM-with-tools planner (the routing-planner has access to additional tools beyond producing the plan)

The planner is an LLM that can issue tool calls (e.g., `list_ensembles`, `query_knowledge_graph`, `web_search`) to inform its routing decision, not just produce a single JSON output.

**Rejected because:** this re-introduces the bundled-reasoning surface AS-9 names as the structural failure source. The routing-planner-as-LLM-with-tools would bundle the routing decision with multi-step tool-use reasoning — exactly the configuration the orchestrator-LLM-as-decider failed on (per Cycle 6 PLAY note 22; Spike λ-paid F-paid-4). The structural property AS-9 codifies is bounded-role-shape; an LLM-with-tools planner inflates the role-shape back to the orchestrator-LLM-shape.

The framework's existing infrastructure provides the inputs an LLM-with-tools planner would gather — the capability list is supplied in the input contract; the deployment's knowledge graph (per ADR-009 — Plexus integration) is queried at session level, not per-request. The bounded-role planner is empirically reliable (Spike ζ); the LLM-with-tools planner is structurally the failure surface the cycle's architecture exists to address.

### No planner — orchestrator-LLM stays as the routing decider

The chat-completions surface continues to route through the `OrchestratorRuntime` ReAct loop with the orchestrator-LLM as the routing decider; no separate planner ensemble is introduced.

**Rejected because:** this is the alternative ADR-027 rejected as a whole. The orchestrator-LLM-as-decider is the structurally-recurring failure surface across three documented failure modes (C1 composition confabulation; Spike δ positive control via removal; Spike λ-paid F-paid-4 post-dispatch protocol-format failure). AS-9's codification names the structural property; ADR-027's primary direction implements the property; ADR-028 (this ADR) specifies the routing-planner ensemble as the role that satisfies the property on the Plan stage.

---

## Consequences

### Positive

- **Plan-stage reliability is empirically grounded.** Spike ζ established 100% JSON conformance + 100% schema validity + 90% strict capability-match across the 20-prompt battery at qwen3:8b. The planner's behavior on the routing-decision task is characterized; production traffic provides corroborating evidence.
- **The planner satisfies AS-9 structurally.** The role is single-decision-shaped (produce JSON dispatch plan from request content + capability list); the LLM does not chain through multi-step reasoning, tool calls, or narration. The structural-bounding property AS-9 names is preserved by the planner's contract.
- **The planner operates within AS-10's constitutional scope.** Input is chat-completions request content + capability list; no out-of-band signals are consumed. The transparent-endpoint promise (per ADR-026) extends through the Plan stage.
- **Operator override is preserved at the model-profile layer.** Deployments choosing a different cheap-tier model are accommodated via the ensemble's YAML; the framework's empirical-grounding scope is the default profile, not a lock-in.
- **The 20-prompt battery serves as a regression test suite.** BUILD-phase work promotes Spike ζ's battery to an integration test; future changes to the planner ensemble (model substitution, prompt refinement, schema evolution) are tested against the empirical baseline.
- **Bonus-path layering is structurally clean.** Explicit-naming extractor and `tool_choice` interception (per ADR-030 disposition) intercept before the planner runs; the planner's contract is unchanged. Operators can adopt or omit bonus paths without restructuring the Plan stage.

### Negative

- **Latency floor adds approximately one bare-LLM-completion to every chat-completions request.** The p50=10s / p90=13s profile at qwen3:8b is the empirical baseline; tuning axes (per ADR-031) address this but the latency cost is structural until tuning is applied. Cline's 30s hard default breaches the single-step floor by 6s every request without operator tuning (per OQ #20).
- **Cheap-tier reliability beyond qwen3:8b is plausible-but-untested.** Operators substituting models bear the characterization burden; the framework's empirical floor is qwen3:8b-grounded.
- **Routing-planner failure is a chat-completions-surface failure.** The planner is invoked on every request; outages or schema-non-conformance propagate. The existing infrastructure (calibration gate, tier-escalation router, autonomy policy, budget controller) operates within the planner ensemble as in any other ensemble dispatch — but the planner's operational profile is elevated relative to capability ensembles.
- **Multi-step composition (OQ #21) is deferred to BUILD design.** The initial single-step planner + framework chain-heuristic default may not handle complex composition shapes Population A clients produce; BUILD/PLAY characterizes the population.
- **The 20-prompt battery is small.** 20 prompts is sufficient to establish mechanism viability per Spike ζ's purpose but is not production-scale coverage. The battery is a regression-test floor, not a comprehensive capability characterization. Production traffic diversity (OQ #25) provides the broader evidence base.

### Neutral

- **The routing-planner ensemble is a system ensemble.** Operators do not author it; it ships as part of the agentic-serving framework under the `agentic-` prefix convention. Operator-facing complexity grows in the capability list (which the operator authors and maintains per ADR-019), not in the planner itself.
- **Spike ζ's empirical baseline is retained as the BUILD starting point.** The scratch ensemble YAML migrates to production; no fresh ensemble is authored. Test suite continuity is preserved.
- **The planner's `topaz_skill` is `tool_use`** per ADR-015's 8-skill taxonomy. The tier-escalation router treats the planner as a tool-use ensemble; the planner's calibration verdicts feed the router's drift criteria (per ADR-018) along with other tool-use ensembles.

## Provenance check

- **Routing-planner mechanism viability**: Spike ζ research log `cycle-7-spike-zeta-routing-planner.md` (driver). Driver chain: same-cycle empirical spike.
- **AS-9 structural-bounding property the planner satisfies**: domain-model §AS-9 (driver). Driver chain: MODEL-phase codification 2026-05-22; same-cycle invariant.
- **AS-10 request-shape commitment the planner operates within**: ADR-026 (driver). Driver chain: same-cycle ADR.
- **Input contract (ORIGINAL REQUEST + CAPABILITY LIST)**: Spike ζ's tested input shape (driver). Driver chain: same-cycle spike.
- **Output contract (`action` / `ensemble` / `rationale` / `input` JSON schema)**: Spike ζ's tested output schema (driver). Driver chain: same-cycle spike.
- **Multi-step composition deferred to BUILD design via single-step-planner + framework-chain-heuristic default**: OQ #21 (driver, open question carried at MODEL) + Spike δ's `resolve_input(step.input, results)` pattern (driver, prior-cycle spike). Driver chain: same-cycle MODEL + prior-cycle spike.
- **Latency posture as tuning concern not structural blocker**: DISCOVER 2026-05-21 practitioner framing (driver) + Essay-Outline 006 §C3 W3.3 (driver). Driver chain: practitioner-voice settled at DISCOVER; recorded in essay.
- **Tuning axes (classifier pre-filter, cached decisions, smaller model, streaming synthesizer)**: DISCOVER 2026-05-21 framing (driver). Driver chain: same as above.
- **Bonus-path layering (explicit-naming extractor, `tool_choice` interception)**: Essay-Outline 006 §C3 W3.2 (driver) + ADR-030 (driver for `tool_choice` interception). Driver chain: same-cycle essay + same-cycle ADR.
- **Rejected alternative — rule-based classifier**: drafting-time synthesis examining the alternative against Spike ζ's empirical capability range. Driver chain: spike-derived + drafting-time analytical engagement. The classifier-as-pre-filter tuning-axis carve-out is drafting-time positioning.
- **Rejected alternative — embedding-similarity router**: drafting-time synthesis examining the alternative against Spike ζ's NL-vocabulary-bridging behavior + the no-capability-match floor problem. Driver chain: drafting-time analytical engagement.
- **Rejected alternative — LLM-with-tools planner**: domain-model §AS-9 (driver, structural-bounding property) + Cycle 6 PLAY note 22 (driver, bundled-reasoning failure surface) + Spike λ-paid F-paid-4 (driver, post-dispatch protocol-format failure). Driver chain: invariant + prior-cycle PLAY + same-cycle spike.
- **Rejected alternative — no planner**: ADR-027 (driver, primary direction). Driver chain: same-cycle ADR.
