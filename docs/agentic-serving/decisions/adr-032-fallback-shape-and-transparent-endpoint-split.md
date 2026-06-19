# ADR-032: Fallback Shape and Transparent-Endpoint Promise Split

> **Largely dormant after ADR-043 on 2026-06-18.** This policy split the transparent-endpoint promise into (1) configuration honesty (the response declares its served-by path: `action: dispatch` / `direct` / intercept) and (2) cost-distribution accountability (capability-matched requests result in ensemble dispatch). Both were built on the Dispatch Pipeline (the routing-planner's action labeling; strict-dispatch-when-capability-matched), which ADR-043 retired. **Sub-promise (2)'s intent is preserved** by the unified loop: capability-matched requests still reach an ensemble via the seat-filler's uniform delegation — ADR-043 §F-ι.1 Resolution B keeps delegation available for every request, toolless included — so the cost-distribution commitment carries forward; only its *mechanism* shifts from planner strict-dispatch to seat-filler delegation. **Sub-promise (1)'s served-by labeling is dormant**: the pipeline's `action`-path declaration no longer exists; whether and how the loop labels a delegated-vs-text turn is undecided and would be a new design if the transparency promise is wanted on the unified surface. Body preserved as architecture-of-record.

**Status:** Proposed; largely dormant after ADR-043 (2026-06-18) — see header

**Date:** 2026-05-22

---

## Context

The framework-driven dispatch pipeline (per ADR-027) routes every chat-completions request through the routing-planner ensemble (per ADR-028); the planner emits either `action: "dispatch"` (a capability match exists; the framework dispatches the named ensemble) or `action: "direct"` (no capability match; the framework falls through to the response-synthesizer's direct-completion path per ADR-029). The fallback shape — what happens on `action: "direct"` — is the architecturally consequential surface for requests that don't match a capability ensemble.

Cycle 7 RESEARCH established that the empirical fallback (the orchestrator-LLM choosing direct completion under NL framing) is in tension with the project's value proposition (Essay-Outline 006 §C6 W6.2; cost-distribution lens framing per practitioner DISCOVER 2026-05-21 verbatim *"The user trusts that llm-orc would use / create ensembles effectively. Full stop."*). The Cycle 6 PLAY Skill Orchestration User stakeholder super-objective was empirically unmet — capability-matched requests under NL framing reliably fell through to direct completion rather than dispatching the matching ensemble.

Tranche 1 research note `cycle-7-oq-18-cost-distribution-validation.md` validated the cost-distribution lens framing against Population A voice (Aider, Cline, OpenCode, Cursor). The finding: **partial corroboration with reframe**. The transparent-endpoint promise splits into two distinct sub-promises that downstream artifacts must keep separate:

1. **Configuration honesty** — the response truthfully reports what served it (direct completion vs. ensemble dispatch vs. multi-dispatch pipeline; configured model honored or substituted). Population A voice corroborates directly: Cline #10551 (DeepSeek V4 Pro 1M context capped at 128K by silent fallback; user filed as bug); OpenCode #20859 (subagent model selection silently overridden by GitHub Copilot provider; user filed as bug). Population A's degradation signal is configuration *dishonesty*.

2. **Cost-distribution accountability** — the project-developer expectation that ensembles do dispatch on capability-matched requests, so the framework distributes load across cheaper tier models rather than burning frontier-tier tokens for capability-matchable work. Population A voice is **silent** on this sub-promise (Population A clients locate cost-distribution decisions inside the client and treat the endpoint as a transparent single-model proxy at the configured model ID). The framing remains practitioner-voiced + project-developer-lens grounded; this is honest residual uncertainty.

Essay-Outline 006 §C6 (Amendment A2 + A2.1) names the split as load-bearing for Cycle 7 ADR drafting: ADRs must keep the two sub-promises in distinct decision surfaces because they rest on different evidence bases and operationalize via different mechanisms. Bundling them in a single ADR risks conflating evidentiary confidence levels.

This ADR specifies the fallback shape — the response on `action: "direct"` from the routing-planner — and the four mechanisms delivering the transparent-endpoint promise's two sub-promises: honest response labeling, strict-dispatch-when-capability-matched, capability-list discovery, and structured advisory for Population B.

---

## Decision

**The agentic-serving chat-completions surface treats the transparent-endpoint promise as two distinct architectural commitments, delivered by four mechanisms.** Each commitment has its own mechanism + evidence basis + operational signal.

### Sub-promise (1): Configuration honesty — delivered by honest response labeling

When the framework processes a chat-completions request, the response declares its served-by path. Specifically:

- When `action: "dispatch"` and dispatch succeeded — the response declares the dispatched ensemble's name and the operative path (`served_by: "ensemble:<name>"` or equivalent typed signal in response headers and/or response metadata).
- When `action: "dispatch"` and dispatch failed irrecoverably (e.g., infrastructure error, schema-non-conformance after Calibration Gate retries) — the response declares the failure and the fallback path taken (`served_by: "direct_fallback"; dispatch_failed: "<failure-type>"`).
- When `action: "direct"` (no capability match) — the response declares the direct-completion path (`served_by: "direct"`) and the response-synthesizer's Rule 5 framing (per ADR-029) appears in `message.content` ("this answer was generated directly without dispatching a specialist ensemble"). The Rule 5 BUILD-default + falsification trigger pattern (per ADR-029) governs whether the framing remains in content or migrates to headers/metadata if production evidence warrants.
- When the framework intercepts `tool_choice` (per ADR-030 disposition (i) follow-on) and routes via the deterministic path — the response declares the explicit-naming path (`served_by: "tool_choice:<ensemble>"`) and bypasses the routing-planner.
- When the framework processes the bridge advisory for unsupported `tool_choice` (per ADR-030 Cycle 7 BUILD bridge) — the response declares the bridge state (`tool_choice_handling: "deferred"; <bridge-advisory-content>`). **Per ADR-030 §Bridge advisory specification, the content-layer (Rule 5-adjacent) acknowledgment applies only to `action: "direct"` responses; dispatch responses carry the bridge signal at the headers and body metadata layers only, with no content-layer noise.**

**Implementation mechanisms:**

- **Response headers** carry typed served-by signals. The header name and shape are BUILD-phase design; candidates include `X-LLM-Orc-Served-By`, `Served-By`, or a standardized field within an existing OpenAI-compatible response-metadata mechanism. The header is the universally-visible-to-clients signal.
- **Response body metadata** carries the same typed signal in a structured field (e.g., a `metadata.served_by` field in the chat-completion response shape, if extending the OpenAI response semantics is appropriate). The body-metadata mechanism is the fallback for clients that don't surface response headers usefully.
- **Response-synthesizer Rule 5 framing** (per ADR-029) carries the direct-completion declaration in `message.content` for direct-mode responses. This is the content-layer signal that pairs with the header/metadata-layer signal.

Multiple layers carry the same signal because Population A client behavior in surfacing response metadata varies (Aider surfaces some; OpenCode's per-provider tool framework may or may not; Cursor's surfacing is opaque). The honest-response-labeling commitment is layered to maximize the likelihood the signal reaches the user across client populations.

**Evidence basis:** Population A voice via OQ #18 directly corroborates configuration honesty as Population A's degradation signal (Cline #10551 + OpenCode #20859). The configuration-honesty sub-promise has empirical Population A validation.

### Sub-promise (2): Cost-distribution accountability — delivered by strict-dispatch-when-capability-matched

The framework's commitment is that **capability-matched requests result in ensemble dispatch**. The routing-planner ensemble (per ADR-028) operationalizes this: when a capability match exists in the request, the planner emits `action: "dispatch"`; the framework dispatches the named ensemble. The fallback to `action: "direct"` is reserved for requests where no capability match exists — not as a silent default for capability-matched requests the routing-planner missed.

**Operational implication for the routing-planner:** the planner's reliability profile (Spike ζ — 90% strict capability-match + 100% defensible-judgment-match across the 20-prompt battery at qwen3:8b) is the empirical floor for sub-promise (2). **Clarification of the 90% figure:** Spike ζ's 20-prompt battery produced 20/20 `action: "dispatch"` outcomes on capability-matched prompts (and `action: "direct"` outcomes on the no-capability-match prompts in the battery, which is the correct routing for that prompt class). The 10% non-strict-match cases were instances where the planner chose a *defensible alternative ensemble* rather than the *single ideal ensemble* — not cases where the planner routed a capability-matched request to `action: "direct"`. Sub-promise (2)'s failure mode (`action: "direct"` on a capability-matched request) was not empirically observed in Spike ζ's battery, though that absence is a 20-prompt empirical floor rather than a structural guarantee. Production traffic surfacing systematic capability-match misses by the planner (either as `action: "direct"` on capability-matched requests or as inappropriate alternative-ensemble routing) triggers the existing Tier-Router Audit drift criteria (per ADR-018) and feeds the operator-observable degradation signaling mechanism below.

**Evidence basis:** Population A voice via OQ #18 is **silent** on this sub-promise. The framing is project-developer-lens grounded with honest residual uncertainty. The cycle accepts the residual uncertainty visibly rather than collapsing it into Population A endorsement. If production evidence surfaces a class of Population A users for whom cost-distribution-accountability is salient (vs. project-developer-only), the evidence basis strengthens; until then, the sub-promise remains practitioner-voiced + project-developer-grounded.

### Mechanism: Capability-list discovery

The framework advertises available capabilities via an OpenAI-protocol-compatible mechanism (per AS-10 — ADR-026). The mechanism is **first-order requirement, not documentation work** (per Essay-Outline 006 §C6 E6.3.3): clients can discover the framework's capabilities through the same surface they already use.

Three candidate implementation surfaces (BUILD-phase design picks one or more):

- **`/v1/models` extension** — the existing `/v1/models` endpoint includes ensemble identifiers as model entries (with a capability-marker field distinguishing ensembles from underlying models). Clients listing models see the capability surface.
- **Sibling endpoint** (e.g., `/v1/ensembles`) — a new endpoint dedicated to capability listing, with structured ensemble metadata (name, description, topaz_skill, calibration status). Clients invoking the endpoint receive the full capability list.
- **Response metadata** — capability list is included in chat-completion response metadata for clients that opt in via a request flag. Lower-bandwidth than the dedicated endpoint; relevant for clients that don't pre-discover.

The discovery mechanism is **available, not required** — Population A clients are free to not discover; the routing-planner ensemble's NL inference operates regardless. Discovery is the surface for clients that *want* to construct explicit `tool_choice` shapes (per ADR-030 follow-on) or that maintain library-topology knowledge per ADR-021's preferred dispatch shape (explicit ensemble naming).

### Mechanism: Structured advisory for Population B

Population B (developer/script clients with alternative-surface access — per ADR-026's Population B concept) on the chat-completions surface receive a structured advisory in the response on `action: "direct"`, pointing them toward `llm-orc invoke` or the direct ensemble HTTP API for capability-matched workflows.

**Advisory shape:** when `served_by: "direct"` and the request content shape matches Population-B-style patterns (script-shaped requests, programmatic content, explicit naming attempts the routing-planner couldn't bind), a typed advisory field in the response metadata recommends the alternative surface. The advisory is informational; it does not change the response content (the response-synthesizer's normal output applies).

**Population B detection** — Population B is empirically distinct from Population A by client family (e.g., requests from `llm-orc-py` library scripts; bare `requests` library payloads; absence of tool-call-aware client headers) but the detection is not load-bearing. The structured advisory is **safe to send to Population A** — it appears as response metadata; Population A clients that don't surface or use the metadata are unaffected. The advisory is *targeted* at Population B but *visible* universally.

**Evidence basis:** Population B has alternative llm-orc surfaces (per ADR-026 scope + ADR-001 ReAct loop surfaces) where explicit capability naming is the normal mode. Advisory is the bridge between the transparent-endpoint surface (Population A primary) and the explicit-capability surface (Population B primary).

### Mechanism: Operator-observable degradation signaling

The framework emits typed observability events (per ADR-023 observability event routing) when chat-completions requests fall through to `action: "direct"`. The event shape (BUILD-phase design):

- `direct_completion_fallback` event with fields: request shape category (NL prose, script-shaped, mixed, ambiguous); detected client population (Population A vs. Population B based on available signals); routing-planner rationale (the planner's emitted rationale for why no capability match).
- `direct_completion_rate` rolling metric: percentage of chat-completions requests resulting in `action: "direct"` over a sliding window. High rates indicate either (a) the deployment's capability library is too narrow for client request shapes, or (b) the routing-planner is systematically missing capability matches.

The signals feed operator dashboards / logs / Tier-Router Audit drift criteria (per ADR-018). Operators with high `direct_completion_rate` and a Population A-heavy deployment have actionable diagnostic — either the capability library needs expansion or the routing-planner needs tuning. Operators with high rates and a Population B-heavy deployment may be observing normal behavior (Population B requests legitimately don't have capability matches and are routed to `direct` honestly).

**The signaling intersects ADR-023's observability event surface.** ADR-032 names the new event types; ADR-023's routing mechanism delivers them to operator destinations. The two ADRs are orthogonal (event content vs. event routing).

#### Refutation threshold for the cost-distribution accountability sub-promise

The `direct_completion_rate` metric needs a refutation threshold that distinguishes expected operational baseline (some chat-completions requests legitimately have no capability match) from real failure (the C1 surface persisting under ADR-027's pipeline). The cycle does not have production data to pin an absolute threshold; what it commits to is a **deployment-relative structural relationship**:

- The expected baseline `direct_completion_rate` for a deployment equals approximately `(100% - operator-estimated-capability-coverage)` — i.e., a deployment whose capability library is designed to cover 80% of expected chat-completions traffic should observe a baseline `direct_completion_rate` of ~20%.
- **Refutation threshold (rough; production evidence refines):** sustained `direct_completion_rate` more than **~15 percentage points above the deployment's expected baseline** over a 24-hour rolling window is evidence that the C1 failure signal is firing under ADR-027's pipeline. Below this delta is expected operational variance (ambiguous request shapes; legitimately uncovered topics; client request shapes not anticipated by the capability library's design).
- **Investigation pathway when the threshold trips:** the diagnostic sequence below uses the tuning axes named in ADR-028 + ADR-031 but **reorders them for the refutation-threshold context**. ADR-028 + ADR-031 sequence the tuning axes for *latency reduction* (classifier pre-filter first, then caching, then smaller-faster planner model); the refutation threshold is a *reliability/coverage signal* — when it trips, routing-decision quality is the diagnostic frame, not latency. The reordered sequence: (1) **routing-planner model profile substitution** (operator override; replace qwen3:8b with a more capable cheap-tier model) — addresses planner reliability directly; (2) **classifier pre-filter** to short-circuit clearly-determinable cases — reduces planner load on the population the classifier covers; (3) **caching planner decisions** on repeated request shapes — reduces planner invocations on the population of repeated shapes; (4) **capability library expansion** (ensemble authoring work) — addresses the upstream root cause if the planner's decisions correctly reflect a library scoping mismatch with the deployment's traffic shape.
- **If the tuning playbook is insufficient** and the C1 signal persists at refutation-threshold levels, the cycle's structural commitment to ADR-027 is empirically falsified on the deployment's traffic shape. The natural reach is the embedding-similarity router (rejected in ADR-028 as primary on capability-coverage grounds; revisitable if production evidence shows the LLM-reasoning planner is the bottleneck). Pushing the failure surface to a frontier-tier LLM is **not** the natural reach per AS-9 — the orchestrator-LLM-as-decider was the failure surface across tier classes in Cycle 6/7 spikes; tier escalation of that role does not change its shape.

The deployment-relative framing acknowledges that operator deployments vary substantially in capability-library coverage and Population A vs. Population B traffic mix. The ~15 percentage points threshold is a starting heuristic the cycle's first deployments calibrate against; PLAY phase field notes are the empirical surface that refines the threshold for downstream cycles.

### Fallback response shape (when `action: "direct"`)

The fallback response is **not** a placeholder or refusal. The framework processes the request via the response-synthesizer's direct-completion path (per ADR-029):

- The synthesizer reads the ORIGINAL REQUEST (no PLAN.action: "dispatch"; no DISPATCH RESULTS).
- The synthesizer produces a useful response from its own training-data + reasoning (substantively answering the request — Spike ε' A1/A2/A3 + C1 validation).
- Rule 5 framing per ADR-029 declares the direct-completion mode.
- Tier escalation per ADR-031 may escalate the synthesizer's tier for substantive direct-completion responses.

The fallback is "honest direct completion under the response-synthesizer's structurally-bounded role," not "I cannot help with that" or "Please use `llm-orc invoke` instead." The framework serves the user's request; the configuration-honesty signal declares the served-by path; the cost-distribution-accountability signal is the operator-observable degradation signaling that fires for the operator to act on (capability library expansion; planner tuning), not for the user to detect.

### Out of scope for ADR-032

- **Specific capability-list-discovery endpoint shape choice.** ADR-032 names three candidate surfaces (`/v1/models` extension, sibling endpoint, response metadata). The choice is BUILD-phase design; multiple surfaces may coexist.
- **Population A vs. Population B detection mechanism.** ADR-032 names that detection is *not load-bearing* (the advisory is safe to send universally; the operator-observable signaling is informational). BUILD-phase design picks the detection heuristic.
- **Specific operator-dashboard layout for degradation signaling.** ADR-032 names the event types; presentation is BUILD-phase + operator-deployment-specific concerns.

---

## Rejected alternatives

### Single transparent-endpoint promise without sub-promise split

The transparent-endpoint promise is treated as a single architectural commitment: "the endpoint behaves predictably and honestly given its configured state." Configuration honesty and cost-distribution accountability are bundled.

**Rejected because:** OQ #18 Population A voice validation established that the two sub-promises rest on different evidence bases — configuration honesty has Population A direct corroboration (Cline #10551 + OpenCode #20859); cost-distribution accountability has Population A silence + project-developer-lens grounding. Bundling them in a single ADR conflates evidentiary confidence levels and obscures the susceptibility-audit lens that downstream auditors should apply per sub-promise.

The bundled framing also opens an evidentiary-confusion failure mode: a future cycle critiquing the cost-distribution-accountability claim could inadvertently weaken the configuration-honesty commitment (which is empirically grounded) by treating the bundle as a single warrant. Keeping the sub-promises distinct lets future audits engage each one's evidence basis on its own terms.

### Configuration-honesty as the only commitment; cost-distribution-accountability dropped

The cycle commits to configuration honesty (Population A directly corroborates); cost-distribution accountability is treated as an internal project value rather than an architectural commitment surfaced in the chat-completions surface design.

**Rejected because:** the cost-distribution-accountability sub-promise is load-bearing for the project-developer value proposition (per cost-distribution lens framing — Essay-Outline 006 §C6 + practitioner DISCOVER 2026-05-21 verbatim). Dropping it would mean the chat-completions surface is honestly labeled but not architecturally committed to ensemble dispatch on capability-matched requests — a transparent endpoint that admits it's not delivering the cost-distribution architecture is honest-but-not-aligned with the project's value proposition.

The honest-residual-uncertainty path (per OQ #18 recommendation) is the right reading: commit to both sub-promises with explicit per-sub-promise evidence-basis labeling; let future audits and production evidence engage each on its merits. Dropping cost-distribution-accountability would defeat the value proposition; bundling it with configuration-honesty would conflate evidence.

### `action: "direct"` returns 4xx error (refuse to handle non-capability-matched requests)

The framework's chat-completions surface refuses to handle requests with no capability match; clients receive HTTP 400 and are pointed to `llm-orc invoke`.

**Rejected because:** this breaks Population A's trust contract — the transparent-endpoint promise (per ADR-026) is that the endpoint behaves as a transparent OpenAI-compatible endpoint at the configured base URL. Refusing non-capability-matched requests means Population A clients can't use llm-orc as a drop-in chat-completions endpoint; they'd need to detect and re-route capability-mismatch cases themselves. The cost-distribution-accountability sub-promise is operationalized by routing capability-matched requests to ensembles, not by refusing capability-mismatched requests.

The fallback response shape (direct completion via response-synthesizer per ADR-029) honors Population A's trust contract while configuration honesty signals when dispatch did not fire.

### Defer capability-list discovery to a follow-on cycle

The chat-completions surface does not expose capability discovery in Cycle 7; clients that want to construct explicit `tool_choice` shapes or maintain library-topology knowledge use external documentation.

**Rejected because:** per Essay-Outline 006 §C6 E6.3.3 + DISCOVER 2026-05-21, capability-list discovery is **first-order requirement, not documentation work**. AS-10's transparent-endpoint promise requires that clients can discover capabilities through the OpenAI-protocol-native surface; documentation-only discovery defeats the surface-level promise. Cycle 7 BUILD ships at least one of the three candidate surfaces (`/v1/models` extension is the lowest-cost candidate); follow-on cycles may add the others.

### Bundle structured-advisory-for-Population-B with the response-synthesizer ensemble's behavior

The response-synthesizer ensemble's system prompt is extended to include Population B redirection logic; the advisory appears in `message.content` directly.

**Rejected because:** the advisory's content-layer placement creates the same friction Rule 5 framing's load-bearing-default surfaces (per ADR-029) — Population B clients may surface the advisory as user-facing noise. The response-metadata layer is the right home for client-routing advisory; the synthesizer's content surface is for the substantive response to the user's request.

Bundling the advisory with the synthesizer also couples the synthesizer's prompt-design to Population A vs. Population B detection, which ADR-032 names as not-load-bearing. Keeping the advisory at the metadata layer preserves the synthesizer's clean structural-bounding role.

---

## Consequences

### Positive

- **Configuration honesty is delivered structurally** via three layered mechanisms (response headers, response metadata, response-synthesizer Rule 5 framing in content). Population A's degradation signal (configuration dishonesty per Cline #10551 + OpenCode #20859) is structurally prevented; the framework cannot silently disguise direct completion as ensemble dispatch.
- **Cost-distribution accountability is operationalized** via strict-dispatch-when-capability-matched. The routing-planner ensemble's emission of `action: "dispatch"` on capability matches is the architectural mechanism; the Tier-Router Audit's drift criteria (per ADR-018) detect systematic misses; the operator-observable degradation signaling surfaces the rate for operator action.
- **The two sub-promises remain distinct.** Future audits (susceptibility-snapshot, argument-audit, conformance-scan) can engage each sub-promise on its evidence basis. Configuration honesty has Population A corroboration; cost-distribution accountability has project-developer-lens grounding + honest residual uncertainty. The bundling failure mode (conflating evidentiary confidence) is structurally prevented.
- **Capability-list discovery is first-order, OpenAI-protocol-native.** Clients (Population A's `tool_choice`-aware sub-segment; Population A's library-topology-aware sub-segment per ADR-021) can discover capabilities through the OpenAI surface they already use. AS-10's transparent-endpoint promise extends to discovery.
- **Population B accommodation does not require detection logic to be load-bearing.** The structured advisory is safe to send universally; Population A clients that don't surface it are unaffected; Population B clients that do surface it receive guidance to the alternative surface. Detection-mechanism complexity is deferred to BUILD without creating a Population A regression risk.
- **Operator-observable degradation signaling closes the cost-distribution-accountability loop.** Operators with high `direct_completion_rate` have actionable diagnostic surfaces. The signaling reuses ADR-023's observability event routing; no new architectural surface required.

### Negative

- **Configuration honesty is multi-layered, raising operational complexity.** Three layers (headers + metadata + Rule 5 content framing) deliver the same signal; downstream debugging and client-integration work hold multiple signal surfaces in mind. The redundancy is intentional (Population A clients vary in metadata surfacing) but the cost is real.
- **Cost-distribution-accountability evidence basis remains project-developer-lens grounded.** Population A voice is silent on this sub-promise; future production evidence may strengthen or weaken the grounding. The cycle commits to the sub-promise with the honest-residual-uncertainty label; if a follow-on cycle finds Population A evidence contradicting cost-distribution accountability, the sub-promise revises.
- **Capability-list discovery surface choice is BUILD-phase open.** Three candidate surfaces; the choice ripples through client-integration documentation, operator-deployment configuration, and the framework's API surface. BUILD design may pick multiple surfaces; the operational complexity is unbounded by ADR-032.
- **Population B advisory is universally visible.** While safe-to-send, Population A clients that surface response metadata may show the advisory to users for whom it's not relevant. The "use `llm-orc invoke` instead" message could confuse Population A users whose deployment specifically configured the chat-completions surface for capability dispatch. BUILD-phase advisory-content design picks careful wording.
- **The fallback response on `action: "direct"` is the response-synthesizer's direct-completion output.** The user gets a useful answer from the cheap-tier synthesizer; the operator pays the synthesizer cost for a request that didn't dispatch. The cost-distribution-accountability sub-promise is conditional on the deployment's capability library matching client request patterns; mismatched deployments pay synthesizer cost on every `action: "direct"` request.

### Neutral

- **The two sub-promises are amendable independently.** Future cycles may strengthen configuration honesty (new mechanisms, finer-grained signaling) without modifying cost-distribution accountability, or vice versa. ADR-032's split structure enables independent evolution.
- **The fallback shape preserves the OpenAI chat-completions API contract.** External clients see standard OpenAI-compatible responses on all paths; the served-by signal is in headers/metadata/content layers but does not break the protocol contract.
- **Capability-list discovery is independent of the routing-planner's behavior.** Discovery surfaces the framework's capability list; the routing-planner consumes the same list. Operators can update the capability list and both surfaces reflect the change consistently.
- **Operator-observable degradation signaling is observability, not control.** Operators see the rate but the framework doesn't automatically take action on high rates. Manual operator intervention is required for capability library expansion or planner tuning; this aligns with the cycle's autonomy-level commitments (per ADR-008).

## Provenance check

- **Two-sub-promise split (configuration honesty + cost-distribution accountability)**: Tranche 1 research note `cycle-7-oq-18-cost-distribution-validation.md` (driver, Population A voice validation finding) + Essay-Outline 006 §C6 Amendment A2 + A2.1 (driver, Cycle 7 DISCOVER + DECIDE sharpening). Driver chain: same-cycle research + same-cycle essay amendment.
- **Configuration honesty Population A evidence (Cline #10551, OpenCode #20859)**: OQ #18 research note Q3 finding (driver). Driver chain: same-cycle DECIDE-entry research; external sources cited.
- **Cost-distribution accountability project-developer-lens grounding**: practitioner DISCOVER 2026-05-21 verbatim ("The user trusts that llm-orc would use / create ensembles effectively. Full stop.") (driver) + Essay-Outline 006 §C6 W6.2 (driver). Driver chain: practitioner-voice + same-cycle essay.
- **Honest-residual-uncertainty disposition**: OQ #18 recommendation (driver, "Sharpen, do not replace"). Driver chain: same-cycle research.
- **Honest response labeling at multiple layers (headers, metadata, content)**: drafting-time synthesis bridging the configuration-honesty sub-promise to Population A client metadata-surfacing variance. Driver chain: same-cycle research (Population A client variance) + drafting-time analytical engagement.
- **Strict-dispatch-when-capability-matched delivers cost-distribution accountability**: Essay-Outline 006 §C6 E6.3.2 (driver). Driver chain: same-cycle essay.
- **Capability-list discovery as first-order requirement (not documentation work)**: Essay-Outline 006 §C6 E6.3.3 (driver) + ADR-026 AS-10 (driver, transparent-endpoint promise structural-consequence). Driver chain: same-cycle essay + same-cycle ADR.
- **Three candidate capability-list-discovery surfaces (`/v1/models` extension, sibling endpoint, response metadata)**: drafting-time synthesis enumerating the OpenAI-protocol-compatible surface options. Driver chain: drafting-time analytical engagement; the three candidates are conventional OpenAI-protocol-compatible patterns.
- **Structured advisory for Population B**: Essay-Outline 006 §C6 E6.3.4 (driver). Driver chain: same-cycle essay.
- **Advisory-is-safe-to-send-universally framing (not load-bearing detection)**: drafting-time synthesis on Population A metadata-surfacing-variance. Driver chain: drafting-time analytical engagement.
- **Operator-observable degradation signaling**: Essay-Outline 006 §C6 E6.3.5 (driver) + ADR-023 (driver, observability event routing). Driver chain: same-cycle essay + prior-ADR.
- **Fallback response shape (direct-completion via response-synthesizer per ADR-029)**: ADR-029 (driver) + Spike ε' A1/A2/A3 + C1 (driver, direct-completion path validation across 4 request shapes). Driver chain: same-cycle ADR + same-cycle spike.
- **Tier escalation reference for substantive direct-completion**: ADR-031 (driver). Driver chain: same-cycle ADR.
- **Rejected alternative — single transparent-endpoint promise without split**: OQ #18 recommendation explicitly (driver, "ADRs that conflate them will fail the susceptibility audit Population A would apply"; the A2.1 framing was softened per argument-audit P2-3 to "the basis is evidentiary clarity, not predictive failure" but the recommendation to keep ADRs separate is unchanged). Driver chain: same-cycle research + same-cycle argument-audit.
- **Rejected alternative — configuration-honesty as only commitment**: drafting-time synthesis weighing the dropped sub-promise against the cost-distribution lens framing's load-bearing role for the project value proposition. Driver chain: drafting-time engagement.
- **Rejected alternative — `action: "direct"` returns 4xx error**: ADR-026 AS-10 transparent-endpoint promise (driver, refuses-non-capability-matched-requests breaks transparent endpoint). Driver chain: same-cycle ADR.
- **Rejected alternative — defer capability-list discovery**: Essay-Outline 006 §C6 E6.3.3 (driver, first-order requirement framing). Driver chain: same-cycle essay.
- **Rejected alternative — bundle Population B advisory in synthesizer content**: ADR-029 (driver, synthesizer structural-bounding role) + drafting-time synthesis (advisory content-layer placement creates Population A friction). Driver chain: same-cycle ADR + drafting-time engagement.
