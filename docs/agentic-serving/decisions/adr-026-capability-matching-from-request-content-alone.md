# ADR-026: Capability Matching from Request Content Alone

**Status:** Proposed

**Date:** 2026-05-22

---

## Context

The agentic-serving project identity is the **transparent OpenAI-compatible endpoint**: any tool that can speak the OpenAI chat-completions protocol should be able to use llm-orc as a drop-in endpoint (per domain-model §Concepts "Transparent OpenAI-compatible endpoint"; practitioner-voiced at DISCOVER 2026-05-21). Population A clients (Aider, Cline, OpenCode, Cursor — tool-call-aware OpenAI-family clients with no alternative llm-orc surface) configure llm-orc by base-URL only; they do not adopt a llm-orc-specific SDK, header convention, or skill-framework manifest. Population A's trust contract is at the architectural layer — that the endpoint behaves as a transparent proxy honoring the configured model ID, with the endpoint exercising as little discretion as possible (per OQ #18 validation: Cline #10551 + OpenCode #20859 are explicit Population A degradation signals against configuration-dishonest endpoints).

Within this project identity, the **routing decision** (which capability ensemble handles a given chat-completions request, or whether the request falls through to direct completion) must work without any client-side opt-in mechanism. The skill-framework that the Skill Orchestration User authors (per ADR-019, ADR-021) runs entirely on the client side; the orchestrator sees only the resulting chat-completion request — `messages[]`, `model`, optional `tools[]`, optional (per Tension 19 disposition) `tool_choice`. Capability inference must operate on that surface alone.

Domain-model OQ #17 names this as a constitutional candidate: *"Capability matching works from request content alone with no client-side opt-in."* MODEL deliberately did not codify the candidate as an invariant — the codification decision is DECIDE work (per Amendment Log entry #12). The candidate is distinct from AS-9 in scope:

- **AS-9** names the **role-shape** property — structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks. AS-9 constrains *who* makes the routing decision (a structurally-bounded role rather than the orchestrator-LLM-as-decider) and *how* the decision is shaped (single-decision-shaped task).
- **OQ #17** names the **request-shape** commitment — the routing-decision input is the chat-completions request body alone; there is no header, no opt-in flag, no skill-framework manifest the client declares. OQ #17 constrains *what surface area* the routing decision operates on.

The two are independent. A framework-driven dispatch pipeline (per ADR-027) satisfies both: the routing-planner ensemble (AS-9-bounded role) reads the chat-completions request content (OQ #17-bounded surface) and produces a JSON dispatch plan. A Tier 1 hybrid (per ADR-030 `tool_choice` interception, if adopted) also satisfies both within its narrower scope.

DECIDE's decision is whether OQ #17 codifies as a project-level invariant, narrows to a Population-A-specific scope, or defers further.

---

## Decision

**Codify OQ #17 as invariant AS-10 for the agentic-serving surface**, stated as:

> **AS-10 (Cycle 7, added 2026-05-22). Capability matching on the agentic-serving chat-completions surface works from request content alone.** The routing decision — which capability ensemble handles a chat-completions request, or whether the request falls through to direct completion — is computed from the request body alone (`messages[]`, `model`, optional `tools[]`, optional `tool_choice` if implemented per ADR-030). No client-side opt-in mechanism is required, accepted, or used: no llm-orc-specific HTTP header, no skill-framework manifest field, no metadata convention the client must adopt. The agentic-serving endpoint presents as a transparent OpenAI-compatible endpoint at the configured base URL.

**Scope.** AS-10 governs the agentic-serving chat-completions surface (`/v1/chat/completions`, `/v1/models`, and any sibling endpoints that compose against the same routing decision, e.g., a `/v1/ensembles` capability-list discovery endpoint per ADR-032). AS-10 does *not* govern the `llm-orc invoke` CLI surface, the direct ensemble HTTP API (when surfaced for Population B), or future non-chat-completions surfaces — those surfaces accept explicit capability identifiers as their normal mode of operation. The invariant is scoped to the surface where the transparent-endpoint promise applies.

**Operational consequences.**

- The routing-planner ensemble (per ADR-028) takes the chat-completions request content + the framework's capability list as input. Its decision must not depend on any out-of-band signal from the client (no header inspection beyond what OpenAI semantics natively allow).
- The capability-list discovery endpoint (per ADR-032) advertises ensembles via OpenAI-protocol-compatible mechanisms (`/v1/models` extension or a sibling endpoint) so clients can discover capabilities through the same surface they already use; the discovery is available, not required.
- `tool_choice` (if implemented per ADR-030) is permitted as a client signal because it is an OpenAI-protocol-native field — sending `tool_choice` is not opting into a llm-orc-specific mechanism. Clients that send `tool_choice` are using the OpenAI contract; the framework's interpretation of the field is the framework's design choice (per ADR-030), not a client-side opt-in.
- Client populations that need ensemble routing without the transparent-endpoint promise (Population B — developer/script clients) have alternative surfaces (`llm-orc invoke`; direct ensemble HTTP API) where explicit capability naming is the normal mode. ADR-032 specifies the structured advisory mechanism that redirects Population B from chat-completions toward those surfaces.

---

## Rejected alternatives

### Narrow to Population-A-only scope (constitutional rule applies only when Population A is detected)

The invariant would apply conditionally — for Population A clients, capability matching works from request content alone; for other clients, an opt-in mechanism is permitted.

**Rejected because** *(analytically derived; the argument follows from the constitutional rule's own surface rather than from prior empirical evidence)*: Population A is not detectable from request content alone. The chat-completions endpoint receives a request body that conforms to the OpenAI protocol; no field reliably identifies the requesting client family. User-Agent headers are unreliable, easily spoofed, and absent from many tool configurations. Conditioning the invariant on Population A detection creates a routing decision that depends on the Population A detection's reliability — which is itself a request-content-alone problem in disguise. The narrower scope re-introduces the constitutional question one layer down.

The narrower scope also leaks the transparent-endpoint promise. A Population A client cannot verify that the endpoint will treat its requests transparently — the endpoint's behavior depends on its (Population A's) detection result, which the client cannot inspect. The transparent-endpoint promise requires that the endpoint behavior is uniform across client populations on the chat-completions surface; the variation surface is the alternative-surface affordance (`llm-orc invoke` exists for Population B), not the chat-completions surface itself.

### Defer further (carry forward to next cycle as candidate)

OQ #17 remains an open question; DECIDE codifies neither the invariant nor its rejection.

**Rejected because:** Cycle 7's downstream ADRs (ADR-027 framework-driven dispatch pipeline; ADR-028 routing-planner ensemble; ADR-030 `tool_choice` disposition; ADR-032 fallback shape) all assume capability matching from request content alone. The routing-planner ensemble's input contract (chat-completions request content + capability list) presupposes the constitutional rule; the `tool_choice` disposition is meaningful only within an OpenAI-protocol-native surface; the fallback shape's "honest response labeling" mechanism presupposes that no client-side opt-in distinguishes honest from dishonest dispositions.

Deferring AS-10 while drafting ADRs that depend on it would create a hidden constitutional dependency — the ADRs would carry an implicit assumption AS-10 names. The cycle's discipline of making invariants explicit at the constitutional layer (per project-level Invariant 14: "constitutional rules are explicit, not implied by downstream artifacts") argues against deferral when the dependent ADRs are being drafted in the same phase.

### Codify as a project-level invariant rather than agentic-serving-scoped

OQ #17 would become a project-level invariant applying to all llm-orc surfaces, not just the chat-completions surface.

**Rejected because:** the `llm-orc invoke` CLI surface and the direct ensemble HTTP API (where surfaced) accept explicit capability identifiers as their normal mode of operation — that is the point of those surfaces. A project-level invariant that "capability matching works from request content alone" would either contradict those surfaces' design (forcing a rewrite of `llm-orc invoke` to infer capability from prose rather than accept it as an argument) or require an exception clause that defeats the invariant's clarity.

The transparent-endpoint promise is an *agentic-serving* identity claim, not a *llm-orc project* identity claim. Other llm-orc surfaces have different contracts. AS-10 scoped to the agentic-serving chat-completions surface preserves the project-level architecture while making the agentic-serving identity explicit and verifiable.

---

## Consequences

### Positive

- **The transparent-endpoint promise becomes a verifiable constitutional commitment.** Population A clients can rely on the chat-completions surface to behave uniformly without inspecting their User-Agent headers or sending out-of-band signals. The promise that "any OpenAI-compatible tool works against llm-orc agentic-serving" is encoded as an invariant downstream artifacts must satisfy.
- **The routing-planner ensemble's input contract is constitutionally bounded.** ADR-028's specification of the routing-planner's input (chat-completions request content + capability list) is a constitutional consequence, not a design choice that could drift. Future changes to the routing-planner that add header inspection or other out-of-band signals are surfaced as constitutional amendments (Amendment Log entries on AS-10) rather than silent design shifts.
- **The Population A vs. Population B partition is preserved at the surface boundary.** Population B accommodation routes through alternative surfaces (`llm-orc invoke`, structured advisory per ADR-032), not through chat-completions-specific opt-in mechanisms. The two populations remain architecturally distinct without the chat-completions surface having to detect them.
- **Capability-list discovery becomes a first-order requirement.** Per ADR-032, the framework advertises available capabilities via OpenAI-protocol-compatible mechanisms — clients can discover capabilities through the same surface they already use. AS-10 makes this requirement constitutional rather than a documentation concern.

### Negative

- **Skill-framework-specific routing optimizations cannot land on the chat-completions surface.** A skill framework that wants the orchestrator to prefer specific capability ensembles in specific phases (e.g., "in the RDD argument-audit phase, prefer the `argument-mapper` ensemble") cannot signal that preference through a custom header or manifest field on the chat-completions surface. Skill frameworks needing fine-grained routing control compose explicitly (per ADR-021's per-capability dispatch contract — `invoke_ensemble("argument-mapper", ...)`) rather than relying on NL routing.
- **Routing-planner ensemble reliability is the load-bearing mechanism.** With no client-side opt-in to short-circuit the routing decision, the routing-planner ensemble's reliability profile (Spike ζ-validated 100% JSON conformance + 90% strict capability-match at qwen3:8b across the 20-prompt battery) is the empirical floor the constitutional rule rests on. Routing-planner degradation under production traffic diversity (per OQ #25) is a constitutional concern, not just an operational one.
- **The capability-list discovery endpoint becomes a deployment-time concern.** Operators deploying llm-orc agentic-serving must populate and maintain the capability list visible to clients. AS-10 does not specify the discovery mechanism's authoring model; ADR-032 addresses the mechanism, but operator-facing complexity grows with library size.

### Neutral

- **AS-10's scope-of-claim partition** (chat-completions surface only; not `llm-orc invoke` or direct ensemble HTTP API) preserves the architectural surface where explicit capability identifiers are the normal mode. The invariant's narrow scope is a feature of its precision, not a weakness.
- **`tool_choice` (if implemented per ADR-030)** is permitted as a client signal under AS-10 because it is an OpenAI-protocol-native field — sending it is not opting into a llm-orc-specific mechanism. AS-10's scope is "no client-side opt-in to a *llm-orc-specific* mechanism," not "no client-supplied routing signals at all." OpenAI-protocol-native signals are within the transparent-endpoint promise.
- **The invariant is amendable.** If future evidence (e.g., production deployment surfaces a class of routing failures that no request-content-alone mechanism can resolve) warrants relaxing the constitutional rule, the Amendment Log mechanism is the path. The codification is not an irreversible commitment; it is the cycle's most-informed reading of the project identity given the artifacts produced.

## Provenance check

- **Project identity framing as "transparent OpenAI-compatible endpoint"**: practitioner DISCOVER 2026-05-21 verbatim ("The point is to make llm-orc agentic-serving compatible with any tool that would want to use an OpenAI-compatible chat completions endpoint. The user trusts that llm-orc would use / create ensembles effectively. Full stop."). Driver chain: practitioner-voice settled at DISCOVER; recorded in domain-model §Concepts "Transparent OpenAI-compatible endpoint"; recorded in product-discovery.md Cycle 7 update.
- **AS-9 vs. OQ #17 distinction (role-shape vs. request-shape)**: domain-model §Open Questions #17 (driver) + domain-model AS-9 §Scope (driver). Driver chain: MODEL-phase deliberation 2026-05-22; preserved at MODEL gate as positive signal per snapshot evaluation.
- **Population A trust contract as "transparent proxy at configured model ID"**: OQ #18 Population A voice validation (driver). Driver chain: same-cycle research note `cycle-7-oq-18-cost-distribution-validation.md` Q3 finding — Population A's degradation signal is configuration dishonesty (Cline #10551; OpenCode #20859).
- **Scope-of-claim partition (chat-completions surface only)**: drafting-time synthesis aligned with the existing surface-boundary architecture (`llm-orc invoke` CLI per ADR-001; ensemble surfaces per ADR-021). Driver chain: prior-ADR-derived; the scope precision is drafting-time judgment bridging the constitutional candidate to the existing surface architecture.
- **Capability-list discovery as first-order constitutional consequence**: drafting-time synthesis. Essay-Outline §C6 E6.3.3 names capability-list discovery as first-order requirement (not documentation work); AS-10's codification makes this constitutional rather than design-choice. The structural-consequence framing is drafting-time bridging.
- **Rejection of Population-A-only narrowing (Population A is not detectable from request content alone)**: drafting-time synthesis examining the alternative against the constitutional rule's own surface. The argument that detection is itself a request-content-alone problem in disguise is drafting-time analytical engagement; not directly cited from prior artifacts.
- **Rejection of project-level scope (other llm-orc surfaces accept explicit capability identifiers)**: ADR-001 (driver, ReAct loop execution model with `invoke_ensemble` taking ensemble name as argument) + ADR-021 (driver, per-capability dispatch contract with explicit ensemble naming as preferred). Driver chain: prior-ADR-derived.
