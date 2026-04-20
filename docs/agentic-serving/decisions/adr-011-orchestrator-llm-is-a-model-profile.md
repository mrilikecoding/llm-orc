# ADR-011: Orchestrator LLM Is a Model Profile

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

Essay §Which LLM Powers the Orchestrator? observes that the orchestrator agent is itself parameterized by the same system it orchestrates. llm-orc already has model profiles — a named model configuration (provider + model + defaults) usable across ensembles, profiles, scripts. Cloud vs. local selection, provider abstraction, and parameter defaults are already solved by the profile system.

The essay discusses a tiered approach — local model for triage, cloud model when the task exceeds local capability — mirroring the conductor skill's routing logic. Essay §Cost implications note that a well-populated knowledge graph could enable a smaller, cheaper model (consumer hardware, 7B-class) to make routing decisions that would otherwise require a frontier model. The economic value proposition of agentic serving depends on the orchestrator's LLM being readily swappable.

An alternative framing — baking a tiered fallback mechanism into the orchestrator (e.g., hard-coded "try local first, then escalate to cloud") — is available but would create a special case that other ensembles do not have.

---

## Decision

The orchestrator agent's LLM is configured via a standard **Model Profile**. The profile system handles provider selection, model selection, credentials, and parameter defaults. No hard-coded tiered fallback exists in the orchestrator.

If tiered behavior is desired (local triage with cloud escalation), it is expressed as a **composed ensemble** invokable by the orchestrator — not as a mechanism special to the orchestrator. In other words: the orchestrator calls `invoke_ensemble("triage-route")`, and that ensemble implements the tier decision internally. The orchestrator itself runs on one model profile per session.

Switching the orchestrator LLM (e.g., experimenting with a smaller model once the knowledge graph is populated) is a configuration change, not a code change.

---

## Consequences

**Positive:**
- Uniformity — the orchestrator uses the same mechanism as every other llm-orc component. No special case
- Trivial to swap orchestrator LLMs for experimentation, supporting OQ #1 (knowledge-compensated model selection) as a testable hypothesis
- Provider abstraction (cloud, local via Ollama/vLLM/SGLang, custom) is inherited from the existing profile system
- The "retrieval-shifts-the-capability-frontier" economic argument is testable — run identical tasks across model profiles, with and without Plexus context

**Negative:**
- Tiered behavior, if desired, requires composing a triage ensemble — slightly more work than a built-in escalation
- A single profile per session means the orchestrator itself cannot dynamically switch LLMs mid-session. This is intentional: LLM swap at the orchestrator level is a session-boundary event, not a runtime decision. (Tiered *routing* of individual tasks — the essay's local-triage-then-escalate pattern — is expressible as the triage-route ensemble pattern invoked by the orchestrator, but only once such an ensemble has been composed and promoted to the library. For fresh deployments without a triage-route ensemble, cross-tier escalation is not available; what is session-scoped is only the orchestrator's own LLM, not the library's eventual capacity to route across tiers)
- Operators who expect a built-in fallback must discover that it is an ensemble composition pattern, not a toggle

**Neutral:**
- The choice of default orchestrator profile is a deployment decision, not an architecture decision
- The profile system's existing contract (Invariants 1-3: agent type mutual exclusivity, inline completeness, profile XOR inline) applies to the orchestrator's configuration

---

## Provenance Check

The explicit rejection of a built-in tiered fallback is drafting-time synthesis. The essay surveys tiered approaches favorably and does not commit to either "build it in" or "express it as an ensemble." This ADR picks the latter to preserve uniformity and avoid a special case. Surfaced here because the rejection framing originates in drafting rather than in the essay's recommendation chain.
