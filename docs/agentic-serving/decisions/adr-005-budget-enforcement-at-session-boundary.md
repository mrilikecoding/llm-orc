# ADR-005: Budget Enforcement at Session Boundary

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

AS-3 establishes that budget enforcement is a control plane concern: turn limits and token limits are enforced at the session level, checked at each iteration of the ReAct loop, regardless of what the orchestrator LLM decides. These are harness-level circuit breakers, not model-level parameters.

Essay §Context Management cites claw-code's precedent (`max_turns=8`, `max_budget_tokens=2000`). Those defaults are calibrated for short task-specific agent sessions. They are too tight for the use cases agentic serving targets — OpenCode and Roo Code sessions routinely run dozens of turns across an extended work session. An outer anchor for sizing: a session should accommodate work on the order of running a full RDD phase (research, decide, or build) through the agentic interface — many turns over extended wall-clock time.

The specific turn and token numbers are build-phase concerns. What the ADR commits to is the enforcement point, the observable behavior on exceeding, and the configurability.

---

## Decision

Each Session carries a Budget (turn limit + token limit) enforced at the session boundary:

1. **Check point.** Before each iteration of the orchestrator's ReAct loop, the control plane checks turn count and cumulative token spend against the Budget. Checks occur outside the orchestrator LLM's reach.
2. **Behavior on exceeding.** The session terminates gracefully. The orchestrator's last complete turn is returned to the client; no partial tool call spans the boundary. The response indicates budget exhaustion explicitly (exhaustion is not silent).
3. **Defaults.** Sized for sustained agentic coding comparable to running an RDD phase within a single session. Specific numbers (order of hundreds of turns, large token ceiling) are set at build and tunable without ADR change.
4. **Configurability.** Budgets are configurable at session start via serving layer configuration and may be overridden per-request within operator-set bounds.

Budget enforcement is model-agnostic. The orchestrator LLM does not see the Budget and cannot negotiate it. (It may observe cumulative spend indirectly through `query_knowledge`, but this is information, not control.)

---

## Consequences

**Positive:**
- Runaway sessions are bounded regardless of orchestrator behavior
- The operator has a hard control surface for cost, consistent with product discovery's cost-control job
- Enforcement at each iteration keeps the maximum overrun small (one turn, one tool call)

**Negative:**
- Defaults may be too tight or too loose for actual workloads until observed in build. Tuning is expected
- Graceful termination may leave complex reasoning chains incomplete — clients must tolerate budget-exhausted responses
- A budget sized for long sessions permits expensive accidents — the Budget is a ceiling, not a cost optimizer
- A single orchestrator turn may trigger multiple sub-invocations: an ensemble execution + mandatory result summarization (ADR-004) + calibration check if the ensemble is in calibration (ADR-007). Each of these is itself LLM work that counts against the session token budget. Turn-count sizing guidance assumes a single LLM call per turn; token-ceiling sizing must account for this multiplier or the effective turn budget is smaller than expected

**Neutral:**
- Token accounting across providers is a known complexity (tokenizer differences, streaming partials). The enforcement mechanism handles this; its specifics are an implementation concern

---

## Provenance Check

The default sizing framing ("sustained agentic coding comparable to running an RDD phase within a single session") does not trace directly to the essay or domain model. It originated in the DECIDE-phase conversation on 2026-04-17 when the user named the outer anchor as analogous to OpenCode sessions and capable of running an RDD phase end-to-end. This is user-provided design intent, not essay evidence. Surfaced here so the framing's origin is visible — the enforcement mechanism traces to AS-3; the sizing target does not.
