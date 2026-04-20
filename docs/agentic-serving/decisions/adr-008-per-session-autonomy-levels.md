# ADR-008: Per-Session Autonomy Levels

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

Essay §Autonomy Boundary surveyed three options for what the orchestrator does without operator approval: fully autonomous within guardrails; surface every decision for confirmation; a configurable level per session. The essay described "a reasonable starting position" but did not commit.

Product discovery (Ensemble Author / Operator Jobs) established that the operator wants visibility *and* control of cost/resource boundaries, but also wants the system to get better over time without hand-tuning everything. These pull in opposite directions: high autonomy amortizes operator effort but reduces per-decision control; high supervision preserves control but defeats the learning-through-use value proposition.

The domain concept *Autonomy Level* names this as a per-session configuration.

---

## Decision

Each Session carries an **Autonomy Level** that governs what the orchestrator can do without operator approval. The Autonomy Level is set at session start (via serving layer configuration or per-request override, within operator-set bounds).

A baseline level is defined, with both tighter and looser levels configurable:

**Baseline (default):**
- Invoke existing ensembles — unrestricted (within Budget per ADR-005)
- Compose new ensembles from library primitives — allowed, subject to validation (ADR-006) and calibration (ADR-007)
- Author new scripts — forbidden (AS-6)
- Author new model profiles — forbidden (AS-6)
- Promote a composed ensemble from local to global or library tier — requires explicit operator approval

**Tighter levels** add approval gates (e.g., approve-before-invoke for uncalibrated ensembles; approve-before-compose for new compositions).

**Looser levels** remove gates (e.g., auto-promote after a calibration quality threshold is met; auto-retry on failure).

The baseline is chosen such that the stateless system works usefully from session one without operator attention per action, but irreversible changes to the library (promotion, new primitive creation) always require operator approval. Autonomy Level does not override AS-6: no configuration can permit script or profile authorship by the orchestrator.

---

## Consequences

**Positive:**
- Operators who want hands-off agentic serving get it at the default level
- Operators who want supervised operation can tighten without code changes
- Irreversible library changes (promotion) are gated by default, supporting the cost/quality-control job
- AS-6 is enforced regardless of level — the operator cannot accidentally (or by mistake) open a script-authorship path

**Negative:**
- Introduces a configuration surface that operators must understand — the Autonomy Level becomes part of the operator's mental model
- Defaults encode a stance that some operators may find too restrictive and others too permissive. The baseline is calibrated for the ensemble-author-as-tool-user pattern from product discovery (operator and tool user are often the same person). A tool user who is *not* also an operator — the "endpoint is a model" mental model — may find silent composition of new ensembles surprising. For deployments targeting pure tool-user sessions, a tighter default that restricts or surfaces compositions is appropriate; the Autonomy Level configuration surface supports this, but operators must set it deliberately
- Looser levels depend on quality signals being reliable; combined with AS-5 and OQ #5 (organic stabilization mechanism), auto-promotion may not be viable until stabilization is well-understood

**Neutral:**
- Autonomy Level is session-scoped. Changing it mid-session would mix authority contexts; not supported

---

## Provenance Check

The essay §Autonomy Boundary presented three options but did not commit. This ADR commits to the third option (configurable level) and composes a specific baseline definition that is not in the essay. The baseline — invoke freely, compose with calibration, never author primitives, never promote without approval — is synthesized from the essay's "reasonable starting position," the operator jobs in product discovery, and AS-6/AS-5. Surfaced here because the specific baseline policy is drafting-time synthesis, not directly attributable to a single driver.
