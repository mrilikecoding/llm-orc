# ADR-007: Calibration Gate for Composed Ensembles

> **Superseded by ADR-046 on 2026-07-01.** The cross-session calibration/trust mechanism (first-N invocations result-checked, verdicts accumulating to a promotable library-tier entry) governed the now-dissolved runtime self-composition surface (`compose_ensemble`, superseded with ADR-006). **AS-5 survives as an invariant:** its per-dispatch quality-gating re-homes to seat contracts (ADR-046 §2); its cross-session stabilization/promotion half is out of scope — there is no self-composition surface left to promote into. Body immutable.

**Status:** Superseded by ADR-046 on 2026-07-01 (was: Accepted)

**Date:** 2026-04-17

---

## Context

AS-5 establishes that stabilization is governed by quality signals, not frequency. Frequency without quality is noise. A pattern does not become a reliable library entry just because it was used many times.

The domain concept *Calibration* defines the evaluation period for a newly composed ensemble: during calibration (first N invocations), results are always checked. The ensemble must demonstrate quality before earning trust.

Essay §Guardrails identifies sandboxing — new ensembles execute in an evaluation mode during their first N invocations, results always checked — as a precondition for safe self-building. Without calibration, a composed ensemble that happens to execute often accumulates frequency signals that (under AS-5) are not trust-conferring, but without an explicit gate the orchestrator could treat frequency as a trust proxy anyway.

---

## Decision

Every ensemble produced by `compose_ensemble` enters a **calibration** state. During calibration:

1. The ensemble's first N invocations are always result-checked. The check produces a quality signal attached to that invocation (Routing Decision in the knowledge graph, if Plexus is active).
2. Calibration does not prevent invocation — a calibrated ensemble is usable; it is simply watched.
3. Quality signals accumulated during calibration determine whether the ensemble transitions to trusted status. Frequency alone does not (AS-5).
4. If Plexus is absent, calibration still runs within the session — quality signals accumulate in the session's execution artifacts and the ensemble may transition to trusted status within the session. Trust does not persist across sessions in stateless mode; a composed ensemble re-enters calibration at the start of the next session. Persisting calibration state across stateless sessions would require a separate persistence mechanism (a local-tier metadata store), which is not introduced by this ADR.
5. N is configurable; a default is set at build. The check mechanism (a check ensemble, orchestrator self-check, operator review, or composition thereof) is an implementation detail — what the ADR commits to is that *some* check runs on every calibration invocation.

A composed ensemble does not become eligible for promotion to global or library tier (per ADR-008) until it has cleared calibration.

---

## Consequences

**Positive:**
- Quality signals govern trust (AS-5) — a brittle composed ensemble that executes often cannot accumulate undeserved trust
- Makes the first N invocations observable — supports the operator's visibility job (DISCOVER #11)
- Integrates with Plexus when available; degrades to session-scoped calibration when absent (consistent with ADR-002)

**Negative:**
- First N invocations of every composed ensemble incur checking cost (latency, tokens)
- Choice of N is a tunable with real cost/risk tradeoffs; a wrong default may be too lax or too expensive
- The check mechanism's own quality determines the calibration's value. A weak checker fails to distinguish signal from noise

**Neutral:**
- Calibration status is ensemble-level state. When Plexus is active, it persists there; when Plexus is absent, it lives only in session memory and is rebuilt each session
- Open Question #5 (organic stabilization mechanism) remains open — what *counts* as a quality signal is not fixed by this ADR. Calibration is the gate; the signal definition is deferred to build iteration and OQ resolution
