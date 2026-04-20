# ADR-006: Full Composition Palette with Reference Graph Validation

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

The essay (§Tension) surveyed the interaction between Invariant 7 (static ensemble references) and dynamic composition. It proposed two safety options: restrict orchestrator-composed ensembles to profile-and-script compositions only (no ensemble-to-ensemble references), or add an explicit validation step that checks references against the existing ensemble reference graph before loading.

DISCOVER feed-forward signal #10 explicitly rejected the first option: the orchestrator needs the full composition palette, including ensemble-to-ensemble references. The tool user and ensemble author are often the same person, and restricting the palette for the agentic use case would leave value on the table.

AS-2 formalizes the second option as the invariant: composed ensembles must be validated before loading. AS-6 restricts composition to existing primitives — the orchestrator cannot generate new scripts or profiles, only arrangements.

---

## Decision

The orchestrator agent composes new ensembles from the **full** library palette: model profiles, scripts, and ensembles. The `compose_ensemble` tool (per ADR-003) is the sole entry point.

Composition produces a new ensemble configuration that:

1. References only primitives that already exist in the library (AS-6)
2. If it references other ensembles, passes validation against the existing ensemble reference graph (AS-2) — the proposed ensemble must not introduce a cycle (Invariant 5), must not exceed the depth limit (Invariant 8), and must resolve every ensemble reference to an existing static entry (Invariant 7)
3. Is stored to the local tier (not global, not library) until explicit promotion (per ADR-008)
4. Satisfies all project-level Invariants 1-14 at load time

Validation occurs at composition time — before the ensemble is written to the local tier and before any `invoke_ensemble` call can target it. A composition that fails validation returns an error to the orchestrator; no partial or pending ensemble state persists.

The orchestrator cannot author new scripts or new model profiles through any tool.

---

## Consequences

**Positive:**
- Full palette accessible — supports the ensemble-author-as-tool-user pattern (DISCOVER #10)
- Invariant 7 remains intact for static composition; AS-2 governs composition-time validation; no modification to the ensemble execution engine
- Scripts and profiles remain operator-curated — no agent-generated executable code (AS-6)

**Negative:**
- Composition-time validation must reproduce the load-time validation logic (risk of divergence if maintained separately). Implementation should reuse the load-time validator, not duplicate it
- A composition that would cycle through an ensemble deep in the reference graph fails at composition time — the orchestrator must reason about or retry; it cannot see the full graph cheaply without `list_ensembles`

**Neutral:**
- Composition produces a local-tier ensemble; promotion is governed separately (ADR-008). Calibration happens after composition, before trust (ADR-007)

---

## Provenance Check

The essay's §Tension suggested "orchestrator-created ensembles should be restricted to pure profile-and-script compositions (no ensemble-to-ensemble references)" as one of two safety options. This ADR chooses the *other* option — full palette with validation. The override came from DISCOVER feed-forward signal #10, produced during the discovery phase when product considerations (the operator-as-tool-user pattern) surfaced. Essay language that suggests palette restriction is now superseded by AS-2 and this ADR. Surfaced here because the chosen framing is not the essay's leading proposal.
