# Gate Reflection: Cycle 8 — The declarative-ensemble collapse ARCHITECT → BUILD

**Date:** 2026-07-02
**Phase boundary:** ARCHITECT → BUILD
**Cycle:** Cycle 8 — The declarative-ensemble collapse

## Belief-mapping question composed for this gate

Warrant elicitation on the most-consolidated boundary in the light additive decomposition: the agent had folded the retired Artifact Bridge (fidelity marshalling) and Client-Tool-Action Terminal (`tool_calls` emission + the ADR-035 form gate) into a single "marshal" node. Question posed: what makes those one responsibility rather than two, given that if the executor-delegation fork (ODP-1) lands, the accept gate's executor runs tests *through the client-tool emission surface* (marshal's territory), so marshal would serve two callers?

## User's response

"Hm - seems like the spirit of an ensemble implies separating the responsibilities across nodes."

The agent incorporated this: marshal decomposed into three bounded nodes — **shape** (fidelity marshalling; Artifact Bridge re-home), **form-gate** (deterministic destination-validity; ADR-035 re-home; a deterministic verifier node, the cheapest rung of the same verification ladder the accept gate extends), and **emit** (client `tool_call` / prose emission; Client-Tool-Action Terminal re-home). This resolved ODP-1 cleanly: **emit is the reusable node** — a client-delegated executor surfaces a test-run by reusing emit, without shape or form-gate.

Then the architect→build susceptibility snapshot (`housekeeping/audits/susceptibility-snapshot-cycle-8-architect.md`) returned a Grounding Reframe: the same warrant-probe was NOT applied to two structurally identical boundaries asserted in the same pass (Serving Registry as a standalone module; Accept Gate as a peer Module), plus a live cross-document inconsistency (system-design "Module" vs roadmap WP-D8 "build-shape segment" for the Accept Gate) and an intra-document propagation gap (Test Architecture still said "marshal test"). The agent presented the reframe with a proposed resolution — apply the ensemble-spirit principle consistently: Accept Gate is a build-shape composition (sub-graph), not a module; the ADR-047 registry decomposes into existing surfaces (Capability List Builder parts + Composition Validator admission) plus a thin new Shape Catalog, not a standalone module. The practitioner confirmed: "Yep, that lands." The artifacts were revised to consistency (Accept Gate demoted to a build-shape composition; Serving Registry recast; system-design ↔ roadmap reconciled; Test Architecture propagated to the three marshal nodes).

## Pedagogical move selected

Challenge (warrant elicitation on the marshal boundary), followed by a Grounding Reframe (ADR-059) after the isolated snapshot flagged the two un-probed boundaries + the live inconsistency. The reframe resolution was derived by applying the practitioner's own stated principle consistently.

## Commitment gating outputs

**Settled premises (building on going into BUILD):**
- The Cycle-8 target architecture is a light, additive decomposition on the surviving engine: the L2 imperative Runtime cluster dissolves; the serving turn is ONE declarative ensemble (classify → seat → shape/form-gate/emit) on the shipped primitives.
- Genuinely-new framework surface is minimal: a Shape Catalog (ADR-047's new piece) + declarative nodes; parts reuse Capability List Builder, admission reuses Composition Validator, Seat Contract re-points `core/validation/`.
- The Accept Gate is a build-shape composition (executor/judge/gate seats), not a framework module. classify routes build turns to the gated shape, non-build turns to the ungated shape.
- The marshal sub-sequence is three bounded nodes (shape / form-gate / emit); emit is the reusable permission seam.
- Deferral discipline holds: current-state sweep (system-design.agents.md, ORIENTATION, roadmap Cycle-4/6/7 WPs, field-guide) + the `agentic/` code deletion land at BUILD (WP-F8).

**Open questions (holding open going into BUILD):**
- ODP-1 (executor home: internal sandbox vs client-delegated, reusing emit) and ODP-2 (accept-loop location: internal bounded `loop:` vs client outer loop) — resolve against round-budget + flow-vs-autonomous evidence.
- ODP-3 (serving default shape) and ODP-4 (Q5 removal-sequencing detail within WP-F8; envelope-relocation-before-deletion is fixed).
- ADR-048 Conditional-Acceptance validation targets (live-builder independence + artifact-injection channel; judge reliability; sandboxed execution; unstated-input oracle rung).
- ADR-046 §Open: decider-as-seat swap (unspiked); seat-contract wiring (WP-E8); AS-3/AS-7 concern re-homing.

**Specific commitments carried forward to BUILD:**
- **Grounding Reframe owed at WP-D8 entry:** rerun ADR-048's grounding-spike fixtures with criteria withheld/thinned, before wiring the accept gate unconditional (carried from the DECIDE gate).
- BUILD-mode: gated (design-alternative-heavy; declared at cycle open).
- WP sequence honors the two hard prerequisites: WP-B8 (envelope relocation) before WP-F8 (deletion); parity (WP-A8/C8/D8/E8) before WP-F8.
- PLAY-facing forward thread parked (flow-engagement; assume-guarantee/hierarchical-confidence; verifiability-as-invariant) — not Cycle-8 build scope; resurfaces at PLAY + the composer-ensemble direction.
