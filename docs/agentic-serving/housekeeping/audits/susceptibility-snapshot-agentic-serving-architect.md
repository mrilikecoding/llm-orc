# Susceptibility Snapshot

**Phase evaluated:** ARCHITECT
**Artifact produced:** system-design.md (v1.0), roadmap.md
**Date:** 2026-04-20

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable from DECIDE | User's turns were terse ("The drivers look good. 2 - agreed. 3 -- ultimately a vision of success is..."). Three substantive exchanges: driver acceptance, cycle-validator extraction classified as build-time refactor, client-tool-surface redirect. No escalating declarative closure. |
| Solution-space narrowing | Clear (module count) | Rising from DECIDE | 13 modules proposed and accepted without examining merger or split alternatives. The only visible scope discussion was Orchestrator Configuration being absorbed into the Serving Layer group — but system-design.md lists it as a standalone module anyway, and no merge rationale appears. Splits (e.g., Context Injection Stage as a function rather than a module, Bootstrapping Pipeline merged into Plexus Adapter) were not surfaced. |
| Framing adoption | Clear (two cases) | Stable from DECIDE | The four-layer ADR framing was adopted wholesale as the module layering structure (L0–L3 maps directly onto ADR-002's Layers 1-4). No examination of alternative decomposition axes (e.g., by data flow, by deployment unit, by change-rate). ADR-009's "structurally reserved" clause was translated into a standalone module (Context Injection Stage) without questioning whether the reservation warranted a module vs. a function signature or a configuration key. |
| Confidence markers | Absent | Stable | No escalating certainty language observed. The Client Tool Surface Commitment section includes explicit provenance ("committed in ARCHITECT 2026-04-20 on user direction") and a reopening condition ("future finding... would reopen this as Option D territory"). This is confidence with stated scope conditions, not closure. |
| Alternative engagement | Ambiguous | Declining from DECIDE | Four client-tool-surface options (A–D) were surfaced with belief-mapping on Option D — this is genuine alternative engagement. However, alternative module decompositions were not surfaced at all. The DECIDE snapshot's recommendation that ADR-001 OQ #3 (external loop substitutability) remain architecturally live does not appear in any module's inversion note or fitness criterion — the system design has no FC that verifies the serving layer remains substitutable if OQ #3 resolves unfavorably. |
| Embedded conclusions | Clear (one case) | Stable from DECIDE | Context Injection Stage is implemented as a module with defined contract, a named no-op pass-through, and integration tests — all before knowing whether the reserved hook warrants this weight. The module exists to prevent "accidental deletion," but a reserved function signature or a config-file position would accomplish the same structural reservation at zero runtime overhead and zero BUILD cost. The module-as-reservation choice was not examined. |

---

## Interpretation

**Overall pattern.** ARCHITECT sits at middle risk on the sycophancy gradient — less susceptible than DECIDE (driver synthesis) but more susceptible than BUILD (code reality tests assumptions). The phase's specific risk, named in the brief, is that module boundaries encode unexamined assumptions. That risk materialized in one moderate case and a sharper case, against a background of generally solid work.

**What is earned.** The Inversion Principle was applied to every module. Most inversion notes are substantive: the Budget Controller note explicitly justifies separation by change-rate; the Orchestrator Runtime note justifies exclusion of Plexus and Calibration from the LLM's context as alignment with the LLM's own mental model; the Plexus Adapter note explains why AS-8 is structurally enforceable only if Plexus-aware code is bounded to a single module. These notes affected boundary definitions — FC-4 (Runtime imports no Plexus, no config, no Autonomy) is a direct product of the Runtime inversion note. FC-4 would not exist without the inversion. This is a real check, not a perfunctory one.

The client-tool-surface decision was handled with genuine engagement: four options presented, belief-mapping applied to Option D, a reopening condition stated. The commitment is Option C (turn-boundary delegation), derived from a directional user signal — see the narrowing question below.

The fitness criteria table is a strength: 13 automatable criteria, not vague quality goals. FC-4, FC-5, FC-6, FC-8 are structurally enforcing decisions that BUILD would otherwise negotiate away. This is the kind of artifact-encoded constraint that earns confidence.

**Two cases that are not earned:**

**Case 1 — Context Injection Stage as a module (moderate).** ADR-009's post-gate reframe said "structurally reserve a pre-orchestration stage." ARCHITECT translated "reserve a stage" into a module with a named contract, integration test, Phase 1 no-op, and FC-9 (exactly one call per session start). The DECIDE snapshot's Item 1 grounding action asked whether the deferral "belongs in ADR-009 or in the serving-layer module's scope document" and flagged that "the absence of even a Phase 2 stub interface may indicate the decision is underspecified." The ARCHITECT response went further in the other direction: a full module rather than a stub. The structural reservation question was not examined — the agent implemented the heavier option without asking whether a reserved function in the Serving Layer would serve the same constraint at lower BUILD cost. The consequence of getting this wrong is concrete: BUILD must create, test, and maintain a module whose Phase 1 implementation is `return []`. If Phase 2 never materializes (OQ #4 and OQ #7 do not resolve favorably, per ADR-009's own negative consequence), this module is permanent overhead.

**Case 2 — Module count without examined alternatives (moderate).** 13 modules across 4 layers arrived at in one pass. This is not inherently wrong — the domain model has 19 concepts, and a one-concept-cluster-to-one-module ratio is a defensible heuristic. But no merger candidates were surfaced to the user for consideration. The most obvious unexamined merger is Orchestrator Configuration into Session Registry: Orchestrator Configuration owns per-session config resolution; Session Registry owns per-session state; the split is justified by "operator-visible vs. internally-derived," but this rationale appears nowhere in the system design. A second unexamined question: why is Bootstrapping Pipeline a separate module rather than an operator CLI command wired directly to Plexus Adapter's ingestion path? Bootstrapping Pipeline has two dependencies (Plexus Adapter, Ensemble Engine) and no dependents except the operator CLI — it has the dependency signature of a script, not a module. Neither alternative was surfaced.

**Prior snapshot follow-through.** The DECIDE snapshot recommended two targeted grounding actions:

- Item 1 (ADR-009 phase 2 hook point): Partially addressed. The hook point was reserved. The question of *module vs. function-signature* was not examined — the heavier option was taken without deliberation.
- Item 2 (ADR-008 pure-tool-user persona gap): Addressed. Autonomy Policy's module description explicitly calls out cycle-status §FF 25 and states the module exposes this as a configuration surface. The Autonomy Policy → Orchestrator Tool Dispatch contract includes a `tool-user-persona` flag. This grounding action was completed and the gap is now assessable as addressed.

ADR-001 OQ #3 (external loop substitutability): The DECIDE snapshot's feed-forward signal 6 asked that "the serving layer be substitutable." No FC enforces this and no inversion note in Serving Layer or Orchestrator Runtime addresses substitutability. This was not a grounding reframe item in DECIDE, but the snapshot noted it. It has not been picked up.

**Client-tool-surface option narrowing.** The user's redirect was: "needs to be a step that direction" of enabling OpenCode to run an RDD pipeline against llm-orc. This is directional, not a specific option pick. Option C (turn-boundary delegation) and Option D (direct client-tool plumbing with a later bridge) both move in that direction. The agent committed Option C and recorded it as "committed in ARCHITECT 2026-04-20 on user direction." The roadmap's WP-F states that before WP-F is built, concrete BDD scenarios must be written into scenarios.md — the agent thus did not write the scenarios that would have tested whether Option C matches the user's intent or only approximates it. The deferral of WP-F scenario-writing to a "DECIDE mini-cycle or inline scenario-write before starting WP-F" means the commitment and its validation are separated by a full BUILD phase. The risk is not large — the commitment has a stated reopening condition — but the inference from "a step that direction" to "Option C is settled" is narrower than the evidence supports.

---

## Recommendation

**Grounding Reframe recommended — two targeted items.**

**Item 1 — Context Injection Stage: module or reservation?**

What is uncertain: whether the structural reservation for Phase 2 context injection warrants a standalone module with a defined contract and integration test, or whether a reserved function signature in the Serving Layer (or a no-op config key) accomplishes the same prevention-of-accidental-deletion at lower BUILD cost.

Grounding action: before BUILD begins WP-J, ask one explicit question: what does "structurally reserved" mean in the deployment? If the answer is "the Serving Layer's session-start path has an explicit call site that returns empty in Phase 1 and is replaced in Phase 2," that is a function, not a module — and it can live inside Serving Layer rather than as a separate import. If the answer is "Phase 2 may be authored by a different team or merged independently," a module with a named contract is warranted. The distinction determines whether WP-J is 30 lines (function + stub test) or 150 lines (module + contract + integration test). The user's vision — OpenCode running an RDD pipeline against llm-orc — is unlikely to interact with Context Injection until Phase 2. Starting with the lighter reservation defers the module decision to when Phase 2 has a concrete implementation plan.

What the user would be building on without grounding: a module that assumes Phase 2 is a distinct, bounded feature rather than an inline enhancement, locked in by FC-9 and a named integration test from day one of BUILD.

**Item 2 — Client-tool-surface option narrowing (low priority).**

What is uncertain: whether the user's directional signal ("a step that direction") committed Option C specifically or endorsed the direction. Options C and D both serve the OpenCode vision; their tradeoffs (latency profile, semantic coupling, future Option D territory) were described to the user but no scenario was written that would reveal the preference empirically.

Grounding action: before WP-F implementation, write two concrete scenarios — one that only Option C satisfies and one that only Option D satisfies — and present them to the user as a choice. The roadmap already mandates scenario-writing before WP-F; the grounding action is to make those scenarios test the C-vs-D distinction explicitly rather than only formalizing Option C's behavior. This converts a directional user signal into an informed commitment.

What the user would be building on without grounding: Option C as settled architecture when the user may not have distinguished between C and D as specific choices, only endorsed the general direction.

---

## Feed-Forward Signals for BUILD

1. **WP-J weight decision.** Before starting WP-J, resolve Context Injection Stage as module vs. function. The roadmap lists WP-J as independent once WP-B is done — this resolution does not block the critical path but should precede WP-J implementation.

2. **WP-F scenario gap.** The roadmap correctly mandates new BDD scenarios before WP-F. Those scenarios should explicitly test the C-vs-D distinction: does the user's intent require the orchestrator to close the turn before any client-side action, or is there a case where mid-turn delegation matters? The scenario outcome is the option commitment, not a prior architectural conclusion.

3. **OQ #3 substitutability.** The DECIDE snapshot flagged that ADR-001 OQ #3 (external loop as alternative) should remain architecturally live. No fitness criterion enforces this. BUILD should not treat the serving layer as permanently coupled to the internal ReAct loop — at minimum, the Serving Layer → Orchestrator Runtime contract should be an interface, not a concrete type, so OQ #3 can be revisited without a rewrite.

4. **Bootstrapping Pipeline module vs. CLI script.** Before implementing WP-K, evaluate whether Bootstrapping Pipeline warrants a module or whether it is a CLI command (a script wired to Plexus Adapter ingestion). A CLI command that calls Plexus Adapter directly satisfies every stated requirement and removes one module from the dependency graph. No architectural constraint in system-design.md requires it to be a module.

5. **DECIDE snapshot Item 2 (ADR-008 autonomy) is resolved.** The Autonomy Policy module's `tool-user-persona` flag addresses the DECIDE snapshot's P2-B gap. BUILD should verify via `test_autonomy_gate_fires_before_every_dispatch` that the flag is exercised in both the operator-as-tool-user and pure-tool-user paths, not only the default path.
