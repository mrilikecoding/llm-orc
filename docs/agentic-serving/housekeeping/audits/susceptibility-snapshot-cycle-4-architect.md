# Susceptibility Snapshot

**Phase evaluated:** ARCHITECT (Cycle 4)
**Artifact produced:** system-design.md v3.0, system-design.agents.md v3.0, roadmap.md (8 new Cycle 4 WPs), ORIENTATION.md update
**Date:** 2026-05-08

---

## Susceptibility Risk Summary (Overall Verdict)

**Low-to-moderate. The dominant signal is inherited framing — not active narrowing within the ARCHITECT phase.** The ARCHITECT phase itself produced a structurally clean integration of six ADRs, a cycle-free dependency graph, a fully populated responsibility matrix, and an ADR-076 qualitative-claim decomposition with no deferred items. The three-property Grounding Reframe test produces two advisory carry-forwards and one conditional finding — none reaching the threshold for a triggered Grounding Reframe in-cycle.

The most consequential susceptibility risk at this boundary is not what ARCHITECT introduced but what it inherited: the DECIDE-gate's asymmetric-grounding finding (OQ #14) was carried forward as Cycle 5+ territory, and ARCHITECT chose option (c) — "note that BUILD evidence will inform what grounding the other stages need" — for four of the five under-grounded cross-layer stages. That choice is architecturally defensible but means the dependency graph encodes six cross-layer edges whose operational grounding is thinner than the ADR-016 edge that anchors the graph.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining from DECIDE peak | The ARCHITECT artifact does not introduce new declarative claims beyond those carried in the ADRs. Module descriptions rephrase ADR text rather than extending it. Assertion density at module-description level is stable; no new unqualified confidence claims appear. |
| Solution-space narrowing | Clear (inherited) | Stable — carried from DECIDE | All three module boundaries (Conversation Compaction separate from Runtime; Tier-Escalation Router separate from Tool Dispatch; Calibration Signal Channel separate from Calibration Gate) were established in the ADRs before ARCHITECT began. ARCHITECT did not independently examine whether the module separations were necessary; it allocated responsibility from the ADR framing. |
| Framing adoption | Clear | Stable | ADR-016's "five mechanisms within L0-L3" framing is present verbatim in the module decomposition and the dependency graph. The Calibration Signal Channel module's own description reads the framing back without noting the "operationalized-within-L1" assumption is still logically-validated-only as of ARCHITECT close. No visible ARCHITECT-phase examination of the framing inheritance. |
| Confidence markers | Absent-to-ambiguous | Declining | The conditional-acceptance notation is present in every surface that touches the Calibration Signal Channel (system-design.md module table; system-design.agents.md module decomposition; dependency graph annotation; layer table). The phrase "first-deployment evidence pending" appears consistently. No confidence escalation observed within ARCHITECT. |
| Alternative engagement | Absent | Declining from DECIDE | ARCHITECT produced no examination of the three module-separation questions named in the dispatch prompt. The Inversion Principle check appears as a per-module Inversion Note (operator mental model alignment) but does not address the question of whether the separations were alternatives rather than givens. The Rejected Alternatives sections live in the ADRs; ARCHITECT did not revisit them at the boundary. |
| Embedded conclusions at artifact-production moments | Clear — three instances | New at ARCHITECT boundary | Three module boundaries are embedded as conclusions in the module table at the moment the ARCHITECT artifact was produced: Conversation Compaction as a separate L2 module (not examined as "could Runtime own this more cheaply"), Tier-Escalation Router as a separate L2 module (not examined as "could Tool Dispatch own this more cheaply"), Calibration Signal Channel as L1 (not examined as "does this module have enough owned surface to warrant existence"). Each conclusion originated in an ADR framing; ARCHITECT instantiated it without independent examination. |

---

## Finding 1 — Solution-Space Narrowing on Module Boundaries

**Finding:** The three new module separations were not examined as alternatives during ARCHITECT. They were carried from DECIDE-phase ADR framings as settled premises.

**Evidence:**

- ADR-012's framing: *"the pipeline operates under ADR-002's four-layer architecture without amendment — Conversation Compaction is an L2 elaboration."* This appears in ADR-012 §Consequences §Neutral. ARCHITECT's module decomposition reads: "No L3 dependencies — receives messages array as parameter from Runtime." The separation is implemented cleanly, but the question of whether Runtime ownership was the simpler design (Runtime already owns the ReAct loop and the messages array) is not examined.

- ADR-015's framing: *"The framing commitment from research-gate Grounding Action 2 holds: tier-escalation responsibilities concentrate within Tool Dispatch (L2 interposition) rather than warranting a dedicated routing module orthogonal to L2."* ARCHITECT created a separate Tier-Escalation Router module. The module's Inversion Note reads: *"Per-ensemble tier alternatives — rejected per ADR-015 — would have served developer convenience (granularity) over operator's mental model (per-skill defaults)."* This inversion note addresses a configuration-model question, not the module-separation question. The ADR's own framing said responsibilities would "concentrate within Tool Dispatch" — but ARCHITECT placed the Router as a separate module depended on by Tool Dispatch. This is a legitimate architectural choice (owned concepts: skill metadata schema, verdict-to-tier mapping, `escalation_bypass` and `missing_skill_metadata` errors — substantial enough to separate), but no explicit examination of whether this contradicts ADR-015's "concentrate within Tool Dispatch" framing appears in the artifact.

- ADR-014 §Rejected Alternatives (e): *"a dedicated trajectory-monitor module orthogonal to Calibration Gate rejected per elaboration-by-evidence."* Calibration Signal Channel is an analogous separation. No note in ARCHITECT's module decomposition explains why the Signal Channel warrants module status when a trajectory-monitor module was explicitly rejected on analogous grounds.

**Risk class:** Advisory. The three separations are architecturally coherent and owned-concept coverage is adequate to justify each. The missing element is the reasoning trace connecting the ADR framings to the final module-separation decisions. This is a documentation gap, not an evidence gap, at the ARCHITECT boundary.

---

## Finding 2 — Framing Adoption from DECIDE-Phase ADRs

**Finding:** The elaboration-by-evidence framing commitment is present in system-design.md as stated ADR-016 inheritance without examining whether ARCHITECT's responsibility-allocation work surfaced any strain on the framing.

**Evidence:**

- system-design.md §Layering Rule: *"If BUILD or first-deployment evidence finds that mechanism (b) time-decay windowing or mechanism (d) periodic out-of-band audit dispatch cannot be operationalized within ADR-002's L0-L3 structure... the elaboration-by-evidence framing commitment is invalidated."*

- The cycle-status ARCHITECT-entry directive (OQ #14 obligation) stated: *"ARCHITECT may surface concrete grounding-mechanism gaps as it allocates responsibilities... ARCHITECT decides whether to (a) surface the gaps for Cycle 5+ research, (b) propose grounding mechanisms inline as architectural-driver entries, or (c) note that BUILD evidence will inform."*

- The ARCHITECT artifact chose option (c) uniformly across all five under-grounded cross-layer stages (L1→L2 verdict→router; L3 cross-session artifact set; intra-L2 conversation-history boundary; orchestrator-response→tool-dispatch boundary; L1→L4 Plexus integration). The fitness criteria for these stages are structurally-typed property checks, not operational-grounding mechanisms. The dependency graph entry for the L1→L2 edge (Tier-Escalation Router → Calibration Gate) reads: "Cycle 4 — new; verdict consumer" with no bounding-mechanism notation comparable to the L0→L1 edge's five-mechanism annotation.

- This is framing adoption: the five-mechanism asymmetry was noted as an open question in DECIDE, and ARCHITECT's responsibility allocation defaulted to the less-rigorously-grounded option (c) without surfacing any reason why the five under-grounded stages do not need mechanism-analogs given OQ #14's explicit flag.

**Risk class:** Advisory carry-forward. The option-(c) choice is defensible at ARCHITECT — BUILD evidence is the natural validation surface, and the falsification trigger on ADR-016 covers the most load-bearing case. The asymmetry is not hidden; it is visible in the dependency graph annotation differential and is logged as OQ #14. But the ARCHITECT artifact does not record *why* option (c) was chosen over option (b) for the four unboxed stages, which means the next agent entering BUILD will not see the reasoning.

---

## Finding 3 — Product-Facing Readability of Module Boundaries

**Finding:** The module separation between Calibration Gate and Calibration Signal Channel does not track the operator's mental model and may create operator-confusion at deployment.

**Evidence:**

- The Calibration Signal Channel's Inversion Note: *"The channel is intrinsically an architectural concept... The operator-facing surface is audit verdicts and parameter-tuning recommendations — operator interacts with the audit verdict, not the channel internals. Naming the module Signal Channel leans architectural; the operator-facing visibility comes through the audit dispatch's diagnostic plus the periodic operator review of advisory drift recommendations."*

- The Inversion Note acknowledges the operator-facing surface lives elsewhere (the audit verdict) while the module name reflects developer architecture. The operator's mental model per product-discovery tension #5 is *"what is the orchestrator doing, and why?"* — the Signal Channel module answers *"how does cross-layer data flow"*, not *"what does the operator see or configure."*

- The Tier-Escalation Router's Inversion Note: *"Operator's mental model — 'I configure per-skill tier defaults at deployment; ensembles declare their primary skill; the system routes by skill+confidence.'"* This one tracks the operator's mental model well; the module boundary aligns with the Topaz skill-taxonomy configuration surface.

- The Session Registry extension: *"The methodology-voice term externalized structured state does not appear in operator-facing surfaces — operators work with the artifact set's named components."* This explicitly surfaces the operator-vs-developer vocabulary gap and resolves it. Strong product-facing orientation.

- The Conversation Compaction module: *"Operator's mental model — 'the orchestrator stays coherent over long sessions because something compacts the conversation.' Module name and owned thresholds match."* Adequate product-facing alignment.

**Risk class:** Advisory. Three of the four new modules or extensions have adequate product-facing orientation; the Calibration Signal Channel is the exception. The module's conditional-acceptance status means it may not be built in Cycle 4 anyway, but if it is, the operator-facing surface gap noted in its own Inversion Note should be resolved before deploying the module.

---

## Finding 4 — Asymmetric-Grounding Coverage in the Architecture (OQ #14)

**Finding:** The dependency graph records six new cross-layer edges in Cycle 4. One (the L0→L1 upward exception) has five bounding mechanisms, a conditional-acceptance flag, a falsification trigger, and a concrete monitoring specification. The other five new cross-layer edges have typed-error contracts and fitness criteria only.

**Evidence:**

The system-design.agents.md dependency graph annotation differential:

- `Ensemble Engine → Calibration Signal Channel`: annotated with a boxed exception block, five bounding mechanisms, read-only-calibration-data-only restriction, structural typing, conditional acceptance.

- `Tier-Escalation Router → Calibration Gate`: annotated "Cycle 4 — new; verdict consumer." No bounding mechanisms. No falsification trigger. The decide-gate note flags: *"L1→L2 verdict→router: Lower — no operational verification of verdict-action mapping under deployment bias dynamics."*

- `Orchestrator Runtime → Conversation Compaction`: annotated "Cycle 4 — new." The decide-gate note: *"Intra-L2 conversation-history boundary: Lower — no verification of cheapest-first ordering's coherence property under deployment-realistic context shapes."*

- `Orchestrator Tool Dispatch → Tier-Escalation Router`: annotated "Cycle 4 — new." The decide-gate note: *"Orchestrator-response→tool-dispatch boundary: Lower — no mechanism for verifying pattern-set adequacy as deployment evidence accumulates."*

- `Conversation Compaction → Ensemble Engine` (new; Layer 4 summarizer): annotated "Cycle 4 — new." No grounding-mechanism notation. This edge introduces an L2→L0 composition where a new module reaches into the Ensemble Engine for a summarizer ensemble — a significant new dependency given Ensemble Engine is L0.

The asymmetry is structurally encoded in the dependency graph: the ADR-016 edge has a visual callout box; the five new downward edges have one-line notations. This is not hidden, but a BUILD engineer arriving without cycle history will not see why the five downward edges did not receive equivalent grounding treatment.

**Risk class:** Advisory carry-forward. The asymmetry is documented and OQ #14 is logged. The risk at the BUILD boundary is that BUILD engineers pick up the five lightly-annotated edges and treat them as fully-grounded, implementing them in order (per the BUILD sequencing recommendation) without the corresponding grounding-mechanism verification loop that ADR-016's edge has. The `Tier-Escalation Router → Calibration Gate` edge is the highest-priority instance because the verdict-to-tier mapping is the operationally load-bearing coupling: if the verdict is unreliable (Sub-Q6 routing gap), the Router's escalation decisions are noise, and no grounding mechanism is specified to detect this.

---

## Specific Findings Summary

| # | Name | Evidence quote | Risk class | Classification |
|---|------|---------------|------------|----------------|
| F1 | Conversation Compaction module-separation not examined | "Conversation Compaction is an L2 elaboration" (ADR-012) — ARCHITECT implemented; no examination of whether Runtime ownership was simpler | Advisory | Advisory carry-forward |
| F2 | Tier-Escalation Router separation contradicts ADR-015 framing without noted reconciliation | ADR-015: "responsibilities concentrate within Tool Dispatch"; ARCHITECT: separate Router module depended on by Tool Dispatch — no reconciliation trace | Advisory | Advisory carry-forward |
| F3 | Calibration Signal Channel separation lacks reasoning trace | ADR-014 rejected a trajectory-monitor module on analogous grounds; no note explains why Signal Channel separation is warranted by contrast | Advisory | Advisory carry-forward |
| F4 | Elaboration-by-evidence framing inherited without examination at ARCHITECT | OQ #14 obligation: ARCHITECT could surface gaps or propose mechanisms; all five stages defaulted to option (c) with no recorded rationale | Advisory | Advisory carry-forward |
| F5 | Calibration Signal Channel module name leaks architectural vocabulary into operator surface | Inversion Note self-identifies: "Naming the module Signal Channel leans architectural" | Advisory | Advisory carry-forward |
| F6 | `Tier-Escalation Router → Calibration Gate` edge has no grounding-mechanism analog | Graph annotation is "Cycle 4 — new; verdict consumer"; decide-gate: "no operational verification of verdict-action mapping under deployment bias dynamics" | Advisory | Highest-priority carry-forward for BUILD entry |
| F7 | `Conversation Compaction → Ensemble Engine` edge (Layer 4 summarizer) introduced without grounding-mechanism notation | New L2→L0 composition; no annotation analogous to the upward-edge exception box | Advisory | Advisory carry-forward |

---

## Recommendations

**No Grounding Reframe warranted at the ARCHITECT boundary.**

The phase is at the BUILD-adjacent end of the sycophancy gradient (ARCHITECT most resistant to framing-adoption risk; BUILD most resistant overall). The signals are inherited, not newly introduced. The conditional-acceptance annotations and falsification triggers from DECIDE are present and intact throughout the ARCHITECT artifacts. None of the seven findings above meets all three properties of the Grounding Reframe test at full strength: each is specific and advisory, but operationally applicable only at BUILD entry — not at the ARCHITECT gate.

**Two advisory carry-forwards recommended for the BUILD-entry handoff prompt:**

**Advisory A — OQ #14 carry-forward with BUILD-specific framing.** The BUILD-entry context should name the `Tier-Escalation Router → Calibration Gate` edge explicitly as requiring a grounding-verification step proportionate to ADR-016's monitoring specification. The existing OQ #14 note targets Cycle 5+ research; the BUILD-level exposure is immediate: the first BUILD work package that implements the Router (WP-E4 per the roadmap sequencing) will produce the first first-deployment evidence on whether the verdict-to-tier mapping is reliable under multi-iteration routing dynamics. That first-deployment evidence is also the Sub-Q6 evidence the decide-gate carried forward. The BUILD handoff prompt should record: *"WP-E4 (Tier-Escalation Router) produces the first evidence on ADR-015's routing-reliability assumption (Sub-Q6). At WP-E4 close, record whether escalation-rate evidence is interpretable as tier-configuration signal or routing-noise. This is OQ #14's BUILD-phase proxy for the L1→L2 verdict→router stage."*

**Advisory B — Calibration Signal Channel module-name resolution before BUILD.** If WP-H4 (Calibration Signal Channel; the last and conditional BUILD work package) is in-scope for this cycle, the module should either (a) be renamed to an operator-vocabulary term that matches the audit-verdict-and-parameter-tuning-recommendation surface the operator actually sees, or (b) the operator-facing surface should be explicitly split from the channel infrastructure — a naming decision that should be made before the first BUILD PR touches the module, not after. The Inversion Note's self-identified gap is the decision surface; the current name is architectural notation, not operator vocabulary.

**No action warranted on Findings F1, F2, F3, F4:** these are documentation gaps in the reasoning trace, not evidence gaps. The DECIDE artifacts contain the reasoning (ADR-012 Consequences §Neutral; ADR-015 Context; ADR-014 Rejected Alternatives (e)); a note in the system-design.agents.md module decompositions pointing to the relevant ADR section would close each gap at low cost, but none creates operational risk that warrants blocking BUILD entry.
