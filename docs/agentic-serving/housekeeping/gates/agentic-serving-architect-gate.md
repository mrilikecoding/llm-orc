# Gate Reflection: Agentic Serving ARCHITECT → BUILD

**Date:** 2026-04-20
**Phase boundary:** ARCHITECT → BUILD
**Cycle:** Agentic Serving (scoped, `docs/agentic-serving/`)

## Belief-mapping question composed for this gate

Two forms were composed. The primary form was Grounding Reframe, derived from the susceptibility-snapshot-agentic-serving-architect.md findings; the secondary form was belief-mapping on the module decomposition.

**Grounding Reframe (two items):**

> Reframe 1 — Context Injection Stage as a module: ADR-009 says the Phase 2 hook point must be *structurally reserved* so silent deletion is prevented. The question is whether that reservation warrants module weight, or whether a lighter structural commitment would satisfy the same intent — a reserved function signature at session start, a reserved config-file position, or a named no-op in a larger module's pipeline. Without grounding, the design would carry a module with contract, integration test, fitness criterion, and roadmap slot that stays permanent overhead if OQ #4 and OQ #7 never resolve favorably.
>
> Reframe 2 — Client-tool surface, Option C specifically or direction that admits C and D? The user's direction ("a step that direction" of OpenCode support) describes a destination but does not pick between Options C (turn-boundary delegation) and D (mid-execution callback). Both get you there. Without grounding, the serving layer commits to C when the actual RDD-via-OpenCode scenarios may stress the C-vs-D boundary. Grounding action: write 2-3 stress scenarios in advance of WP-F that exercise the boundary.

**Belief-mapping probe (consolidations):**

> The system design landed on 13 modules in one pass. Two plausible consolidations were not surfaced: Orchestrator Configuration could merge into Serving Layer; Calibration Gate could be a subcomponent of Orchestrator Runtime. What would you need to believe for *either* consolidation to be correct, and does that belief match or contradict your sense of how these concepts want to sit?

## User's response

On consolidations: "While those seem like reasonable consolidations, do you think they would be best left separate so that we have clear separation of concerns? Or would it be more cohesive to include them?"

On Context Injection: "yes that seems realistic to keep open: context injection stage. How is it done in single-agent paradigms? What serves our ability to answer the question more faithfully later?"

On Plexus: "For the Plexus question, I think allowing the system to be extensible to it is important. Does that serve the reframe questions here? Happy to discuss further."

Follow-up after the agent's substantive engagement with each: "Great. That reads cleanly to me. Layering concern is resolved for me based on what you say."

## Pedagogical move selected

**Grounding Reframe** (two items, per ADR-059) paired with **belief-mapping** (on the consolidation question).

The Grounding Reframe was dispatched from susceptibility-snapshot findings meeting all three significance properties (specific, actionable, in-cycle applicable). Belief-mapping on consolidations was composed to test whether the 13-module decomposition survived a contested reading rather than merely getting waved through.

## Commitment gating outputs

**Settled premises (the user is building on these going into BUILD):**

- Architectural drivers (as presented, unchanged).
- 4-layer module layering (L0 Core / L1 Domain Policy / L2 Runtime / L3 Entry); 12 modules post-amendment.
- FC-4 is load-bearing: Orchestrator Runtime imports only Budget Controller, Tool Dispatch, and Summarizer Harness.
- Responsibility Matrix allocations (post-amendment: Context Injection action now belongs to Serving Layer's `resolve_session_start_context` function, not a standalone module).
- 13 fitness criteria (FC-9 amended to check the typed function signature and call site rather than a module).
- Dependency graph direction; no cycles; L1-through-L3 layering rule.
- Retrofit triage: cycle-validator extraction (`_validate_cross_ensemble_cycles` → public `validate_ensemble_reference_graph`) is a build-time refactor in WP-A, scoped as a `refactor:` commit.
- Client Tool Surface Commitment: Option C (turn-boundary delegation) is the current architectural answer, but is **scenario-gated** — WP-F does not start until stress scenarios in `scenarios.md` have been written and Option C has been shown to handle them; if any scenario requires mid-execution callback, C is insufficient and the commitment is amended.
- Orchestrator Configuration and Calibration Gate both stay as separate L1/L3 modules — consolidation into their respective neighbors would either invert the layering (Config) or break FC-4 (Calibration).
- Phase 2 Plexus context injection is reserved structurally via a typed function `resolve_session_start_context(session: SessionContext) -> list[PromptFragment]` in Serving Layer, not a module. Phase 1 body returns `[]`; Phase 2 is a function-body change, not a structural addition.

**Open questions (the user is holding these open going into BUILD):**

- Client-tool BDD scenarios do not yet exist in `scenarios.md`. WP-F is blocked on them; they are blocked on a short scenario-write (potentially a DECIDE mini-cycle).
- Visibility form (OQ #2 in domain model): WP-E defaults to structured SSE events; final form may be a dashboard, structured log, or tool-user-inline surfacing.
- Budget specific numbers: ADR-005 defers to build; outer anchor is "comparable to running an RDD phase."
- Calibration N (ADR-007): default count deferred to build.
- Session identity mechanism (WP-B): message-history-derivation with OpenAI `user` field correlation as default; custom header/cookie may become necessary for tightened Autonomy deployments.
- `record_outcome` payload schema (ADR-009): minimum payload deferred to build.
- Open Questions 1, 3, 4, 5, 6, 7 from the domain model remain open (OQ #2 and #8 already addressed at prior gates).

**Specific commitments carried forward to BUILD:**

- Before any WP starts: the builder invokes `/rdd-build` and reads `system-design.md` as the primary context document (not the full artifact set); the roadmap provides sequencing context.
- Before WP-F specifically: builder requests scenario-write for client-tool delegation (4 target scenarios captured in roadmap Open Decision Point #1).
- WP-A (cycle validator extraction) is a `refactor:` commit; it may run before or in parallel with WP-B; it blocks WP-G.
- FC-1 through FC-13 are enforced as build-time fitness criteria; violations are stewardship-checkpoint concerns.
- Each WP's stewardship checkpoint verifies participating-module coverage against claimed scenarios (roadmap Participating-Modules check).
