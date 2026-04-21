# Active RDD Cycle: Agentic Serving

**Started:** 2026-03-20
**Current phase:** BUILD (in progress — WP-A complete)
**Artifact base:** `docs/agentic-serving/`
**Essay:** `../essays/001-agentic-serving-architecture.md`

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ✅ Complete | `../essays/001-agentic-serving-architecture.md` | The DAG engine and ReAct loop are complementary, not competing. Plexus as knowledge graph doesn't just add memory -- it lowers the capability threshold for the orchestrator LLM, converting reasoning into retrieval. The cost framing may matter more than the capability framing. |
| DISCOVER | ✅ Complete | `../product-discovery.md` | Plexus as lib (push model) means ingestion boundary is file content, not LLM output -- quality emerges from enrichment, not curation. Tool user and ensemble author often the same person -- visibility is tinkering, not just debugging. Conductor ceiling unknown; orchestration may require frontier models regardless of graph population. Bootstrapping pipeline has a shape but quality gate is the enrichment layer, not upstream curation. |
| MODEL | ✅ Complete | `../domain-model.md` | Plexus should be optional (AS-8) -- design for stateless, benefit from Plexus when available. Enrichment pipeline maturity is an open question that determines whether the learning-system value proposition is real. Two-tier architecture: stateless orchestrator as baseline, Plexus as upgrade to learning system. |
| DECIDE | ✅ Complete | `../decisions/adr-001..011-*.md`, `../scenarios.md`, `../interaction-specs.md` | Plexus's more compelling frame is intra-session multi-agent substrate via consumer-registered lens grammars, not only cross-session memory. Per-ensemble lens registration would make the orchestrator's access polyglot. AS-4 preserved (lens is query-surface grammar applied during enrichment). Reframe is forward signal, not a current-cycle driver -- Plexus's lens design is in-progress. Captured as OQ #8 and essay reflection; folds back in a later cycle. |
| ARCHITECT | ✅ Complete | `../system-design.md`, `../roadmap.md`, `../ORIENTATION.md` (regenerated) | Retrofit mode: ensemble engine stays Layer 3 unchanged; 12 modules across 4 layers plus typed `resolve_session_start_context` function in Serving Layer; 13 fitness criteria. Client tool surface: Option C (turn-boundary delegation) is the commitment, scenario-gated — WP-F does not start until stress scenarios exercise the C/D distinction. Context Injection demoted from module to typed function (ADR-009 reservation is satisfied by signature + call site, not by a module). Consolidations of Orchestrator Configuration into Serving Layer and Calibration Gate into Runtime rejected: the former would invert layering; the latter would break FC-4. Roadmap has 10 WPs, 3 classified transition states; TS-1 (stateless orchestrator serving OpenCode) is the vision-named intermediate target. |
| BUILD | ▶ In Progress | WP-A complete (commits `8a0f5d6`, `0980323`); WP-B Groups 1-2 complete (commits `188b93f`, `111a026`, `59b9053`, `b1e3c54`) — see roadmap Completed Work Log | -- |
| PLAY | ☐ Optional | -- | -- |
| SYNTHESIZE | ☐ Optional | -- | -- |

## Feed-Forward Signals

### From RESEARCH
1. The orchestrator agent pattern (ReAct loop with ensemble invocation as primary tool) is the pragmatic entry point -- not the hybrid model
2. Invariant 7 (static ensemble references) does not govern orchestrator tool-call invocations, but orchestrator-created ensembles must still satisfy it. Validation mechanism needed at creation time
3. Plexus integration is the differentiator: no comparable system provides persistent structured memory across agentic sessions
4. The knowledge-compensated model selection hypothesis (populated graph enables cheaper models) is commercially significant and testable but unvalidated
5. Context management (summarization, compaction, Plexus offloading) is a correctness requirement, not an optimization
6. Budget enforcement (turn limits, token budgets) is a control plane concern validated by both OpenHands and claw-code

### From RESEARCH (reflections)
7. The cost framing (economics over capability) may be the stronger value proposition -- "80% quality at 10% cost" resonates more than "marginally better quality"
8. Open question for domain model: can well-orchestrated smaller models + populated knowledge graph compete with frontier models on quality while winning on cost?

### From DISCOVER
9. Plexus operates as a lib (push model), not a server -- the client drives ingestion of file content, enrichments extract signal. Quality gate is the enrichment pipeline, not upstream curation
10. The orchestrator needs the full composition palette (ensembles + profiles + scripts), including ensemble-to-ensemble references -- the "restrict to profile-and-script only" fallback from the essay is insufficient
11. Tool user and ensemble author are often the same person -- visibility is part of the value proposition (tinkering), not just an operational concern
12. Organic stabilization over explicit curation -- repeated patterns should surface on their own, but quality signals (not just frequency) must inform what stabilizes
13. The conductor skill's ceiling is unknown -- orchestration is broad AND deep, and may require frontier models regardless of graph population. The knowledge-compensated model selection hypothesis remains unvalidated
14. Bootstrapping pipeline: llm-orc artifacts → Plexus ingestion (file content) → enrichment → queryable graph. Background/async ingestion is practical. "Garbage in" concern is addressed by ingesting source material, not LLM summaries

### From MODEL
15. AS-8: Plexus is optional -- the orchestrator works statelessly without it. Design for stateless, benefit from Plexus when available
16. Two-tier architecture: stateless orchestrator (serving layer + ReAct loop + ensemble engine) as baseline product; Plexus as upgrade to learning system
17. Enrichment pipeline maturity (open question #7) determines whether the learning-system value proposition is real. Not a blocker but load-bearing for AS-4 and AS-5
18. 8 invariants (AS-1 through AS-8), 17 concepts, 13 actions, 7 open questions in the scoped model

### From DECIDE
19. 11 ADRs accepted, cycle-scoped at `../decisions/`. All post-audit revisions applied; three argument-audit rounds closed clean
20. Budget (ADR-005) sized for long agentic sessions — outer bound of "running an RDD phase within a session." Specific turn/token numbers deferred to build; enforcement mechanism fixed
21. Full composition palette with validation (ADR-006) overrides the essay's "restrict to profile-and-script" fallback. Conformance debt: cross-ensemble cycle validator is currently private in `EnsembleLoader` — must be extracted to a public function before `compose_ensemble` is built. Captured as refactor scenarios 1-3 in `scenarios.md`
22. Fixed orchestrator tool surface of exactly 5 tools (ADR-003). No dynamic tool extension
23. Result summarization is mandatory and has a Plexus-active vs. stateless split (ADR-004). In stateless mode, lost summarization detail is unrecoverable by the orchestrator
24. Calibration is session-scoped when Plexus is absent; persists across sessions when Plexus is active (ADR-007)
25. Autonomy Level baseline is calibrated for the operator-as-tool-user persona from DISCOVER. Pure tool-user deployments (FF-2) may warrant a tighter default that surfaces composition events — add as architectural configuration surface
26. Plexus integration phased (ADR-009): Phase 1 tool-first; Phase 2 context injection deferred. **Phase 2 hook point is structurally reserved** (post-gate reframe applied): ARCHITECT's session-start flow design must include a pre-orchestration stage where injection can be inserted without modifying the ReAct loop (ADR-001) or tool surface (ADR-003). Phase 1 leaves the stage empty; Phase 2 populates it. Technical rationale for Phase 1 sufficiency: linked to orchestrator profile capability (ADR-011) and OQ #1 (knowledge-compensated model selection)
27. Orchestrator LLM is a Model Profile (ADR-011); no hard-coded tiered fallback. Tiered routing is expressible as a composed ensemble — but only once such an ensemble has been composed and promoted
28. Client tool surface boundary is an open decision discovered while writing interaction specs. How the orchestrator handles client-declared tools (bash, file-edit, etc.) vs. its internal tool surface (ADR-003) needs resolution in ARCHITECT or a follow-up DECIDE mini-cycle

### From DECIDE (reflection gate)
29. Plexus's more compelling frame (per user, 2026-04-19) is intra-session multi-agent substrate via consumer-registered lens grammars. Per-ensemble lens registration would make orchestrator access polyglot — each subagent writes and queries through its own grammar over a shared enrichment. AS-4 is preserved under this reading (lens is enrichment-time query grammar, not LLM-summary ingestion). In-design on the Plexus side; captured as OQ #8 and essay reflection; folds back in a later cycle. Not a driver for current-cycle ADR changes

### From BUILD
39. **Budget defaults sized for local-orchestration value prop** (commit `b1e3c54`, 2026-04-20). Default `turn_limit=500`, `token_limit=10_000_000`; override bounds raised proportionally. Reframing: llm-orc's value proposition is that orchestration with local-hardware compute trades tokens-for-quality against a single frontier-API call. The token ceiling is a pathology circuit breaker for the local-orchestration-heavy case, not a cost ceiling for frontier-API pricing. Frontier-mix deployments tighten via `config.yaml`. ADR-005 flagged this as a build-phase tuning decision; the new numbers track cost-framing reflection (FF #7) and OQ #1 (knowledge-compensated model selection). ADR-005's sizing framing could be amended later to make the local-first default explicit, but the numbers speak for themselves for now
40. **`DEFAULT_MODEL_PROFILE = "default"` is a placeholder in WP-B** (`orchestrator_config.py:22`). Resolves to the literal string "default" until WP-B Group 3 (`/v1/models`) wires it against the actual Model Profile library via `ConfigurationManager.get_model_profiles()`. At that point the resolver either lands on a real profile or raises "no default configured" — Group 3's decision
41. **WP-B Phase 1 defers per-request override *application*.** The `OverrideBounds` config surface is in place (operators can express bounds); the mechanism that applies request overrides and clamps/rejects against bounds is deferred until a concrete scenario demands it. `OrchestratorConfigResolver.resolve()` is parameterless in Phase 1
42. **Session Registry is in-process memory for WP-B.** No cross-process persistence. Sufficient while Autonomy Level and Calibration do not require it (baseline deployment). Persistence is added when AS-3/AS-5/AS-7 enforcement demands it — the contract already accommodates the change (system-design §Session Registry "persistence of Session state across HTTP requests when persistence is required by Autonomy Level or Calibration state")
43. **Tier 1 stewardship clean at WP-B Groups 1-2 boundary.** Dependency direction respects L3 → L0 layering; FC-1/FC-2/FC-3 pass; lifecycle-composition test added for SessionRegistry's shared-reference pattern (caller mutation visible through subsequent lookup by the same identity — commit `59b9053`). Boundary integration tests deferred to Group 4+ (Serving Layer construction). No ADR or domain-model amendments triggered

### From ARCHITECT
30. Retrofit mode — llm-orc has existing FastAPI server, MCP handlers (ExecutionHandler, ValidationHandler, ensemble_crud_handler, promotion_handler, validation_handler, script_handler), ensemble engine, config manager, auth, and artifact system. Agentic serving is additive; Layer 3 (Ensemble Engine) stays unchanged per ADR-001/002
31. 12-module decomposition across 4 layers (L0 Core / L1 Domain Policy / L2 Runtime / L3 Entry) plus typed `resolve_session_start_context` function in Serving Layer. Originally 13 modules; Context Injection Stage demoted to function per ADR-066 gate-reflection amendment #1
32. FC-4 is load-bearing: Orchestrator Runtime imports *only* Budget Controller, Tool Dispatch, and Summarizer Harness. No Plexus, Autonomy, or Calibration leak into the reasoning space. This preserves the orchestrator LLM's mental model ("I emit tool calls and observe results") and structurally enforces ADR-003's closed tool-set property
33. Client tool surface commitment: Option C — turn-boundary delegation via `finish_reason: tool_calls`. Internal action space stays at 5 tools (ADR-003); client tools (bash, file_edit, etc. from OpenCode / Roo Code / Cline) flow through the response surface. Commitment is **scenario-gated** — WP-F does not start until stress scenarios (4 targets in roadmap Open Decision Point #1) test the C vs. D distinction. If any requires mid-execution callback, C is insufficient
34. Phase 2 Plexus context injection is reserved via typed function signature `resolve_session_start_context(session: SessionContext) -> list[PromptFragment]` in Serving Layer — not a module. ADR-009's "structurally reserved" clause satisfied by signature + call site; Phase 1 returns `[]`; Phase 2 is a function-body change. This matches the single-agent paradigm practice (Claude Code, OpenCode, claw-code)
35. Retrofit debt: WP-A extracts `_validate_cross_ensemble_cycles` + `_build_reference_graph` from `EnsembleLoader` private helpers to public `validate_ensemble_reference_graph` in `core/config/ensemble_config.py`. Both load-time and composition-time validation share the single routine (FC-6). Hard-blocks WP-G (composition)
36. Vision-named transition state TS-1 (WP-A + WP-B + WP-C + WP-D + WP-E + WP-F) is the stateless orchestrator that serves OpenCode — the intermediate target for "I can use OpenCode and run a version of this RDD pipeline with it"
37. Roadmap has 10 WPs; 9 hard deps, 4 implied; 3 transition states; 7 open decision points carried forward as build-time latitude
38. Consolidations probed at gate: Orchestrator Configuration → Serving Layer rejected (inverts layering — L1/L3 config is read by L1 modules like Budget Controller and Autonomy Policy via Session Registry). Calibration Gate → Orchestrator Runtime rejected (breaks FC-4; calibration is ensemble-state not runtime-state, and requires Plexus Adapter which Runtime must not import)

## Context for Resumption

**WP-B resumption pointer (2026-04-20).** WP-B split across sessions. Groups 1-2 (Session Registry + Orchestrator Configuration) are complete on branch `agentic-serving`. A fresh session should pick up at **Group 3: `/v1/models` endpoint** in the Serving Layer (resolves orchestrator Model Profile names — lists the IDs from `ConfigurationManager.get_model_profiles()`, filtered by operator configuration). This is the group that decides what `DEFAULT_MODEL_PROFILE = "default"` resolves against (FF #40). Remaining WP-B groups: 3 (`/v1/models`), 4 (`/v1/chat/completions` non-streaming skeleton + `resolve_session_start_context`), 5 (SSE streaming skeleton + tool-call formatting), 6 (integration verification — FC-9, session identity across requests). No downstream dependencies; pure plumbing ahead.

This is a scoped RDD cycle for the agentic-serving feature of llm-orc, a declarative DAG-based LLM orchestration engine. The research phase investigated whether llm-orc can serve as the backend for agentic coding tools via OpenAI-compatible endpoints. Six questions were explored across two research cycles: API surface requirements, DAG-to-ReAct mapping, self-building ensembles, Plexus as memory layer, OpenHands architecture, and claw-code architecture.

The essay concludes with a four-layer architecture: API surface, orchestrator agent, ensemble engine (existing), and knowledge graph (Plexus). Product discovery surfaced seven value tensions and six assumption inversions. Key product insight: Plexus operates as a push-model lib where the client drives ingestion of file content (not LLM summaries), and enrichments extract signal -- quality emerges from the enrichment pipeline, not upstream curation. The tool user and ensemble author are often the same person, making visibility a product feature (tinkering) not just an operational concern.

Citation and argument audits have been run. All P1 issues were remediated in the essay. Some P2 issues remain unresolved. Domain model establishes 8 invariants (AS-1 through AS-8) with AS-8 (Plexus is optional) as the key architectural constraint: design for stateless operation, benefit from Plexus when available. The next phase is DECIDE -- ADRs and behavior scenarios using the domain vocabulary.

## Conformance Notes

**Corpus migrated from RDD v0.4.2 to v0.7.3 on 2026-04-17.** See `.migration-version` and `.migration-rollback.json` for migration details.

**Retroactive Tier 1 audits are absent.** RESEARCH, DISCOVER, and MODEL phases were produced under v0.4.2 which did not have the following v0.7.x unconditional mechanisms: research-methods-reviewer dispatch (ADR-060), per-phase susceptibility-snapshot-evaluator dispatches, and gate reflection notes (ADR-066). These are not being fabricated retrospectively. Tier 1 mechanisms will fire properly starting at DECIDE.

**Deferred conformance items** (to be picked up opportunistically):
- **Framing audit on essay.** v0.7.3 argument-audit dispatches include a framing-audit section; `housekeeping/audits/argument-audit-001.md` has argument-audit only. Pick up in continued work if framing tension surfaces.
- **Concepts table column header.** `domain-model.md` §Concepts column 4 is labeled "Avoid (synonyms)"; v0.7.3 template says "Related Terms". Fix when the table is next edited.
- **First-person plural in research log.** `essays/research-logs/001b-agentic-serving-architecture.md:3,47` uses "we" in question titles. Cross-cutting third-person rule applies.
- **Value tensions phrasing.** `product-discovery.md` §Value Tensions stated as declarative prose rather than open questions per v0.7.3 discover template.
