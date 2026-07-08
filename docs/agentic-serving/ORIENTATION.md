# Agentic Serving -- Orientation

## What this system is

Agentic serving extends llm-orc (a declarative DAG-based LLM orchestration engine) to serve as the backend for agentic coding tools via OpenAI-compatible endpoints. As of Cycle 8, every request to `/v1/chat/completions` is handled by **one declarative Serving Ensemble** -- classify (route the turn) -> seat (dispatch to the resolved capability) -> marshal (shape the deliverable, gate its validity, emit it) -- executed by the same L0 Ensemble Engine that runs any llm-orc ensemble, on primitives that already ship (guard/branch, bounded loop, dynamic dispatch). There is no persistent internal orchestrator: the client (OpenCode, or any OpenAI-compatible caller) owns the multi-turn agentic loop, and each request is a single declarative pass. The north star is full model parity through composition -- the endpoint should do everything a single model does behind a coding tool (explain, fix, edit, run tests, build) with no capability loss, and composing ensembles should widen what's possible, never narrow it.

## Who it serves

- **Tool user** -- a developer using an OpenAI-family agentic coding client (OpenCode, Aider, Cline, Cursor) pointed at llm-orc. Trusts the endpoint to behave transparently and, since Cycle 8, to handle any turn a real model would (not just new-file builds). Reading path: `product-discovery.md` (Tool User jobs) -> `system-design.md` §Cycle 8 -> `.llm-orc/ensembles/agentic-serving/serving.yaml`.
- **Ensemble author / operator** -- authors capability ensembles and composition shapes, runs the server. Curates the Topaz-keyed registry and Shape Catalog declaratively; the old promotion/calibration surface is gone. Reading path: `product-discovery.md` (Operator jobs) -> ADR-047 -> `src/llm_orc/core/serving/`.
- **Skill orchestration user** -- operates a client-side skill framework (RDD, Anthropic Skills, MCP-based frameworks) that decomposes work into capability-typed sub-tasks. The classify seat still routes by the shared Topaz 8-skill taxonomy. Reading path: `product-discovery.md` -> `domain-model.md` (AS-10) -> `interaction-specs.md` Cycle 8 additions.
- **Orchestrator LLM -- dissolved.** Older docs describe a standing ReAct-looping actor that held conversation state and composed ensembles at runtime. That actor no longer exists; its jobs re-home to the classify seat (routing), the operator-curated registry (composition), and substrate reads/writes (state) under the client-owned loop. See `interaction-specs.md`, "Stakeholder: Orchestrator LLM -- dissolved."

## Key constraints

Drawn from `domain-model.md` invariants and `system-design.md` §Cycle 8:

- **AS-11 -- declarative-ensemble-native; extend the engine, never a parallel layer.** Control flow lives in ensemble DAGs and engine primitives, not bespoke Python. This invariant drove the Cycle-8 collapse and the deletion of the old `agentic/` layer.
- **AS-2 -- validate-before-load is the registry's single admission gate.** Every capability part and composition shape passes a shared reference-graph check (no cycle, within depth, resolves) before it becomes dispatchable.
- **Full model parity via composition, not degradation.** The serve must do everything a single model does -- explain, fix, edit, run tests, build -- with no capability loss, and composition should increase the surface of what's possible, never shrink it while complicating the mechanism.
- **Grounded acceptance is a composed verification gate, independent of the builder.** A build deliverable is accepted only when a deterministic executor (runs code + tests) and an isolated adequacy/coverage judge both pass -- catching orthogonal failure modes, never the builder grading itself.
- **Interactive latency on the 32GB rig is first-class, not a tuning afterthought.** Thinking-mode is a per-seat routing decision (easy turns thinking-off, hard turns thinking-on), keeping ordinary turns interactive on the permanent target hardware.

## How the artifacts fit together

**Tier 1 -- Entry point:** `ORIENTATION.md` (this document).

**Tier 2 -- Primary readables (read end-to-end):**
- `product-discovery.md` -- stakeholder needs, value tensions, assumption inversions; Cycle 8's full-model-parity settlement and interactive-speed tension are woven into the existing sections.
- `system-design.md` (v6.7) -- **read `## Cycle 8 -- Target Architecture` first**: the current, self-contained architecture (modules, dependency graph, fitness criteria, open decision points). Everything below that section (`Architecture at a glance`, the L2 Runtime cluster) describes the now-dissolved orchestrator-actor architecture, retained as historical record pending rewrite.
- `roadmap.md` -- work packages. Cycle 8's WP-A8 through WP-F8 are all complete in the actual codebase, though the roadmap's Completed-Work-Log table and Cycle-4/6/7 sections haven't been swept to reflect that yet (see Current State).

**Tier 3 -- Supporting material (consult as needed):**
- `domain-model.md` -- vocabulary and invariants AS-1..AS-11; §"Cycle-8 orchestrator-actor dissolution" records what survives, supersedes, or re-homes.
- `decisions/adr-044..048-*.md` -- the Cycle 8 ADRs: 044 (AS-11), 045 (clean-slate collapse), 046 (target architecture / actor dissolution), 047 (extensibility -- registry + Shape Catalog), 048 (grounded acceptance). Older ADRs implementing the dissolved layer carry supersession headers pointing here -- check an ADR's header before trusting its body.
- `system-design.agents.md` -- dense pre-Cycle-8 module companion, now banner-marked historical (Cycle-8 WP-F8); prefer `system-design.md` §Cycle 8 and `field-guide.md` for current module detail.
- `scenarios.md` -- Given/When/Then scenarios; the "Cycle 8 -- Target Architecture Scenarios" section at the top is current, earlier sections are banner-marked historical.
- `interaction-specs.md` -- stakeholder task decomposition; "Cycle 8 interaction-spec additions" at the end is the current surface, including the Orchestrator LLM dissolution.
- `field-guide.md` -- the current module-to-code map (regenerated post-WP-F8, 2026-07-08): covers the Serving Ensemble pipeline, the Shape Catalog / capability registry / admission (`core/serving/`), the Accept Gate build-shape, the seat contract (`core/validation/`), the session substrate (`core/session/`), and the L0 engine. The primary reference for "where does X live in code."
- `essays/` + `essays/research-logs/` -- research history; Cycle 8 skipped formal RESEARCH (the Cycle-7 PLAY spikes served that role), so the three `cycle-8-spike-*.md` logs are its grounding evidence.
- `housekeeping/cycle-status.md` -- the fullest narrative of decisions and rationale; dense, and currently a few commits behind the actual repo state (see Current State).
- `housekeeping/audits/`, `.rdd/gates/` -- argument/susceptibility audits and phase-boundary gate notes, including the Cycle-8 set.
- `architecture-map.md` -- an 8-cycle multi-fidelity map, now banner-marked historical (its "ACTIVE" loop-driver was deleted in the Cycle-8 collapse; read `field-guide.md` for the current map). `benchmark-design.md`, `skill-framework-capability-registry.md` -- supporting reference material.

## Current state

**DECIDE and ARCHITECT are complete.** Cycle 8's five ADRs (044-048) are Accepted (048 Conditionally), each argument-audit-converged. ARCHITECT closed with an additive `system-design.md` §Cycle 8: the L2 imperative Runtime cluster dissolves into one declarative Serving Ensemble on the shipped L0 engine.

**BUILD is complete through WP-F8, the cycle's last work package.** WP-A8 (Serving Ensemble skeleton), WP-B8 (relocating survivors -- session registry, artifact store, validator, guard, I/O envelope -- out of `agentic/`), WP-C8 (Topaz-keyed registry + Shape Catalog, replacing the hardcoded intent-to-shape map), WP-D8 (the Accept Gate, default-on for build turns), WP-E8 (wiring `core/validation/` as each seat's admission contract), and WP-F8 (deleting `src/llm_orc/agentic/`, ~12.7K lines / 32 files, plus its tests) are all landed. The bespoke imperative serving layer no longer exists; the declarative Serving Ensemble is the only serving path, grounded against a live endpoint and a literal `opencode run`.

**Capability coverage today is build (accept-gated) and explain**, both real-endpoint-grounded. Fix, edit-existing, and run-tests are the named frontier -- future default seats/shapes grown the same way, PLAY-validated per the full-model-parity requirement. Compose-at-runtime and composer-ensembles remain a deferred, not-ruled-out direction (ADR-047 §Deferred); any such mechanism must stay a declarative engine capability, never a revived orchestrator-LLM authoring step.

**Doc-sweep status:** the Cycle-8 current-state sweep (WP-F8, 2026-07-08) reconciled the corpus to the post-collapse state -- `cycle-status.md`, `roadmap.md` (WP-C8/E8/F8 now in the Completed Work Log), `field-guide.md`, and this document are current; `system-design.md` §Cycle 8 is the current architecture, with its pre-Cycle-8 sections plus `system-design.agents.md` and `architecture-map.md` banner-marked historical. The deep historical companion content is retained, not rewritten, for the graduate-time research branch.

**Next step is graduate-time:** fold this corpus into native project docs with a clean merge to `main` (native docs only), while the full 8-cycle corpus survives on a research branch. PLAY (validating the fix/edit/run-tests frontier) and the remaining doc sweep are open, not blocking.
