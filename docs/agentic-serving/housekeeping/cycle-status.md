# RDD Cycle Status — Agentic Serving (Scoped)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle open:** v0.8.5
**Migration version:** 0.8.5 (`housekeeping/.migration-version`)

## Cycle Stack

### Active: Cycle 5 — Agentic-serving library structure (capability ensembles + multi-methodology-consumer surface)

**Cycle number:** 5
**Started:** 2026-05-12
**Current phase:** build (closed 2026-05-12; play / synthesize / graduate available as next moves)
**Cycle type:** mini-cycle
**Plugin version:** v0.8.5
**Artifact base:** `docs/agentic-serving/`
**Skipped phases:** research, model, architect, play, synthesize
**BUILD mode:** auto (declared 2026-05-12 per ADR-091; mechanical-character YAML/config authoring suits autonomous execution after high-level direction; review concentrated at start-and-end. Practitioner accepts the mode-selection tradeoff: design-alternative examination and scoping-judgment surfacing are gated-mode capabilities.)

**Origin:** Cycle 4 PLAY (2026-05-12) chose option (c) — follow-up cycle DECIDE/BUILD pickup — with the play-derived proposal `proposals/agentic-serving-library-structure.md` as substrate. Cycle 4 status archived at `cycle-archive/cycle-4-cheap-orchestrator-and-ensembles.md`.

**Recommended cycle shape (per proposal §"RDD-phase routing"):** DISCOVER (update mode against multi-methodology-consumer pattern) → DECIDE (resolve OD-1 through OD-6) → BUILD (mechanical authoring of capability ensembles, profile file, subdirectory layout, README).

**MODEL handling for Cycle 5:** Skipped as a standalone phase per user-selected Mode D shape. New vocabulary that surfaces in DISCOVER (e.g., "methodology layer", "dispatch layer", "execution layer", "capability ensemble", "operation-named ensemble", "methodology consumer") will be folded into DISCOVER's tail as Amendment Log entries on `domain-model.md`. If DECIDE deliberation reveals vocabulary territory that warrants a dedicated MODEL phase, the cycle's `Skipped phases:` field can be amended mid-cycle.

**ARCHITECT handling for Cycle 5:** Skipped per the proposal recommendation — the existing `system-design.md` and `system-design.agents.md` modules accommodate the proposed library structure without re-allocation. If DECIDE outcomes (particularly OD-3 methodology-layer composition shape) introduce module-shape change, the cycle's `Skipped phases:` field can be amended to insert ARCHITECT before BUILD.

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| DISCOVER | ✅ Complete (2026-05-12; update mode; gate closed with belief-mapping on Methodology Consumer framing) | updated `product-discovery.md` (new candidate stakeholder confirmed as distinct role; 2 new jobs for Ensemble Author / Operator; new Skill Orchestration User role with jobs + mental model; 3 new value tensions; 6 new assumption inversions with attribution discipline; 7 new vocabulary entries with three-tier settled/candidate disposition) + susceptibility snapshot `housekeeping/audits/susceptibility-snapshot-cycle-5-discover.md` (no Grounding Reframe; 2 advisories integrated into DECIDE entry commitments) + gate reflection `housekeeping/gates/cycle-5-discover-gate.md` | Practitioner refined the architectural commitment from the proposal's "methodology-agnostic orchestrator" to **"skill-framework-agnostic orchestrator"** — broader: covers RDD, Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks, and emerging skill standards. Practitioner verbatim: *"Better is to be able to leverage any agentic skill (as 'skills' are more-or-less standardized these days). [Encoding] a specific flow into llm-orc... that's not my first choice."* Methodology Consumer renamed / generalized to **Skill Orchestration User**, confirmed as distinct role (humans wear multiple roles; concerns are distinct). Topaz 8-skill taxonomy is the lingua franca between skill frameworks (decomposing) and capability dispatch (routing). Capability library is capability-fine-grained / operation-named. Working defaults are in Cycle 5 BUILD scope. Reliability profile observation captured for Orchestrator LLM (high on derivable, low on integration). |
| DECIDE | ✅ Complete (2026-05-12; gate closed with belief-mapping on parameterized-capability-ensembles timing; conjunctive falsification standard practitioner-generated) | 3 new ADRs (019, 020, 021) + ADR-015 partial-update header + `skill-framework-capability-registry.md` artifact + scenarios.md additions (3 feature blocks + Cycle 5 Cycle Acceptance Criteria Table + preservation scenarios) + interaction-specs.md additions (new Skill Orchestration User stakeholder + 4 new Ensemble Author / Operator tasks) + 3-round argument audit (clean at round 3) + conformance scan (zero violations; 10 BUILD-scope gaps; 3 compatible notes) + susceptibility snapshot (no Grounding Reframe; 2 advisories integrated into BUILD carry-forwards) + gate reflection `housekeeping/gates/cycle-5-decide-gate.md` | Practitioner refined ADR-021's falsification trigger from output-quality divergence at sub-task verdict level to a **conjunctive standard at long-horizon task outcome level**: (a) generalized agnostic scheme fails to produce good long-horizon results under cheap-cloud-orchestrator + local-free-model leverage AND (b) framework-encoding into agentic serving is empirically the *only* way to recover good results. Targets the premature-inversion failure mode where one capability ensemble serving one skill framework better than another might be mistaken for the agnostic commitment being wrong. Value-proposition framing: cost savings via local-free-model leverage under cheap-cloud orchestration; long-horizon task outcomes as measurement surface (not per-sub-task verdicts). Skill frameworks are *pluggable consumers* of the generalized orchestration scheme, not *modalities* of the orchestrator. |
| BUILD | ✅ Complete (2026-05-12; auto mode; phase-close susceptibility snapshot + gate reflection note written) | 7 per-file Model Profiles in `.llm-orc/profiles/agentic-*.yaml` (WP-A5; profile-file format refined from proposal's single-aggregate-file shape to one-file-per-profile to match the loader's `name:`-discriminated format); `.llm-orc/ensembles/agentic-serving/` subdirectory with 8 ensembles — 6 capability (`code-generator` promoted from `agentic-coding-helper` per ADR-019 §"Working defaults"; `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`, `web-searcher`) + 2 system (moved with names preserved) — and operator-facing README (WP-B5 through WP-F5); `.llm-orc/scripts/agentic_serving/web_searcher.py` (Tavily adapter; 3 error paths smoke-tested) (WP-E5); `.llm-orc/config.yaml` `agentic_serving:` section rewritten to reference `agentic-*` profile names (WP-G5); downstream sweep applied — system-design.md Amendment Log entry 8; domain-model.md Amendment Log entry 9; scenarios.md preservation scenario amended to reflect agentic-coding-helper → code-generator promotion; ORIENTATION.md current-state regenerated for BUILD-close milestone (WP-H5); ADR-019 §Consequences §Positive scope-of-claim qualifier added at snapshot remediation; README authoring guide distinguishes LLM-agent vs. script-agent schemas at snapshot remediation. Verification: `llm-orc list-ensembles` discovers all 8 agentic-serving ensembles; `ConfigurationManager.get_model_profiles()` resolves all 7 `agentic-*` profiles. Susceptibility snapshot `housekeeping/audits/susceptibility-snapshot-cycle-5-build.md` (no Grounding Reframe; 3 advisory carry-forwards — 2 applied in-cycle as remediations, 1 — preservation-scenario-amendment pattern — recorded as auto-mode feed-forward); gate reflection `housekeeping/gates/cycle-5-build-gate.md`. | Practitioner-implicit signal at BUILD entry: auto-mode declaration warranted by the WPs' mechanical character (YAML/config authoring + Python adapter script). BUILD ran without per-scenario-group gates; close-out reflection-time at cycle close confirmed two in-cycle remediations (snapshot findings 2 + 3) and one auto-mode feed-forward (snapshot finding 1). Cycle 4 PLAY's n=1 evidence-basis qualifier on the proposal's framings is confirmed settled-by-use at Cycle 5 close — the operation-named principle, capability-fine-grained naming, three-layer architecture, and skill-framework-agnostic dispatch survived concrete BUILD-phase authoring as operator/structural vocabulary. Cycle acceptance criteria Layer-match `no` entries (multi-skill-framework deployment evidence; fresh-clone first-encounter live exercise; integration scenario through five capability dispatches) remain unsatisfied at the named layer; deferred to operator-driven empirical work per ADR-019 §Negative. |

## Carry-forward signals from Cycle 4

### Substrate

**Load-bearing handoff artifact:** `proposals/agentic-serving-library-structure.md` (2026-05-12). Cycle 5 takes this as the directionally-strong starting point for DISCOVER, not as architectural settlement. Per Cycle 4 PLAY susceptibility snapshot advisory #2, all framings in the proposal derive from a single inhabitation session (n=1) and have not been examined under DISCOVER-phase assumption inversion.

### DISCOVER-routed PLAY notes

From Cycle 4 `essays/reflections/field-notes.md`:

- **Note 1** — auto-mode of BUILD shipped mechanism architecture but did not ship operator-facing working defaults; ADR-015 §Negative's "operator-driven library migration" reads cleanly at decide time but produced a configuration-surface gap at first-encounter (practitioner verbatim: *"the agentic-serving config is to me part of the build"*). **DISCOVER attends:** how should the Ensemble Author / Operator stakeholder's super-objective or jobs be refined to capture the working-defaults-at-first-encounter requirement?
- **Note 8** — usability friction observation (DISCOVER as candidate value tension).
- **Note 11** — usability friction observation (DISCOVER as candidate value tension).
- **Note 14** (extended routing per Cycle 4 PLAY snapshot advisory #1) — RDD's tier-1 Architectural Isolation mechanism maps cleanly to `invoke_ensemble`'s fresh-context dispatch property. Multi-methodology-consumer pattern (the orchestrator is a substrate). **DISCOVER attends:** does the stakeholder model need a "Methodology Consumer" stakeholder role distinct from Ensemble Author / Operator? Are RDD-as-methodology, code-review-as-methodology, security-review-as-methodology separate stakeholder instances or one parameterized role?
- **Note 18** — usability friction observation (DISCOVER as candidate value tension).

### Three-layer framing under examination (proposal substrate, n=1 evidence)

The proposal proposes a three-layer architecture (methodology layer / capability dispatch layer / capability instance layer). DISCOVER should examine this framing under assumption inversion:

- What would have to be true for the three-layer separation to be the wrong abstraction?
- What would have to be true for "operation-named ensembles" to be wrong (vs. methodology-named ensembles)?
- What would have to be true for the `agentic-` prefix / `agentic-serving/` subdirectory convention to be wrong?
- What would the right ensemble decomposition look like if the orchestrator were *not* methodology-agnostic?

### DECIDE-phase open decisions (resolved at DECIDE)

Carried forward from `proposals/agentic-serving-library-structure.md` §"Open decisions":

| OD | Topic | DECIDE territory |
|----|-------|------------------|
| OD-1 | `mathematical_reasoning` slot strategy | Minor ADR or domain-model amendment; proposal recommends option (b) leave unauthored |
| OD-2 | `tool_use` ensemble shape | ADR-003 amendment territory if option (b) MCP; BUILD-mechanical if option (a) script-agent or option (c) client-side delegation |
| OD-3 | Methodology-layer composition shape | Multi-ADR territory; affects scenarios + may reopen ARCHITECT |
| OD-4 | Web-search backend (Brave / Tavily / Exa / Serper / DDG) | Operational decision; possibly ADR |
| OD-5 | Placement of general-purpose ensembles (e.g., `development/code-review.yaml`) | Style decision; can defer to BUILD |
| OD-6 | Methodology-skill / capability-ensemble naming registry | New corpus artifact territory |

### BUILD-phase mechanical work (queued for after DECIDE)

Per `proposals/agentic-serving-library-structure.md` §"RDD-phase routing":

- Author capability ensemble YAMLs: `code-generator`, `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`.
- Create `.llm-orc/profiles/agentic-serving-profiles.yaml` (isolates all agentic-serving Model Profiles for easy model-swap).
- Create `.llm-orc/ensembles/agentic-serving/` subdirectory; move `agentic-result-summarizer.yaml` and `agentic-calibration-checker.yaml` into it.
- Rewrite `.llm-orc/config.yaml` `agentic_serving:` section to reference renamed profiles and the new subdirectory layout.
- Write `agentic-serving/README.md` (structure + extension guide).

Specific work packages (WP-A5, WP-B5, ...) will be enumerated in DECIDE-phase scenarios once OD-1 through OD-6 resolve.

### Advisory carry-forwards from Cycle 4 PLAY susceptibility snapshot

- **Evidence-basis is n=1.** Cycle 4 PLAY produced one inhabitation session. The directionally-strong framings in the proposal warrant DISCOVER-phase assumption inversion, not direct acceptance.
- **Attribution discipline.** Cycle 4 PLAY's snapshot caught agent-introduced framings recorded as practitioner-generated discoveries (notes 14, 19). Cycle 5 DISCOVER should attend: when reading proposal framings, distinguish *empirically-grounded findings* (MissingSkillMetadataError recovery path works on first encounter; multi-methodology pattern feasible on existing primitives; tier-routing activated cleanly through OpenCode) from *agent-analytical-synthesis* (three-layer framing labels; operation-named-vs-methodology-named principle; capability-fine-grained naming pattern).
- **Plexus-conditional value claim** (carried from DECIDE-gate snapshot of Cycle 4). If Cycle 5 DECIDE deliberation on OD-2/OD-3 surfaces cross-session calibration needs, the value claim is Plexus-conditional; in-session value preserves without Plexus.

## Context for Resumption

A fresh session resuming Cycle 5 DISCOVER should read in this order:

1. This file (Cycle 5 cycle-status.md) — current state.
2. `proposals/agentic-serving-library-structure.md` — load-bearing handoff substrate.
3. `cycle-archive/cycle-4-cheap-orchestrator-and-ensembles.md` — Cycle 4's complete state at close, including PLAY field-notes routing summary and susceptibility snapshot disposition.
4. `essays/reflections/field-notes.md` Cycle 4 section — original PLAY notes 1–19 (the proposal's evidence base).
5. `product-discovery.md` — current product discovery state (target for DISCOVER update).
6. `housekeeping/audits/susceptibility-snapshot-cycle-4-play.md` — advisory context that shaped the proposal's evidence-basis qualifier and the four advisory integrations recorded against `field-notes.md` and the proposal itself.

## DECIDE-entry context (carry-forwards from DISCOVER gate close)

### Settled premises going into DECIDE

1. **Skill Orchestration User / Methodology Consumer is a distinct role** — humans wear multiple roles concurrently; concerns are distinct.
2. **The orchestrator commitment is skill-framework-agnostic** (broader than the proposal's "methodology-agnostic" framing) — covers RDD, Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks, and emerging skill standards.
3. **Topaz 8-skill taxonomy is the lingua franca** between skill frameworks (decomposing) and capability dispatch (routing).
4. **Capability library is capability-fine-grained / operation-named**, not methodology-coarse.
5. **Working defaults are in Cycle 5 BUILD scope** — operator-facing deployment shape lands, not just mechanism architecture. Practitioner verbatim from Cycle 4 PLAY note 1: *"the agentic-serving config is to me part of the build."*
6. **Reliability profile observation for Orchestrator LLM** — high on derivable claims; low on integration claims; consistent across task types.

### Open questions DECIDE must address

1. **OD-1 through OD-6** from the proposal — all six need DECIDE resolution.
2. **No-dispatch fallback (note 19)** — coverage gap (Cycle 5+ ADR) or intended scope (orchestrator narration *is meant* to bypass dispatch for tasks no ensemble matches)? Framing examination at DECIDE.
3. **Calibration on orchestrator-own-narration (note 16)** — hold; do not force resolution this cycle.
4. **Vocabulary disposition** — does "three-layer architecture" / "capability ensemble" / "operation-named ensemble" survive as operator voice through DECIDE work, or relocate to research voice?
5. **`compose_ensemble` primitives misunderstanding (note 13)** — does this need a DECIDE scenario, and what?

### Specific commitments carried forward to DECIDE (from snapshot advisories)

1. **DECIDE OD-3 acknowledges the skill-framework-agnostic commitment is provisionally settled** (snapshot Advisory 1). OD-3's deliberation includes seam-case inversions: does Topaz-skill routing produce routing-quality parity across skill-framework contexts, or do framework-specific dispatch needs surface?

2. **DECIDE explicitly dispatches the four inversion questions** named in §"Three-layer framing under examination" (snapshot Advisory 2) to specific OD slots:
   - What would have to be true for the three-layer separation to be the wrong abstraction?
   - What would have to be true for "operation-named ensembles" to be wrong (vs. methodology-named)?
   - What would have to be true for the `agentic-` prefix / `agentic-serving/` subdirectory convention to be wrong?
   - What would the right ensemble decomposition look like if the orchestrator were *not* skill-framework-agnostic?

3. **Cycle 4 PLAY attribution discipline holds at DECIDE entry** — load-bearing empirical observations (n=1 evidence) are usable substrate; framings (especially agent-introduced "fallback" / "two gaps" characterizations) require examination, not direct adoption.

## BUILD-entry context (carry-forwards from DECIDE gate close)

### Settled premises going into BUILD

1. **Skill-framework-agnostic dispatch is the architectural commitment**; falsification standard is conjunctive at long-horizon task outcome level (general scheme fails AND framework-encoding only path).
2. **Value proposition**: cost savings via local-free-model leverage under cheap-cloud orchestration; long-horizon task outcomes as measurement surface.
3. **Minimum-viable capability ensemble set (5 ensembles)** is BUILD-scope authoring target — RDD-research-workflow-representative initial shape.
4. **On-ramp authoring is in BUILD scope**: profile file + subdirectory + README + rewritten config section.
5. **`web-searcher` with Tavily default** is the `tool_use` slot's authored capability.
6. **Per-capability dispatch contract** with explicit-naming preferred + natural-language supported.

### Open commitments BUILD must honor

1. **Snapshot Advisory 1 — scope-claim breadth (BUILD-phase settlement)**: skill-framework-agnostic commitment is grounded in RDD only (n=1). BUILD scope is RDD-decomposition exercise. If BUILD-phase work produces decomposition evidence for any non-RDD framework, capture it; otherwise scope claim persists as candidate at BUILD close.

2. **Snapshot Advisory 2 — no-dispatch-fallback resolution durability**: the "intended scope" resolution closes the discover-gate examination commitment at minimum threshold. BUILD work surfacing orchestrator-natural-language-response errors should be recorded as candidate evidence — either for continued "intended scope" or future-cycle reconsideration as coverage-gap territory.

3. **Vocabulary candidacy resolution at BUILD close**: "capability ensemble", "operation-named ensemble", "three-layer architecture" enter BUILD as candidates pending settled-by-use confirmation. Product-discovery vocabulary table updates at BUILD close — settled (survived concrete authoring) or relocated to research-voice in domain-model.md.

4. **Cycle Acceptance Criteria Table integration verification**: 4 emergent/aggregate criteria identified; 3 with Layer-match "no" requiring integration tests or live-deployment evidence at BUILD Step 5.5.

5. **Downstream-artifact sweep for ADR-019's update of ADR-015**: four-artifact sweep (system-design.md, ORIENTATION.md, domain-model.md, field-guide.md) deferred to BUILD close. Likely brief — reframing is at proposal/product-discovery characterization layer, not at system-design module layer.

6. **BUILD mode declaration**: defaults to `gated`; practitioner declares at BUILD entry. Cycle 5 BUILD's mechanical-character work (YAML/config authoring) may suit `auto` mode; practitioner's call.

### BUILD work packages

Per ADR-019 §"Working defaults are in Cycle 5 BUILD scope" and ADR-020:

- **WP-A5**: `.llm-orc/profiles/agentic-serving-profiles.yaml` new file (all agentic-serving Model Profiles isolated; single-file model swaps).
- **WP-B5**: `.llm-orc/ensembles/agentic-serving/` subdirectory + move `agentic-result-summarizer` and `agentic-calibration-checker` into it.
- **WP-C5**: `code-generator.yaml` capability ensemble (promoted from `agentic-coding-helper`, code_generation).
- **WP-D5**: `claim-extractor.yaml` (factual_knowledge), `argument-mapper.yaml` (logical_reasoning), `prose-improver.yaml` (writing_quality), `text-summarizer.yaml` (summarization). Four single-agent ensembles authored in one work package.
- **WP-E5**: `web-searcher.yaml` script-agent ensemble + Tavily adapter Python script + environment-variable wiring (per ADR-020).
- **WP-F5**: `.llm-orc/ensembles/agentic-serving/README.md` operator-facing extension guide.
- **WP-G5**: Rewrite `.llm-orc/config.yaml` `agentic_serving:` section to reference renamed profiles and new subdirectory.
- **WP-H5**: Downstream sweep (system-design.md, ORIENTATION.md, domain-model.md, field-guide.md) for ADR-019's update of ADR-015; cycle-status archive prep at BUILD close.
- **WP-I5** (optional): integration scenarios at Step 5.5 — fresh-clone first-encounter live exercise verifying the cycle's working-defaults claim.

Work-package dependencies: WP-A5 (profile file) and WP-B5 (subdirectory) before WP-G5 (config rewrite references them). WP-C5/D5/E5 (capability ensembles) before WP-G5 (config references them). WP-F5 (README) can land in parallel.

## Feed-Forward Signals

### From DECIDE (Cycle 5)

1. **Conjunctive falsification standard for the agnostic commitment** — long-horizon task outcome layer is the measurement surface; sub-task verdict divergence alone (ADR-018 drift criteria) does not invalidate the contract. BUILD-phase work that surfaces *task-outcome* signal feeds this measurement directly.
2. **Three resolution paths under falsification** — parameterized capability ensembles (lightest, preserves operation-named principle); per-skill-framework capability ensembles (reopens ADR-019); explicit acceptance the agnostic commitment was over-broad. Cycle 5 commits to neither under falsification — names them as available paths.
3. **`web-searcher`'s tier-escalation degeneracy** — script-agent ensembles have no-op tier escalation (same cheap and escalated profile reference). The README documents this; future capability ensembles with similar shapes (script-agents for other tool_use cases) inherit the same property.
4. **n=1 skill-framework scope gap** — RDD is the only framework structurally verified. BUILD's RDD-decomposition exercise is the natural empirical anchor; non-RDD framework integration evidence is desired but not in BUILD scope.

### From DISCOVER (Cycle 5)

1. **Skill-framework-agnostic commitment** is the load-bearing architectural refinement of the proposal's framing. DECIDE OD-3 deliberation is the practical test surface.
2. **Operation-named / capability-fine-grained library** is the working organizational principle for BUILD-phase ensemble authoring; survives DECIDE if concrete authoring decisions hold the principle.
3. **Working defaults in BUILD scope** removes the cycle-shape ambiguity that produced Cycle 4 PLAY note 1's first-encounter gap. BUILD scope includes the agentic-serving profile file, tagged capability ensembles, subdirectory layout, and README.
4. **Reliability profile observation** for Orchestrator LLM is feed-forward for any DECIDE scenario that involves orchestrator-natural-language-claims (note 16 / note 19 territory if a future cycle takes it up).
5. **Vocabulary three-tier disposition** (settled at gate / candidate under DECIDE / candidate agent-introduced) is the working frame for any new vocabulary that surfaces at DECIDE; preserve attribution discipline across the cycle.

### From Cycle 4 (closed at PLAY)

The Cycle 4 archive carries forward five load-bearing findings for Cycle 5 DISCOVER attention:

1. **Note 1: configuration-surface gap** — operator-driven library migration produced a first-encounter gap; the agentic-serving config is part of what BUILD ships in operator perception (whether or not the ADR-015 §Negative reading places it elsewhere).
2. **Note 14: multi-methodology-consumer feasibility** — RDD-via-agentic-serving (and other methodology consumers) is structurally feasible on the existing primitive surface; no ADR amendment needed.
3. **Note 15: methodology-layer separation** — methodology / dispatch / execution as three substrate layers; the operation-level discovery question is downstream of slot-level Topaz framing.
4. **Notes 16 + 19: quality-infrastructure coverage gap** — Calibration Gate covers dispatched-ensemble outputs and ADR-017 covers tool-call patterns, but neither inspects the orchestrator's natural-language output to the client, and the no-dispatch fallback path bypasses the entire quality infrastructure. **This may be DECIDE territory in Cycle 5, not DISCOVER — note as candidate ADR if OD-2 or OD-3 deliberation surfaces the gap.**
5. **Six Open Decisions (OD-1 through OD-6)** — see DECIDE territory table above.
