# Cycle 5 Archive — Agentic-serving library structure (capability ensembles + multi-methodology-consumer surface)

**Archived:** 2026-05-13 (PLAY close → Path 1 decision → Cycle 6 opens on routing + observability)
**Plugin version:** v0.8.5
**Migration version:** 0.8.5
**Artifact base:** `docs/agentic-serving/`

**Cycle disposition at close:** PLAY produced 20 field notes across two stakeholder seats (gamemaster reconnaissance + Skill Orchestration User inhabitation via OpenCode). Practitioner chose **Path 1** — Thread A defects (4 broken capability ensembles + result-summarizer compression + code-generator coder timeout) treated as normal llm-orc dev work outside the methodology cycle; routing + observability axes opened as Cycle 6 scoped mini-cycle. SYNTHESIZE deferred (available as later move). Graduate deferred.

---

## Cycle 5 — Agentic-serving library structure (capability ensembles + multi-methodology-consumer surface)

**Cycle number:** 5
**Started:** 2026-05-12
**Closed:** 2026-05-13 (PLAY complete; cycle archived)
**Cycle type:** mini-cycle
**Plugin version:** v0.8.5
**Artifact base:** `docs/agentic-serving/`
**Skipped phases:** research, model, architect, synthesize
**BUILD mode:** auto (declared 2026-05-12 per ADR-091; mechanical-character YAML/config authoring suits autonomous execution after high-level direction; review concentrated at start-and-end. Practitioner accepted the mode-selection tradeoff: design-alternative examination and scoping-judgment surfacing are gated-mode capabilities.)

**Origin:** Cycle 4 PLAY (2026-05-12) chose option (c) — follow-up cycle DECIDE/BUILD pickup — with the play-derived proposal `proposals/agentic-serving-library-structure.md` as substrate. Cycle 4 status archived at `cycle-archive/cycle-4-cheap-orchestrator-and-ensembles.md`.

**Recommended cycle shape (per proposal §"RDD-phase routing"):** DISCOVER (update mode against multi-methodology-consumer pattern) → DECIDE (resolve OD-1 through OD-6) → BUILD (mechanical authoring of capability ensembles, profile file, subdirectory layout, README).

**MODEL handling:** Skipped as standalone phase per Mode D shape. New vocabulary folded into DISCOVER's tail as Amendment Log entries on `domain-model.md`.

**ARCHITECT handling:** Skipped per proposal recommendation — existing `system-design.md` and `system-design.agents.md` modules accommodated the library structure without re-allocation.

## Phase Status (final)

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| DISCOVER | ✅ Complete (2026-05-12; update mode; gate closed with belief-mapping on Methodology Consumer framing) | updated `product-discovery.md` + susceptibility snapshot `housekeeping/audits/susceptibility-snapshot-cycle-5-discover.md` + gate reflection `housekeeping/gates/cycle-5-discover-gate.md` | Practitioner refined the architectural commitment from the proposal's "methodology-agnostic orchestrator" to **"skill-framework-agnostic orchestrator"** — broader: covers RDD, Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks, and emerging skill standards. Methodology Consumer renamed / generalized to **Skill Orchestration User**, confirmed as distinct role. Topaz 8-skill taxonomy is the lingua franca. Capability library is capability-fine-grained / operation-named. Working defaults are in BUILD scope. Reliability profile observation captured (high on derivable, low on integration). |
| DECIDE | ✅ Complete (2026-05-12; gate closed with belief-mapping on parameterized-capability-ensembles timing; conjunctive falsification standard practitioner-generated) | 3 new ADRs (019, 020, 021) + ADR-015 partial-update header + `skill-framework-capability-registry.md` + scenarios.md additions + interaction-specs.md additions + 3-round argument audit + conformance scan + susceptibility snapshot + gate reflection `housekeeping/gates/cycle-5-decide-gate.md` | Practitioner refined ADR-021's falsification trigger to a **conjunctive standard at long-horizon task outcome level**: (a) generalized agnostic scheme fails to produce good long-horizon results under cheap-cloud-orchestrator + local-free-model leverage AND (b) framework-encoding into agentic serving is empirically the *only* way to recover good results. Value-proposition: cost savings via local-free-model leverage; long-horizon task outcomes as measurement surface. Skill frameworks are *pluggable consumers*, not *modalities*. |
| BUILD | ✅ Complete (2026-05-12; auto mode; phase-close susceptibility snapshot + gate reflection note written) | 7 per-file Model Profiles in `.llm-orc/profiles/agentic-*.yaml`; `.llm-orc/ensembles/agentic-serving/` subdirectory with 6 capability + 2 system ensembles and operator-facing README; `.llm-orc/scripts/agentic_serving/web_searcher.py` (Tavily adapter); `.llm-orc/config.yaml` `agentic_serving:` section rewritten; downstream sweep applied. Verification: `llm-orc list-ensembles` discovers all 8 agentic-serving ensembles; `ConfigurationManager.get_model_profiles()` resolves all 7 `agentic-*` profiles. Susceptibility snapshot `housekeeping/audits/susceptibility-snapshot-cycle-5-build.md`; gate reflection `housekeeping/gates/cycle-5-build-gate.md`. | Auto-mode declaration warranted by WPs' mechanical character. Cycle 4 PLAY's n=1 evidence-basis qualifier confirmed settled-by-use at Cycle 5 close. Cycle acceptance criteria Layer-match `no` entries (multi-skill-framework deployment evidence; fresh-clone first-encounter live exercise; integration scenario through five capability dispatches) remained unsatisfied at the named layer; deferred to operator-driven empirical work per ADR-019 §Negative. |
| PLAY | ✅ Complete (2026-05-13; two-phase method: gamemaster reconnaissance via `curl` produced 9 observations; Skill Orchestration User inhabitation via OpenCode produced 10 observations + cross-cutting reflection; post-reflection coda (note 20) closed the snapshot's named NL-with-client-tools gap. 20 field notes total. Susceptibility snapshot complete with no Grounding Reframe and 3 advisory carry-forwards.) | `essays/reflections/field-notes.md` (Cycle 5 PLAY section + post-reflection coda) + `housekeeping/audits/susceptibility-snapshot-cycle-5-play.md` | Practitioner verbatim at close: *"the observability of the routing still seems lacking to me, but in the serving console process and in the output from the orchestrator — I'd like to have more visibility into what the routing is doing, even if output is the most important metric. This was flagged last time."* Snapshot reframed note 19's "Cycle 1 → 4 → 5 unchanged" as **infrastructure-complete / routing-incomplete** (Cycle 5 BUILD shipped new internal events; operator-terminal routing of those events remains deferred). BUILD-regression batch (notes 1, 4, 5) discloses 4 of 6 capability ensembles runtime-broken (NoneType+str on the single-agent-no-system_prompt YAML shape); `code-generator` partial (coder timeout + critic-on-empty-output hallucination + summarizer content-stripping + error-status-inversion). Note 20 confirms ADR-021's natural-language-supported clause is unsupported under both tested client configurations; operational routing preference is direct → client-tools → ensemble. |
| SYNTHESIZE | ☐ Deferred (available as later move; not rejected) | — | — |

## Carry-forward signals to Cycle 6 (and to Thread A hotfix work)

### Thread A — Operationally blocking defects (handled as normal llm-orc dev work, OUTSIDE the methodology cycle)

Per Cycle 5 PLAY susceptibility snapshot reframe: note 1's "structurally inadequate" framing overstates the diagnosis. The evidence supports a single scenario addition + mechanical fix, not tooling redesign.

1. **4 broken capability ensembles share a YAML shape the executor rejects at runtime**: `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`. Single-agent + ensemble-level `default_task:` + no agent-level `system_prompt:` produces `unsupported operand type(s) for +: 'NoneType' and 'str'` at agent setup; `agents_count: 0`; zero tokens consumed. Fix is either executor-defensible-default (treat missing system_prompt as empty string or compose from default_task) or YAML patch (add explicit system_prompt to each).
2. **`code-generator` coder agent timeout** under qwen3:8b at 180s — tunable via timeout extension or model swap; consequence-cascade (critic hallucinates success on empty input; synthesizer produces independent output) survives the timeout.
3. **`agentic-result-summarizer` (ADR-004/AS-7) content-stripping + error-status-inversion** — qwen3:0.6b strips load-bearing code and reports `status: success` when execution dict carries `status: error`. Either tune the summarizer's system prompt to preserve code blocks and not invert error status, or replace with a passthrough-with-summary-coda shape that preserves the original result alongside the summary.
4. **Add a runtime-dispatch test scenario to `scenarios.md`** mandating each capability ensemble is exercised end-to-end (not just discovered + schema-validated) before BUILD declares close. The snapshot's specific remediation — single scenario addition, not tooling redesign.

### Cycle 6 territory — Routing surface + observability (scoped mini-cycle)

**Routing axis** (note 20's challenged assumption): ADR-021's natural-language-supported clause is unsupported under both tested client configurations. The operational routing preference is **direct → client-tools → ensemble**, not ensemble-first-when-slot-fits. Cycle 6 DECIDE: re-ground the contract via system-prompt work, OR explicitly narrow the supported-routing surface and document the operative routing preference.

**Observability axis** (note 19's reframed framing): **infrastructure-complete / routing-incomplete.** Cycle 5 BUILD shipped new internal events (verdicts, tier-routing decisions, audit consumption, signal-channel aggregation); the architecture has the telemetry; what's missing is the routing of telemetry to human-visible surfaces. Cycle 6 DECIDE: wire existing internal events to (a) operator terminal — colored logs or TUI dashboard per Cycle 4 note 7's concrete framing, and (b) orchestrator's reasoning context so it can answer the timing/graph questions a Skill Orchestration User asks.

The two axes are linked: the operator can't tell what routing decision happened without observability; the orchestrator can't refine its routing decisions without visibility into its own dispatches.

### Cycle 5 PLAY susceptibility snapshot — three advisory carry-forwards

1. **Note 1's framing overstates** the diagnosis. Cycle 6 should treat the BUILD-regression remediation as a scenario addition + mechanical fix, not a tooling redesign.
2. **Note 19's "unchanged across cycles" is misleading**; the sharper framing is "infrastructure-complete / routing-incomplete." Cycle 6 DECIDE target is wiring existing telemetry, not designing observability from scratch.
3. **Note 15's compound framing** ("fabrication while critiquing fabrication") may direct DECIDE toward one architectural response when two failure modes are warranted — (a) hallucination presented as ensemble output (calibration coverage gap), and (b) accurate critique of ensemble dispatch-surface behavior (recovery-mode reasoning working as designed). Cycle 6 DECIDE should treat them as distinct.

### Cycle 5 BUILD susceptibility snapshot — three advisory carry-forwards (still active)

1. **Preservation-scenario amendment pattern** (auto-mode feed-forward) — BUILD updating `scenarios.md` to match implementation rather than surfacing the conflict for practitioner review. Future auto-mode BUILDs should flag scenario-rewrite events in the session record.
2. **Script-agent YAML schema constraint documentation** — the BUILD-time discovery that `ScriptAgentConfig` has no `system_prompt:` field is documented implicitly in the `web-searcher.yaml` reference instance; should be explicit in any next operator-facing documentation pass.
3. **ADR-019 §Consequences §Positive n=1 scope qualifier** — ORIENTATION.md qualifies to n=1 (RDD only structurally verified); ADR-019 §Consequences §Positive does not carry the qualification inline. Cycle 6 should either act on this (extend evidence base via non-RDD framework integration) or explicitly defer with rationale.

### Settled by use at Cycle 5 close

- The operation-named, capability-fine-grained library principle (notes 14, 15 from Cycle 4 PLAY) survived concrete BUILD-phase authoring as operator/structural vocabulary.
- The skill-framework-agnostic architectural commitment (DISCOVER refinement of the proposal's "methodology-agnostic" framing) is the load-bearing architectural posture.
- The three-layer separation (methodology layer / dispatch layer / execution layer) is operator vocabulary at Cycle 5 close, not research vocabulary.
- Script-agent error-path dispatch surface works end-to-end (PLAY note 7 — `web-searcher`'s `authentication_failed` shape).
- Composition pipeline respects dependencies (PLAY note 8 — orchestrator halted downstream dispatches when web-searcher failed auth).
- Explicit-naming dispatch contract per ADR-021 works under explicit naming (PLAY note 3).
- Multi-turn memory works (PLAY note 17 — orchestrator retained file-read state across queries).

### Notes routed to RESEARCH (Cycle 6+ research territory)

- **PLAY note 14** — orchestrator's own remediation predictions (system-prompt tuning recommendations) did not bear out in subsequent dispatches. New question: does the orchestrator's self-modeling of dispatch reliably predict dispatch behavior?
- **PLAY note 12** (also informs note 18) — orchestrator's structural blindness to its own execution graph. New question: at what threshold of self-knowledge does an orchestrator stop fabricating recovery-narrations of dispatch outcomes it cannot observe?

### Notes routed to SYNTHESIS (if a synthesis essay is later written)

- Cycle 5 PLAY observations 3 (explicit-naming dispatch works), 7 (web-searcher error path), 8 (composition dependencies), 17 (multi-turn memory)
- The "validation-vs-execution gap" pattern (snapshot-narrowed framing: discovery-layer pre-shipping verification was structurally distinct from runtime-correctness verification)
- The two-phase PLAY method (gamemaster reconnaissance + stakeholder inhabitation) as complementary epistemic acts — methodology-relevant pattern that could transfer to the `/rdd-play` skill itself
- The "infrastructure-complete / routing-incomplete" reframe (snapshot's correction of the cross-cycle persistence framing)

## Context for Resumption (if Cycle 5 is ever reopened)

A fresh session reopening Cycle 5 (e.g., for synthesis essay work) should read in this order:

1. This archive file (Cycle 5 cycle-status snapshot at close).
2. `essays/reflections/field-notes.md` Cycle 5 PLAY section (20 observations + cross-cutting reflection + post-reflection coda + routing summary).
3. `housekeeping/audits/susceptibility-snapshot-cycle-5-play.md` (the snapshot that reframed three of the field notes' aggregate framings).
4. `housekeeping/audits/susceptibility-snapshot-cycle-5-build.md` (BUILD-phase advisory carry-forwards; three remain active going into Cycle 6).
5. `housekeeping/gates/cycle-5-{discover,decide,build}-gate.md` (gate reflection notes).
6. `decisions/adr-019-*.md`, `adr-020-*.md`, `adr-021-*.md` (the cycle's three new ADRs).
7. `product-discovery.md` §Stakeholder Map + §Jobs and Mental Models — Skill Orchestration User role.
8. `interaction-specs.md` §Stakeholder: Skill Orchestration User + §Ensemble Author / Operator additions.

## Cycle 5 deliverables summary

**Decisions:**
- 3 new ADRs (019 — skill-framework-agnostic library; 020 — `tool_use` script-agent shape; 021 — per-capability dispatch contract)
- ADR-015 partial-update header
- `skill-framework-capability-registry.md` new artifact

**Scenarios:**
- 3 feature blocks added to `scenarios.md`
- Cycle 5 Cycle Acceptance Criteria Table
- 1 preservation scenario amendment (recorded in BUILD snapshot as auto-mode feed-forward)

**Interaction specs:**
- New Skill Orchestration User stakeholder section
- 4 new Ensemble Author / Operator tasks

**Code/configuration:**
- 7 `agentic-*` Model Profiles in `.llm-orc/profiles/`
- 8 ensembles in `.llm-orc/ensembles/agentic-serving/` (6 capability + 2 system)
- `.llm-orc/scripts/agentic_serving/web_searcher.py` (Tavily adapter; 3 error paths smoke-tested)
- `.llm-orc/ensembles/agentic-serving/README.md` operator extension guide
- `.llm-orc/config.yaml` `agentic_serving:` section rewritten

**Downstream sweep:**
- `system-design.md` Amendment Log entry 8
- `domain-model.md` Amendment Log entry 9
- `scenarios.md` preservation scenario amendment
- `ORIENTATION.md` current-state regenerated for BUILD-close milestone

**Audits and snapshots:**
- 3 susceptibility snapshots (discover, decide, build, play)
- 1 conformance scan (decide)
- 3-round argument audit (decide; clean at round 3)
- 4 gate reflection notes (discover, decide, build; PLAY exempt per skill)

**PLAY findings:**
- 20 field notes (9 from gamemaster reconnaissance via `curl`; 10 from Skill Orchestration User inhabitation via OpenCode; 1 post-reflection coda)
- Routing distribution: 3 BUILD-regression, 5 DECIDE, 5 DISCOVER, 1 RESEARCH, 4 SYNTHESIS, 2 interaction-specs notes
- Cross-cutting reflection: routing observability gap is the persistent carry-forward (sharpened by snapshot to "infrastructure-complete / routing-incomplete")

## Path forward chosen at close (2026-05-13)

**Path 1** (per practitioner verbatim: *"I think path 1 is the way forward. Routing + observability need to be addressed."*):

- **Thread A defects** → normal llm-orc dev work; not a methodology cycle
- **Cycle 6** → scoped mini-cycle on **routing + observability** (Mode D shape; DISCOVER → DECIDE → ARCHITECT (likely) → BUILD; gated mode for BUILD given design-alternative examination matters here)
- **Synthesize Cycle 5** → deferred (available as later move)
- **Graduate** → deferred (Cycle 6 close is a natural revisit point)
