# ADR-047: Serving extensibility — capability registry, composition-shape catalog, and declarative binding

**Status:** Accepted (2026-07-02)

## Context

Q4 (extensibility) is the DECIDE item after ADR-046. ADR-046 settled the per-turn
handler (classify → seat → marshal, one declarative ensemble) and dynamic dispatch
(a runtime-resolved `ensemble:` target, shipped). Q4 decides how operators register
and curate candidate seats per capability, the serving default, and the binding-time
question. It must do so without reintroducing the orchestrator actor ADR-046
dissolved or the AS-5 trust/promotion machinery it retired.

Two grounding inputs shape this decision:

- **Runtime-composition feasibility spike** (`essays/research-logs/cycle-8-spike-runtime-composition-feasibility.md`, 2026-07-02). A fixed-shape pipeline with runtime-filled parts is expressible today on the shipped primitives (chained `dispatch` nodes wired by `depends_on`, or dispatch-to-a-pre-authored-wrapper). A runtime-*authored* arbitrary DAG shape (a decider emits the structure and the engine materializes it) is NOT expressible without a new primitive, and the spike named four precise gaps.
- **llm-conductor mining** (the practitioner's local skill; `~/.claude/skills/llm-conductor/`). It is a catalog-anchored hybrid: a catalog of named composition shapes selected and customized per task, with novel-DAG design as the author-time fallback that graduates into a pattern library (its ADR-013 + ADR-019). Its essay-004 **Strategy A vs Strategy B** frame is the useful result: Strategy A (routing/composition intelligence *outside* the ensembles, over stateless instruments) beats Strategy B (a general ensemble with an *internal* small-model self-router) for small models, on cited evidence (AutoMix; small-LM-verifier work; DeepMind scaling-agents). The failure is not the patterns, it is placing the routing where a small model must execute it. Its trap is the `Design → Calibrate → Establish → Trust → Promote` populator, which is exactly the retired AS-5 machinery.

**Standing principle (practitioner, 2026-07-02).** Whenever the reasoning reaches "a
small model cannot do this," the first question is what **ensemble orchestration**
(bounded roles + structure + deterministic verification) can achieve, before
defaulting to "a frontier model should do it." Composition is the capability lever
(the corpus north star: the serve does everything a single model does, and
composition increases the surface). A frontier model is a fallback and a benchmark,
not the reflex.

## Decision

### 1. The registry is a Topaz-keyed library of building-block parts plus an operator-curated catalog of composition shapes.

- **Parts** are capability ensembles keyed by Topaz skill (`code_generation`, `tool_use`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`, `writing_quality`, `instruction_following`, `summarization`). The classify decider emits a target; dynamic dispatch resolves it against the registry.
- **Shapes** are named composition patterns authored declaratively as ensemble skeletons or wrappers (solo; gen → review; gather → analyze → synthesize; fan-out → merge; and so on). The catalog *structure* is borrowed from llm-conductor's pattern library (keyed by capability × shape × output-type); only the structure, not its populator.
- Both parts and shapes are **operator-curated**. **AS-2 (validate-before-load) is the admission gate**: every registered part and every registered shape is validated against the ensemble reference graph (no cycle, within the depth limit, every reference resolves to an existing entry) before it can fill a seat or run.

### 2. Binding is declarative and load-time-first.

The classify decider selects a shape and fills its slots; dynamic dispatch binds the
runtime-chosen parts. Selection lives in the declarative structure, not inside a
small model's self-routing. This runs on the shipped primitives today (feasibility
spike). Turn-time selection among several candidates per capability is a hybrid that
a guard/branch or a richer decider can express. Both load-time curation and turn-time
selection keep routing *external* to the capability ensembles: essay-004's Strategy-A
criterion is about external-vs-internal placement, not binding time, so both satisfy
it. The default is **load-time curation + classify-selection** because it needs no new
primitive, not because turn-time selection would be less "Strategy A".

### 3. Invention of new shapes is an author-time activity that grows the catalog.

New composition shapes are authored (by an operator, or by a capable design-time
process) and registered into the catalog after AS-2 validation and review. Runtime
shape-*invention* by the serve is out of scope for the near-term: the Strategy-A/B
evidence names small-model runtime shape-invention as the documented failure mode.

### 4. Serving default: one general shape operators extend.

A default generalist composition shape ships as the baseline (candidate shapes: the
plain classify → seat → marshal handler with a general capability seat, or a
gather → build → verify shape; chosen at BUILD). Operators extend by adding parts and
shapes, never by editing the engine (AS-11).

### 5. Not reintroduced: trust-promotion.

The catalog is operator-curated, not calibration-grown. There is no
accumulate-quality-then-auto-promote loop (retired AS-5). A composition is admitted by
AS-2 plus its contract (the seat contract ADR-046 §Open still tracks as designed but
unwired and unvalidated), not by earned standing trust. Per-dispatch quality signal is
Q2's concern, kept separate.

## Deferred — the composer-ensemble path (named forward direction, not built here)

The practitioner's forward vision is **specialized ensembles that make other ensembles
from building-block ensembles** — the declarative rebuild of llm-conductor's *designer*
role as an ensemble. It is deferred, with a starting strategy so the deferral is
bounded, not vague:

- **compose-at-runtime primitive.** A runner that resolves a structured composition
  spec and materializes it as a child sub-DAG (closing the four gaps the feasibility
  spike named: single-target dispatch; scalar-only `${dep.field}` resolver; DAG frozen
  after parse; no runtime-spec-to-`EnsembleConfig` path). Built TDD like
  guard/loop/dispatch, **when a flow demonstrably needs a shape the catalog lacks**
  (AS-11).
- **composer ensembles**, with these strategy pillars:
  - (a) compose from the registry's validated parts;
  - (b) AS-2 validates the composer's output (the newly composed ensemble's
    reference-graph structure: no cycle, within depth, every reference resolves to an
    existing entry) before it registers; per-dispatch output quality stays Q2's concern
    (§5), unchanged for composer-produced ensembles;
  - (c) the composer is a **verified ensemble, not a lone model**. Whether the compose
    step itself runs as an **orchestration of bounded small-model roles plus
    deterministic verification**, rather than a single capable reasoning process, is an
    open **hypothesis** (the standing ensemble-over-frontier principle, §Context), not
    an evidence-backed claim. essay-004's
    cited evidence backs ensemble-first for *routing/selection*; it assigns *design*
    work (choosing DAG shapes, composing from parts) to the more capable tier and
    reports mixed-model synthesis underperforming on open-ended generation. So the
    baseline and benchmark for the compose step is a capable-model-composed structure,
    and ensemble-decomposition of that step is the standing-principle bet to *validate*
    against the baseline at BUILD/PLAY: held open, not ruled out. The
    attempt-then-escalate decision, where used, is gated by (d)'s deterministic checks,
    never by a small model judging its own output;
  - (d) acceptance is **deterministic** (contract + verification), never
    trust-accumulation;
  - (e) **author-time-with-review first**; per-request runtime composition (on the
    primitive) is the further-out version, gated on evidence it runs reliably
    unattended.

## Rejected alternatives

- **Strategy B (a general ensemble with internal small-model self-routing).** Rejected
  on the evidence llm-conductor's essay-004 cites: the failure is placing routing where
  a small model must execute it. The classify seat is external and declarative
  (Strategy A). *Extension flagged (standing principle, beyond the cited evidence):*
  essay-004's validated Strategy A puts a *capable* model (Claude) in the external
  router seat, and no source it cites tests a small model there. This ADR's position
  that the external router may itself be an orchestration of bounded small-model roles
  is a considered extension of essay-004's external-vs-internal placement axis, driven
  by the standing principle. It is not a claim the cited evidence already establishes.
- **Build the compose-at-runtime primitive now.** Rejected as premature. The catalog
  plus runtime-fill covers the near-term, and AS-11 says extend the engine when a flow
  needs a shape the catalog lacks. None does yet. Held as the named deferred extension.
- **Calibration-grown catalog (llm-conductor's Design → Calibrate → Trust → Promote).**
  Rejected: it is the retired AS-5 trust/promotion machinery wearing a design-laboratory
  coat. Copy the catalog structure, reject the populator.
- **Frontier-model default for routing/selection steps.** Rejected as the default per
  both essay-004's evidence (external SLM routing beats internal self-routing) and the
  standing principle: question what bounded-role ensemble orchestration can do first;
  frontier is a fallback and a benchmark, not the reflex. **For composition/design
  steps this rejection does not hold on the evidence** (essay-004 assigns design to the
  capable tier): there a capable-model-composed structure is the baseline and benchmark,
  and ensemble-first-for-composition is the open hypothesis (§Deferred pillar c), not a
  settled rejection.

## Consequences

**Positive:**
- Near-term extensibility ships on already-shipped primitives; no new engine work is required to reach a curated, composable serving surface.
- The composer-ensemble vision has a bounded path rather than an open question: the compose-at-runtime primitive has a grounded spec (the feasibility spike's four named gaps), while ensemble-first-for-composition (pillar c) remains an open, un-grounded hypothesis under the standing ensemble-over-frontier principle (§Context). The other strategy pillars are design guardrails, not empirical bets. Guardrails (AS-2, deterministic acceptance) keep any future work on the direction clear of the retired machinery.
- The ensemble-over-frontier principle is applied as a design lens, wired into the composer's compose step.

**Negative / cost:**
- The shape-catalog + registry-curation surface (schema, operator UX, AS-2 wiring) is real BUILD work.
- The compose-at-runtime primitive and composer-ensembles are deferred; runtime shape-authorship is a genuine capability gap until they are built.

**Neutral:**
- Turn-time selection (the binding-time hybrid) is available but not the default; revisitable when a deployment shows a need.

## Provenance check

- **Driver-derived:** the registry-plus-catalog shape and the load-time-first binding follow from the runtime-composition feasibility spike and the llm-conductor mining (Strategy A/B; the ADR-019 pattern-library structure). The dissolution, AS-11, and AS-2's survival are from ADR-046 and the domain model.
- **Drafting-time synthesis:** the composer-ensemble strategy pillars are a synthesis of the practitioner's stated vision with the dissolution constraints; the serving-default candidate shapes are not yet chosen (BUILD); the standing ensemble-over-frontier principle is the practitioner's directive applied as a lens.
- **Evidence scope (refined after the R1 argument audit, 2026-07-02):** essay-004's Strategy-A / SLM-routing evidence grounds the *routing/selection* decision (§1–§3) only. It does **not** ground ensemble-first-for-*composition* (pillar c): essay-004 assigns design work to the capable tier and reports mixed-model synthesis underperforming on open-ended generation. Ensemble-first-for-composition is held as the standing-principle hypothesis (ensemble-over-frontier, §Context), deferred and not ruled out. The domain model's AS-6 disposition (2026-07-02 forward note) is the anchor: what is retired is the orchestrator-LLM `compose_ensemble` actor; runtime composition *as a declarative engine capability* remains a live Q4+ direction.
- **Empirical-Grounding Filter (ADR-097):** the near-term decision (catalog + runtime-fill) is spike-grounded (feasibility PASS for fixed-shape, runtime-filled compositions). The composer-ensemble direction is a **named forward direction, not a commitment**: deferred and un-grounded; the compose-at-runtime primitive is spec'd from the spike's four gaps but unbuilt. No feature is committed on research-surfaced possibility alone.
